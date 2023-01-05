#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"

#include <memory>

using namespace mlir;
namespace {
#include "TritonGPUCombine.inc"
using triton::DotOp;
using triton::gpu::ConvertLayoutOp;
using triton::gpu::DotOperandEncodingAttr;
using triton::gpu::MmaEncodingAttr;

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// convert(blocked, dot_operand) ->
// convert(blocked, mma) + convert(mma,  dot_operand)
// if this value is itself the result of a dot operation
// this is a heuristic to accommodate some pattern seen in fused attention
// kernels.
// TODO: replace this by something more generic, i.e. layout-aware CSE
class DecomposeDotOperand : public mlir::RewritePattern {

public:
  explicit DecomposeDotOperand(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<triton::gpu::ConvertLayoutOp>(op))
      return mlir::failure();
    auto convert = llvm::cast<triton::gpu::ConvertLayoutOp>(op);
    auto srcType = convert.getOperand().getType().cast<RankedTensorType>();
    auto dstType = convert.getType().cast<RankedTensorType>();
    if (srcType.getEncoding().isa<triton::gpu::BlockedEncodingAttr>() &&
        dstType.getEncoding().isa<triton::gpu::DotOperandEncodingAttr>()) {
      auto dstDotOperand =
          dstType.getEncoding().cast<triton::gpu::DotOperandEncodingAttr>();
      auto dstParent = dstDotOperand.getParent();
      if (dstDotOperand.getOpIdx() == 1 ||
          !dstParent.isa<triton::gpu::MmaEncodingAttr>())
        return mlir::failure();
      auto dstParentMma = dstParent.cast<triton::gpu::MmaEncodingAttr>();
      if (dstParentMma.isVolta() || dstParentMma.getWarpsPerCTA()[1] > 1)
        return mlir::failure();
      SetVector<Operation *> bwdSlices;
      mlir::getBackwardSlice(convert.getResult(), &bwdSlices);
      if (llvm::find_if(bwdSlices, [](Operation *op) {
            return isa<triton::DotOp>(op);
          }) == bwdSlices.end())
        return mlir::failure();

      auto tmpType = RankedTensorType::get(
          dstType.getShape(), dstType.getElementType(), dstParentMma);
      auto tmp = rewriter.create<triton::gpu::ConvertLayoutOp>(
          convert.getLoc(), tmpType, convert.getOperand());
      auto newConvert = rewriter.create<triton::gpu::ConvertLayoutOp>(
          convert.getLoc(), dstType, tmp);
      rewriter.replaceOp(op, {newConvert});
      return mlir::success();
    }
    return mlir::failure();
  }
};

class SimplifyReduceCvt : public mlir::RewritePattern {
public:
  explicit SimplifyReduceCvt(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::ReduceOp::getOperationName(), 2, context) {
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto reduce = cast<triton::ReduceOp>(*op);
    auto reduceArg = dyn_cast<triton::gpu::ConvertLayoutOp>(
        reduce.getOperand().getDefiningOp());
    if (!reduceArg)
      return mlir::failure();
    // this may generate unsupported conversions in the LLVM codegen
    if (reduceArg.getOperand()
            .getType()
            .cast<RankedTensorType>()
            .getEncoding()
            .isa<triton::gpu::MmaEncodingAttr>())
      return mlir::failure();
    auto newReduce = rewriter.create<triton::ReduceOp>(
        op->getLoc(), reduce.redOp(), reduceArg.getOperand(), reduce.axis());
    if (isa<triton::gpu::ConvertLayoutOp>(
            *reduceArg.getOperand().getDefiningOp()))
      return mlir::failure();
    Value newRet = newReduce.getResult();
    // it's still beneficial to move the conversion
    // to after the reduce if necessary since it will be
    // done on a rank-reduced tensor hence cheaper
    if (newRet.getType() != reduce.getResult().getType())
      newRet = rewriter.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), reduce.getResult().getType(), newRet);
    rewriter.replaceOp(op, newRet);

    return success();
  }
};

// Layout conversions can't deduce their return type automatically.
// IIUC they are therefore not handled by DRR right now
class SimplifyConversion : public mlir::RewritePattern {
public:
  explicit SimplifyConversion(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             4, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<triton::gpu::ConvertLayoutOp>(op))
      return mlir::failure();
    auto convert = llvm::cast<triton::gpu::ConvertLayoutOp>(op);
    // we don't handle conversions to DotOperandEncodingAttr
    // this is a heuristics to accommodate fused attention
    auto srcType = convert.getOperand().getType().cast<RankedTensorType>();
    auto dstType = convert.getType().cast<RankedTensorType>();
    if (dstType.getEncoding().isa<triton::gpu::DotOperandEncodingAttr>() &&
        srcType.getEncoding().isa<triton::gpu::MmaEncodingAttr>())
      return mlir::failure();
    // convert to the same layout -- we can delete
    if (op->getResultTypes() == op->getOperandTypes()) {
      rewriter.replaceOp(op, op->getOperands());
      return mlir::success();
    }
    Operation *arg = op->getOperand(0).getDefiningOp();
    // block argument
    if (!arg)
      return mlir::failure();
    // cvt(alloc_tensor(x), type2) -> alloc_tensor(x, type2)
    auto alloc_tensor = dyn_cast<triton::gpu::AllocTensorOp>(arg);
    if (alloc_tensor) {
      if (!isSharedEncoding(op->getResult(0))) {
        return mlir::failure();
      }
      rewriter.replaceOpWithNewOp<triton::gpu::AllocTensorOp>(
          op, op->getResult(0).getType());
      return mlir::success();
    }
    // cvt(insert_slice(x), type2) -> insert_slice(cvt(x, type2))
    auto insert_slice = dyn_cast<triton::gpu::InsertSliceAsyncOp>(arg);
    if (insert_slice) {
      if (!isSharedEncoding(op->getResult(0))) {
        return mlir::failure();
      }
      auto newType = op->getResult(0).getType().cast<RankedTensorType>();
      // Ensure that the new insert_slice op is placed in the same place as the
      // old insert_slice op. Otherwise, the new insert_slice op may be placed
      // after the async_wait op, which is not allowed.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(insert_slice);
      auto newArg = rewriter.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), newType, insert_slice.dst());
      rewriter.replaceOpWithNewOp<triton::gpu::InsertSliceAsyncOp>(
          op, newType, insert_slice.src(), newArg.getResult(),
          insert_slice.index(), insert_slice.mask(), insert_slice.other(),
          insert_slice.cache(), insert_slice.evict(), insert_slice.isVolatile(),
          insert_slice.axis());
      return mlir::success();
    }
    // cvt(extract_slice(x), type2) -> extract_slice(cvt(x, type2))
    auto extract_slice = dyn_cast<tensor::ExtractSliceOp>(arg);
    if (extract_slice) {
      if (!isSharedEncoding(op->getResult(0))) {
        return mlir::failure();
      }
      auto origType = extract_slice.source().getType().cast<RankedTensorType>();
      auto newType = RankedTensorType::get(
          origType.getShape(), origType.getElementType(),
          op->getResult(0).getType().cast<RankedTensorType>().getEncoding());
      auto origResType = op->getResult(0).getType().cast<RankedTensorType>();
      auto resType = RankedTensorType::get(
          origResType.getShape(), origResType.getElementType(),
          extract_slice.getType().cast<RankedTensorType>().getEncoding());
      // Ensure that the new extract_slice op is placed in the same place as the
      // old extract_slice op. Otherwise, the new extract_slice op may be placed
      // after the async_wait op, which is not allowed.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(extract_slice);
      auto newArg = rewriter.create<triton::gpu::ConvertLayoutOp>(
          op->getLoc(), newType, extract_slice.source());
      rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
          op, resType, newArg.getResult(), extract_slice.offsets(),
          extract_slice.sizes(), extract_slice.strides(),
          extract_slice.static_offsets(), extract_slice.static_sizes(),
          extract_slice.static_strides());
      return mlir::success();
    }

    // cvt(cvt(x, type1), type2) -> cvt(x, type2)
    if (llvm::isa<triton::gpu::ConvertLayoutOp>(arg)) {
      if (arg->getOperand(0).getDefiningOp() &&
          !isSharedEncoding(arg->getOperand(0)) &&
          isSharedEncoding(convert.getOperand()) &&
          !isSharedEncoding(convert.getResult())) {
        return mlir::failure();
      }
      if (isSharedEncoding(convert.getOperand()) &&
          isSharedEncoding(convert.getResult())) {
        return mlir::failure();
      }
      auto srcType = convert.getOperand().getType().cast<RankedTensorType>();
      auto srcShared =
          srcType.getEncoding().dyn_cast<triton::gpu::SharedEncodingAttr>();
      if (srcShared && srcShared.getVec() > 1)
        return mlir::failure();
      rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(
          op, op->getResultTypes().front(), arg->getOperand(0));
      return mlir::success();
    }
    // cvt(type1, splat(type2, x)) -> splat(type1, x)
    if (auto splat = llvm::dyn_cast<triton::SplatOp>(arg)) {
      rewriter.replaceOpWithNewOp<triton::SplatOp>(op, op->getResultTypes(),
                                                   splat.src());
      return mlir::success();
    }
    // cvt(type1, make_range(type2, x)) -> make_range(type1, x)
    if (auto range = llvm::dyn_cast<triton::MakeRangeOp>(arg)) {
      rewriter.replaceOpWithNewOp<triton::MakeRangeOp>(
          op, op->getResultTypes(), range.start(), range.end());
      return mlir::success();
    }
    // cvt(type, constant) -> constant
    if (auto cst = llvm::dyn_cast<arith::ConstantOp>(arg))
      if (auto ret = cst.getValue().dyn_cast<SplatElementsAttr>()) {
        auto newRet = SplatElementsAttr::get(op->getResultTypes().front(),
                                             ret.getSplatValue<Attribute>());
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newRet);
        return mlir::success();
      }
    return mlir::failure();
  }
};

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

// TODO: Interface
LogicalResult invertEncoding(Attribute targetEncoding, Operation *op,
                             Attribute &ret) {
  ret = targetEncoding;
  if (auto expand_dims = dyn_cast<triton::ExpandDimsOp>(op)) {
    ret = triton::gpu::SliceEncodingAttr::get(
        op->getContext(), expand_dims.axis(), targetEncoding);
  }
  if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
    auto sliceEncoding =
        targetEncoding.dyn_cast<triton::gpu::SliceEncodingAttr>();
    if (!sliceEncoding)
      return failure();
    ret = sliceEncoding.getParent();
  }
  return success();
}

// TODO: Interface
LogicalResult getForwardEncoding(Attribute sourceEncoding, Operation *op,
                                 Attribute &ret) {
  if (op->hasTrait<mlir::OpTrait::Elementwise>()) {
    ret = sourceEncoding;
    return success();
  }
  if (isa<triton::ReduceOp>(op)) {
    ret = Attribute();
    return success();
  }
  return failure();
}

inline bool expensive_to_remat(Operation *op) {
  if (!op)
    return true;
  if (isa<tensor::ExtractSliceOp, triton::gpu::AllocTensorOp,
          triton::gpu::InsertSliceAsyncOp, triton::LoadOp, triton::StoreOp,
          triton::AtomicRMWOp, triton::AtomicCASOp, triton::DotOp>(op))
    return true;
  if (isa<scf::YieldOp, scf::ForOp>(op))
    return true;
  return false;
}

LogicalResult simulateBackwardRematerialization(
    Operation *initOp, SetVector<Operation *> &processed,
    SetVector<Attribute> &layout, llvm::MapVector<Value, Attribute> &toConvert,
    Attribute targetEncoding) {
  // DFS
  std::vector<std::pair<Operation *, Attribute>> queue;
  queue.emplace_back(initOp, targetEncoding);
  // We want to see the effect of converting `initOp` to a new layout
  // so we initialize `numCvts = 1`.
  int numCvts = 1;
  while (!queue.empty()) {
    Operation *currOp;
    Attribute currLayout;
    std::tie(currOp, currLayout) = queue.back();
    queue.pop_back();
    // If the current operation is expensive to rematerialize,
    // we stop everything
    if (expensive_to_remat(currOp))
      return mlir::failure();
    // we would propagate the conversion here
    numCvts -= 1;
    // check if the conversion could be folded at this operation
    if (isa<triton::gpu::ConvertLayoutOp, arith::ConstantOp,
            triton::MakeRangeOp, triton::SplatOp>(*currOp))
      continue;
    // done processing
    processed.insert(currOp);
    layout.insert(currLayout);
    // add all operands to the queue
    for (Value argI : currOp->getOperands()) {
      Attribute newEncoding;
      // cannot invert the current encoding for this operand
      // we stop everything
      if (failed(invertEncoding(currLayout, currOp, newEncoding))) {
        return mlir::failure();
      }
      if (toConvert.count(argI) && toConvert[argI] != newEncoding)
        return mlir::failure();
      //
      Operation *opArgI = argI.getDefiningOp();
      toConvert.insert({argI, newEncoding});
      if (!opArgI || processed.contains(opArgI) ||
          (opArgI->getBlock() != initOp->getBlock()))
        continue;
      // we add one expensive conversion for the current operand
      numCvts += 1;
      queue.emplace_back(opArgI, newEncoding);
    }
  }
  // if rematerialization would add more conversions than it removes
  // then we don't do it
  if (numCvts > 0)
    return mlir::failure();
  return mlir::success();
}

//

Operation *cloneWithInferType(mlir::PatternRewriter &rewriter, Operation *op,
                              BlockAndValueMapping &mapping) {
  Operation *newOp = rewriter.clone(*op, mapping);
  auto origType = op->getResult(0).getType().cast<RankedTensorType>();
  auto newType = RankedTensorType::get(
      origType.getShape(), origType.getElementType(),
      newOp->getOperand(0).getType().cast<RankedTensorType>().getEncoding());
  newOp->getResult(0).setType(newType);
  auto typeInfer = dyn_cast<InferTypeOpInterface>(newOp);
  if (typeInfer) {
    SmallVector<Type, 1> newType;
    auto success = typeInfer.inferReturnTypes(
        newOp->getContext(), newOp->getLoc(), newOp->getOperands(),
        newOp->getAttrDictionary(), newOp->getRegions(), newType);
    if (succeeded(success))
      newOp->getResult(0).setType(newType.front());
  }
  return newOp;
}

//
class MoveConvertOutOfIf : public mlir::RewritePattern {
public:
  explicit MoveConvertOutOfIf(mlir::MLIRContext *context)
      : mlir::RewritePattern(scf::IfOp::getOperationName(), 2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ifOp = cast<scf::IfOp>(*op);
    auto thenYield = ifOp.thenYield();
    auto elseYield = ifOp.elseYield();
    int numOps = thenYield.getNumOperands();
    SmallVector<Value> newThenYieldOps = thenYield.getOperands();
    SmallVector<Value> newElseYieldOps = elseYield.getOperands();
    SetVector<Operation *> thenCvts;
    SetVector<Operation *> elseCvts;
    SmallVector<Type> newRetTypes;

    BlockAndValueMapping mapping;
    for (size_t i = 0; i < numOps; i++) {
      auto thenCvt = dyn_cast<triton::gpu::ConvertLayoutOp>(
          thenYield.getOperand(i).getDefiningOp());
      auto elseCvt = dyn_cast<triton::gpu::ConvertLayoutOp>(
          elseYield.getOperand(i).getDefiningOp());
      if (thenCvt && elseCvt &&
          std::distance(thenCvt->user_begin(), thenCvt->user_end()) == 1 &&
          std::distance(elseCvt->user_begin(), elseCvt->user_end()) == 1 &&
          thenCvt.getOperand().getType() == elseCvt.getOperand().getType()) {
        mapping.map(thenCvt.getResult(), thenCvt.getOperand());
        mapping.map(elseCvt.getResult(), elseCvt.getOperand());
        newRetTypes.push_back(thenCvt.getOperand().getType());
        thenCvts.insert((Operation *)thenCvt);
        elseCvts.insert((Operation *)elseCvt);
      } else
        newRetTypes.push_back(thenYield.getOperand(i).getType());
    }
    if (mapping.getValueMap().empty())
      return mlir::failure();

    rewriter.setInsertionPoint(op);
    auto newIfOp = rewriter.create<scf::IfOp>(ifOp.getLoc(), newRetTypes,
                                              ifOp.getCondition(), true);
    // rematerialize `then` block
    rewriter.setInsertionPointToEnd(newIfOp.thenBlock());
    for (Operation &op : ifOp.thenBlock()->getOperations()) {
      if (thenCvts.contains(&op)) {
        mapping.map(op.getResult(0), mapping.lookup(op.getOperand(0)));
        continue;
      }
      rewriter.clone(op, mapping);
    }
    // rematerialize `else` block
    rewriter.setInsertionPointToEnd(newIfOp.elseBlock());
    for (Operation &op : ifOp.elseBlock()->getOperations()) {
      if (elseCvts.contains(&op)) {
        mapping.map(op.getResult(0), mapping.lookup(op.getOperand(0)));
        continue;
      }
      rewriter.clone(op, mapping);
    }

    rewriter.setInsertionPointAfter(newIfOp);
    SmallVector<Value> newRetValues = newIfOp.getResults();
    for (size_t i = 0; i < numOps; i++) {
      if (newIfOp.getResult(i).getType() != ifOp.getResult(i).getType()) {
        newRetValues[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
            newIfOp.getLoc(), ifOp.getResult(i).getType(),
            newIfOp.getResult(i));
      }
    }

    rewriter.replaceOp(op, newRetValues);
    return mlir::success();
  }
};

//
class FoldConvertAndReduce : public mlir::RewritePattern {
public:
  explicit FoldConvertAndReduce(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             1, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *cvtOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(*cvtOp);
    auto srcEncoding =
        cvt.getOperand().getType().cast<RankedTensorType>().getEncoding();
    auto dstEncoding =
        cvt.getResult().getType().cast<RankedTensorType>().getEncoding();
    if (srcEncoding.isa<triton::gpu::SliceEncodingAttr>())
      return failure();
    SetVector<Operation *> cvtSlices;
    auto filter = [&](Operation *op) {
      return op->getBlock() == cvt->getBlock() &&
             !(isa<triton::ReduceOp>(op) &&
               !op->getResult(0).getType().isa<RankedTensorType>()) &&
             !isa<scf::YieldOp>(op);
    };
    mlir::getForwardSlice(cvt.getResult(), &cvtSlices, filter);
    if (cvtSlices.empty())
      return failure();

    llvm::MapVector<Value, Attribute> toConvert;
    for (Operation *op : cvtSlices) {
      // don't rematerialize anything expensive
      if (expensive_to_remat(op))
        return failure();
      // don't rematerialize non-element-wise
      if (!op->hasTrait<mlir::OpTrait::Elementwise>())
        return failure();
      Attribute dstEncoding =
          cvt.getOperand().getType().cast<RankedTensorType>().getEncoding();
      // don't rematerialize if it adds an extra conversion that can't
      // be removed
      for (Value arg : op->getOperands()) {
        Operation *argOp = arg.getDefiningOp();
        SetVector<Operation *> processed;
        SetVector<Attribute> layout;
        llvm::MapVector<Value, Attribute> toConvert;
        if (argOp && (argOp != cvt) && cvtSlices.count(argOp) == 0 &&
            failed(simulateBackwardRematerialization(argOp, processed, layout,
                                                     toConvert, dstEncoding))) {
          return failure();
        }
      }
    }

    BlockAndValueMapping mapping;
    auto op = cvtSlices.front();
    for (Value arg : op->getOperands()) {
      if (arg.getDefiningOp() == cvt)
        mapping.map(arg, cvt.getOperand());
      else {
        auto cvtI = rewriter.create<triton::gpu::ConvertLayoutOp>(
            arg.getLoc(), cvt.getOperand().getType(), arg);
        if (Operation *argOp = arg.getDefiningOp())
          cvtI->moveAfter(argOp);
        mapping.map(arg, cvtI);
      }
    }
    rewriter.setInsertionPoint(op);
    Operation *newOp = rewriter.clone(*op, mapping);
    auto oldType = op->getResult(0).getType().cast<RankedTensorType>();
    auto newType = RankedTensorType::get(
        oldType.getShape(), oldType.getElementType(),
        cvt.getOperand().getType().cast<RankedTensorType>().getEncoding());

    newOp->getResult(0).setType(newType);
    auto newCvtType = RankedTensorType::get(
        oldType.getShape(), oldType.getElementType(),
        cvt.getResult().getType().cast<RankedTensorType>().getEncoding());
    auto newCvt = rewriter.create<triton::gpu::ConvertLayoutOp>(
        newOp->getLoc(), newCvtType, newOp->getResult(0));
    rewriter.replaceOp(op, newCvt->getResults());
    return success();
  }
};

// Layout conversions are expensive. They require going through
// shared memory, which is orders of magnitude slower than
// other non-i/o operations in the dialect.
// It therefore makes sense to remove them whenever possible,
// even if it means rematerializing all values whose definitions
// are reachable from it without passing through any memory operation.
class RematerializeBackward : public mlir::RewritePattern {
public:
  explicit RematerializeBackward(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *cvt,
                  mlir::PatternRewriter &rewriter) const override {
    if (!llvm::isa<triton::gpu::ConvertLayoutOp>(cvt))
      return mlir::failure();
    // we don't touch block arguments
    Operation *op = cvt->getOperand(0).getDefiningOp();
    if (!op)
      return mlir::failure();
    // we don't want to rematerialize any conversion to/from shared
    if (isSharedEncoding(cvt->getResults()[0]) ||
        isSharedEncoding(cvt->getOperand(0)))
      return mlir::failure();
    // we don't handle conversions to DotOperandEncodingAttr
    // this is a heuristics to accommodate fused attention
    auto targetType = cvt->getResultTypes()[0].cast<RankedTensorType>();
    if (targetType.getEncoding().isa<triton::gpu::DotOperandEncodingAttr>())
      return mlir::failure();
    // DFS
    SetVector<Operation *> processed;
    SetVector<Attribute> layout;
    llvm::MapVector<Value, Attribute> toConvert;
    std::vector<std::pair<Operation *, Attribute>> queue;
    queue.emplace_back(cvt, targetType.getEncoding());
    int numCvts = 1;
    while (!queue.empty()) {
      Operation *currOp;
      Attribute currLayout;
      std::tie(currOp, currLayout) = queue.back();
      queue.pop_back();
      // If the current operation is expensive to rematerialize,
      // we stop everything
      if (expensive_to_remat(currOp))
        break;
      // a conversion will be removed here (i.e. transferred to operands)
      numCvts -= 1;
      // done processing
      processed.insert(currOp);
      layout.insert(currLayout);
      // add all operands to the queue
      for (Value argI : currOp->getOperands()) {
        Attribute newEncoding;
        // cannot invert the current encoding for this operand
        // we stop everything
        if (failed(invertEncoding(currLayout, currOp, newEncoding)))
          return mlir::failure();
        if (toConvert.count(argI) && toConvert[argI] != newEncoding)
          return mlir::failure();
        //
        Operation *opArgI = argI.getDefiningOp();
        toConvert.insert({argI, newEncoding});
        if (!opArgI || processed.contains(opArgI) ||
            (opArgI->getBlock() != cvt->getBlock()))
          continue;
        // if the conversion can be folded into opArgI then
        // we don't count this conversion as expensive
        if (isa<triton::gpu::ConvertLayoutOp, arith::ConstantOp,
                triton::MakeRangeOp, triton::SplatOp>(*opArgI))
          continue;
        // we add one expensive conversion for the current operand
        numCvts += 1;
        queue.emplace_back(opArgI, newEncoding);
      }
    }
    // if rematerialization would add more conversions than it removes
    // then we don't do it
    if (numCvts > 0)
      return mlir::failure();

    SmallVector<Value, 4> sortedValues;
    SetVector<Operation *> tmp;
    for (auto &item : toConvert) {
      Value v = item.first;
      if (v.getDefiningOp())
        tmp.insert(v.getDefiningOp());
      else
        sortedValues.push_back(v);
    }
    tmp = mlir::topologicalSort(tmp);
    for (Operation *op : tmp)
      sortedValues.push_back(op->getResult(0));

    BlockAndValueMapping mapping;
    for (Value currOperand : sortedValues) {
      // unpack information
      Attribute targetLayout = toConvert.lookup(currOperand);
      // rematerialize the operand if necessary
      Operation *currOperation = currOperand.getDefiningOp();
      if (processed.contains(currOperation)) {
        currOperation = cloneWithInferType(rewriter, currOperation, mapping);
        currOperand = currOperation->getResult(0);
      }
      // compute target type for the layout cast
      auto currType = currOperand.getType().cast<RankedTensorType>();
      auto newType = RankedTensorType::get(
          currType.getShape(), currType.getElementType(), targetLayout);
      auto newOperand = rewriter.create<triton::gpu::ConvertLayoutOp>(
          currOperand.getLoc(), newType, currOperand);
      if (currOperation)
        newOperand->moveAfter(currOperation);
      mapping.map(currOperand, newOperand);
    }
    rewriter.replaceOp(cvt, mapping.lookup(cvt->getOperand(0)));
    return mlir::success();
  }
};

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

class MoveConvertOutOfLoop : public mlir::RewritePattern {
public:
  explicit MoveConvertOutOfLoop(mlir::MLIRContext *context)
      : mlir::RewritePattern(scf::ForOp::getOperationName(), 1, context) {}

  SmallVector<Value, 4>
  rematerializeForLoop(mlir::PatternRewriter &rewriter, scf::ForOp &forOp,
                       size_t i, RankedTensorType newType,
                       triton::gpu::ConvertLayoutOp origConversion) const {
    // Rewrite init argument
    Type origType = forOp.getInitArgs()[i].getType();
    SmallVector<Value, 4> newInitArgs = forOp.getInitArgs();
    newInitArgs[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
        newInitArgs[i].getLoc(), newType, newInitArgs[i]);
    // Clone for loop
    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInitArgs);
    newForOp->moveBefore(forOp);
    rewriter.setInsertionPointToStart(newForOp.getBody());
    BlockAndValueMapping mapping;
    for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
      mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
    mapping.map(origConversion.getResult(), newForOp.getRegionIterArgs()[i]);
    // the iter arg of interest may have other uses than the conversion
    // we're hoisting out of the loop. If that's the case we will
    // need to add extra conversions for all uses... which is only useful
    // if these extra conversions can be removed by another pattern
    auto oldArg = forOp.getRegionIterArgs()[i];
    auto newArg = newForOp.getRegionIterArgs()[i];
    auto newArgFallback = rewriter.create<triton::gpu::ConvertLayoutOp>(
        newForOp.getLoc(), origType, newArg);

    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (&op == (Operation *)(&origConversion))
        continue;
      Operation *newOp = rewriter.clone(op, mapping);
      if (find(oldArg.getUsers(), &op) != oldArg.getUsers().end())
        newOp->replaceUsesOfWith(newArg, newArgFallback);
    }

    // create yield, inserting conversions if necessary
    auto yieldOp = forOp.getBody()->getTerminator();
    SmallVector<Value, 4> newYieldArgs;
    for (Value arg : yieldOp->getOperands())
      newYieldArgs.push_back(mapping.lookup(arg));
    newYieldArgs[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
        yieldOp->getLoc(), newType, newYieldArgs[i]);
    rewriter.create<scf::YieldOp>(forOp.getLoc(), newYieldArgs);

    // replace
    SmallVector<Value, 4> newResults = newForOp->getResults();
    newResults[i] = rewriter.create<triton::gpu::ConvertLayoutOp>(
        rewriter.getUnknownLoc(), origType, newForOp->getResult(i));
    newResults[i].getDefiningOp()->moveAfter(newForOp);
    return newResults;
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto forOp = cast<scf::ForOp>(op);
    auto iterArgs = forOp.getRegionIterArgs();
    for (const auto &iterArg : llvm::enumerate(iterArgs)) {
      // if (iterArg.index() != 1)
      //   continue;
      // skip non-tensor types
      if (!iterArg.value().getType().isa<RankedTensorType>())
        continue;
      // we only move `iterArg` out of the loop if
      //   - there is only a single conversion use
      //   - moving this conversion out of the loop will not generate
      //     any extra non-removable conversion
      auto users = iterArg.value().getUsers();
      // check first condition
      SetVector<Type> cvtTargetTypes;
      for (auto user : users) {
        if (isa<triton::gpu::ConvertLayoutOp>(user)) {
          auto newType =
              user->getResults()[0].getType().cast<RankedTensorType>();
          auto oldType = user->getOperand(0).getType().cast<RankedTensorType>();
          if (oldType.getEncoding().isa<triton::gpu::SharedEncodingAttr>() &&
              newType.getEncoding()
                  .isa<triton::gpu::DotOperandEncodingAttr>()) {
            continue;
          }
          if (newType.getEncoding().isa<triton::gpu::SharedEncodingAttr>()) {
            if (newType.getEncoding()
                    .cast<triton::gpu::SharedEncodingAttr>()
                    .getVec() == 1)
              continue;
          }
          cvtTargetTypes.insert(newType);
        }
      }
      if (cvtTargetTypes.size() != 1)
        continue;
      // TODO: check second condition
      for (auto user : users) {
        if (isa<triton::gpu::ConvertLayoutOp>(user))
          continue;
      }
      // check
      for (auto op : iterArg.value().getUsers()) {
        auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op);
        if (!cvt)
          continue;
        auto targetType = op->getResultTypes()[0].cast<RankedTensorType>();
        auto newFor = rematerializeForLoop(rewriter, forOp, iterArg.index(),
                                           targetType, cvt);
        rewriter.replaceOp(forOp, newFor);
        return success();
      }
    }
    return failure();
  }
};

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------

class RematerializeForward : public mlir::RewritePattern {
public:
  explicit RematerializeForward(mlir::MLIRContext *context)
      : mlir::RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(),
                             2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *_cvtOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto cvt = cast<triton::gpu::ConvertLayoutOp>(_cvtOp);
    auto forOp = dyn_cast<scf::ForOp>(cvt->getParentOp());
    if (!forOp)
      return mlir::failure();
    auto isInLoop = [&](Operation *op) { return op->getParentOp() == forOp; };

    SetVector<Operation *> cvtSlices;
    auto filter = [&](Operation *op) {
      return isInLoop(op) &&
             !isa<triton::LoadOp, triton::StoreOp, triton::AtomicRMWOp,
                  triton::AtomicCASOp>(op) &&
             !isa<triton::DotOp>(op) && !isa<scf::YieldOp>(op) &&
             !isa<triton::gpu::ConvertLayoutOp>(op);
    };
    mlir::getForwardSlice(cvt.getResult(), &cvtSlices, filter);
    if (cvtSlices.empty())
      return failure();

    for (Operation *op : cvtSlices) {
      if (!op->hasTrait<mlir::OpTrait::SameOperandsAndResultEncoding>() &&
          !op->hasTrait<mlir::OpTrait::SameOperandsAndResultType>())
        return failure();
      for (Value arg : op->getOperands()) {
        Operation *argOp = arg.getDefiningOp();
        if (argOp && (argOp != cvt) &&
            !isa<arith::ConstantOp, triton::SplatOp>(argOp)) {
          return failure();
        }
      }
    }

    // otherwise, we push the conversion forward
    // since we'll be able to move it out of
    // the loop once it reaches the yield op
    // op(cvt(arg_0), arg_1, ..., arg_n)
    // -> cvt(op(arg_0, cvt(arg_1), ..., cvt(arg_n)))
    BlockAndValueMapping mapping;
    auto op = cvtSlices.front();
    for (Value arg : op->getOperands()) {
      if (arg.getDefiningOp() == cvt)
        mapping.map(arg, cvt.getOperand());
      else {
        auto cvtI = rewriter.create<triton::gpu::ConvertLayoutOp>(
            arg.getLoc(), cvt.getOperand().getType(), arg);
        mapping.map(arg, cvtI);
      }
    }
    Operation *newOp = rewriter.clone(*op, mapping);
    newOp->getResult(0).setType(cvt.getOperand().getType());
    auto newCvt = rewriter.create<triton::gpu::ConvertLayoutOp>(
        newOp->getLoc(), cvt.getResult().getType(), newOp->getResult(0));
    rewriter.replaceOp(op, newCvt->getResults());
    return success();
  }
};

// -----------------------------------------------------------------------------
//
// -----------------------------------------------------------------------------
namespace {
int computeCapabilityToMMAVersion(int computeCapability) {
  if (computeCapability < 70) {
    return 0;
  } else if (computeCapability < 80) {
    return 1;
  } else if (computeCapability < 90) {
    return 2;
  } else {
    assert(false && "computeCapability > 90 not supported");
    return 3;
  }
}

SmallVector<int64_t, 2> mmaVersionToShapePerWarp(int version) {
  if (version == 1)
    return {16, 16};
  else if (version == 2)
    return {16, 8};
  else {
    assert(false && "version not supported");
    return {0, 0};
  }
}

SmallVector<unsigned, 2> warpsPerTileV1(const ArrayRef<int64_t> shape,
                                        int numWarps) {
  SmallVector<unsigned, 2> ret = {1, 1};
  SmallVector<int64_t, 2> shapePerWarp =
      mmaVersionToShapePerWarp(1 /*version*/);
  bool changed = false;
  do {
    changed = false;
    int pre = ret[0];
    if (ret[0] * ret[1] < numWarps) {
      ret[0] = std::clamp<unsigned>(ret[0] * 2, 1, shape[0] / shapePerWarp[0]);
      changed = pre != ret[0];
    }
    if (ret[0] * ret[1] < numWarps) {
      pre = ret[1];
      ret[1] = std::clamp<unsigned>(ret[1] * 2, 1, shape[1] / shapePerWarp[1]);
      changed = pre != ret[1];
    }
  } while (changed);
  return ret;
}

SmallVector<unsigned, 2> warpsPerTileV2(triton::DotOp dotOp,
                                        const ArrayRef<int64_t> shape,
                                        int numWarps) {
  SetVector<Operation *> slices;
  mlir::getForwardSlice(dotOp.getResult(), &slices);
  if (llvm::find_if(slices, [](Operation *op) {
        return isa<triton::DotOp>(op);
      }) != slices.end())
    return {(unsigned)numWarps, 1};

  SmallVector<unsigned, 2> ret = {1, 1};
  SmallVector<int64_t, 2> shapePerWarp = {16, 8};
  bool changed = false;
  // TODO (@daadaada): double-check.
  // original logic in
  // https://github.com/openai/triton/blob/master/lib/codegen/analysis/layout.cc#L252
  // seems buggy for shape = [32, 16] ?
  do {
    changed = false;
    if (ret[0] * ret[1] >= numWarps)
      break;
    if (shape[0] / shapePerWarp[0] / ret[0] >=
        shape[1] / (shapePerWarp[1] * 2) / ret[1]) {
      if (ret[0] < shape[0] / shapePerWarp[0]) {
        ret[0] *= 2;
      } else
        ret[1] *= 2;
    } else {
      ret[1] *= 2;
    }
  } while (true);
  return ret;
}

} // namespace

class OptimizeBlockedToShared : public mlir::RewritePattern {
public:
  explicit OptimizeBlockedToShared(mlir::MLIRContext *context)
      : RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(), 1,
                       context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cvt = cast<triton::gpu::ConvertLayoutOp>(op);
    auto srcType = cvt.getOperand().getType().cast<RankedTensorType>();
    auto dstType = cvt.getResult().getType().cast<RankedTensorType>();
    auto srcBlockedLayout =
        srcType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>();
    auto dstSharedLayout =
        dstType.getEncoding().dyn_cast<triton::gpu::SharedEncodingAttr>();
    if (!srcBlockedLayout || !dstSharedLayout)
      return failure();
    if (srcBlockedLayout.getOrder() == dstSharedLayout.getOrder())
      return failure();
    // For now only works if single use is transpose
    // TODO: rematerialize #shared uses
    auto users = op->getUsers();
    if (std::distance(users.begin(), users.end()) != 1 ||
        !isa<triton::TransOp>(*users.begin()))
      return failure();

    auto tmpShared = triton::gpu::SharedEncodingAttr::get(
        op->getContext(), dstSharedLayout.getVec(),
        dstSharedLayout.getPerPhase(), dstSharedLayout.getMaxPhase(),
        srcBlockedLayout.getOrder());
    auto tmpType = RankedTensorType::get(srcType.getShape(),
                                         srcType.getElementType(), tmpShared);
    auto tmpCvt = rewriter.create<triton::gpu::ConvertLayoutOp>(
        op->getLoc(), tmpType, cvt.getOperand());

    auto newDstType = RankedTensorType::get(
        users.begin()->getResultTypes()[0].cast<RankedTensorType>().getShape(),
        srcType.getElementType(), dstSharedLayout);

    auto newTrans = rewriter.create<triton::TransOp>(op->getLoc(), newDstType,
                                                     tmpCvt.getResult());

    rewriter.replaceOp(*users.begin(), newTrans.getResult());
    return success();
  }
};

class OptimizeConvertToDotOperand : public mlir::RewritePattern {
public:
  explicit OptimizeConvertToDotOperand(mlir::MLIRContext *context)
      : RewritePattern(triton::gpu::ConvertLayoutOp::getOperationName(), 1,
                       context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cvt = cast<triton::gpu::ConvertLayoutOp>(op);
    auto srcType = cvt.getOperand().getType().cast<RankedTensorType>();
    auto dstType = cvt.getResult().getType().cast<RankedTensorType>();
    // order
    ArrayRef<unsigned> order;
    if (auto srcBlockedLayout =
            srcType.getEncoding().dyn_cast<triton::gpu::BlockedEncodingAttr>())
      order = srcBlockedLayout.getOrder();
    else if (auto srcSharedLayout =
                 srcType.getEncoding()
                     .dyn_cast<triton::gpu::SharedEncodingAttr>())
      order = srcSharedLayout.getOrder();
    else
      return failure();
    // dot operand output
    auto dstDotOperandLayout =
        dstType.getEncoding().dyn_cast<triton::gpu::DotOperandEncodingAttr>();
    if (!dstDotOperandLayout)
      return failure();
    if (!dstDotOperandLayout.getIsMMAv1Row())
      return failure();
    bool isMMAv1Row =
        dstDotOperandLayout.getIsMMAv1Row().cast<BoolAttr>().getValue();
    if ((order[0] == 1 && isMMAv1Row) || (order[0] == 0 && !isMMAv1Row))
      return failure();

    auto newIsRow = BoolAttr::get(op->getContext(), !isMMAv1Row);
    auto newDstEncoding = triton::gpu::DotOperandEncodingAttr::get(
        op->getContext(), dstDotOperandLayout.getOpIdx(),
        dstDotOperandLayout.getParent(), newIsRow);
    auto newDstType = RankedTensorType::get(
        dstType.getShape(), dstType.getElementType(), newDstEncoding);
    auto newCvt = rewriter.create<triton::gpu::ConvertLayoutOp>(
        op->getLoc(), newDstType, cvt.getOperand());
    rewriter.replaceOp(op, newCvt.getResult());
    return success();
  }
};

class BlockedToMMA : public mlir::RewritePattern {
  int computeCapability;

public:
  BlockedToMMA(mlir::MLIRContext *context, int computeCapability)
      : mlir::RewritePattern(triton::DotOp::getOperationName(), 2, context),
        computeCapability(computeCapability) {}

  static SmallVector<unsigned, 2> getWarpsPerTile(triton::DotOp dotOp,
                                                  const ArrayRef<int64_t> shape,
                                                  int version, int numWarps) {
    switch (version) {
    case 1:
      return warpsPerTileV1(shape, numWarps);
    case 2:
      return warpsPerTileV2(dotOp, shape, numWarps);
    default:
      assert(false && "not supported version");
      return {0, 0};
    }
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto dotOp = cast<triton::DotOp>(op);
    // TODO: Check data-types and SM compatibility
    auto oldRetType = dotOp.getResult().getType().cast<RankedTensorType>();
    if (!oldRetType.getEncoding() ||
        oldRetType.getEncoding().isa<triton::gpu::MmaEncodingAttr>())
      return failure();

    auto AType = dotOp.getOperand(0).getType().cast<RankedTensorType>();
    auto BType = dotOp.getOperand(1).getType().cast<RankedTensorType>();

    // for FMA, should retain the blocked layout.
    int versionMajor = computeCapabilityToMMAVersion(computeCapability);
    if (!supportMMA(dotOp, versionMajor))
      return failure();

    auto AOrder = AType.getEncoding()
                      .cast<triton::gpu::DotOperandEncodingAttr>()
                      .getParent()
                      .cast<triton::gpu::BlockedEncodingAttr>()
                      .getOrder();
    auto BOrder = BType.getEncoding()
                      .cast<triton::gpu::DotOperandEncodingAttr>()
                      .getParent()
                      .cast<triton::gpu::BlockedEncodingAttr>()
                      .getOrder();

    // get MMA encoding for the given number of warps
    auto retShape = oldRetType.getShape();
    auto mod = op->getParentOfType<mlir::ModuleOp>();
    int numWarps = triton::gpu::TritonGPUDialect::getNumWarps(mod);

    auto warpsPerTile =
        getWarpsPerTile(dotOp, retShape, versionMajor, numWarps);
    triton::gpu::MmaEncodingAttr mmaEnc;
    if (versionMajor == 1) {
      auto shapeA = AType.getShape();
      auto shapeB = BType.getShape();
      bool isARow = AOrder[0] != 0;
      bool isBRow = BOrder[0] != 0;
      mmaEnc = triton::gpu::MmaEncodingAttr::get(
          oldRetType.getContext(), versionMajor, warpsPerTile, shapeA, shapeB,
          isARow, isBRow);
    } else if (versionMajor == 2) {
      mmaEnc = triton::gpu::MmaEncodingAttr::get(
          oldRetType.getContext(), versionMajor, 0 /*versionMinor*/,
          warpsPerTile);
    } else {
      assert(false && "Mma layout only support versionMajor of 1 or 2");
    }
    auto newRetType =
        RankedTensorType::get(retShape, oldRetType.getElementType(), mmaEnc);

    // convert accumulator
    auto oldAcc = dotOp.getOperand(2);
    auto newAcc = rewriter.create<triton::gpu::ConvertLayoutOp>(
        oldAcc.getLoc(), newRetType, oldAcc);
    Value a = dotOp.a();
    Value b = dotOp.b();
    auto oldAType = a.getType().cast<RankedTensorType>();
    auto oldBType = b.getType().cast<RankedTensorType>();
    auto oldAOrder = oldAType.getEncoding()
                         .cast<triton::gpu::DotOperandEncodingAttr>()
                         .getParent()
                         .cast<triton::gpu::BlockedEncodingAttr>()
                         .getOrder();
    auto oldBOrder = oldBType.getEncoding()
                         .cast<triton::gpu::DotOperandEncodingAttr>()
                         .getParent()
                         .cast<triton::gpu::BlockedEncodingAttr>()
                         .getOrder();
    Attribute isMMAv1RowA;
    Attribute isMMAv1RowB;
    if (versionMajor == 1) {
      isMMAv1RowA = BoolAttr::get(getContext(), oldAOrder[0] == 1);
      isMMAv1RowB = BoolAttr::get(getContext(), oldBOrder[0] == 1);
    }

    auto newAType = RankedTensorType::get(
        oldAType.getShape(), oldAType.getElementType(),
        triton::gpu::DotOperandEncodingAttr::get(
            oldAType.getContext(), 0, newRetType.getEncoding(), isMMAv1RowA));
    auto newBType = RankedTensorType::get(
        oldBType.getShape(), oldBType.getElementType(),
        triton::gpu::DotOperandEncodingAttr::get(
            oldBType.getContext(), 1, newRetType.getEncoding(), isMMAv1RowB));

    a = rewriter.create<triton::gpu::ConvertLayoutOp>(a.getLoc(), newAType, a);
    b = rewriter.create<triton::gpu::ConvertLayoutOp>(b.getLoc(), newBType, b);
    auto newDot = rewriter.create<triton::DotOp>(dotOp.getLoc(), newRetType, a,
                                                 b, newAcc, dotOp.allowTF32());

    rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(
        op, oldRetType, newDot.getResult());
    return success();
  }
};

class FixupLoop : public mlir::RewritePattern {

public:
  explicit FixupLoop(mlir::MLIRContext *context)
      : mlir::RewritePattern(scf::ForOp::getOperationName(), 2, context) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto forOp = cast<scf::ForOp>(op);

    // Rewrite init argument
    SmallVector<Value, 4> newInitArgs = forOp.getInitArgs();
    bool shouldRematerialize = false;
    for (size_t i = 0; i < newInitArgs.size(); i++) {
      auto initArg = newInitArgs[i];
      auto regionArg = forOp.getRegionIterArgs()[i];
      if (newInitArgs[i].getType() != forOp.getRegionIterArgs()[i].getType() ||
          newInitArgs[i].getType() != forOp.getResultTypes()[i]) {
        shouldRematerialize = true;
        break;
      }
    }
    if (!shouldRematerialize)
      return failure();

    scf::ForOp newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInitArgs);
    newForOp->moveBefore(forOp);
    rewriter.setInsertionPointToStart(newForOp.getBody());
    BlockAndValueMapping mapping;
    for (const auto &arg : llvm::enumerate(forOp.getRegionIterArgs()))
      mapping.map(arg.value(), newForOp.getRegionIterArgs()[arg.index()]);
    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());

    for (Operation &op : forOp.getBody()->getOperations()) {
      rewriter.clone(op, mapping);
    }
    rewriter.replaceOp(forOp, newForOp.getResults());
    return success();
  }
};

// This pattern collects the wrong Mma those need to update and create the right
// ones for each.
class CollectMmaToUpdateForVolta : public mlir::RewritePattern {
  DenseMap<MmaEncodingAttr, MmaEncodingAttr> &mmaToUpdate;

public:
  CollectMmaToUpdateForVolta(
      mlir::MLIRContext *ctx,
      DenseMap<MmaEncodingAttr, MmaEncodingAttr> &mmaToUpdate)
      : mlir::RewritePattern(triton::DotOp::getOperationName(), 1, ctx),
        mmaToUpdate(mmaToUpdate) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {

    auto dotOp = cast<triton::DotOp>(op);
    auto *ctx = dotOp->getContext();
    auto AT = dotOp.a().getType().cast<RankedTensorType>();
    auto BT = dotOp.b().getType().cast<RankedTensorType>();
    auto DT = dotOp.d().getType().cast<RankedTensorType>();
    if (!DT.getEncoding())
      return failure();
    auto mmaLayout = DT.getEncoding().dyn_cast<MmaEncodingAttr>();
    if (!(mmaLayout && mmaLayout.isVolta()))
      return failure();

    // Has processed.
    if (mmaToUpdate.count(mmaLayout))
      return failure();

    auto dotOperandA = AT.getEncoding().cast<DotOperandEncodingAttr>();
    auto dotOperandB = BT.getEncoding().cast<DotOperandEncodingAttr>();
    bool isARow = dotOperandA.getIsMMAv1Row().cast<BoolAttr>().getValue();
    bool isBRow = dotOperandB.getIsMMAv1Row().cast<BoolAttr>().getValue();
    auto [isARow_, isBRow_, isAVec4, isBVec4] =
        mmaLayout.decodeVoltaLayoutStates();
    if (isARow_ == isARow && isBRow_ == isBRow) {
      return failure(); // No need to update
    }

    auto newMmaLayout = MmaEncodingAttr::get(
        ctx, mmaLayout.getVersionMajor(), mmaLayout.getWarpsPerCTA(),
        AT.getShape(), BT.getShape(), isARow, isBRow);

    // Collect the wrong MMA Layouts, and mark need to update.
    mmaToUpdate.try_emplace(mmaLayout, newMmaLayout);

    return failure();
  }
};

// Correct the versionMinor field in MmaEncodingAttr for Volta.
class UpdateMMAVersionMinorForVolta : public mlir::RewritePattern {
  const DenseMap<MmaEncodingAttr, MmaEncodingAttr> &mmaToUpdate;
  enum class Kind {
    kUnk,
    kCvtToMma,
    kCvtToDotOp,
    kDot,
    kConstant,
  };
  mutable Kind rewriteKind{Kind::kUnk};

public:
  UpdateMMAVersionMinorForVolta(
      mlir::MLIRContext *ctx, llvm::StringRef opName,
      const DenseMap<MmaEncodingAttr, MmaEncodingAttr> &mmaToUpdate)
      : RewritePattern(opName, 1 /*benefit*/, ctx), mmaToUpdate(mmaToUpdate) {}

  LogicalResult match(Operation *op) const override {
    MmaEncodingAttr mma;
    if (mmaToUpdate.empty())
      return failure();
    if (op->getNumResults() != 1)
      return failure();
    auto tensorTy = op->getResult(0).getType().dyn_cast<RankedTensorType>();
    if (!tensorTy)
      return failure();

    // ConvertLayoutOp
    if (auto cvt = llvm::dyn_cast<ConvertLayoutOp>(op)) {
      // cvt X -> dot_operand
      if (auto dotOperand =
              tensorTy.getEncoding().dyn_cast<DotOperandEncodingAttr>()) {
        mma = dotOperand.getParent().dyn_cast<MmaEncodingAttr>();
        rewriteKind = Kind::kCvtToDotOp;
        if (mma && mmaToUpdate.count(mma))
          return success();
      }
      if ((mma = tensorTy.getEncoding().dyn_cast<MmaEncodingAttr>())) {
        // cvt X -> mma
        rewriteKind = Kind::kCvtToMma;
        if (mma && mmaToUpdate.count(mma))
          return success();
      }
    } else if (auto dot = llvm::dyn_cast<DotOp>(op)) {
      // DotOp
      mma = dot.d()
                .getType()
                .cast<RankedTensorType>()
                .getEncoding()
                .dyn_cast<MmaEncodingAttr>();
      rewriteKind = Kind::kDot;
    } else if (auto constant = llvm::dyn_cast<arith::ConstantOp>(op)) {
      // ConstantOp
      mma = tensorTy.getEncoding().dyn_cast<MmaEncodingAttr>();
      rewriteKind = Kind::kConstant;
    }

    return success(mma && mmaToUpdate.count(mma));
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    switch (rewriteKind) {
    case Kind::kDot:
      rewriteDot(op, rewriter);
      break;
    case Kind::kConstant:
      rewriteConstant(op, rewriter);
      break;
    case Kind::kCvtToDotOp:
      rewriteCvtDotOp(op, rewriter);
      break;
    case Kind::kCvtToMma:
      rewriteCvtToMma(op, rewriter);
      break;
    default:
      llvm::report_fatal_error("Not supported rewrite kind");
    }
  }

private:
  void rewriteCvtDotOp(Operation *op, PatternRewriter &rewriter) const {
    auto *ctx = op->getContext();
    auto cvt = llvm::cast<ConvertLayoutOp>(op);
    auto tensorTy = cvt.result().getType().cast<RankedTensorType>();
    auto dotOperand = tensorTy.getEncoding().cast<DotOperandEncodingAttr>();
    MmaEncodingAttr newMma =
        mmaToUpdate.lookup(dotOperand.getParent().cast<MmaEncodingAttr>());
    auto newDotOperand = DotOperandEncodingAttr::get(
        ctx, dotOperand.getOpIdx(), newMma, dotOperand.getIsMMAv1Row());
    auto newTensorTy = RankedTensorType::get(
        tensorTy.getShape(), tensorTy.getElementType(), newDotOperand);
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(op, newTensorTy,
                                                 cvt.getOperand());
  }

  void rewriteDot(Operation *op, PatternRewriter &rewriter) const {
    auto *ctx = op->getContext();
    auto dot = llvm::cast<DotOp>(op);
    auto tensorTy = dot.d().getType().cast<RankedTensorType>();
    auto mma = tensorTy.getEncoding().cast<MmaEncodingAttr>();
    auto newMma = mmaToUpdate.lookup(mma);
    auto newTensorTy = RankedTensorType::get(tensorTy.getShape(),
                                             tensorTy.getElementType(), newMma);
    rewriter.replaceOpWithNewOp<DotOp>(op, newTensorTy, dot.a(), dot.b(),
                                       dot.c(), dot.allowTF32());
  }

  void rewriteCvtToMma(Operation *op, PatternRewriter &rewriter) const {
    auto *ctx = op->getContext();
    auto cvt = llvm::cast<ConvertLayoutOp>(op);
    auto tensorTy = cvt.result().getType().cast<RankedTensorType>();
    auto mma = tensorTy.getEncoding().cast<MmaEncodingAttr>();
    auto newMma = mmaToUpdate.lookup(mma);
    auto newTensorTy = RankedTensorType::get(tensorTy.getShape(),
                                             tensorTy.getElementType(), newMma);
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(op, newTensorTy,
                                                 cvt.getOperand());
  }

  void rewriteConstant(Operation *op, PatternRewriter &rewriter) const {
    auto *ctx = op->getContext();
    auto constant = llvm::cast<arith::ConstantOp>(op);
    auto tensorTy = constant.getResult().getType().dyn_cast<RankedTensorType>();
    auto mma = tensorTy.getEncoding().cast<MmaEncodingAttr>();
    auto newMma = mmaToUpdate.lookup(mma);
    auto newTensorTy = RankedTensorType::get(tensorTy.getShape(),
                                             tensorTy.getElementType(), newMma);
    if (auto attr = constant.getValue().dyn_cast<SplatElementsAttr>()) {
      auto newRet =
          SplatElementsAttr::get(newTensorTy, attr.getSplatValue<Attribute>());
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, newTensorTy, newRet);
      return;
    }

    assert(false && "Not supported ConstantOp value type");
  }
};

} // namespace

#define GEN_PASS_CLASSES
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUCombineOpsPass
    : public TritonGPUCombineOpsBase<TritonGPUCombineOpsPass> {
public:
  TritonGPUCombineOpsPass() = default;
  TritonGPUCombineOpsPass(int computeCapability) {
    this->computeCapability = computeCapability;
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();

    mlir::RewritePatternSet patterns(context);

    patterns.add<OptimizeBlockedToShared>(context);
    patterns.add<OptimizeConvertToDotOperand>(context);
    patterns.add<SimplifyConversion>(context);
    patterns.add<SimplifyReduceCvt>(context);
    patterns.add<FoldConvertAndReduce>(context);
    patterns.add<DecomposeDotOperand>(context);
    patterns.add<RematerializeBackward>(context);
    patterns.add<RematerializeForward>(context);
    patterns.add<MoveConvertOutOfLoop>(context);
    patterns.add<MoveConvertOutOfIf>(context);
    patterns.add<BlockedToMMA>(context, computeCapability);

    if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
    }

    llvm::DenseMap<MmaEncodingAttr, MmaEncodingAttr> mmaToUpdate;
    {
      mlir::RewritePatternSet patterns(context);
      patterns.add<CollectMmaToUpdateForVolta>(context, mmaToUpdate);
      if (applyPatternsAndFoldGreedily(m, std::move(patterns)).failed())
        signalPassFailure();
    }
    {
      mlir::RewritePatternSet patterns(context);
      patterns.add<UpdateMMAVersionMinorForVolta>(
          context, DotOp::getOperationName(), mmaToUpdate);
      patterns.add<UpdateMMAVersionMinorForVolta>(
          context, ConvertLayoutOp::getOperationName(), mmaToUpdate);
      patterns.add<UpdateMMAVersionMinorForVolta>(
          context, arith::ConstantOp::getOperationName(), mmaToUpdate);
      mlir::GreedyRewriteConfig config;
      config.useTopDownTraversal = true;

      if (applyPatternsAndFoldGreedily(m, std::move(patterns), config).failed())
        signalPassFailure();
    }

    mlir::RewritePatternSet loopFixup(context);
    loopFixup.add<FixupLoop>(context);
    if (applyPatternsAndFoldGreedily(m, std::move(loopFixup)).failed()) {
      signalPassFailure();
    }
  }
};

std::unique_ptr<Pass>
mlir::createTritonGPUCombineOpsPass(int computeCapability) {
  return std::make_unique<TritonGPUCombineOpsPass>(computeCapability);
}
