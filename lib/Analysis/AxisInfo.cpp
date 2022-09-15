#include "mlir/Analysis/DataFlowAnalysis.h"
#include "llvm/Support/raw_ostream.h"
#include <iostream>

#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// AxisInfo
//===----------------------------------------------------------------------===//

// Function for extended Euclidean Algorithm
static int gcd_impl(int a, int b, int *x, int *y) {
  // Base Case
  if (a == 0) {
    *x = 0;
    *y = 1;
    return b;
  }
  int x1, y1; // To store results of recursive call
  int gcd = gcd_impl(b % a, a, &x1, &y1);
  // Update x and y using results of
  // recursive call
  *x = y1 - (b / a) * x1;
  *y = x1;
  return gcd;
}

static int gcd(int a, int b) {
  int x, y;
  return gcd_impl(a, b, &x, &y);
}

AxisInfo AxisInfo::getPessimisticValueState(Value value) {
  size_t rank = 1;
  if (TensorType ty = value.getType().dyn_cast<TensorType>())
    rank = ty.getRank();
  int divHint = 1;
  if (BlockArgument blockArg = value.dyn_cast<BlockArgument>()) {
    Operation *op = blockArg.getOwner()->getParentOp();
    if (FuncOp fun = dyn_cast<FuncOp>(op)) {
      Attribute attr =
          fun.getArgAttr(blockArg.getArgNumber(), "tt.divisibility");
      if (attr)
        divHint = attr.cast<IntegerAttr>().getValue().getZExtValue();
    }
  }
  DimVectorT contiguity(rank, 1);
  DimVectorT divisibility(rank, divHint);
  DimVectorT constancy(rank, 1);
  return AxisInfo(contiguity, divisibility, constancy);
}

// The gcd of both arguments for each dimension
AxisInfo AxisInfo::join(const AxisInfo &lhs, const AxisInfo &rhs) {
  DimVectorT retContiguity;
  DimVectorT retDivisibility;
  DimVectorT retConstancy;
  for (size_t d = 0; d < lhs.getRank(); d++) {
    retContiguity.push_back(gcd(lhs.getContiguity(d), rhs.getContiguity(d)));
    retDivisibility.push_back(
        gcd(lhs.getDivisibility(d), rhs.getDivisibility(d)));
    retConstancy.push_back(gcd(lhs.getConstancy(d), rhs.getConstancy(d)));
  }
  return AxisInfo(retContiguity, retDivisibility, retConstancy);
}

//===----------------------------------------------------------------------===//
// AxisInfoAnalysis
//===----------------------------------------------------------------------===//

AxisInfo AxisInfoAnalysis::visitBinaryOp(
    Operation *op, AxisInfo lhsInfo, AxisInfo rhsInfo,
    const std::function<int(AxisInfo, AxisInfo, int)> &getContiguity,
    const std::function<int(AxisInfo, AxisInfo, int)> &getDivisibility,
    const std::function<int(AxisInfo, AxisInfo, int)> &getConstancy) {
  int rank = lhsInfo.getRank();
  AxisInfo::DimVectorT newContiguity;
  AxisInfo::DimVectorT newDivisibility;
  AxisInfo::DimVectorT newConstancy;
  for (size_t d = 0; d < rank; d++) {
    newContiguity.push_back(getContiguity(lhsInfo, rhsInfo, d));
    newDivisibility.push_back(getDivisibility(lhsInfo, rhsInfo, d));
    newConstancy.push_back(getConstancy(lhsInfo, rhsInfo, d));
  }
  return AxisInfo(newContiguity, newDivisibility, newConstancy);
}

ChangeResult AxisInfoAnalysis::visitOperation(
    Operation *op, ArrayRef<LatticeElement<AxisInfo> *> operands) {
  AxisInfo curr;
  // This preserves the input axes (e.g., cast):
  if (llvm::isa<arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
                triton::PtrToIntOp, triton::IntToPtrOp,
                triton::gpu::ConvertLayoutOp>(op))
    curr = operands[0]->getValue();
  // Constant ranges
  if (triton::MakeRangeOp make_range =
          llvm::dyn_cast<triton::MakeRangeOp>(op)) {
    int start = make_range.start();
    int end = make_range.end();
    AxisInfo::DimVectorT contiguity = {end - start};
    AxisInfo::DimVectorT divisibility = {highestPowOf2Divisor(start)};
    AxisInfo::DimVectorT constancy = {1};
    curr = AxisInfo(contiguity, divisibility, constancy);
  }
  // Constant
  if (arith::ConstantOp constant = llvm::dyn_cast<arith::ConstantOp>(op)) {
    auto intAttr = constant.getValue().dyn_cast<IntegerAttr>();
    if (intAttr) {
      size_t val = intAttr.getValue().getZExtValue();
      curr = AxisInfo({1}, {highestPowOf2Divisor(val)}, {1});
    }
    // TODO: generalize to dense attr
    auto splatAttr = constant.getValue().dyn_cast<SplatElementsAttr>();
    if (splatAttr && splatAttr.getElementType().isInteger(32)) {
      auto value = splatAttr.getSplatValue<int>();
      TensorType ty = splatAttr.getType().cast<TensorType>();
      curr = AxisInfo(
          AxisInfo::DimVectorT(ty.getRank(), 1),
          AxisInfo::DimVectorT(ty.getRank(), highestPowOf2Divisor(value)),
          AxisInfo::DimVectorT(ty.getShape().begin(), ty.getShape().end()));
    }
  }
  // Addition
  if (llvm::isa<arith::AddIOp, triton::AddPtrOp>(op)) {
    auto newContiguity = [&](AxisInfo lhs, AxisInfo rhs, int d) {
      return std::max(gcd(lhs.getContiguity(d), rhs.getConstancy(d)),
                      gcd(lhs.getConstancy(d), rhs.getContiguity(d)));
    };
    auto newConstancy = [&](AxisInfo lhs, AxisInfo rhs, int d) {
      return gcd(lhs.getConstancy(d), rhs.getConstancy(d));
    };
    auto newDivisibility = [&](AxisInfo lhs, AxisInfo rhs, int d) {
      return gcd(lhs.getDivisibility(d), rhs.getDivisibility(d));
    };
    curr = visitBinaryOp(op, operands[0]->getValue(), operands[1]->getValue(),
                         newContiguity, newDivisibility, newConstancy);
  }
  // Multiplication
  if (llvm::isa<arith::MulIOp>(op)) {
    auto newContiguity = [](AxisInfo lhs, AxisInfo rhs, int d) { return 1; };
    auto newConstancy = [](AxisInfo lhs, AxisInfo rhs, int d) {
      return gcd(lhs.getConstancy(d), rhs.getConstancy(d));
    };
    auto newDivisibility = [](AxisInfo lhs, AxisInfo rhs, int d) {
      return lhs.getDivisibility(d) * rhs.getDivisibility(d);
    };
    curr = visitBinaryOp(op, operands[0]->getValue(), operands[1]->getValue(),
                         newContiguity, newDivisibility, newConstancy);
  }
  // Splat
  if (llvm::isa<triton::SplatOp>(op)) {
    Type _retTy = *op->result_type_begin();
    TensorType retTy = _retTy.cast<TensorType>();
    AxisInfo opInfo = operands[0]->getValue();
    AxisInfo::DimVectorT contiguity;
    AxisInfo::DimVectorT divisibility;
    AxisInfo::DimVectorT constancy;
    for (size_t d = 0; d < retTy.getRank(); d++) {
      contiguity.push_back(1);
      divisibility.push_back(opInfo.getDivisibility(0));
      constancy.push_back(retTy.getShape()[d]);
    }
    curr = AxisInfo(contiguity, divisibility, constancy);
  }
  // expandDims
  if (auto expandDims = llvm::dyn_cast<triton::ExpandDimsOp>(op)) {
    Type _retTy = *op->result_type_begin();
    Type _opTy = *op->operand_type_begin();
    TensorType retTy = _retTy.cast<TensorType>();
    TensorType opTy = _opTy.cast<TensorType>();
    ArrayRef<int64_t> retShape = retTy.getShape();
    ArrayRef<int64_t> opShape = opTy.getShape();
    AxisInfo opInfo = operands[0]->getValue();
    AxisInfo::DimVectorT contiguity = opInfo.getContiguity();
    AxisInfo::DimVectorT divisibility = opInfo.getDivisibility();
    AxisInfo::DimVectorT constancy = opInfo.getConstancy();
    contiguity.insert(contiguity.begin() + expandDims.axis(), 1);
    divisibility.insert(divisibility.begin() + expandDims.axis(), 1);
    constancy.insert(constancy.begin() + expandDims.axis(), 1);
    curr = AxisInfo(contiguity, divisibility, constancy);
  }
  // Broadcast
  if (llvm::isa<triton::BroadcastOp>(op)) {
    Type _retTy = *op->result_type_begin();
    Type _opTy = *op->operand_type_begin();
    TensorType retTy = _retTy.cast<TensorType>();
    TensorType opTy = _opTy.cast<TensorType>();
    ArrayRef<int64_t> retShape = retTy.getShape();
    ArrayRef<int64_t> opShape = opTy.getShape();
    AxisInfo opInfo = operands[0]->getValue();
    AxisInfo::DimVectorT contiguity;
    AxisInfo::DimVectorT divisibility;
    AxisInfo::DimVectorT constancy;
    for (size_t d = 0; d < retTy.getRank(); d++) {
      contiguity.push_back(opShape[d] == 1 ? 1 : opInfo.getContiguity(d));
      divisibility.push_back(opInfo.getDivisibility(d));
      constancy.push_back(opShape[d] == 1 ? retShape[d] : 1);
    }
    curr = AxisInfo(contiguity, divisibility, constancy);
  }
  if (curr.getRank() == 0) {
    return markAllPessimisticFixpoint(op->getResults());
  }
  // join all latice elements
  ChangeResult result = ChangeResult::NoChange;
  for (Value value : op->getResults()) {
    result |= getLatticeElement(value).join(curr);
  }
  return result;
}

} // namespace mlir
