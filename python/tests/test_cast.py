import triton
import triton.language as tl


# TODO: function with no arguments don't work
@triton.jit
def cast_check_binop(X):
    zero_0d = tl.zeros([], dtype=tl.float32)
    zero_1d = tl.zeros([2], dtype=tl.float32)
    zero_2d_21 = tl.zeros([2, 1], dtype=tl.float32)
    zero_2d_22 = tl.zeros([2, 2], dtype=tl.float32)

    # scalar + scalar -> scalar
    a0 = 0.0 + 0.0
    # scalar + 0D -> 0D
    a1 = 0.0 + zero_0d
    a2 = zero_0d + 0.0
    # scalar + 1D -> 1D
    a3 = 0.0 + zero_1d
    a4 = zero_1d + 0.0
    # scalar + 2D -> 2D
    a5 = 0.0 + zero_2d_22
    a6 = zero_2d_22 + 0.0

    # 0D + 0D -> 0D
    b1 = zero_0d + zero_0d
    # 0D + 1D -> 1D
    b2 = zero_0d + zero_1d
    b3 = zero_1d + zero_0d
    # 0D + 2D -> 2D
    b4 = zero_0d + zero_2d_22
    b5 = zero_2d_22 + zero_0d

    # 1D + 1D -> 1D
    c1 = zero_1d + zero_1d
    # 1D + 2D -> 2D
    c2 = zero_1d + zero_2d_21
    c3 = zero_1d + zero_2d_22
    c4 = zero_2d_21 + zero_1d
    c5 = zero_2d_22 + zero_1d

    # 2D + 2D -> 2D
    d1 = zero_2d_21 + zero_2d_21
    d2 = zero_2d_22 + zero_2d_22
    d3 = zero_2d_21 + zero_2d_22
    d4 = zero_2d_22 + zero_2d_21

    return a0, a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, c1, c2, c3, c4, c5, d1, d2, d3, d4


def test_cast_check_binop():
    kernel = triton.compiler._compile(cast_check_binop,
                                      signature="*fp32",
                                      device=0,
                                      output="ttgir")
    assert (kernel)
    # TODO: Check types of the results


@triton.jit
def cast_check_0dtensor(ptr, val):
    val_sc = val
    val_0d = tl.zeros([], dtype=tl.float32)
    ptr_sc = ptr
    ptr_0d = tl.broadcast_to(ptr, [])
    mask_sc = val == 0
    mask_0d = tl.broadcast_to(mask_sc, [])

    # Load
    # val_0d += tl.load(ptr_sc)
    val_0d += tl.load(ptr_0d)
    # val_0d += tl.load(ptr_sc, mask_sc)
    val_0d += tl.load(ptr_sc, mask_0d)
    val_0d += tl.load(ptr_0d, mask_sc)
    val_0d += tl.load(ptr_0d, mask_0d)
    # val_0d += tl.load(ptr_sc, mask_sc, val_sc)
    val_0d += tl.load(ptr_sc, mask_0d, val_sc)
    val_0d += tl.load(ptr_0d, mask_sc, val_sc)
    val_0d += tl.load(ptr_0d, mask_0d, val_sc)
    val_0d += tl.load(ptr_sc, mask_sc, val_0d)
    val_0d += tl.load(ptr_sc, mask_0d, val_0d)
    val_0d += tl.load(ptr_0d, mask_sc, val_0d)
    val_0d += tl.load(ptr_0d, mask_0d, val_0d)

    # Store
    tl.store(ptr_sc + 0, val_sc)
    tl.store(ptr_0d + 1, val_sc)
    tl.store(ptr_sc + 2, val_0d)
    tl.store(ptr_0d + 3, val_0d)
    tl.store(ptr_sc + 4, val_sc, mask_sc)
    tl.store(ptr_0d + 5, val_sc, mask_sc)
    tl.store(ptr_sc + 6, val_sc, mask_0d)
    tl.store(ptr_0d + 7, val_sc, mask_0d)
    tl.store(ptr_sc + 8, val_0d, mask_sc)
    tl.store(ptr_0d + 9, val_0d, mask_sc)
    tl.store(ptr_sc + 10, val_0d, mask_0d)
    tl.store(ptr_0d + 11, val_0d, mask_0d)


def test_cast_check_0dtensor():
    kernel = triton.compiler._compile(cast_check_0dtensor,
                                      signature="*fp32,fp32",
                                      device=0,
                                      output="ttir")
    assert (kernel)
    # TODO: Use ttgir or lower
    # TODO: Check types of the results
