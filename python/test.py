import itertools
import re
from typing import Optional, Union

import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton._C.libtriton.triton as _triton
import triton.language as tl
from triton.runtime.jit import JITFunction, TensorWrapper, reinterpret

int_dtypes = ['int8', 'int16', 'int32', 'int64']
uint_dtypes = ['uint8', 'uint16', 'uint32', 'uint64']
float_dtypes = ['float16', 'float32', 'float64']
dtypes = int_dtypes + uint_dtypes + float_dtypes
dtypes_with_bfloat16 = dtypes + ['bfloat16']
torch_dtypes = ['bool'] + int_dtypes + ['uint8'] + float_dtypes + ['bfloat16']

def numpy_random(shape, dtype_str, rs: Optional[RandomState] = None, low=None, high=None):
    """
    Override `rs` if you're calling this function twice and don't want the same
    result for both calls.
    """
    if isinstance(shape, int):
        shape = (shape, )
    if rs is None:
        rs = RandomState(seed=17)
    if dtype_str in int_dtypes + uint_dtypes:
        iinfo = np.iinfo(getattr(np, dtype_str))
        low = iinfo.min if low is None else max(low, iinfo.min)
        high = iinfo.max if high is None else min(high, iinfo.max)
        dtype = getattr(np, dtype_str)
        x = rs.randint(low, high, shape, dtype=dtype)
        x[x == 0] = 1  # Hack. Never return zero so tests of division don't error out.
        return x
    elif dtype_str in float_dtypes:
        return rs.normal(0, 1, shape).astype(dtype_str)
    elif dtype_str == 'bfloat16':
        return (rs.normal(0, 1, shape).astype('float32').view('uint32')
                & np.uint32(0xffff0000)).view('float32')
    elif dtype_str in ['bool', 'int1', 'bool_']:
        return rs.normal(0, 1, shape) > 0.0
    else:
        raise RuntimeError(f'Unknown dtype {dtype_str}')

def to_triton(x: np.ndarray, device='cuda', dst_type=None) -> Union[TensorWrapper, torch.Tensor]:
    '''
    Note: We need dst_type because the type of x can be different from dst_type.
          For example: x is of type `float32`, dst_type is `bfloat16`.
          If dst_type is None, we infer dst_type from x.
    '''
    t = x.dtype.name
    if t in uint_dtypes:
        signed_type_name = t.lstrip('u')  # e.g. "uint16" -> "int16"
        x_signed = x.astype(getattr(np, signed_type_name))
        return reinterpret(torch.tensor(x_signed, device=device), getattr(tl, t))
    else:
        if t == 'float32' and dst_type == 'bfloat16':
            return torch.tensor(x, device=device).bfloat16()
        return torch.tensor(x, device=device)

def to_numpy(x):
    if isinstance(x, TensorWrapper):
        return x.base.cpu().numpy().astype(getattr(np, torch_dtype_name(x.dtype)))
    elif isinstance(x, torch.Tensor):
        if x.dtype is torch.bfloat16:
            return x.cpu().float().numpy()
        return x.cpu().numpy()
    else:
        raise ValueError(f"Not a triton-compatible tensor: {x}")


if False:
    def test_dot(epilogue, allow_tf32, dtype, device='cuda'):
        M, N, K = 128, 128, 64
        num_warps = 8
        trans_a, trans_b = False, False

        # triton kernel
        @triton.jit
        def kernel(X, stride_xm, stride_xk,
                  Y, stride_yk, stride_yn,
                  W, stride_wn, stride_wl,
                  Z, stride_zm, stride_zn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                  ADD_MATRIX: tl.constexpr, ADD_ROWS: tl.constexpr, ADD_COLS: tl.constexpr,
                  ALLOW_TF32: tl.constexpr,
                  DO_SOFTMAX: tl.constexpr, CHAIN_DOT: tl.constexpr,
                  TRANS_A: tl.constexpr, TRANS_B: tl.constexpr):
            off_m = tl.arange(0, BLOCK_M)
            off_n = tl.arange(0, BLOCK_N)
            off_l = tl.arange(0, BLOCK_N)
            off_k = tl.arange(0, BLOCK_K)
            Xs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
            Ys = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn
            Ws = W + off_n[:, None] * stride_wn + off_l[None, :] * stride_wl
            Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
            z = tl.dot(tl.load(Xs), tl.load(Ys), trans_a=TRANS_A, trans_b=TRANS_B, allow_tf32=ALLOW_TF32)
            if ADD_MATRIX:
                z += tl.load(Zs)
            if ADD_ROWS:
                ZRs = Z + off_m * stride_zm
                z += tl.load(ZRs)[:, None]
            if ADD_COLS:
                ZCs = Z + off_n * stride_zn
                z += tl.load(ZCs)[None, :]
            if DO_SOFTMAX:
                max = tl.max(z, 1)
                z = z - max[:, None]
                num = tl.exp(z)
                den = tl.sum(num, 1)
                z = num / den[:, None]
            if CHAIN_DOT:
                # tl.store(Zs, z)
                # tl.debug_barrier()
                z = tl.dot(z.to(tl.float16), tl.load(Ws), trans_a=TRANS_A)
            tl.store(Zs, z)

        # input
        rs = RandomState(17)
        x = numpy_random((K, M) if trans_a else (M, K), dtype_str=dtype, rs=rs) * .1
        print(x)
        y = numpy_random((N, K) if trans_b else (K, N), dtype_str=dtype, rs=rs) * .1
        w = numpy_random((N, N), dtype_str=dtype, rs=rs) * .1
        print("###1")
        print(x.shape)
        print(y.shape)
        print(w.shape)
        if allow_tf32:
            x = (x.view('uint32') & np.uint32(0xffffe000)).view('float32')
            y = (y.view('uint32') & np.uint32(0xffffe000)).view('float32')
            w = (w.view('uint32') & np.uint32(0xffffe000)).view('float32')
        print("###2")
        print(x.shape)
        print(y.shape)
        print(w.shape)
        x_tri = to_triton(x, device=device)
        y_tri = to_triton(y, device=device)
        w_tri = to_triton(w, device=device)
        # triton result
        z = 1 + numpy_random((M, N), dtype_str=dtype, rs=rs) * .1
        z_tri = to_triton(z, device=device)
        if epilogue == 'trans':
            z_tri = torch.as_strided(z_tri, (M, N), z_tri.stride()[::-1])
        pgm = kernel[(1, 1)](x_tri, x_tri.stride(0), x_tri.stride(1),
                            y_tri, y_tri.stride(0), y_tri.stride(1),
                            w_tri, w_tri.stride(0), w_tri.stride(1),
                            z_tri, z_tri.stride(0), z_tri.stride(1),
                            TRANS_A=trans_a, TRANS_B=trans_b,
                            BLOCK_M=M, BLOCK_K=K, BLOCK_N=N,
                            ADD_MATRIX=epilogue == 'add-matrix',
                            ADD_ROWS=epilogue == 'add-rows',
                            ADD_COLS=epilogue == 'add-cols',
                            DO_SOFTMAX=epilogue == 'softmax',
                            CHAIN_DOT=epilogue == 'chain-dot',
                            ALLOW_TF32=allow_tf32,
                            num_warps=num_warps)
        # torch result
        print("###3")
        x_ref = x.T if trans_a else x
        y_ref = y.T if trans_b else y
        print(x_ref.shape)
        print(y_ref.shape)
        z_ref = np.matmul(x_ref, y_ref)
        if epilogue == 'add-matrix':
            z_ref += z
        if epilogue == 'add-rows':
            z_ref += z[:, 0][:, None]
        if epilogue == 'add-cols':
            z_ref += z[0, :][None, :]
        if epilogue == 'softmax':
            num = np.exp(z_ref - np.max(z_ref, axis=-1, keepdims=True))
            denom = np.sum(num, axis=-1, keepdims=True)
            z_ref = num / denom
        if epilogue == 'chain-dot':
            z_ref = np.matmul(z_ref.T if trans_a else z_ref, w)
        # compare
        # print(z_ref[:,0], z_tri[:,0])
        np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)
        # make sure ld/st are vectorized
        # ptx = pgm.asm['ptx']
        # assert 'ld.global.v4' in ptx
        # assert 'st.global.v4' in ptx
        # if allow_tf32:
        #     assert 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32' in ptx
        # elif dtype == 'float32':
        #     assert 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32' not in ptx
        # elif dtype == 'int8':
        #     assert 'mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32' in ptx

def test_dot(epilogue, allow_tf32, dtype, device='cuda'):
    M, N, K = 32, 16, 16# 32, 32, 32
    num_warps = 8
    trans_a, trans_b = False, False

    # triton kernel
    @triton.jit
    def kernel(X, stride_xm, stride_xk,
              Y, stride_yk, stride_yn,
              W, stride_wn, stride_wl,
              Z, stride_zm, stride_zn,
              Z2, stride_z2m, stride_z2n,
              BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
              ADD_MATRIX: tl.constexpr, ADD_ROWS: tl.constexpr, ADD_COLS: tl.constexpr,
              ALLOW_TF32: tl.constexpr,
              DO_SOFTMAX: tl.constexpr, CHAIN_DOT: tl.constexpr,
              TRANS_A: tl.constexpr, TRANS_B: tl.constexpr):
        off_m = tl.arange(0, BLOCK_M)
        off_n = tl.arange(0, BLOCK_N)
        off_l = tl.arange(0, BLOCK_N)
        off_k = tl.arange(0, BLOCK_K)
        Xs = X + off_m[:, None] * stride_xm + off_k[None, :] * stride_xk
        Ys = Y + off_k[:, None] * stride_yk + off_n[None, :] * stride_yn
        Ws = W + off_n[:, None] * stride_wn + off_l[None, :] * stride_wl
        Zs = Z + off_m[:, None] * stride_zm + off_n[None, :] * stride_zn
        Z2s = Z2 + off_m[:, None] * stride_z2m + off_n[None, :] * stride_z2n
        z = tl.dot(tl.load(Xs), tl.load(Ys), trans_a=TRANS_A, trans_b=TRANS_B, allow_tf32=ALLOW_TF32)
        if ADD_MATRIX:
            z += tl.load(Zs)
        if ADD_ROWS:
            ZRs = Z + off_m * stride_zm
            z += tl.load(ZRs)[:, None]
        if ADD_COLS:
            ZCs = Z + off_n * stride_zn
            z += tl.load(ZCs)[None, :]
        if DO_SOFTMAX:
            max = tl.max(z, 1)
            z = z - max[:, None]
            num = tl.exp(z)
            den = tl.sum(num, 1)
            z = num / den[:, None]
        if CHAIN_DOT:
            # tl.store(Zs, z)
            tl.debug_barrier()
            tl.debug_barrier()
            tl.debug_barrier()
            tl.debug_barrier()
            loadWs = tl.load(Ws)
            tl.debug_barrier()
            tl.debug_barrier()
            tl.debug_barrier()
            tl.debug_barrier()
            tl.store(Z2s, z)
            tl.debug_barrier()
            tl.debug_barrier()
            tl.debug_barrier()
            tl.debug_barrier()
            z_to = z.to(tl.float16)
            tl.debug_barrier()
            tl.debug_barrier()
            tl.debug_barrier()
            tl.debug_barrier()
            z = tl.dot(z_to, loadWs, trans_a=TRANS_A)
        tl.store(Zs, z)
    is_print = os.getenv("PRINT", False)
    # input
    rs = RandomState(17)
    x = numpy_random((K, M) if trans_a else (M, K), dtype_str=dtype, rs=rs) * .1
    y = numpy_random((N, K) if trans_b else (K, N), dtype_str=dtype, rs=rs) * .1
    w = numpy_random((N, N), dtype_str=dtype, rs=rs) * .1
    x *= 0.0
    x[0][0] = 1.0
    y *= 0.0
    y += 1.0
    w *= 0.0
    # w += 1.0
    w[0][13] = 1.0
    w[0][15] = 1.0
    w[1][13] = 1.0
    w[1][15] = 1.0
    print("###1")
    print(x.shape)
    print(y.shape)
    print(w.shape)
    if allow_tf32:
        x = (x.view('uint32') & np.uint32(0xffffe000)).view('float32')
        y = (y.view('uint32') & np.uint32(0xffffe000)).view('float32')
        w = (w.view('uint32') & np.uint32(0xffffe000)).view('float32')
    print("###2")
    print(x.shape)
    print(y.shape)
    print(w.shape)
    x_tri = to_triton(x, device=device)
    y_tri = to_triton(y, device=device)
    w_tri = to_triton(w, device=device)
    # triton result
    z = 1 + numpy_random((M, N), dtype_str=dtype, rs=rs) * .1
    z_tri = to_triton(z, device=device)
    z2_tri = to_triton(z, device=device)
    if epilogue == 'trans':
        z_tri = torch.as_strided(z_tri, (M, N), z_tri.stride()[::-1])
    pgm = kernel[(1, 1)](x_tri, x_tri.stride(0), x_tri.stride(1),
                        y_tri, y_tri.stride(0), y_tri.stride(1),
                        w_tri, w_tri.stride(0), w_tri.stride(1),
                        z_tri, z_tri.stride(0), z_tri.stride(1),
                        z2_tri, z2_tri.stride(0), z2_tri.stride(1),
                        TRANS_A=trans_a, TRANS_B=trans_b,
                        BLOCK_M=M, BLOCK_K=K, BLOCK_N=N,
                        ADD_MATRIX=epilogue == 'add-matrix',
                        ADD_ROWS=epilogue == 'add-rows',
                        ADD_COLS=epilogue == 'add-cols',
                        DO_SOFTMAX=epilogue == 'softmax',
                        CHAIN_DOT=epilogue == 'chain-dot',
                        ALLOW_TF32=allow_tf32,
                        num_warps=num_warps)
    # torch result
    x_ref = x.T if trans_a else x
    y_ref = y.T if trans_b else y
    print("###3")
    print(x_ref.shape)
    print(y_ref.shape)
    if is_print:
        print(pgm.asm['llir'])
        print(pgm.asm['ttir'])
    z_ref = np.matmul(x_ref, y_ref)
    if epilogue == 'add-matrix':
        z_ref += z
    if epilogue == 'add-rows':
        z_ref += z[:, 0][:, None]
    if epilogue == 'add-cols':
        z_ref += z[0, :][None, :]
    if epilogue == 'softmax':
        num = np.exp(z_ref - np.max(z_ref, axis=-1, keepdims=True))
        denom = np.sum(num, axis=-1, keepdims=True)
        z_ref = num / denom
    if epilogue == 'chain-dot':
        if is_print:
            print(f"w = {w}")
            print(f"z_tri = {to_numpy(z_tri)}")
            print(f"z2_tri = {to_numpy(z_tri)}")
            print(f"pre_zref = {z_ref}")
        z_ref = np.matmul(z_ref.T if trans_a else z_ref, w)
        if is_print:
            print(f"post_zref = {z_ref}")
    # compare
    # print(z_ref[:,0], z_tri[:,0])
    np.testing.assert_allclose(z_ref, to_numpy(z_tri), rtol=0.01)
    # make sure ld/st are vectorized
    # ptx = pgm.asm['ptx']
    # assert 'ld.global.v4' in ptx
    # assert 'st.global.v4' in ptx
    # if allow_tf32:
    #     assert 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32' in ptx
    # elif dtype == 'float32':
    #     assert 'mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32' not in ptx
    # elif dtype == 'int8':
    #     assert 'mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32' in ptx

import os
os.system("rm -rf /home/siwasaki/.triton/cache")
for epilogue in ['chain-dot']: # , 'none', 'trans', 'add-matrix', 'add-rows', 'add-cols', 'softmax']:
    for allow_tf32 in [True, False]:
        for dtype in ['float16']:
            if not (allow_tf32 and (dtype in ['float16'])):
                print((epilogue, allow_tf32, dtype))
                test_dot(epilogue, allow_tf32, dtype)
