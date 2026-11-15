"""第 10 章：minillm 的 kernel 注册成 PyTorch 算子以后，作为算子是否合格。

  - torch.library.opcheck：schema、fake tensor（torch.compile 要用）、autograd 注册是否正确；
  - 反向：和 PyTorch 参考实现的梯度对比；
  - torch.compile(fullgraph=True)：整个小网络不 graph break，结果和 eager 一致。
"""
import pytest
import torch
import torch.nn.functional as F

import minillm.ops as ops


def rand(*shape, device, dtype=torch.float32):
    return torch.randn(*shape, device=device, dtype=dtype, requires_grad=True)


def ref_rmsnorm(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


CASES = {
    "add": (lambda d: (rand(1000, device=d), rand(1000, device=d)), lambda x, y: x + y, ops.add, {}),
    "silu_mul": (lambda d: (rand(64, 300, device=d), rand(64, 300, device=d)),
                 lambda g, u: F.silu(g) * u, ops.silu_mul, {}),
    "rmsnorm": (lambda d: (rand(2, 33, 512, device=d), rand(512, device=d), 1e-6), ref_rmsnorm, ops.rmsnorm, {}),
    "softmax": (lambda d: (rand(16, 1000, device=d),), lambda x: torch.softmax(x, -1), ops.softmax, {}),
    "mm": (lambda d: (rand(128, 256, device=d), rand(256, 192, device=d)), lambda a, b: a @ b, ops.mm,
           {"atol": 2e-2, "rtol": 2e-2}),                 # fp32 输入的 tl.dot 默认是 TF32（第 7 章）
}


@pytest.mark.parametrize("name", CASES)
def test_opcheck(device, name):
    make, _, op, tol = CASES[name]
    torch.library.opcheck(op, make(device), **tol)


@pytest.mark.parametrize("name", CASES)
def test_backward(device, name):
    make, ref_fn, op, tol = CASES[name]
    args = make(device)
    ref_args = [a.detach().clone().requires_grad_(True) if isinstance(a, torch.Tensor) else a for a in args]
    out, ref = op(*args), ref_fn(*ref_args)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad)
    torch.testing.assert_close(out, ref, **tol)
    for a, r in zip(args, ref_args):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a.grad, r.grad, **(tol or {"atol": 1e-4, "rtol": 1e-4}))


def mlp_block(x, w_norm, w1, w3, w2):
    """minillm 的 MLP 半层：h = x + W2 · (silu(W1 · rmsnorm(x)) ⊙ (W3 · rmsnorm(x)))"""
    h = ops.rmsnorm(x, w_norm, 1e-6)
    return ops.add(x, ops.mm(ops.silu_mul(ops.mm(h, w1), ops.mm(h, w3)), w2))


def test_compile_mlp_block(device):
    torch.manual_seed(0)
    x = torch.randn(256, 512, device=device, dtype=torch.float16)
    ws = [torch.randn(512, device=device, dtype=torch.float16),
          torch.randn(512, 1024, device=device, dtype=torch.float16) * 0.05,
          torch.randn(512, 1024, device=device, dtype=torch.float16) * 0.05,
          torch.randn(1024, 512, device=device, dtype=torch.float16) * 0.05]
    params = [w.clone().requires_grad_(True) for w in ws]
    params_c = [w.clone().requires_grad_(True) for w in ws]
    out = mlp_block(x, *params)
    out_c = torch.compile(mlp_block, fullgraph=True)(x, *params_c)     # fullgraph：不允许 graph break
    torch.testing.assert_close(out_c, out, atol=1e-2, rtol=1e-2)
    out.sum().backward()
    out_c.sum().backward()
    for p, pc in zip(params, params_c):
        torch.testing.assert_close(pc.grad, p.grad, atol=5e-2, rtol=5e-2)
