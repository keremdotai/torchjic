# TorchJiC - Just-in-CUDA: A runtime Python-to-CUDA compiler for PyTorch

TorchJiC is an experimental framework for compiling Python functions or PyTorch operations directly into CUDA kernels — at runtime.
Inspired by Just-In-Time (JIT) compilation techniques, TorchJiC aims to bring low-latency, GPU-accelerated execution to arbitrary Python logic,
without the need to manually write CUDA code.

> [!NOTE]  
> TorchJiC is currently a concept under development. Contributions, feedback, and ideas are welcome!

## ✨ Why TorchJiC?

Many advanced deep learning workflows involve custom computation logic — especially in areas like time series modeling, physics-informed networks, or scientific machine learning.
While PyTorch offers torch.jit and tools like TorchDynamo to accelerate models, these tools have limitations.

Even with TorchScript, many functions — especially those involving explicit Python loops or control flow — still execute parts of their logic in Python.
This can be painfully slow on the GPU.

#### Example: Custom RNN Cell

```python
@torch.jit.script
def my_recurrent_cell(x_seq, h_0):
    h = h_0
    for t in range(x_seq.size(1)):
        x = x_seq[:, t]
        h = torch.tanh(x + h)  # simplified update
    return h
```

Even though this is TorchScript-compiled, the iteration over time steps (for t in range(...)) is still performed in Python space and cannot be fused or parallelized on the GPU.
In contrast, built-in CUDA-optimized GRU/LSTM implementations use kernel fusion and loop unrolling in highly-tuned CUDA code.

TorchJiC targets these exact scenarios by enabling:

- ✅ Full compilation of Python-level control flow (e.g. loops, if-statements)
- ✅ Custom CUDA kernel generation from arbitrary PyTorch functions
- ✅ Batch-level and loop fusion of repeated operations

Instead of tracing operations after they’re executed, TorchJiC generates CUDA kernels that eliminate the Python loop entirely — resulting in true GPU parallelism.

## 🚧 Roadmap

- Basic operator tracing and dispatch (e.g. element-wise math ops)
- CUDA kernel generation using templates or Triton
- Compilation and caching mechanism
- Support for PyTorch Tensor inputs/outputs
- Automatic fallback to eager mode on unsupported ops
- Integration with PyTorch modules (nn.Module)
- Benchmarking vs. TorchScript / Triton / handwritten CUDA

## 📦 Goals

- Developer-friendly decorator API
- Minimal configuration
- Clear diagnostics for unsupported operations
- Option to pre-compile kernels (optional AOT path)

## 🧪 Experimental Areas

- Kernel generation with Triton or NVRTC
- Torch FX / bytecode / AST-level tracing
- Type inference for CUDA codegen
- Custom memory layout optimizations

## 🤝 Contributing

> [!WARNING]
> This project is currently in the ideation phase — there is no code, no prototype, and no concrete implementation yet.
> It’s a vision for a system that enables true Just-in-Time CUDA compilation for PyTorch.

If you’re interested in:
- Discussing the feasibility of dynamic CUDA codegen
- Exploring tracing pipelines (Torch FX, AST, or bytecode level)
- Investigating Triton or NVRTC as backends
- Prototyping design directions or API ergonomics
- Or simply brainstorming with a shared interest in accelerating PyTorch…

You’re very welcome to open an issue or start a discussion!
This project needs minds, not PRs (yet).
