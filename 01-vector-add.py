from pathlib import Path

import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()
FIG_DIR = Path(__file__).parent / "fig" / "01-vector-add"
FIG_DIR.mkdir(exist_ok=True, parents=True)


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert x.device == y.device == DEVICE, "Input tensors must be on the same device as the kernel"

    output = torch.empty_like(x)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # torch.tensor objects are implicitly converted to pointers to their first element
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["size"],
        x_vals=[2**i for i in range(12, 28, 1)],
        x_log=True,
        line_arg="provider",
        line_vals=["torch", "triton"],
        line_names=["PyTorch", "Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="GB/s",
        plot_name="vector-add-performance",
        args={},
    )
)
def benchmark(size, provider):
    x = torch.randn(size, device=DEVICE, dtype=torch.float32)
    y = torch.randn(size, device=DEVICE, dtype=torch.float32)

    quantiles = [0.5, 0.2, 0.8]

    if provider == "torch":
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)

    gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(min_ms), gbps(max_ms)


if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    x = torch.randn(size, device=DEVICE)
    y = torch.randn(size, device=DEVICE)
    output_torch = x + y
    output_triton = add(x, y)
    print(output_torch)
    print(output_triton)
    print(f"max diff: {torch.max(torch.abs(output_torch - output_triton))}")

    benchmark.run(print_data=True, show_plots=True, save_path=FIG_DIR)
