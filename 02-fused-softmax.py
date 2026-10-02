from pathlib import Path

import torch

import triton
import triton.language as tl
from triton.runtime import driver

DEVICE = triton.runtime.driver.active.get_active_torch_device()
FIG_DIR = Path(__file__).parent / "fig" / "02-fused-softmax"
FIG_DIR.mkdir(exist_ok=True, parents=True)

def is_hip() -> bool:
    return triton.runtime.driver.active.get_current_target() == "hip"


def is_cdna() -> bool:
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ("gfx908", "gfx90a", "gfx940", "gfx941")


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Compute row-wise softmax of X using native pytorch"""
    # read MN elements; write M elements
    x_max = x.max(dim=1)[0]
    # read MN + M elements; write MN elements
    z = x - x_max[:, None]
    # read MN elements; write MN elements
    numerator = torch.exp(z)
    # read MN elements; write M elements
    denominator = numerator.sum(dim=1)
    # read MN + M elements; write MN elements
    ret = numerator / denominator[:, None]
    # in total: read 5MN + 2M elements; wrote 3MN + 2M elements
    return ret


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    BLOCKSIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCKSIZE)
        input_ptrs = row_start_ptr + col_offsets
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


properties = driver.active.utils.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()


def softmax(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2, "Input tensor must be 2D"
    n_rows, n_cols = x.shape

    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 8
    num_stages = 4 if SIZE_SMEM > 200000 else 2
    y = torch.empty_like(x)

    kernel = softmax_kernel.warmup(y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCKSIZE=BLOCK_SIZE, num_stages=num_stages, num_warps=num_warps, grid=(1,))

    kernel._init_handles()
    n_regs = kernel.n_regs
    size_smem = kernel.metadata.shared
    if is_hip():
        NUM_GPRS = NUM_REGS
        if is_cdna():
            NUM_GPRS = NUM_REGS * 2

        MAX_NUM_THREADS = properties["max_threads_per_sm"]
        max_num_waves = MAX_NUM_THREADS // WARP_SIZE
        occupancy = min(NUM_GPRS // WARP_SIZE // n_regs, max_num_waves) // num_warps
    else:
        occupancy = NUM_REGS // (n_regs * WARP_SIZE * num_warps)
    occupancy = min(occupancy, SIZE_SMEM // size_smem)
    num_programs = NUM_SM * occupancy
    num_programs = min(num_programs, n_rows)
    kernel[(num_programs, 1, 1)](y, x, x.stride(0), y.stride(0), n_rows, n_cols, BLOCK_SIZE, num_stages)
    return y


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * i for i in range(2, 100)],
        line_arg="provider",
        line_vals=["triton", "torch", "naive_softmax"],
        line_names=["Triton", "Torch", "Naive Softmax"],
        styles=[("blue", "-"), ("green", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name="softmax-performance",
        args={"M": 4096},
    )
)
def benchmark(N, M, provider):
    x = torch.randn(N, M, device=DEVICE)
    if provider == "triton":
        ms = triton.testing.do_bench(lambda: softmax(x))
    elif provider == "torch":
        ms = triton.testing.do_bench(lambda: torch.nn.functional.softmax(x, dim=-1))
    elif provider == "naive_softmax":
        ms = triton.testing.do_bench(lambda: naive_softmax(x))
    else:
        raise ValueError(f"Unknown provider: {provider}")
    gbps = lambda ms: (x.numel() * x.element_size() * 2) / (ms * 1e6)
    return gbps(ms)


if __name__ == "__main__":
    benchmark.run(show_plots=True, print_data=True, save_path=FIG_DIR)
