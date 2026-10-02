import tabulate
import torch

import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _dropout(x_ptr, x_keep_ptr, output_ptr, n_elements, p, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_keep = tl.load(x_keep_ptr + offsets, mask=mask)
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)

def dropout(x, x_keep, p):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements= x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _dropout[grid](x, x_keep, output, n_elements, p, BLOCK_SIZE=1024)
    return output


def run_dropout_example():
    x = torch.randn(size=(10,), device=DEVICE)
    p = 0.5
    x_keep = (torch.rand(size=(10,), device=DEVICE) > p).to(torch.int32)

    output = dropout(x, x_keep, p)
    print(tabulate.tabulate([
        ["input"] + x.tolist(),
        ["keep mask"] + x_keep.tolist(),
        ["output"] + output.tolist(),
    ]))


@triton.jit
def _seeded_dropout(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    random = tl.rand(seed, offsets)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seed):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    n_elements= x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _seeded_dropout[grid](x, output, n_elements, p, seed, BLOCK_SIZE=1024)
    return output


def run_seeded_dropout_examples():
    x = torch.randn(size=(10,), device=DEVICE)
    output = seeded_dropout(x, p=0.5, seed=123)
    output2 = seeded_dropout(x, p=0.5, seed=123)
    output3 = seeded_dropout(x, p=0.5, seed=512)

    print(
        tabulate.tabulate([
            ["input"] + x.tolist(),
            ["output (seed = 123)"] + output.tolist(),
            ["output (seed = 123)"] + output2.tolist(),
            ["output (seed = 512)"] + output3.tolist(),
        ]))

# ==== excercises ====
# 1. Extend the kernel to operate over a matrix and use a vector of seeds - one per row.
@triton.jit
def _seeded_dropout_matrix(x_ptr, output_ptr, n_rows, n_cols, p, seed_ptr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_rows * n_cols
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    row_idx = offsets // n_cols
    col_idx = offsets % n_cols
    seed_mask = row_idx < n_rows
    row_seed = tl.load(seed_ptr + row_idx, mask=seed_mask, other=0).to(tl.int32)
    random = tl.rand(row_seed, col_idx)
    x_keep = random > p
    output = tl.where(x_keep, x / (1 - p), 0.0)
    tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout_matrix(x, p, seeds):
    assert x.ndim == 2, "x must be a matrix"
    assert x.is_contiguous()
    output = torch.empty_like(x)
    n_rows, n_cols = x.shape
    grid = lambda meta: (triton.cdiv(n_rows*n_cols, meta["BLOCK_SIZE"]),)
    _seeded_dropout_matrix[grid](x, output, n_rows=n_rows, n_cols=n_cols, p=p, seed_ptr=seeds, BLOCK_SIZE=1024)
    return output


def run_seeded_dropout_matrix_examples():
    x = torch.randn(size=(3, 3), device=DEVICE)
    seeds1 = torch.tensor([123, 123, 123], device=DEVICE)
    seeds2 = torch.tensor([123, 144, 155], device=DEVICE)
    output = seeded_dropout_matrix(x, p=0.5, seeds=seeds1)
    output2 = seeded_dropout_matrix(x, p=0.5, seeds=seeds1)
    output3 = seeded_dropout_matrix(x, p=0.5, seeds=seeds2)

    print(
        tabulate.tabulate([
            ["input"] + x.tolist(),
            ["output (seeds1)"] + output.tolist(),
            ["output (seeds1)"] + output2.tolist(),
            ["output (seeds2)"] + output3.tolist(),
        ]))



if __name__ == "__main__":
    run_dropout_example()
    run_seeded_dropout_examples()
    run_seeded_dropout_matrix_examples()
