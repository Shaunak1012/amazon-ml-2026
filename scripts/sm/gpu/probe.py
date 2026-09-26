"""Training-job entry point for the GPU probe: prints the GPU, torch/CUDA versions and a bf16 matmul timing."""
import subprocess
import time

import torch

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    print(f"bf16 8k matmul x20: {time.time() - t:.2f}s -> {20 * 2 * 8192 ** 3 / (time.time() - t) / 1e12:.1f} TFLOPS")
print("PROBE OK")
