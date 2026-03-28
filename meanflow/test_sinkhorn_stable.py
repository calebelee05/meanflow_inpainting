import torch
import numpy as np
import sys
sys.path.append('/home/jovyan/brandylulu/OT-flow/meanflow')
from models.optimal_transport import OTPlanSampler

batch_size = 128
x0 = torch.randn(batch_size, 3, 32, 32)
x1 = torch.randn(batch_size, 3, 32, 32)

x0 = x0 * 2.0 - 1.0
x1 = x1 * 2.0 - 1.0

print(f"x0 stats: mean={x0.mean():.4f}, std={x0.std():.4f}")
print(f"x1 stats: mean={x1.mean():.4f}, std={x1.std():.4f}")

sampler = OTPlanSampler(
    method="sinkhorn",
    reg=0.05,
    numItermax=100,
    stopThr=1e-7
)

try:
    x0_paired, x1_paired = sampler.sample_plan(x0, x1)
    print("Sinkhorn go!")
    print(f"Paired shapes: {x0_paired.shape}, {x1_paired.shape}")
except Exception as e:
    print(f"Sinkhorn fail: {e}")