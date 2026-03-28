import torch
import torchvision.datasets as datasets
from training.data_transform import get_transform_cifar
import numpy as np

transform = get_transform_cifar(is_for_fid=False)
dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4
)

for samples, _ in data_loader:
    print(f"Original samples shape: {samples.shape}")
    print(f"Original samples range: [{samples.min():.4f}, {samples.max():.4f}]")
    print(f"Original samples mean: {samples.mean():.4f}")
    print(f"Original samples std: {samples.std():.4f}")
    
    samples = samples * 2.0 - 1.0  # [-1, 1]
    
    print(f"\nAfter scaling to [-1, 1]:")
    print(f"Scaled samples range: [{samples.min():.4f}, {samples.max():.4f}]")
    print(f"Scaled samples mean: {samples.mean():.4f}")
    print(f"Scaled samples std: {samples.std():.4f}")
    
    samples_flat = samples.reshape(samples.shape[0], -1)
    print(f"\nFlattened shape: {samples_flat.shape}")
    print(f"Flattened std: {samples_flat.std():.4f}")
    
    suggested_reg = float(samples_flat.std() * 0.1)
    print(f"\nSuggested ot_reg = {suggested_reg:.6f}")
    
    break