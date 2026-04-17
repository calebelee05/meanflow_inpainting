"""
Standalone MeanFlow Inpainting Script
(No call to generate_onestep_mf.py)
"""

import os
import re
import click
import pickle
import random
import zipfile
import glob
import numpy as np
import torch
import PIL.Image
import matplotlib.pyplot as plt
import tqdm
import dnnlib

# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32))
            for seed in seeds
        ]

    def randn(self, size, **kwargs):
        return torch.stack([
            torch.randn(size[1:], generator=g, **kwargs)
            for g in self.generators
        ])

    def randint(self, *args, size, **kwargs):
        return torch.stack([
            torch.randint(*args, size=size[1:], generator=g, **kwargs)
            for g in self.generators
        ])


def parse_int_list(s):
    if isinstance(s, list):
        return s
    result = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            result.extend(range(int(a), int(b)+1))
        else:
            result.append(int(part))
    return result


def load_random_cifar(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        files = [f for f in z.namelist() if f.endswith('.png')]
        selected = random.choice(files)
        with z.open(selected) as f:
            return np.array(PIL.Image.open(f).convert("RGB"))


def run_inference(net, x_orig, seed, H, W, device, sigma_min=0.002, sigma_max=1.0):
    """Run inference for a single seed. Returns out_denorm (H, W) in [0,1]."""
    rnd = StackedRandomGenerator(device, [seed])
    # Full latent noise: (1, 2, H, W)
    latents = rnd.randn([1, 2, H, W], device=device)

    # Model input: [ch0 of original (1 ch), full latent (1 ch)] -> 2 channels
    model_input = torch.cat([
        x_orig[:, :1, :, :],    # channel 0 of original data  (1 ch)
        latents[:, :1, :, :]    # complete latent              (1 ch)
    ], dim=1)                   # -> 2 channels total

    class_labels = torch.zeros(1, net.label_dim, device=device)

    sigma_min = 0.002
    sigma_max = 1.0
    
    with torch.no_grad():
        # Network predicts velocity/residual; subtract to get output
        output = latents[:, :1, :, :] - net(
            model_input,
            t=sigma_max * torch.ones([1, 1, 1, 1], device=device),
            class_labels=class_labels,
            h=(sigma_max - sigma_min) * sigma_max * torch.ones([1, 1, 1, 1], device=device),
            augment_labels=torch.zeros(1, 9).to(device)
        )  # shape: (1, 2, H, W)

    # Denormalise output back to original data range
    out_np = output.cpu().numpy()[0]  # (2, H, W)
    out_denorm = (out_np + 1.0) / 2.0 # (2, H, W)
    return out_denorm[1]


def save_results(outdir, original, out_denorm, label, mse_val):
    """Save prediction and MSE heatmaps."""
    plt.figure(figsize=(6, 6))
    plt.imshow(out_denorm, cmap='viridis')
    plt.colorbar()
    plt.title(f'Output ch1 ({label})')
    plt.axis('off')
    plt.savefig(os.path.join(outdir, f"best_out_ch1.png"), bbox_inches='tight')
    plt.close()

    mse_map = (original[1] - out_denorm) ** 2
    plt.figure(figsize=(6, 6))
    plt.imshow(mse_map, cmap='hot')
    plt.colorbar()
    plt.title(f'MSE ch1 ({label}, MSE={mse_val:.2e})')
    plt.axis('off')
    plt.savefig(os.path.join(outdir, f"best_mse_ch1.png"), bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

@click.command()
@click.option('--network', required=True, help='Network snapshot (.pkl)')
@click.option('--outdir', required=True, help='Output directory')
@click.option('--seeds', default='0', help='Seeds (e.g. 0 or 0-10)')
@click.option('--image', default=None, help='Path to image')
@click.option('--npz_folder', default=None, help='Path to folder with .npz test data')
@click.option('--device', default='cuda')
def main(network, outdir, seeds, image, npz_folder, device):

    device = torch.device(device)
    os.makedirs(outdir, exist_ok=True)
    seeds = parse_int_list(seeds)

    # ---------------------------------------------------------
    # Load network
    # ---------------------------------------------------------

    print("Loading network...")
    with dnnlib.util.open_url(network) as f:
        net = pickle.load(f)['ema'].to(device)

    H = 64
    W = 64

    # ---------------------------------------------------------
    # Collect all npz files
    # ---------------------------------------------------------

    if npz_folder is not None:
        npz_files = sorted(glob.glob(os.path.join(npz_folder, '*.npz')))
        if not npz_files:
            raise ValueError("No .npz files found in npz_folder")
    elif image is not None:
        npz_files = [image]
    else:
        raise ValueError("Provide --npz_folder or --image")

    print(f"Found {len(npz_files)} files. Running inference...")

    all_file_mses = []   # average MSE per file (averaged over seeds)
    best_global = {
        'mse': float('inf'),
        'out_denorm': None,
        'original': None,
        'label': None,
    }

    # ---------------------------------------------------------
    # Loop over all files
    # ---------------------------------------------------------

    for npz_path in tqdm.tqdm(npz_files, desc='Files'):

        # Load data
        data = np.load(npz_path)
        if npz_folder is not None:
            original_flat_init  = data['init_value']
            original_flat_final = data['final_value']
            original_init  = original_flat_init.reshape(H, W).astype(np.float32)
            original_final = original_flat_final.reshape(H, W).astype(np.float32)
            original = np.stack([original_init, original_final], axis=0)  # (2, H, W)
        else:
            original = data[list(data.keys())[0]].astype(np.float32)
            if original.shape == (H, W, 2):
                original = original.transpose(2, 0, 1)
            assert original.shape == (2, H, W)

        # Normalise using same fixed scaling as dataset: [0,1] -> [-1,1]
        original_norm = original * 2 - 1
        x_orig = torch.from_numpy(original_norm).unsqueeze(0).to(device)

        # Run inference for each seed, track best seed for this file
        best_seed_mse = float('inf')
        best_seed_out = None

        for seed in seeds:
            out_denorm = run_inference(net, x_orig, seed, H, W, device)
            mse = float(np.mean((original[1] - out_denorm) ** 2))
            if mse < best_seed_mse:
                best_seed_mse = mse
                best_seed_out = out_denorm

        all_file_mses.append(best_seed_mse)

        # Track globally best prediction across all files
        if best_seed_mse < best_global['mse']:
            best_global['mse'] = best_seed_mse
            best_global['out_denorm'] = best_seed_out
            best_global['original'] = original
            best_global['label'] = os.path.basename(npz_path)

    # ---------------------------------------------------------
    # Report and save
    # ---------------------------------------------------------

    avg_mse = float(np.mean(all_file_mses))
    print(f"\nAverage MSE across {len(npz_files)} files: {avg_mse:.4e}")
    print(f"Best prediction: {best_global['label']} (MSE={best_global['mse']:.4e})")

    # Save original heatmaps for the best file
    for ch, label in enumerate(['ch0', 'ch1']):
        plt.figure(figsize=(6, 6))
        plt.imshow(best_global['original'][ch], cmap='viridis')
        plt.colorbar()
        plt.title(f'Original {label} (best file: {best_global["label"]})')
        plt.axis('off')
        plt.savefig(os.path.join(outdir, f"best_original_{label}.png"), bbox_inches='tight')
        plt.close()

    save_results(
        outdir,
        best_global['original'],
        best_global['out_denorm'],
        best_global['label'],
        best_global['mse'],
    )

    # Save summary text
    with open(os.path.join(outdir, 'summary.txt'), 'w') as f:
        f.write(f"Files evaluated: {len(npz_files)}\n")
        f.write(f"Seeds used: {seeds}\n")
        f.write(f"Average MSE (best seed per file): {avg_mse:.4e}\n")
        f.write(f"Best file: {best_global['label']}\n")
        f.write(f"Best MSE: {best_global['mse']:.4e}\n")

    print("Done.")
    print(f"Results saved in {outdir}")


if __name__ == "__main__":
    main()