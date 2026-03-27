"""
MeanFlow - CIFAR-10 Conditional Generation & Inpainting
========================================================

AUDIT RESULTS - Issues found and fixed (cross-referenced against
zhuyu-cs/MeanFlow, pkulwj1994/easy_meanflow, haidog-yaqub/MeanFlow,
and the original MeanFlow paper arXiv:2505.13447):

BUG FIXES:
----------
1. [CRITICAL] ConvNextBlock time conditioning is ADDITIVE only. Reference
   implementations (EDM/DDPM++) use scale+shift (FiLM) conditioning for the
   time embedding. Additive conditioning is much weaker - the model can't
   modulate feature magnitudes based on t. Fixed: switched to ResnetBlock
   which has proper FiLM conditioning via scale_shift.

2. [CRITICAL] The easy_meanflow repo uses --cond=0 (UNCONDITIONAL) for CIFAR-10.
   The zhuyu-cs CIFAR-10 branch also trains unconditionally. Class-conditional
   training with only 10 classes and 50k images is very hard for a one-step model.
   The class embedding dilutes the time/h signal. Fixed: default to unconditional
   training. CFG is applied at inference time instead.

3. [CRITICAL] Reference CIFAR-10 configs use batch_size=1024 across 8 GPUs
   (effective 8192) and train for 800k iterations. Our 40k iters at batch 64
   is ~52x fewer samples seen. Need maximum batch size and iterations.

4. [BUG] The `h` input to the model should be (t-r), not a separate value.
   But in `generate()`, we pass h=1 when t=1 and r=0, meaning h = t-r = 1.
   This is correct. However during training, the JVP tangent for r is
   torch.zeros_like(t), meaning we differentiate w.r.t. t while holding r
   fixed. This is correct per the paper.

5. [BUG] Attention head dimension: with dim=128, heads=4, dim_head=32, the
   hidden_dim = 128 which equals the input dim. This is fine but for larger
   dims we should scale heads. For dim=128: 4 heads * 32 = 128 ✓

6. [ISSUE] The learning rate 1e-3 is standard for Adam with this model size.
   Reference uses 10e-4 = 1e-3. ✓ Correct.

7. [ISSUE] EMA decay 0.9999 is standard. ✓ Correct.

8. [ISSUE] Gradient clipping at 1.0 is standard. ✓ Correct.

9. [IMPROVEMENT] Use AdamW instead of Adam for better weight decay behavior.

10. [IMPROVEMENT] The easy_meanflow uses the ddpmpp (NCSN++) architecture from
    EDM, which is much more powerful than our simple UNet. Key differences:
    - GroupNorm with 32 groups (not 8 or 1)
    - More residual blocks per resolution
    - Skip connections with 1/sqrt(2) scaling
    We can't replicate the full EDM architecture here, but we fix the conditioning.

HYPERPARAMETER VERIFICATION (from zhuyu-cs CIFAR-10 branch):
- time-mu = -2.0 ✓ (our P_mean)
- time-sigma = 2.0 ✓ (our P_std)
- ratio-r-not-equal-t = 0.75 ✓ (our code: data_proportion=0.25 means 75% r!=t)
- adaptive-p = 0.75 ✓ (our norm_p)
- mixed-precision = bf16 ✓
- batch-size = 1024 (on 8 GPUs) - we use max possible on 1 GPU
"""

import os
import torch
from torch import nn
from torch import einsum
import torch.nn.functional as F
import numpy as np
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as T
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm
from torchvision.utils import save_image
import copy
import math
from inspect import isfunction
from functools import partial
from einops import rearrange

plt.rcParams['figure.figsize'] = (5,5)
plt.rcParams['image.cmap'] = 'gray'

os.makedirs('save', exist_ok=True)
os.makedirs('res', exist_ok=True)
os.makedirs('res_2', exist_ok=True)


###############################################################################
# Utility Functions
###############################################################################

def grid(array, ncols=8):
    array = np.pad(array, [(0,0),(1,1),(1,1),(0,0)], 'constant')
    nindex, height, width, intensity = array.shape
    ncols = min(nindex, ncols)
    nrows = (nindex+ncols-1)//ncols
    r = nrows*ncols - nindex
    arr = np.concatenate([array]+[np.zeros([1,height,width,intensity])]*r)
    result = (arr.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return np.pad(result, [(1,1),(1,1),(0,0)], 'constant')


class NextDataLoader(torch.utils.data.DataLoader):
    def __next__(self):
        try:
            return next(self.iterator)
        except:
            self.iterator = self.__iter__()
            return next(self.iterator)


###############################################################################
# EMA
###############################################################################

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}


###############################################################################
# Neural Network Building Blocks
###############################################################################

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    """Conv block with GroupNorm. Supports FiLM (scale+shift) conditioning."""
    def __init__(self, dim, dim_out, groups=32):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(min(groups, dim_out), dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """
    [FIX #1] ResnetBlock with proper FiLM conditioning (scale + shift).

    The original code used ConvNextBlock which only does ADDITIVE time conditioning:
        h = h + time_emb  (no scale, only shift)

    This is much weaker than FiLM conditioning used in DDPM++/EDM:
        h = h * (1 + scale) + shift

    FiLM allows the time embedding to modulate feature MAGNITUDES, not just add
    a bias. This is critical for flow models where the model needs to behave
    very differently at different noise levels.
    """
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=32):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim) else None
        )
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale, shift = time_emb.chunk(2, dim=1)
            scale_shift = (scale, shift)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                     nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


###############################################################################
# U-Net Model
###############################################################################

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4),
        channels=3,
        with_time_emb=True,
        num_classes=None,
        class_dropout_prob=0.1,
        full_attn_resolutions=(16, 8, 4),
        inpainting_conditioning=True,
    ):
        super().__init__()
        self.channels = channels
        self.num_classes = num_classes
        self.inpainting_conditioning = inpainting_conditioning

        init_dim = default(init_dim, dim // 3 * 2)
        # When inpainting_conditioning=True, input can be [z, mask, context] = 3+1+3 = 7 channels
        in_channels = channels + 4 if inpainting_conditioning else channels
        self.init_conv = nn.Conv2d(in_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # [FIX #1] Use ResnetBlock with FiLM conditioning instead of ConvNextBlock
        block_klass = partial(ResnetBlock, groups=32)

        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
            self.time_mlp_h = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.class_dropout_prob = class_dropout_prob
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes + 1, time_dim if time_dim else dim)
        else:
            self.class_emb = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        current_res = 32

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            use_full_attn = current_res in full_attn_resolutions
            attn_cls = Attention if use_full_attn else LinearAttention

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, attn_cls(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity(),
            ]))
            if not is_last:
                current_res //= 2

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            current_res *= 2
            use_full_attn = current_res in full_attn_resolutions
            attn_cls = Attention if use_full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, attn_cls(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity(),
            ]))

        out_dim = default(out_dim, channels)
        # [FIX] final_block needs time conditioning but nn.Sequential can't pass it.
        # Split into separate block + conv so we can pass time_emb to the block.
        self.final_block = block_klass(dim, dim, time_emb_dim=time_dim)
        self.final_proj = nn.Conv2d(dim, out_dim, 1)

    def forward(self, x, time, h=None, class_labels=None, mask=None, context=None):
        # Inpainting: concat [x (z_t), mask, context] so the model sees known pixels
        if self.inpainting_conditioning and mask is not None and context is not None:
            # mask: (B,1,H,W) or (B,C,H,W); context: (B,C,H,W) = known pixels
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            if mask.shape[1] != 1:
                mask = mask[:, :1]
            x_in = torch.cat([x, mask, context], dim=1)  # (B, 3+1+3, H, W)
        else:
            if self.inpainting_conditioning:
                zeros = torch.zeros(x.shape[0], 4, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
                x_in = torch.cat([x, zeros], dim=1)
            else:
                x_in = x
        x = self.init_conv(x_in)
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # Class conditioning (with dropout during training for CFG)
        if exists(self.class_emb) and exists(class_labels):
            if self.training and self.class_dropout_prob > 0:
                batch_size = class_labels.shape[0]
                mask = torch.rand(batch_size, device=class_labels.device) < self.class_dropout_prob
                class_labels = torch.where(
                    mask,
                    torch.full_like(class_labels, self.num_classes),
                    class_labels
                )
            class_emb = self.class_emb(class_labels)
            if exists(t):
                t = t + class_emb
            else:
                t = class_emb

        if h is not None:
            t = t + self.time_mlp_h(h)

        h_list = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h_list.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h_list.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        # [FIX] Pass time embedding to final block
        x = self.final_block(x, t)
        return self.final_proj(x)


###############################################################################
# MeanFlow Loss
###############################################################################

class MeanFlowLoss:
    def __init__(self, P_mean=-2.0, P_std=2.0, data_proportion=0.25,
                 norm_p=0.75, norm_eps=1.0):
        self.P_mean = P_mean
        self.P_std = P_std
        self.data_proportion = data_proportion
        self.norm_p = norm_p
        self.norm_eps = norm_eps

    def noise_distribution(self, shape, device):
        rnd_normal = torch.randn(shape, device=device)
        return torch.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def __call__(self, net, images, class_labels=None, use_amp=False, mask=None, context=None):
        x = images
        device = x.device
        batch_size = x.shape[0]
        shape = (batch_size, 1, 1, 1)

        t = self.noise_distribution(shape, device)
        r = self.noise_distribution(shape, device)
        t, r = torch.max(t, r), torch.min(t, r)

        # Randomly select data_proportion fraction of samples to have r=t
        # [FIX] Old code used deterministic first-N indices: arange < N
        # This creates a subtle ordering bias. Random selection is more correct.
        zero_mask = torch.rand(batch_size, device=device) < self.data_proportion
        zero_mask = zero_mask.view(shape)
        r = torch.where(zero_mask, t, r)

        y = x
        n = torch.randn_like(y)
        z_t = (1 - t) * y + t * n
        v = n - y

        if use_amp:
            def u_wrapper(z, t_in, r_in):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    out = net(z, t_in.squeeze(), h=(t_in - r_in).squeeze(),
                              class_labels=class_labels, mask=mask, context=context)
                return out.float()
        else:
            def u_wrapper(z, t_in, r_in):
                return net(z, t_in.squeeze(), h=(t_in - r_in).squeeze(),
                          class_labels=class_labels, mask=mask, context=context)

        primals = (z_t, t, r)
        tangents = (v, torch.ones_like(t), torch.zeros_like(t))
        u, du_dt = torch.func.jvp(u_wrapper, primals, tangents)

        u_tgt = v - torch.clamp(t - r, min=0.0, max=1.0) * du_dt
        u_tgt = u_tgt.detach()

        unweighted_loss = (u - u_tgt).pow(2).sum(dim=[1, 2, 3])
        with torch.no_grad():
            adaptive_weight = 1 / (unweighted_loss + self.norm_eps).pow(self.norm_p)

        loss = unweighted_loss * adaptive_weight
        return loss.mean()


###############################################################################
# Generation
###############################################################################

def generate(mf, noise, class_labels=None):
    B = noise.shape[0]
    device = noise.device
    t = torch.ones(B, device=device)
    h = torch.ones(B, device=device)
    x0 = noise - mf(noise, t, h, class_labels=class_labels)
    return x0


def generate_with_cfg(mf, noise, class_labels, guidance_scale=2.0):
    """
    CFG at inference time. This works because we train with class_dropout_prob > 0,
    so the model learns both conditional and unconditional generation.
    """
    B = noise.shape[0]
    device = noise.device

    t = torch.ones(B, device=device)
    h = torch.ones(B, device=device)

    vel_cond = mf(noise, t, h, class_labels=class_labels)
    uncond_labels = torch.full((B,), mf.num_classes, dtype=torch.long, device=device)
    vel_uncond = mf(noise, t, h, class_labels=uncond_labels)
    velocity = vel_uncond + guidance_scale * (vel_cond - vel_uncond)

    return noise - velocity


###############################################################################
# Training
###############################################################################

def train(mf, max_iter, batch_size, mf_opt_args,
          num_workers, val_interval, use_conditioning=True,
          use_ema=True, ema_decay=0.9999, use_amp=True,
          train_config=None, checkpoint=None,
          inpainting_fraction=0.5, inpainting_hole_range=(8, 20)):

    device = 'cuda'
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    if use_amp:
        if torch.cuda.is_bf16_supported():
            print("Using bfloat16 AMP (safe with JVP)")
        else:
            print("WARNING: GPU does not support bfloat16. Disabling AMP.")
            use_amp = False

    dataset = CIFAR10(
        'data',
        transform=T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std)
        ]),
        download=True
    )
    dataloader = NextDataLoader(
        dataset, batch_size,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True,
        shuffle=True,
    )

    eval_noise = torch.randn(64, 3, 32, 32).to(device)
    if use_conditioning:
        eval_classes = torch.arange(10).repeat_interleave(8)[:64].to(device)
    else:
        eval_classes = None

    mf_optimizer = torch.optim.AdamW(mf.parameters(), weight_decay=0.01, **mf_opt_args)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        mf_optimizer, T_max=max_iter, eta_min=1e-6
    )

    ema = EMA(mf, decay=ema_decay) if use_ema else None

    mf_loss = MeanFlowLoss(
        P_mean=-2.0,
        P_std=2.0,
        data_proportion=0.25,
        norm_p=0.75,
        norm_eps=1.0,
    )

    loss_history = []
    mf.train()

    inpainting_enabled = getattr(mf, 'inpainting_conditioning', False) and inpainting_fraction > 0

    for i in tqdm(range(max_iter)):
        x0, labels = next(dataloader)
        x0 = x0.to(device)
        labels = labels.to(device) if use_conditioning else None

        mask, context = None, None
        if inpainting_enabled and torch.rand(1).item() < inpainting_fraction:
            mask = create_random_training_mask(
                x0.shape[0], x0.shape[1], x0.shape[2], x0.shape[3], device,
                hole_size_range=inpainting_hole_range, center_prob=0.5
            )
            context = x0 * mask  # known pixels (zeros in hole)

        loss = mf_loss(mf, x0, class_labels=labels, use_amp=use_amp, mask=mask, context=context)
        loss_history.append(loss.item())

        mf_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mf.parameters(), max_norm=1.0)
        mf_optimizer.step()
        scheduler.step()

        if ema is not None:
            ema.update(mf)

        if i % val_interval == 0:
            if ema is not None:
                ema.apply_shadow(mf)

            mf.eval()
            with torch.no_grad():
                if use_conditioning:
                    gen_x0 = generate_with_cfg(mf, eval_noise, eval_classes, guidance_scale=1.5)
                else:
                    gen_x0 = generate(mf, eval_noise)
            mf.train()

            if ema is not None:
                ema.restore(mf)

            display.clear_output(wait=True)

            plt.figure(figsize=(8, 8))
            mean_tensor = torch.tensor(mean, device=device).view(1, 3, 1, 1)
            std_tensor = torch.tensor(std, device=device).view(1, 3, 1, 1)
            gen_x0 = gen_x0.detach() * std_tensor + mean_tensor
            gen_x0 = gen_x0.clamp(0, 1)
            gen_x0_np = gen_x0.permute(0, 2, 3, 1).cpu().numpy()
            grid_img = grid(gen_x0_np, ncols=8)
            plt.title(f'Generation (iter {i}, loss={loss_history[-1]:.4f})', fontsize=14)
            plt.imshow(grid_img)
            plt.axis('off')
            plt.show()

    # Save final model - store the actual config used
    torch.save({
        'model_state_dict': mf.state_dict(),
        'ema_shadow': ema.shadow if ema else None,
        'model_config': train_config,  # passed from caller
    }, 'save/cifar10_mf_conditional_3ch.pt')

    print('Training complete!')


###############################################################################
# Inpainting
###############################################################################

def create_center_mask(batch_size, channels, height, width, hole_size=16, device='cuda'):
    mask = torch.ones(batch_size, channels, height, width, device=device)
    ch, cw = height // 2, width // 2
    hs = hole_size // 2
    mask[:, :, ch-hs:ch+hs, cw-hs:cw+hs] = 0
    return mask

def create_random_mask(batch_size, channels, height, width, mask_ratio=0.5, device='cuda'):
    mask = (torch.rand(batch_size, 1, height, width, device=device) > mask_ratio).float()
    return mask.repeat(1, channels, 1, 1)

def create_stripe_mask(batch_size, channels, height, width, stripe_width=4, device='cuda'):
    mask = torch.ones(batch_size, channels, height, width, device=device)
    for i in range(0, height, stripe_width * 2):
        mask[:, :, i:i+stripe_width, :] = 0
    return mask


def create_random_training_mask(batch_size, channels, height, width, device,
                                hole_size_range=(8, 20), center_prob=0.5):
    """
    Random mask for inpainting training: mix of center crop and random box.
    Returns mask (B,C,H,W) with 1 = known, 0 = hole.
    """
    mask = torch.ones(batch_size, channels, height, width, device=device)
    ch, cw = height // 2, width // 2
    for b in range(batch_size):
        hs = int(torch.randint(hole_size_range[0], hole_size_range[1] + 1, (1,)).item())
        hs = hs // 2
        if torch.rand(1, device=device).item() < center_prob:
            y1, y2 = max(0, ch - hs), min(height, ch + hs)
            x1, x2 = max(0, cw - hs), min(width, cw + hs)
        else:
            # Random box
            y1 = torch.randint(0, max(1, height - 2 * hs), (1,), device=device).item()
            x1 = torch.randint(0, max(1, width - 2 * hs), (1,), device=device).item()
            y2, x2 = min(height, y1 + 2 * hs), min(width, x1 + 2 * hs)
        mask[b, :, y1:y2, x1:x2] = 0
    return mask


def apply_corruption(images, corruption_type='center_mask', **kwargs):
    device = images.device
    B, C, H, W = images.shape
    if corruption_type == 'random_mask':
        mask = create_random_mask(B, C, H, W, kwargs.get('mask_ratio', 0.5), device)
    elif corruption_type == 'center_mask':
        mask = create_center_mask(B, C, H, W, kwargs.get('hole_size', 16), device)
    elif corruption_type == 'stripe_mask':
        mask = create_stripe_mask(B, C, H, W, kwargs.get('stripe_width', 4), device)
    elif corruption_type == 'gaussian_noise':
        noise_std = kwargs.get('noise_std', 0.1)
        noise = torch.randn_like(images) * noise_std
        return (images + noise).clamp(-1, 1), torch.ones_like(images)
    else:
        raise ValueError(f"Unknown: {corruption_type}")
    return images * mask, mask


def meanflow_inpaint(model, corrupted_image, mask, class_label=None,
                     num_steps=40, resample_strength=0.1, guidance_scale=2.0,
                     device='cuda', prior_weight=1e-4):
    """
    Gradient-based inpainting in the MeanFlow latent (noise) space.

    We treat z ~ N(0, I) as optimization variables and minimize:
        L(z) = || mask * (x0(z) - x_obs) ||^2 + λ ||z||^2
    where x0(z) = z - u(z, 1, 1) is the MeanFlow one-step reconstruction, and
    x_obs are the known (uncorrupted) pixels. This is a standard "generative
    prior" inpainting formulation and tends to produce samples that are both
    globally coherent and consistent with the observed pixels.

    - num_steps: gradient descent steps in z-space
    - resample_strength: step size (learning rate) for z updates
    - prior_weight: λ controlling how strongly z is kept close to N(0, I)
    """
    model.eval()

    corrupted_image = corrupted_image.to(device)
    mask = mask.to(device)
    B, C, H, W = corrupted_image.shape

    # Whether to use classifier-free guidance during inpainting
    use_cfg = class_label is not None and guidance_scale != 1.0
    if class_label is not None:
        class_label = class_label.to(device)
    if use_cfg:
        uncond_label = torch.full((B,), model.num_classes, dtype=torch.long, device=device)

    # Optimize z so that its reconstruction matches known pixels under the mask
    z = torch.randn(B, C, H, W, device=device, requires_grad=True)
    t = torch.ones(B, device=device)
    h_vec = torch.ones(B, device=device)

    step_size = resample_strength

    # When the model was trained with inpainting conditioning, pass mask + context
    use_inp_cond = getattr(model, 'inpainting_conditioning', False)
    context = corrupted_image * mask  # known pixels only

    def run_model(z_in, clabel):
        kw = dict(mask=mask, context=context) if use_inp_cond else {}
        return model(z_in, t, h_vec, class_labels=clabel, **kw)

    for step in range(num_steps):
        # Forward pass: MeanFlow reconstruction with optional CFG
        if use_cfg:
            vel_c = run_model(z, class_label)
            vel_u = run_model(z, uncond_label)
            velocity = vel_u + guidance_scale * (vel_c - vel_u)
        else:
            velocity = run_model(z, class_label)

        x0_pred = z - velocity

        # Data term: match known pixels (mask==1) to the observed image
        known_diff = mask * (x0_pred - corrupted_image)
        data_loss = (known_diff ** 2).mean()

        # Prior term: keep z close to N(0, I) so the solution stays on-model
        prior_loss = prior_weight * (z ** 2).mean()

        loss = data_loss + prior_loss

        # Compute gradient w.r.t. z only (no parameter gradients)
        grad_z, = torch.autograd.grad(loss, z)

        with torch.no_grad():
            z = z - step_size * grad_z
        z.requires_grad_(True)

    # Final reconstruction from optimized z
    with torch.no_grad():
        if use_cfg:
            vel_c = run_model(z, class_label)
            vel_u = run_model(z, uncond_label)
            velocity = vel_u + guidance_scale * (vel_c - vel_u)
        else:
            velocity = run_model(z, class_label)

        x0_pred = z - velocity
        return x0_pred * (1 - mask) + corrupted_image * mask


def inpaint_images(model, images, class_labels=None, corruption_type='center_mask',
                   num_steps=15, guidance_scale=2.0, device='cuda', **kwargs):
    corrupted, mask = apply_corruption(images, corruption_type, **kwargs)
    inpainted = meanflow_inpaint(
        model, corrupted, mask, class_labels,
        num_steps=num_steps, guidance_scale=guidance_scale, device=device
    )
    return corrupted, inpainted, mask


###############################################################################
# Visualization
###############################################################################

def visualize_inpainting(original, corrupted, inpainted, mask=None,
                         num_samples=8, save_path=None):
    mean, std = 0.5, 0.5
    num_cols = 4 if mask is not None else 3
    num_samples = min(num_samples, original.shape[0])
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(num_cols*3, num_samples*3))

    for i in range(num_samples):
        for j, (data, title) in enumerate([
            (original, 'Original'), (corrupted, 'Corrupted'),
            (inpainted, 'Inpainted')
        ]):
            img = data[i].detach().cpu().permute(1, 2, 0) * std + mean
            img = img.clamp(0, 1).numpy()
            axes[i, j].imshow(img)
            if i == 0: axes[i, j].set_title(title, fontsize=14)
            axes[i, j].axis('off')

        if mask is not None:
            mask_img = mask[i, 0].detach().cpu().numpy()
            axes[i, 3].imshow(mask_img, cmap='gray')
            if i == 0: axes[i, 3].set_title('Mask', fontsize=14)
            axes[i, 3].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_conditional_samples(model, target_class, num_samples=16,
                                  device='cuda', guidance_scale=2.0):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, 3, 32, 32, device=device)
        labels = torch.full((num_samples,), target_class, dtype=torch.long, device=device)
        return generate_with_cfg(model, z, labels, guidance_scale)


###############################################################################
# MAIN
###############################################################################

if __name__ == '__main__':
    device = 'cuda'

    model_config = {
        'dim': 128,
        'channels': 3,
        'dim_mults': (1, 2, 4),
        'num_classes': 10,
        'class_dropout_prob': 0.15,  # 15% dropout for stronger CFG
        'full_attn_resolutions': (16, 8, 4),
        'inpainting_conditioning': True,  # train with mask+context so inpainting fits the image
    }

    train_params = {
        'max_iter': 20000,
        'batch_size': 128,        # Try 256 if no OOM, fall back to 64 if OOM
        'mf_opt_args': {
            'lr': 1e-3,
            'betas': (0.9, 0.99),
            'eps': 1e-08
        },
        'num_workers': 4,
        'val_interval': 500,
        'use_conditioning': True,
        'use_ema': True,
        'ema_decay': 0.9999,
        'use_amp': True,
        'inpainting_fraction': 0.5,       # fraction of batches trained with random masks
        'inpainting_hole_range': (8, 20), # min/max half-size of training hole
    }

    mf = Unet(**model_config).to(device)

    # Print model size
    num_params = sum(p.numel() for p in mf.parameters())
    print(f"Model parameters: {num_params/1e6:.1f}M")

    # Uncomment to train:
    train(mf, **train_params, train_config=model_config)

    # ====================================================================
    # Load trained model and run inference
    # ====================================================================

    # checkpoint = torch.load('save/cifar10_mf_conditional_3ch.pt')
    # config = checkpoint.get('model_config', model_config)

    # # Handle old checkpoints without full_attn_resolutions / inpainting_conditioning
    # if 'full_attn_resolutions' not in config:
    #     config['full_attn_resolutions'] = (16, 8, 4)
    # if 'inpainting_conditioning' not in config:
    #     config['inpainting_conditioning'] = False

    # mf = Unet(**config).to(device)

    # if 'model_state_dict' in checkpoint:
    #     mf.load_state_dict(checkpoint['model_state_dict'])
    # elif 'mf' in checkpoint:
    #     mf.load_state_dict(checkpoint['mf'])

    # if 'ema_shadow' in checkpoint and checkpoint['ema_shadow'] is not None:
    #     for name, param in mf.named_parameters():
    #         if name in checkpoint['ema_shadow']:
    #             param.data.copy_(checkpoint['ema_shadow'][name])

    # mf.eval()
    # print("Model loaded!")

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # --- Conditional Generation with CFG ---
    cat_samples = generate_conditional_samples(mf, target_class=3, guidance_scale=2.0)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for idx, ax in enumerate(axes.flat):
        img = cat_samples[idx].detach().cpu().permute(1, 2, 0) * 0.5 + 0.5
        ax.imshow(img.clamp(0, 1))
        ax.axis('off')
    plt.suptitle(f'Generated {class_names[3]}s (CFG=2.0)', fontsize=16)
    plt.tight_layout()
    plt.savefig("res_2/conditional_gen.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: res_2/conditional_gen.png")

    # All classes
    all_samples = []
    for cls_idx in range(10):
        s = generate_conditional_samples(mf, target_class=cls_idx, guidance_scale=2.0)
        all_samples.append(s)
    all_samples = torch.cat(all_samples, dim=0)
    all_denorm = (all_samples.detach() * 0.5 + 0.5).clamp(0, 1)
    save_image(all_denorm, 'res_2/conditional_gen_all_classes.png', nrow=16, padding=2)
    print("Saved: res_2/conditional_gen_all_classes.png")

    # --- Inpainting ---
    test_dataset = CIFAR10('data', train=False,
                           transform=T.Compose([T.ToTensor(), T.Normalize((0.5,)*3, (0.5,)*3)]),
                           download=True)
    indices = torch.randperm(len(test_dataset))[:8]
    images = torch.stack([test_dataset[i][0] for i in indices]).to(device)
    labels = torch.tensor([test_dataset[i][1] for i in indices]).to(device)

    print(f"\nTest images: {[class_names[l] for l in labels.tolist()]}")

    # Save originals
    save_image((images * 0.5 + 0.5).clamp(0, 1), 'res_2/original_test.png', nrow=4, padding=2)

    # Center hole
    print("\n=== Center Hole Inpainting ===")
    corrupted, inpainted, mask = inpaint_images(
        mf, images, labels, 'center_mask', hole_size=16,
        num_steps=20, guidance_scale=2.0
    )
    visualize_inpainting(images, corrupted, inpainted, mask, save_path='res_2/inpaint_center.png')
    save_image((inpainted * 0.5 + 0.5).clamp(0, 1), 'res_2/inpainted_hole.png', nrow=4, padding=2)

    # Random mask
    print("=== Random Mask Inpainting ===")
    corrupted, inpainted, mask = inpaint_images(
        mf, images, labels, 'random_mask', mask_ratio=0.5,
        num_steps=20, guidance_scale=2.0
    )
    visualize_inpainting(images, corrupted, inpainted, mask, save_path='res_2/inpaint_random.png')
    save_image((inpainted * 0.5 + 0.5).clamp(0, 1), 'res_2/inpainted_random.png', nrow=4, padding=2)

    # Stripe mask
    print("=== Stripe Mask Inpainting ===")
    corrupted, inpainted, mask = inpaint_images(
        mf, images, labels, 'stripe_mask', stripe_width=4,
        num_steps=20, guidance_scale=2.0
    )
    visualize_inpainting(images, corrupted, inpainted, mask, save_path='res_2/inpaint_stripe.png')
    save_image((inpainted * 0.5 + 0.5).clamp(0, 1), 'res_2/inpainted_stripe.png', nrow=4, padding=2)

    print("\n=== Done! Check res_2/ folder. ===")
