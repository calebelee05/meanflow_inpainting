import torch
import torch.nn as nn
from math import prod



class BranchMLP(nn.Module):
    """
    Generic vector MLP backbone.
    Input:
    - x: (B, input_dim)
    Output:
    - y: (B, output_dim)
    """

    class _ResidualBlock(nn.Module):
        def __init__(self, hidden_dim: int, dropout: float = 0.0, use_layer_norm: bool = True):
            super().__init__()
            self.norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
            self.fc1 = nn.Linear(hidden_dim, hidden_dim)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.norm(x)
            h = self.fc1(h)
            h = self.act(h)
            h = self.drop(h)
            h = self.fc2(h)
            h = self.drop(h)
            return x + h

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        use_residual: bool = False,
        residual_dropout: float = 0.0,
        residual_layer_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.use_residual = bool(use_residual)

        if not self.use_residual:
            layers = [nn.Linear(self.input_dim, hidden_dim), nn.ReLU()]
            for _ in range(num_hidden_layers - 1):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dim, self.output_dim))
            self.net = nn.Sequential(*layers)
        else:
            self.in_proj = nn.Linear(self.input_dim, hidden_dim)
            self.in_act = nn.ReLU()
            self.blocks = nn.ModuleList(
                [
                    BranchMLP._ResidualBlock(
                        hidden_dim=hidden_dim,
                        dropout=float(residual_dropout),
                        use_layer_norm=bool(residual_layer_norm),
                    )
                    for _ in range(num_hidden_layers)
                ]
            )
            self.out_proj = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x must be (B,{self.input_dim}), got {tuple(x.shape)}")
        if not self.use_residual:
            return self.net(x)

        h = self.in_act(self.in_proj(x))
        for blk in self.blocks:
            h = blk(h)
        return self.out_proj(h)
    

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    Architecture: sinusoidal encoding (fixed) -> 2-layer MLP (learnable)
    """
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 16):
        super().__init__()
        self.time_embedding_dim = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        Args:
            t:          (B,) tensor of timesteps (can be fractional, in [0, 1])
            dim:        output dimension (frequency_embedding_size)
            max_period: controls the minimum frequency
        Returns:
            (B, dim) tensor
        """
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(max_period, dtype=torch.float32)) * torch.arange(half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]           # (B, half)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (B, dim)
        if dim % 2:  # 奇数维时补一列零
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) scalar timesteps
        Returns:
            (B, hidden_size) embedding vector
        """
        t_freq = self.timestep_embedding(t, self.time_embedding_dim)
        return self.mlp(t_freq)


class MFModel(nn.Module):
    """
    Generic D-dimensional MeanFlow wrapper.

    Forward pass layout:
        xt  (B, *shapes) --flatten--> (B, D)  ─────────────────────────────┐
        t   (B,)         --embedder-> (B, H)                                ├─ cat -> (B, D+2H) -> net -> (B, D) -> (B, *shapes)
        h=t-s (B,)       --embedder-> (B, H)  ─────────────────────────────┘

    t and h share a single TimestepEmbedder (same learned embedding space).

    Args:
        shapes (tuple[int, ...]):     spatial/channel dims, e.g. (C, H, W) or (D,).
                                      D = prod(shapes) is the flattened data dimension.
        hidden_size (int):            embedding dim H for TimestepEmbedder.
        net_arch (type):              nn.Module subclass to instantiate as the backbone.
                                      Must accept keyword args `input_dim` and `output_dim`
                                      in addition to whatever is in net_config.
                                      input_dim  = D + 2 * hidden_size  (injected by MFModel)
                                      output_dim = D                    (injected by MFModel)
        net_config (dict):            extra kwargs forwarded to net_arch (e.g. hidden_dim,
                                      num_hidden_layers). Do NOT include input_dim/output_dim.
        frequency_embedding_size (int): sinusoidal frequency dim inside TimestepEmbedder.

    Example:
        model = MFModel(
            shapes=(16,),
            hidden_size=64,
            net_arch=BranchMLP,
            net_config=dict(hidden_dim=256, num_hidden_layers=4, use_residual=True),
        )
    """

    def __init__(self, shapes, hidden_size, net_arch, net_config, time_embedding_dim=16):
        super().__init__()
        self.shapes = shapes
        self.D = prod(shapes)

        self.time_embedder = TimestepEmbedder(hidden_size, time_embedding_dim)

        # MFModel injects input_dim / output_dim so callers never need to compute them.
        self.net = net_arch(
            input_dim=self.D + 2 * hidden_size,
            output_dim=self.D,
            **net_config,
        )

    def forward(self, xt, times, aug_cond=None):
        """
        Args:
            xt:       (B, *shapes)
            times:    tuple (t, h) where t is current time (B,) and h = t - s is the interval (B,)
            aug_cond: ignored, kept for interface compatibility with MeanFlow
        Returns:
            (B, *shapes)
        """
        t, s = times
        h = t-s 
        B = xt.shape[0]
        xt_flat = xt.view(B, -1)               # (B, D)

        t_emb = self.time_embedder(t)          # (B, H)
        h_emb = self.time_embedder(h)          # (B, H)  — shared embedder, different input

        x_in = torch.cat([xt_flat, t_emb, h_emb], dim=-1)  # (B, D + 2H)
        out = self.net(x_in)                   # (B, D)

        return out.view(B, *self.shapes)        # (B, *shapes)
