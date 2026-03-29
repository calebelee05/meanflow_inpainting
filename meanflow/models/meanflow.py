import torch
import torch.nn as nn
from models.time_sampler import sample_two_timesteps
from models.ema import init_ema, update_ema_net
from models.optimal_transport import OTPlanSampler
from models.loss import MeanFlowEuclidean_loss

def reshape_time(tensor, target_shape):
    dims = len(target_shape)
    return tensor.view(-1, *([1] * (dims - 1)))


class FlowSampler:
    """
    Samples (xt, v) given source x0 (data) and target x1 (noise).

    Notation:
        x0 : clean data
        x1 : Gaussian noise
        xt : linear interpolation x0 -> x1 at time t
        v  : conditional velocity  x1 - x0

    Args:
        case (str): coupling strategy.
            "base" — identity coupling (x0, x1 paired as given).
            "ot"   — mini-batch optimal-transport coupling via OTPlanSampler.
        ot_kwargs: forwarded to OTPlanSampler when case="ot".
    """

    def __init__(self, case='base', **ot_kwargs):
        if case not in ('base', 'ot'):
            raise ValueError(f"Unknown case '{case}'. Choose 'base' or 'ot'.")
        self.case = case
        if case == 'ot':
            self.ot_sampler = OTPlanSampler(**ot_kwargs)

    def sample(self, x0, x1, t):
        """
        Apply coupling then compute interpolation.

        Args:
            x0 (Tensor): data samples  [B, ...]
            x1 (Tensor): noise samples [B, ...]
            t  (Tensor): time values   [B, 1, 1, 1] (already reshaped)

        Returns:
            xt (Tensor): interpolated sample  (1-t)*x0 + t*x1
            v  (Tensor): conditional velocity  x1 - x0
        """
        if self.case == 'ot':
            x0, x1 = self.ot_sampler.sample_plan(x0, x1, replace=False)

        # reshape t from (B,) to (B, 1, 1, ...) to broadcast over spatial dims
        t_b = t.view(t.shape[0], *([1] * (x0.dim() - 1)))
        xt = (1 - t_b) * x0 + t_b * x1
        v = x1 - x0
        return xt, v


class MeanFlow(nn.Module):
    def __init__(self, arch, args, net_configs):
        super(MeanFlow, self).__init__()
        self.net = arch(**net_configs)
        self.args = args

        # Put this in a buffer so that it gets included in the state dict
        self.register_buffer("num_updates", torch.tensor(0))

        self.net_ema = init_ema(self.net, arch(**net_configs), args.ema_decay)

        # maintain extra ema nets
        self.ema_decays = args.ema_decays
        for i, ema_decay in enumerate(self.ema_decays):
            self.add_module(f"net_ema{i + 1}", init_ema(self.net, arch(**net_configs), ema_decay))

        # FlowSampler owns the coupling strategy (and OT sampler when case="ot")
        case = getattr(args, 'case', 'base')
        ot_kwargs = dict(
            method=getattr(args, 'ot_method', 'exact'),
            reg=getattr(args, 'ot_reg', 0.05),
            numItermax=getattr(args, 'ot_numItermax', 100),
            stopThr=getattr(args, 'ot_stopThr', 1e-9),
        )
        self.flow_sampler = FlowSampler(case=case, **ot_kwargs) if case == 'ot' else FlowSampler(case=case)

    def update_ema(self):
        self.num_updates += 1
        num_updates = self.num_updates
        update_ema_net(self.net, self.net_ema, num_updates)
        for i in range(len(self.ema_decays)):
            update_ema_net(self.net, self._modules[f"net_ema{i + 1}"], num_updates)

    def loss_normalize(self, loss):
        adp_wt = (loss.detach() + self.args.norm_eps) ** self.args.norm_p
        return loss / adp_wt
    
    def construct_v_avg_func(self, net, aug_cond, version):
        if version in ["mf", "imf"]:
            def v_avg_fun(xt, t, s):
                return net(xt, (t, s), aug_cond)
        elif version == "x-pixel":
            def v_avg_fun(xt, t, s):
                x0_pred = net(xt, (t, s), aug_cond)
                t_b = t.view(t.shape[0], *([1] * (xt.dim() - 1)))
                return (xt - x0_pred) / t_b.clamp(min=1e-4)
        else:
            raise ValueError(f"Unknown version '{version}'")
        return v_avg_fun


    def forward_with_loss(self, x0, x1=None,aug_cond=None, ):
        """
        Args:
            x0 (Tensor): clean data  [B, ...]
            aug_cond   : augmentation conditioning
            x1 (Tensor, optional): Gaussian noise [B, ...]. Sampled fresh if None.
        """
        device = x0.device
        if x1 is None:
            x1 = torch.randn_like(x0).to(device)
        else:
            x1 = x1.to(device)

        # Sample time steps
        t, s = sample_two_timesteps(self.args, num_samples=x0.shape[0], device=device)
        if self.args.version == "x-pixel":
            t= torch.clamp(t, min=0.05) # to avoid division by zero in x-pixel version 

        # Apply coupling and compute interpolation (delegates to FlowSampler)
        xt, v = self.flow_sampler.sample(x0, x1, t)
        v_avg_fun = self.construct_v_avg_func(self.net, aug_cond, version=self.args.version)

        # loss shape: [B], formulation: (v - vtgt)^2 summed over non-batch dims
        loss,dbg = MeanFlowEuclidean_loss(v_avg_fun, xt, v, t, s, verbose=True, version=self.args.version)
        self.dbg = dbg

        loss = self.loss_normalize(loss)
        loss = loss.mean()

        return loss

    def sample(self, x1=None, net=None, device=None, samples_shape=200, T=1):
        net = net if net is not None else self.net_ema

        if x1 is None:
            x1 = torch.randn(samples_shape, dtype=torch.float32, device=device)
        else:
            x1 = x1.to(device)
        batch_size=x1.shape[0]
        v_avg_func = self.construct_v_avg_func(net, aug_cond=None, version=self.args.version)
        xt = x1
        for step in range(T, 0, -1):
            t = torch.full((batch_size,), step / T, device=device)
            s = torch.full((batch_size,), (step - 1) / T, device=device)
            h = reshape_time(t - s, xt.shape)
            u = v_avg_func(xt, t, s)
            xt = xt - u * h
        return xt
