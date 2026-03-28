import torch
import torch.nn as nn

def MeanFlowEuclidean_loss(v_avg_fun, xt, v, t, s, v_avg=None, verbose=False,version="mf"):
    with torch.amp.autocast("cuda", enabled=False):

        if version not in ["mf","imf","x-pixel"]: 
            raise ValueError(f"Unsupported version {version}, expected one of ['mf','imf','x-pixel']")
        
        h = t - s
        dt = torch.ones_like(t)
        ds = torch.zeros_like(s)
        if version == "mf":
            dz= v 
        if version in ["imf","x-pixel"]:
            dz = v_avg_fun(xt, t, t) #Eq (12) in [2]
        _, du_dt = torch.func.jvp(
            v_avg_fun,
            (xt, t, s),
            (dz, dt, ds),
        )

        if v_avg is None:
            v_avg = v_avg_fun(xt, t, s)

        h_b = h.view((h.shape[0],) + (1,) * (xt.ndim - 1))
        vtgt = v_avg + h_b * du_dt.detach()

        diff2 = (v - vtgt) ** 2
        
        sum_dims = tuple(range(1, diff2.dim()))
        loss = diff2.sum(dim=sum_dims)
            

        if verbose:
            dbg = {
                "h": h.detach(),
                "v_avg": v_avg.detach(),
                "du_dt": du_dt.detach(),
                "vtgt": vtgt.detach(),
                "v": v.detach(),
            }
            return loss, dbg
        return loss