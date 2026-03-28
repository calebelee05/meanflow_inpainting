import torch
import numpy as np
import ot 


import jax
import jax.numpy as jnp
from ott.geometry import geometry
from ott.solvers.linear import solve

class JaxLOT:
    """
    Linear OT (LOT) blockwise solver using JAX and ott.
    """

    def __init__(self, X1, X2, X_ref):
        self.X1 = jnp.asarray(X1)
        self.X2 = jnp.asarray(X2)
        self.X_ref = jnp.asarray(X_ref)
        self.n = self.X1.shape[0]
        self.n_ref = self.X_ref.shape[0]
        assert self.n % self.n_ref == 0, "n must be divisible by n_ref"
        self.group_size = self.n // self.n_ref

    def LOT_embed(self):
        """
        Compute OT plans from reference to X1 and X2, and store gamma1, gamma2.
        """
        C_ref_X1 = jnp.linalg.norm(self.X_ref[:, None, :] - self.X1[None, :, :], axis=-1)
        C_ref_X2 = jnp.linalg.norm(self.X_ref[:, None, :] - self.X2[None, :, :], axis=-1)
        p_ref = jnp.ones(self.n_ref) / self.n_ref
        p_X = jnp.ones(self.n) / self.n
        p_Y = jnp.ones(self.n) / self.n
        geom1 = geometry.Geometry(cost_matrix=C_ref_X1)
        geom2 = geometry.Geometry(cost_matrix=C_ref_X2)
        gamma1 = solve(geom1, a=p_ref, b=p_X).matrix
        gamma2 = solve(geom2, a=p_ref, b=p_Y).matrix
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.p_ref = p_ref
        return gamma1, gamma2

    def group_ot_solver(self):
        """
        Use gamma1, gamma2 to group X1, X2, then solve OT for each group and assemble the total plan.
        """
        if not hasattr(self, 'gamma1') or not hasattr(self, 'gamma2'):
            self.LOT_embed()

        gamma1_np = jnp.array(self.gamma1)
        gamma2_np = jnp.array(self.gamma2)
        groups_X1 = [jnp.argsort(-gamma1_np[i])[:self.group_size] for i in range(self.n_ref)]
        groups_X2 = [jnp.argsort(-gamma2_np[i])[:self.group_size] for i in range(self.n_ref)]

        C_X1_X2 = jnp.linalg.norm(self.X1[:, None, :] - self.X2[None, :, :], axis=-1)
        idx1_list = jnp.array(groups_X1)
        idx2_list = jnp.array(groups_X2)
        sub_C_batch = jnp.stack([C_X1_X2[jnp.ix_(idx1, idx2)] for idx1, idx2 in zip(idx1_list, idx2_list)], axis=0)
        sub_p = jnp.ones((self.n_ref, self.group_size)) / self.group_size

        def solve_match_linear(C, a, b):
            geom = geometry.Geometry(cost_matrix=C)
            out = solve(geom, a=a, b=b)
            return out.matrix

        sub_plan_batch = jax.vmap(solve_match_linear)(sub_C_batch, sub_p, sub_p)  # (n_ref, group_size, group_size)

        total_plan = jnp.zeros((self.n, self.n))
        for i in range(self.n_ref):
            idx1 = idx1_list[i]
            idx2 = idx2_list[i]
            total_plan = total_plan.at[jnp.ix_(idx1, idx2)].set(sub_plan_batch[i])
        total_plan = total_plan / self.n_ref
        return total_plan

    def independent_coupling(self):
        """
        Compute the independent coupling: gamma1.T @ (gamma2 / p_ref[:, None])
        Returns a (n, n) matrix.
        """
        if not hasattr(self, 'gamma1') or not hasattr(self, 'gamma2'):
            self.LOT_embed()
        coupling = self.gamma1.T @ (self.gamma2 / self.p_ref[:, None])
        return coupling
    
    
class NumpyLOT:
    """
    Linear OT (LOT) blockwise solver using numpy and POT (ot).
    """

    def __init__(self, X1, X2, X_ref):
        self.X1 = X1
        self.X2 = X2
        self.X_ref = X_ref
        self.n = X1.shape[0]
        self.n_ref = X_ref.shape[0]
        assert self.n % self.n_ref == 0, "n must be divisible by n_ref"
        self.group_size = self.n // self.n_ref

    def LOT_embed(self):
        """
        Compute OT plans from reference to X1 and X2, and store gamma1, gamma2.
        """
        C_ref_X1 = ot.dist(self.X_ref, self.X1,metric='euclidean')
        C_ref_X2 = ot.dist(self.X_ref, self.X2,metric='euclidean')
        p_ref = np.ones(self.n_ref) / self.n_ref
        p_X = np.ones(self.n) / self.n
        p_Y = np.ones(self.n) / self.n
        gamma1 = ot.emd(p_ref, p_X, C_ref_X1)
        gamma2 = ot.emd(p_ref, p_Y, C_ref_X2)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.p_ref = p_ref
        return gamma1, gamma2

    def group_ot_solver(self):
        """
        Use gamma1, gamma2 to group X1, X2, then solve OT for each group and assemble the total plan.
        """
        # Ensure gamma1, gamma2 are computed
        if not hasattr(self, 'gamma1') or not hasattr(self, 'gamma2'):
            self.LOT_embed()

        groups_X1 = [np.argsort(-self.gamma1[i])[:self.group_size] for i in range(self.n_ref)]
        groups_X2 = [np.argsort(-self.gamma2[i])[:self.group_size] for i in range(self.n_ref)]

        C_X1_X2 = ot.dist(self.X1, self.X2,metric='euclidean')
        total_plan = np.zeros((self.n, self.n))
        for i in range(self.n_ref):
            idx1 = groups_X1[i]
            idx2 = groups_X2[i]
            sub_C = C_X1_X2[np.ix_(idx1, idx2)]
            sub_p = np.ones(self.group_size) / self.group_size
            sub_plan = ot.emd(sub_p, sub_p, sub_C)
            total_plan[np.ix_(idx1, idx2)] = sub_plan
        total_plan /= self.n_ref
        return total_plan

    def independent_coupling(self):
        """
        Compute the independent coupling: gamma1.T @ gamma2 / p_ref
        Returns a (n, n) matrix.
        """
        if not hasattr(self, 'gamma1') or not hasattr(self, 'gamma2'):
            self.LOT_embed()
      
        coupling = self.gamma1.T @ (self.gamma2 / self.p_ref[:, None])
        return coupling
    
class OTPlanSampler:
    """
    Optimal Transport plan sampler for minibatch OT.
    Supports exact OT and Sinkhorn algorithm.
    """
    def __init__(self, method="sinkhorn", reg=0.5, numItermax=100, stopThr=1e-9,mass=0.8,numThreads=1):
        """
        Args:
            method: "exact" for linear programming, "sinkhorn" for entropic regularization
            reg: Regularization parameter for Sinkhorn (ignored for exact OT)
            numItermax: Maximum number of iterations for Sinkhorn
            stopThr: Stop threshold for Sinkhorn
        """
        # if not OT_AVAILABLE:
        #     raise ImportError("POT library is required for OT sampling. Install with: pip install pot")
        # print(f"[DEBUG] OTPlanSampler initialized with:")
        # print(f"  method={method}")
        # print(f"  reg={reg}")
        # print(f"  numItermax={numItermax}")
        # print(f"  stopThr={stopThr}")
        self.method = method
        self.reg = reg
        self.numItermax = numItermax
        self.stopThr = stopThr
        self.mass=0.9
        self.numThreads=numThreads
        
    def compute_cost_matrix(self, x0, x1):
        """
        Compute the cost matrix between x0 and x1.
        Using squared Euclidean distance.
        """
        x0_flat = x0.reshape(x0.shape[0], -1)
        x1_flat = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0_flat, x1_flat, p=2)**2 
        return M

    def get_ot_plan(self, x0, x1):
        """
        Compute the optimal transport plan between x0 and x1.
        Returns a coupling matrix π of shape (batch_size, batch_size).
        """
        batch_size = x0.shape[0]
        device = x0.device
        # Compute cost matrix
        M = self.compute_cost_matrix(x0, x1)
        # Convert to numpy for POT
        M_np = M.detach().cpu().numpy().astype(np.float64)
        # Uniform distributions (can be modified if needed)
        a = np.ones(batch_size, dtype=np.float64) / batch_size
        b = np.ones(batch_size, dtype=np.float64) / batch_size
        if self.method == "exact":
            numItermax = max(self.numItermax,1e7, a.shape[0] * 1000)
            # Exact OT using linear programming (POT)
            pi = ot.emd(a, b, M_np, numItermax=numItermax,numThreads=self.numThreads)
        elif self.method == "exact_ott":
            # Exact OT using ott.jax solver

            M_jax = jnp.asarray(M_np)
            a_jax = jnp.asarray(a)
            b_jax = jnp.asarray(b)
            geom = geometry.Geometry(cost_matrix=M_jax)
            out = solve(geom, a=a_jax, b=b_jax)
            pi = np.array(out.matrix)
        elif self.method == "sinkhorn":
            # Sinkhorn algorithm (entropic regularization)
            x0_flat = x0.reshape(batch_size, -1)
            x1_flat = x1.reshape(batch_size, -1)
            data_std = float((x0_flat.std() + x1_flat.std()).item() / 2.0)
            
            dynamic_reg = data_std * 0.1
            
            actual_reg = max(dynamic_reg, 0.05)  # 至少0.05
            
            # print(f"Data std: {data_std:.4f}, Using reg: {actual_reg:.4f}")
            
            M_np = M_np / (M_np.max() + 1e-10)
            
            pi = ot.bregman.sinkhorn_stabilized(
                a, b, M_np,
                reg=actual_reg,
                numItermax=100,
                stopThr=1e-7,
                verbose=False,
                warn=False
            )

                
        elif self.method=='lot-ind':
            # 计算均值和标准差
            mean_x0,mean_x1 = x0.mean(dim=0),x1.mean(dim=0)   
            std_x0,std_x1 = x0.std(dim=0),x1.std(dim=0)
            mean_ref = (mean_x0 + mean_x1) / 2
            std_ref = (std_x0 + std_x1) / 2
            # 生成参考点
            x_ref = torch.randn(batch_size, x0.shape[1], device=x0.device) * std_ref + mean_ref
            LOT_solver = JaxLOT(x0, x1, x_ref)
            pi = LOT_solver.independent_coupling()
        elif self.method=='lot-group':
            # 计算均值和标准差
            mean_x0,mean_x1 = x0.mean(dim=0),x1.mean(dim=0)   
            std_x0,std_x1 = x0.std(dim=0),x1.std(dim=0)
            mean_ref = (mean_x0 + mean_x1) / 2
            std_ref = (std_x0 + std_x1) / 2
            # 生成参考点
            x_ref = torch.randn(batch_size, x0.shape[1], device=x0.device) * std_ref + mean_ref
            LOT_solver = JaxLOT(x0, x1, x_ref)
            pi = LOT_solver.group_coupling()
        elif self.method =='partial_ot':
            numItermax = max(self.numItermax,1e6, a.shape[0] * 1000)
            pi=ot.partial.partial_wasserstein(a, b, M_np, m=self.mass, nb_dummies=1, log=False,numItermax=numItermax)
        else:
            raise ValueError(f"Unknown OT method: {self.method}")
        # Convert back to torch tensor
        pi = torch.from_numpy(pi).float().to(device)
        return pi
    
    def sample_from_plan(self, pi, batch_size=None, replace=True):
        """
        Sample indices according to the OT plan.
        Returns source and target indices for pairing.
        """
        if batch_size is None:
            batch_size = pi.shape[0]
        # Flatten the coupling matrix
        pi_flat = pi.flatten()
        # Sample according to the distribution
        indices = torch.multinomial(pi_flat, batch_size, replacement=replace)
        # Convert flat indices back to (i, j) pairs
        i_indices = indices // pi.shape[1]
        j_indices = indices % pi.shape[1]
        return i_indices, j_indices
    
    def sample_plan(self, x0, x1, replace=False):
        """
        Main interface: compute OT plan and sample pairs.
        Returns reordered x0 and x1 that are optimally coupled.
        """
        # Compute OT plan
        pi = self.get_ot_plan(x0, x1)
        # Sample indices
        if self.method in ['partial_ot']:
            batch_size=int(self.mass*x0.shape[0])
        else:
            batch_size = x0.shape[0]
        
        i_indices, j_indices = self.sample_from_plan(pi, batch_size=batch_size, replace=replace)
        # Return reordered samples
        return x0[i_indices], x1[j_indices]