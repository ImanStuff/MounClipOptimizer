import torch
from torch.optim import Optimizer
import math
from model import CausalSelfAttention

class MuonClipOptimizer(Optimizer):

    def __init__(self, params, lr=1e-4, momentum=0.9, weight_decay=0.1, tau=100.0):
        if not lr >= 0.0: raise ValueError(f"Invalid learning rate: {lr}")
        if not momentum >= 0.0: raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, tau=tau)
        super(MuonClipOptimizer, self).__init__(params, defaults)

    @staticmethod
    def _newton_schulz_5(G, steps=5, eps=1e-7):
        if G.ndim != 2:
            return G

        orig_dtype, orig_device = G.dtype, G.device
        X = G.to(torch.bfloat16) / (torch.norm(G) + eps)

        transposed = G.size(0) > G.size(1)
        if transposed:
            X = X.T

        a, b, c = (3.4445, -4.7750, 2.0315)
        for _ in range(steps):
            A = X @ X.T
            B = b * A + c * (A @ A)
            X = a * X + B @ X

        if transposed:
            X = X.T

        return X.to(orig_device).to(orig_dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            mu = group['momentum']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)

                M_t = state['momentum_buffer'].mul_(mu).add_(grad)

                O_t_base = self._newton_schulz_5(M_t)

                O_t = None
                if p.dim() >= 2:
                    n, m = p.shape
                    scaling_factor = math.sqrt(max(n, m)) * 0.2
                    O_t = O_t_base * scaling_factor
                else:
                    O_t = O_t_base
                update_term = O_t
                if weight_decay > 0:
                    update_term = update_term.add(p, alpha=weight_decay)

                p.add_(update_term, alpha=-lr)

        return loss

    @torch.no_grad()
    def perform_qk_clip(self, model):
        tau = self.param_groups[0]['tau']

        for module in model.modules():
            if isinstance(module, CausalSelfAttention):
                if module.S_max is None:
                    continue

                S_h_max = module.S_max

                for h in range(module.n_head):
                    if S_h_max[h] > tau:
                        gamma = tau / S_h_max[h].item()
                        sqrt_gamma = math.sqrt(gamma)

                        n_embd = module.n_embd
                        head_size = n_embd // module.n_head

                        q_start_index = h * head_size
                        q_end_index = (h + 1) * head_size
                        module.c_attn.weight.data[q_start_index:q_end_index, :].mul_(sqrt_gamma)

                        k_start_index = n_embd + h * head_size
                        k_end_index = n_embd + (h + 1) * head_size
                        module.c_attn.weight.data[k_start_index:k_end_index, :].mul_(sqrt_gamma)