
import torch
from optim_wrappers.decoupled_decay import DecoupledDecay


class SAMBase(torch.optim.Optimizer):
    def __init__(self, params, rho=0.05, scaled_max='gradnorm', sam_ord=2, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        self.adam_initialized = False
        self.sam_ord = sam_ord
        self.scaled_max = scaled_max
        self.sharpness = []
        super().__init__(params, **kwargs)
        for group in self.param_groups:
            group.setdefault("rho", rho)

    def __repr__(self):
        return "SharpnessAware." + super().__repr__()

    def _init_adam(self):
        for group in self.param_groups:
            for p in group["params"]:
                p.grad = torch.zeros_like(p)
        super().step()  # dummy step
        self.adam_initialized = True

    def _grad_norm(self):
        # put everything on the same device, in case of model parallelism
        shared_device = self.param_groups[0]["params"][0].device
        gradnorm = torch.stack([
            p.grad.pow(2).sum().to(shared_device)
            for group in self.param_groups for p in group["params"] if p.grad is not None
        ]).sum().sqrt()
        return gradnorm

    @torch.no_grad()
    def _find_approx_constrained_max(self, eps=1e-8):
        if self.scaled_max == "gradnorm":
            grad_norm = self._grad_norm()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    self.state[p]["old_p"] = None
                    continue

                # store old parameters
                self.state[p]["old_p"] = p.data.clone()

                # how to scale max step
                if self.scaled_max == "none":
                    denom = torch.ones_like(p)
                    g = p.grad
                elif self.scaled_max == "adam":
                    if "exp_avg_sq" in self.state[p]:
                        bias_correction2 = 1 - group["betas"][1] ** self.state[p]["step"]
                        denom = self.state[p]["exp_avg_sq"].div(bias_correction2).sqrt().add(0.1)
                    else:
                        denom = torch.ones_like(p)
                    g = p.grad
                elif self.scaled_max == "gradnorm":
                    denom = grad_norm.add(eps).to(p)
                    g = p.grad
                else:
                    raise NotImplementedError(self.scaled_max)

                # adjust based on ord p (as in ||e||p <= 1)
                if self.sam_ord == 1:
                    denom = torch.ones_like(g)
                    i_max = g.argmax()
                    sign_max = g.view(-1)[i_max].sign()
                    g.zero_()
                    g.view(-1)[i_max] = sign_max
                elif self.sam_ord == 'inf':
                    denom = torch.ones_like(g)
                    g = g.sign()
                else:
                    pass

                # do step
                p.addcdiv_(g, denom, value=group["lr"] * group["rho"])
                p.grad = None

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        if not self.adam_initialized:
            self._init_adam()

        loss = closure()
        self._find_approx_constrained_max()
        # XXX: decay after max step?
        if isinstance(self, DecoupledDecay):
            self._store_decay()
        loss_max = closure()  # get grad at approx max of inner problem (|e|| <= rho)
        self.sharpness.append((loss_max - loss).item())
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or self.state[p]["old_p"] is None:
                    continue
                p.data.copy_(self.state[p]["old_p"])  # set params back to w
                self.state[p]["old_p"] = None

        super().step()  # "sharpness-aware" super()-agnostic update (closure already evaluated)

        return loss


class SAM(SAMBase, torch.optim.SGD):
    pass

class SADAM(SAMBase, torch.optim.Adam):
    pass

