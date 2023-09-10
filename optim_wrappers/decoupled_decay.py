
import torch

class DecoupledDecay(torch.optim.Optimizer):
    def __init__(self, params, decay_type=None, decay_rate=1e-3, decay_threshold=2e-4, **kwargs):
        super().__init__(params, **kwargs)
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.decay_threshold = decay_threshold
        self.decay = {}

    def __repr__(self):
        return "Decayed." + super().__repr__()

    def _store_decay(self):
        if self.decay_type is None:
            return

        for group in self.param_groups:
            for p in group['params']:
                if self.decay_type == "l1":
                    decay = torch.sign(p).clone()

                elif self.decay_type == "hinge":
                    if "exp_avg_sq" in self.state[p]:
                        scale = self.state[p]["exp_avg_sq"].add(1e-8).pow(-0.5)
                    else:
                        scale = torch.ones_like(p)
                    if self.decay_threshold >= 0:
                        decay_region = p.abs() > self.decay_threshold * scale
                    else:
                        decay_region = p.abs() < -self.decay_threshold * scale
                    decay = torch.zeros_like(p)
                    decay[decay_region] = torch.sign(p[decay_region]).clone()

                elif self.decay_type == "l2":
                    decay = p.clone()

                elif self.decay_type == "test":
                    if "old_p" in self.state[p] and self.state[p]["old_p"] is not None:
                        decay = self.state[p]["old_p"].sign() + (p - self.state[p]["old_p"])
                    else:
                        decay = p.sign()
                    decay = decay.clone()

                else:
                    decay = None

                # store decay
                self.decay[p] = decay

    def _apply_decay(self):
        for group in self.param_groups:
            for p in group['params']:
                if self.decay[p] is not None:
                    # XXX: decay proportional to sqrt(v) + 0.1?
                    eps = 1e-1
                    bias_correction2 = 1 - group["betas"][1] ** self.state[p]["step"]
                    vsqrt = self.state[p]["exp_avg_sq"].div(bias_correction2).sqrt().add(eps)
                    p.sub_(self.decay[p] * vsqrt, alpha=group['lr'] * self.decay_rate)
                    self.decay[p] = None

    @torch.no_grad()
    def step(self, closure):
        self._store_decay()
        loss = super().step(closure)
        self._apply_decay()

        return loss
