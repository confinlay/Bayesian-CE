import torch
from torch.optim import Optimizer
from numpy.random import gamma

class H_SA_SGHMC(Optimizer):
    """
    Stochastic Gradient Hamiltonian Monte-Carlo Sampler that uses
    a burn-in procedure to adapt its own hyperparameters during
    the initial stages of sampling.
    """

    def __init__(self, params, lr=1e-2, base_C=0.05,
                 gauss_sig=0.1, alpha0=10, beta0=10):
        """
        Parameters
        ----------
        params : iterable
            Parameters to be optimized.
        lr : float
            Base learning rate for the optimizer.
        base_C : float
            Momentum decay per time-step.
        gauss_sig : float
            Controls the weight_decay = 1/(gauss_sig^2).
        alpha0 : float
            Hyperparameter for prior resampling (Gamma).
        beta0 : float
            Hyperparameter for prior resampling (Gamma).
        """
        self.eps = 1e-6
        self.alpha0 = alpha0
        self.beta0 = beta0

        self.weight_decay = 1.0 / (gauss_sig ** 2) if gauss_sig > 0 else 0.0
        if self.weight_decay <= 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(self.weight_decay))
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if base_C < 0:
            raise ValueError("Invalid friction term: {}".format(base_C))

        defaults = dict(lr=lr, base_C=base_C)
        super(H_SA_SGHMC, self).__init__(params, defaults)

    def step(self, burn_in=False, resample_momentum=False, resample_prior=False):
        """
        Perform one step of SGHMC.
        burn_in : Whether we are in burn-in phase (adapt the variance estimates).
        resample_momentum : Whether to resample the momentum term from N(0, Sigma).
        resample_prior : Whether to resample the weight_decay from a Gamma prior.
        """
        loss = None

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]

                # Initialize state on first usage
                if len(state) == 0:
                    state["iteration"] = 0
                    state["tau"] = torch.ones_like(p)
                    state["g"] = torch.ones_like(p)
                    state["V_hat"] = torch.ones_like(p)
                    state["v_momentum"] = torch.zeros_like(p)
                    state['weight_decay'] = self.weight_decay

                state["iteration"] += 1

                # Optionally resample the prior precision from a Gamma
                if resample_prior:
                    alpha = self.alpha0 + p.data.numel() / 2.0
                    beta = self.beta0 + (p.data ** 2).sum().item() / 2.0
                    gamma_sample = gamma(shape=alpha, scale=1.0 / beta)
                    state['weight_decay'] = gamma_sample

                base_C = group["base_C"]
                lr = group["lr"]
                weight_decay = state["weight_decay"]
                tau, g, V_hat = state["tau"], state["g"], state["V_hat"]

                # Gradient + weight decay
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(p.data, alpha=weight_decay)

                # During burn-in, adapt the variance estimates
                if burn_in:
                    tau.add_(-tau * (g ** 2) / (V_hat + self.eps) + 1.0)
                    tau_inv = 1.0 / (tau + self.eps)
                    g.add_(-tau_inv * g + tau_inv * d_p)
                    V_hat.add_(-tau_inv * V_hat + tau_inv * (d_p ** 2))

                # Preconditioning
                V_sqrt = torch.sqrt(V_hat)
                V_inv_sqrt = 1.0 / (V_sqrt + self.eps)

                # Optionally resample momentum from N(0, Sigma)
                if resample_momentum:
                    state["v_momentum"] = torch.normal(
                        mean=torch.zeros_like(d_p),
                        std=torch.sqrt((lr ** 2) * V_inv_sqrt)
                    )
                v_momentum = state["v_momentum"]

                # Noise term
                noise_var = 2.0 * (lr ** 2) * V_inv_sqrt * base_C - (lr ** 4)
                noise_std = torch.sqrt(torch.clamp(noise_var, min=1e-16))
                noise_sample = torch.normal(mean=torch.zeros_like(d_p), std=noise_std)

                # Update momentum
                v_momentum.add_(
                    - (lr ** 2) * V_inv_sqrt * d_p
                    - base_C * v_momentum
                    + noise_sample
                )

                # Finally update parameter
                p.data.add_(v_momentum)

        return loss
