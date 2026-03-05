from typing import Literal
import pathlib
import time

import numpy as np
import torch
import scipy.stats
import matplotlib.colors
import matplotlib.patches
import matplotlib.collections
import matplotlib.pyplot as plt

from .config import SUPPORTED_GW_EVENTS, CBC_PE_POSTERIOR_FILES
from .utils import _read_prior_or_posterior_file


def learn_flow(
    data: torch.Tensor,
    flow,
    optimizer,
    N_epochs: int,
    batch_size: int,
    save_file: str,
    stop_early_if_no_improvement_in_n_epochs: int = 0, 
) -> list:  
    assert pathlib.Path(save_file).parent.exists(), "directory for given save_file doesn't exist"
    training_summary = f"data.shape: {data.shape}\nflow: {flow}\noptimizer: {optimizer}\nN_epochs: {N_epochs}\nbatch_size: {batch_size}\n"            

    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(data),
        batch_size=batch_size, 
        shuffle=True
    )

    start = time.perf_counter()
    epoch_mean_losses = []
    minimum_epoch_mean_loss = torch.inf
    best_epoch = 0
    for epoch in range(N_epochs + 1):
        losses = []

        for d in train_loader:
            # minimize expected KL divergence
            loss = -flow().log_prob(torch.stack(d)).mean() # -log p(x)
            if not torch.isfinite(loss).item():
                print(f'Aborting: loss = nan at epoch {epoch}.')
                return epoch_mean_losses
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach())
        
        losses = torch.stack(losses)
        epoch_mean_loss = losses.mean().item()
        epoch_mean_losses.append(epoch_mean_loss)

        if epoch % 10 == 0:
            progress = f"[{epoch:6d} / {N_epochs}]\tavg. loss = {epoch_mean_loss:3.4f} +- {losses.std().item():3.4f}"
            training_summary += f"{progress}\n"
            print(f'{progress}')
        
        if stop_early_if_no_improvement_in_n_epochs > 0:
            if epoch_mean_loss < minimum_epoch_mean_loss:
                minimum_epoch_mean_loss = epoch_mean_loss
                best_epoch = epoch
            else:
                if epoch - best_epoch >= stop_early_if_no_improvement_in_n_epochs:
                    print(f'Stopping early - no improvement in loss after {stop_early_if_no_improvement_in_n_epochs} epochs.')
                    break
    
    end = time.perf_counter()
    torch.save(flow, save_file)

    training_summary += f"\ntrain time: {(end - start) / 60:.2f} minutes"
    
    model_save_file = pathlib.Path(save_file)
    summary_save_file = model_save_file.parent.joinpath(model_save_file.stem + '_train_summary.txt')
    summary_save_file.touch(exist_ok=True)
    summary_save_file.write_text(training_summary)

    return epoch_mean_losses


def get_gw_event_pe_posterior_samples(event: str, method: Literal['2D', '3D']):
    posterior_file = CBC_PE_POSTERIOR_FILES[event][method]
    
    samples = _read_prior_or_posterior_file(posterior_file, method)
    
    if method == '2D':
        return torch.tensor(samples['lambdat'], dtype=torch.float32), torch.tensor(samples['q'], dtype=torch.float32)
    elif method == '3D':
        return torch.tensor(samples['lambda1'], dtype=torch.float32), \
            torch.tensor(samples['q'], dtype=torch.float32), \
            torch.tensor(samples['lambda2'], dtype=torch.float32)
    else:
        raise NotImplementedError()


def plot_probability(box_probs: torch.Tensor, title: str, save_file: str | None = None):    
    N_boxes = len(box_probs)
    patches = []
    prob_values = []
    width, height = box_probs[1, 1, 0] - box_probs[0, 0, 0], box_probs[1, 1, 1] - box_probs[0, 0, 1]
    for i in range(N_boxes):
        for j in range(N_boxes):
            box = box_probs[i, j]
            patches.append(
                matplotlib.patches.Rectangle(
                    (box[0].item(), box[1].item()), 
                    width.item(), 
                    height.item()
                )
            )
            prob_values.append(box[2]) # the prob value

    collection = matplotlib.collections.PatchCollection(
        patches, 
        cmap=plt.colormaps['inferno'],
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=box_probs[:, :, 2].max().item())
    )
    collection.set_array(np.array(prob_values))

    fig, ax = plt.subplots()
    ax.add_collection(collection)
    fig.colorbar(collection, label='Probability')
    ax.set_xlim(0., 5000.)
    ax.set_ylim(0., 1.)
    ax.set_xlabel(r'$\tilde{\Lambda}$')
    ax.set_ylabel(r'$q$')
    ax.set_title(title)
    
    if save_file is not None:
        fig.savefig(save_file)
    else:
        plt.show()


def _to_latent_space(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2, 'x should be an (N, D) shaped Tensor'
    assert x.shape[-1] in (2, 3), 'last dimension of x must be size 2 or 3'
    
    if x.shape[-1] == 2:
        z = torch.stack((torch.log(x[:, 0]), torch.logit(x[:, 1])), dim=-1)
    else:
        z = torch.stack((torch.log(x[:, 0]), torch.logit(x[:, 1]), torch.log(x[:, 2])), dim=-1)
    return z


def _to_data_space(z: torch.Tensor):
    assert z.ndim == 2, 'z should be an (N, D) shaped Tensor'
    assert z.shape[-1] in (2, 3), 'last dimension of z must be size 2 or 3'
    
    if z.shape[-1] == 2:
        x = torch.stack((torch.exp(z[:, 0]), torch.sigmoid(z[:, 1])), dim=-1)
    else:
        x = torch.stack((torch.exp(z[:, 0]), torch.sigmoid(z[:, 1]), torch.exp(z[:, 2])), dim=-1)
    return x


def _get_log_abs_det_jacobian(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2, 'x should be an (N, D) shaped Tensor'
    assert x.shape[-1] in (2, 3), 'last dimension of x must be size 2 or 3'
    
    if x.shape[-1] == 2:
        lambdat, q = x[:, 0], x[:, 1]
        ladj = -torch.log(lambdat) - torch.log(q) - torch.log(1 - q)
    else:
        lambda1, q, lambda2 = x[:, 0], x[:, 1], x[:, 2]
        ladj = -torch.log(lambda1) - torch.log(q) - torch.log(1 - q) - torch.log(lambda2)
    return ladj


def _scale_down(x: torch.Tensor, lambdat_max=None, lambda1_max=None, lambda2_max=None) -> torch.Tensor:
    assert x.ndim == 2, 'x should be an (N, D) shaped Tensor'
    assert x.shape[-1] in (2, 3), 'last dimension of x must be size 2 or 3'
    
    if x.shape[-1] == 2:
        assert lambdat_max is not None
        # bring LambdaT to original range
        w = torch.stack((x[:, 0] / lambdat_max, x[:, 1]), dim=-1)
    else:
        assert lambda1_max is not None and lambda2_max is not None
        # bring Lambda1 and Lambda2 to original range
        w = torch.stack(
            (
                x[:, 0] / lambda1_max, 
                x[:, 1], 
                x[:, 2] / lambda2_max
            ),
            dim=-1
        )

    return w

def _scale_up(w: torch.Tensor, lambdat_max=None, lambda1_max=None, lambda2_max=None) -> torch.Tensor:
    assert w.ndim == 2, 'w should be an (N, D) shaped Tensor'
    assert w.shape[-1] in (2, 3), 'last dimension of w must be size 2 or 3'
    
    if w.shape[-1] == 2:
        assert lambdat_max is not None
        # bring LambdaT to original range
        x = torch.stack((w[:, 0] * lambdat_max, w[:, 1]), dim=-1)
    else:
        assert lambda1_max is not None and lambda2_max is not None
        # bring Lambda1 and Lambda2 to original range
        x = torch.stack(
            (
                w[:, 0] * lambda1_max, 
                w[:, 1], 
                w[:, 2] * lambda2_max
            ),
            dim=-1
        )
    return x

def _scale_jacobian(lambdat_max=None, lambda1_max=None, lambda2_max=None):
    if lambdat_max is not None:
        return 1 / lambdat_max
    else:
        assert lambda1_max is not None and lambda2_max is not None
        return 1 / (lambda1_max * lambda2_max)

class NormalizingFlow:
    def __init__(
            self,
            event: str,
            flow_file: str,
            method: Literal['2D', '3D'] = '2D'
    ):    
        assert method in ['2D', '3D'], "method must be one of ['2D', '3D']"
        self.method = method
        
        assert event in SUPPORTED_GW_EVENTS, f"event must be one of {SUPPORTED_GW_EVENTS}."
        self.event = event
        self.flow = torch.load(flow_file, weights_only=False)

        # Need everything else below if reflection method is used for bounding output
        # instead of transformation method (default).
        self.posterior_samples = torch.stack(
            get_gw_event_pe_posterior_samples(event, method),
            dim=-1
        )

        if method == '2D':
            self.low = torch.tensor([0., 0.], dtype=torch.float32)
            self.high = torch.tensor([torch.inf, 1.], dtype=torch.float32)
        
        elif method == '3D':
            self.low = torch.tensor([0., 0., 0.], dtype=torch.float32)
            self.high = torch.tensor([torch.inf, 1., torch.inf], dtype=torch.float32)
            
    def set_flow_model(self, flow_file: str):
        self.flow = torch.load(flow_file, weights_only=False)

    def sample(self, size: int, bound_method="transform") -> torch.Tensor:
        if bound_method == "transform":
            z = self.flow().sample([size])
            x = _to_data_space(z)
        elif bound_method == "reflect":
            w = self.flow().sample([size])
            if w.shape[-1] == 2:
                x = _scale_up(w, lambdat_max=self.posterior_samples[0].max())
            else:
                x = _scale_up(w, lambda1_max=self.posterior_samples[0].max(), lambda2_max=self.posterior_samples[2].max())
        return x

    def log_pdf(self, x: torch.Tensor, bound_method="transform") -> torch.Tensor:
        if bound_method == "transform":
            z = _to_latent_space(x)
            ladj = _get_log_abs_det_jacobian(x)
            lp = self.flow().log_prob(z).detach() + ladj
            lp = torch.nan_to_num(lp, nan=-torch.inf)
            return lp
        elif bound_method == "reflect":
            return torch.log(self.pdf(x, bound_method="reflect"))
        else:
            raise NotImplementedError()
    
    def pdf(self, x: torch.Tensor, bound_method="transform") -> torch.Tensor:
        if bound_method == "transform":
            return torch.exp(self.log_pdf(x))
        elif bound_method == "reflect":
            if x.shape[-1] == 2:
                w = _scale_down(x, lambdat_max=self.posterior_samples[0].max())
            else:
                w = _scale_down(x, lambda1_max=self.posterior_samples[0].max(), lambda2_max=self.posterior_samples[2].max())
            p = torch.exp(self.flow().log_prob(w).detach())
            
            for i, (low, high) in enumerate(zip(self.low, self.high)):                
                if torch.isfinite(low):
                    reflect_w = w.clone()
                    reflect_w[:, i] = 2.0 * low - w[:, i]

                    p += torch.exp(self.flow().log_prob(reflect_w).detach())

                if torch.isfinite(high):
                    reflect_w = w.clone()
                    reflect_w[:, i] = 2.0 * high - w[:, i]

                    p += torch.exp(self.flow().log_prob(reflect_w).detach())

            # jacobian of transformation to whitened space
            if x.shape[-1] == 2:
                p *= _scale_jacobian(lambdat_max=self.posterior_samples[0].max())
            else:
                p *= _scale_jacobian(lambda1_max=self.posterior_samples[0].max(), lambda2_max=self.posterior_samples[2].max())
            return p
        else:
            raise NotImplementedError()
    
    def integrate_pdf_over_box(self, low, high, N_grid: int):
        if self.method == '2D':
            lambdat = torch.linspace(low[0], high[0], N_grid)
            q = torch.linspace(low[1], high[1], N_grid)
            
            lambdat_grid, q_grid = torch.meshgrid([lambdat, q], indexing='xy')
            points = torch.stack((lambdat_grid, q_grid), dim=-1).reshape((N_grid*N_grid), 2)
            
            prob_grid = self.pdf(points).reshape((N_grid, N_grid))

            # Integrates over q (dim=1), then lambdat (dim=0)
            q_integral = torch.trapezoid(prob_grid, q, dim=1)
            prob = torch.trapezoid(q_integral, lambdat)
        
        else:
            raise NotImplementedError()
        
        return prob
    
    def compute_probability_over_grid(self, lower_bounds, upper_bounds, N_grid: int, save_file: str):
        eps = 1e-5

        if self.method == '2D':
            lambdat_grid, q_grid = torch.meshgrid(
                torch.linspace(lower_bounds[0] + eps, upper_bounds[0], N_grid),
                torch.linspace(lower_bounds[1] + eps, upper_bounds[1], N_grid),
                indexing='xy'
            )

            x_grid = torch.stack((lambdat_grid, q_grid), dim=-1)
            box_probs = torch.zeros((N_grid - 1, N_grid - 1, 3))

            for i in range(N_grid - 1):
                for j in range(N_grid - 1):
                    low = x_grid[i, j]
                    high = x_grid[i + 1, j + 1]

                    box_prob = self.integrate_pdf_over_box(low, high, 10)
                    box_probs[i, j] = torch.hstack((low, box_prob))
        else:
            raise NotImplementedError()
        
        torch.save(box_probs, save_file)

    def plot_density(
            self, 
            N_grid: int = 200, 
            save_file: str | None = None
    ):
        if self.method == '2D':        
            lambdat, q = get_gw_event_pe_posterior_samples(self.event, self.method) # type: ignore

            fig, ax = plt.subplots(figsize=(7, 7))
            lt_grid, q_grid = torch.meshgrid([torch.linspace(0., lambdat.max(), N_grid), torch.linspace(0.0, 1.0, N_grid)], indexing='xy')
            points = torch.stack([lt_grid, q_grid], dim=-1).reshape((N_grid**2, 2))
            lp = self.log_pdf(points).reshape((N_grid, N_grid)).detach()
            p = torch.exp(lp)

            qcs = ax.contourf(lt_grid, q_grid, p, levels=50, cmap='inferno')
            plt.colorbar(qcs, ax=ax)
            ax.scatter(x=lambdat, y=q, s=0.18, c='gray', alpha=0.30)
            ax.set_title(self.event)
            ax.set_xlabel(r'$\tilde{\Lambda}$', fontsize=14)
            ax.set_ylabel(r'$q$', fontsize=14)
            plt.legend()

        else:
            lambda1, q, lambda2 = get_gw_event_pe_posterior_samples(self.event, self.method) # type: ignore

            fig, ax = plt.subplots(1, 3, figsize=(18, 6), width_ratios=[0.20, 0.20, 0.20])

            l1_arr = torch.linspace(0., lambda1.max(), N_grid)
            q_arr = torch.linspace(0., 1.0, N_grid)
            l2_arr = torch.linspace(0., lambda2.max(), N_grid)

            l1_grid, q_grid, l2_grid = torch.meshgrid([l1_arr, q_arr, l2_arr], indexing='xy')
            points = torch.stack([l1_grid, q_grid, l2_grid], dim=-1).reshape((N_grid**3, 3))

            p = self.pdf(points).reshape((N_grid, N_grid, N_grid))

            l1_q_p = torch.trapezoid(p, l2_arr, dim=2)
            l1_l2_p = torch.trapezoid(p, l1_arr, dim=0).T
            l2_q_p = torch.trapezoid(p, q_arr, dim=1)

            qcs = ax[0].contourf(l1_arr, q_arr, l1_q_p, cmap='inferno', levels=30)
            plt.colorbar(qcs, ax=ax[0])
            ax[0].scatter(lambda1, q, s=0.1, c='gray', alpha=0.10)
            ax[0].set_xlabel(r"$\Lambda_1$")
            ax[0].set_ylabel(r"$q$")

            qcs = ax[1].contourf(l2_arr, q_arr, l2_q_p, cmap='inferno', levels=30)
            plt.colorbar(qcs, ax=ax[1])
            ax[1].scatter(lambda2, q, s=0.2, c='gray', alpha=0.30)
            ax[1].set_xlabel(r"$\Lambda_2$")
            ax[1].set_ylabel(r"$q$")

            qcs = ax[2].contourf(l1_arr, l2_arr, l1_l2_p, cmap='inferno', levels=30)
            plt.colorbar(qcs, ax=ax[2])
            ax[2].scatter(lambda1, lambda2, s=0.2, c='gray', alpha=0.30)
            ax[2].set_xlabel(r"$\Lambda_1$")
            ax[2].set_ylabel(r"$\Lambda_2$")

        if save_file is not None:
            fig.savefig(save_file, bbox_inches='tight', dpi=300)
        else:
            plt.show()
    

class BoundedKDE:
    def __init__(
            self,
            event: str,
            method: Literal['2D', '3D'] = '2D',
            Ns=None
    ):    
        assert method in ['2D', '3D'], "method must be one of ['2D', '3D']"
        self.method = method
        assert event in SUPPORTED_GW_EVENTS, f"event must be one of {SUPPORTED_GW_EVENTS}."
        self.event = event
        
        self.posterior_samples = torch.stack(
            get_gw_event_pe_posterior_samples(event, method),
            dim=-1
        )

        if Ns is not None:
            self.posterior_samples = self.posterior_samples[:Ns]

        if method == '2D':
            whitened_samples = _scale_down(self.posterior_samples, lambdat_max=self.posterior_samples[0].max())
        else:
            whitened_samples = _scale_down(self.posterior_samples, lambda1_max=self.posterior_samples[0].max(), lambda2_max=self.posterior_samples[2].max())
        self.base_kde = scipy.stats.gaussian_kde(whitened_samples.T.numpy())

        if method == '2D':
            self.low = torch.tensor([0., 0.], dtype=torch.float32)
            self.high = torch.tensor([torch.inf, 1.], dtype=torch.float32)
        
        elif method == '3D':
            self.low = torch.tensor([0., 0., 0.], dtype=torch.float32)
            self.high = torch.tensor([torch.inf, 1., torch.inf], dtype=torch.float32)
    
    def log_pdf(self, x: torch.Tensor, resample: bool = False) -> torch.Tensor:
        return torch.log(self.pdf(x, resample))

    def pdf(self, x: torch.Tensor, resample: bool = False) -> torch.Tensor:
        """Return an estimate of the density evaluated at the given points."""
        if resample:
            X = self.sample(len(self.posterior_samples))
            if X.shape[-1] == 2:
                W = _scale_down(X, lambdat_max=self.posterior_samples[0].max())
            else:
                W = _scale_down(X, lambda1_max=self.posterior_samples[0].max(), lambda2_max=self.posterior_samples[2].max())
            kde = scipy.stats.gaussian_kde(W.T.numpy())
        else:
            kde = self.base_kde               
        
        if x.shape[-1] == 2:
            w = _scale_down(x, lambdat_max=self.posterior_samples[0].max())
        else:
            w = _scale_down(x, lambda1_max=self.posterior_samples[0].max(), lambda2_max=self.posterior_samples[2].max())
        p = torch.tensor(kde(w.T.numpy()).T, dtype=torch.float32)
        
        for i, (low, high) in enumerate(zip(self.low, self.high)):                
            if torch.isfinite(low):
                reflect_w = w.clone()
                reflect_w[:, i] = 2.0 * low - w[:, i]

                p += torch.tensor(kde(reflect_w.T.numpy()).T, dtype=torch.float32)

            if torch.isfinite(high):
                reflect_w = w.clone()
                reflect_w[:, i] = 2.0 * high - w[:, i]
                
                p += torch.tensor(kde(reflect_w.T.numpy()).T, dtype=torch.float32)
        
        # jacobian of transformation to whitened space
        if x.shape[-1] == 2:
            p *= _scale_jacobian(lambdat_max=self.posterior_samples[0].max())
        else:
            p *= _scale_jacobian(lambda1_max=self.posterior_samples[0].max(), lambda2_max=self.posterior_samples[2].max())

        return p
    
    def sample(self, size: int) -> torch.Tensor:
        samples = torch.tensor(self.base_kde.resample(size), dtype=torch.float32).T
        if samples.shape[-1] == 2:
            samples = _scale_up(samples, lambdat_max=self.posterior_samples[0].max())
        else:
            samples = _scale_up(samples, lambda1_max=self.posterior_samples[0].max(), lambda2_max=self.posterior_samples[2].max())

        for i, (low, high) in enumerate(zip(self.low, self.high)):
            if low is not None:
                samples[:, i][samples[:, i] < low] = 2. * low - samples[:, i][samples[:, i] < low]
                    
            if high is not None:
                samples[:, i][samples[:, i] > high] = 2. * high - samples[:, i][samples[:, i] > high]

        return samples