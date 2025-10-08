from typing import Literal
import pathlib
import time
import multiprocessing

import numpy as np
import torch
import scipy.stats

import matplotlib.colors
import matplotlib.patches
import matplotlib.collections
import matplotlib.pyplot as plt

from .config import SUPPORTED_EVENTS, GW_PE_POSTERIOR_FILES, GWXTREME_FLOW_FILES, GWXTREME_KDE_GRID_FILES
from .utils import (
    _read_prior_or_posterior_file, 
    get_eos_interpolant_from_parameters, 
    get_eos_interpolant, 
    get_lambdat_for_eos, 
    apply_mass_constraint,
    get_masses
)


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


def learn_ensemble(
    event: str,
    method: str,
    flow_constructor,
    flow_kwargs: dict,
    optimizer_constructor,
    optimizer_kwargs: dict,
    save_dir: str,
    N_ensemble: int = 32,
    N_epochs: int = 50,
    batch_size: int = 100,
    N_processors: int = 1,
    resample_size: int | None = None,
    native_flow=None
):
    process_args = []
    for i in range(N_ensemble):
        # same flow architecture and optimizer for all in ensemble for now
        flow = flow_constructor(**flow_kwargs)
        optimizer = optimizer_constructor(params=flow.parameters(), **optimizer_kwargs)

        if resample_size is None:
            # Not resampling, using real data
            data = get_gw_event_pe_posterior_samples(event, method)
            X = torch.stack(data, dim=-1)
            Z = _to_latent_space(X)
        else:
            assert native_flow is not None
            Z = native_flow().sample([resample_size])
        
        process_args.append((Z, flow, optimizer, N_epochs, batch_size, f"{save_dir}/ensemble_flow_{i}.pkl"))
    
    if N_processors > 1:
        with multiprocessing.Pool(processes=N_processors) as pool:
            pool.starmap(learn_flow, process_args)
    else:
        for args in process_args:
            learn_flow(*args)


def score_ensemble(event: str, method: str, ensemble_dir: str, save_file: str | None = None):
    if save_file is not None:
        if not pathlib.Path(save_file).parent.exists():
            raise FileNotFoundError(f"Aborting: directory for save_file {save_file} does not exist.")
    
    X = torch.stack(
        get_gw_event_pe_posterior_samples(event, method),
        dim=-1,
    )
    Z = _to_latent_space(X)
    ladj = _get_log_abs_det_jacobian(X)

    kde = ReflectKDE(event, method) # type: ignore

    flow_ensemble = {}
    for file in pathlib.Path(ensemble_dir).iterdir():
        if file.suffix == '.pkl':
            flow_ensemble[file.stem] = torch.load(file, weights_only=False)
    
    losses = {}
    for fname, flow in flow_ensemble.items():
        flow_log_density = flow().log_prob(Z).detach() + ladj
        kde_log_density = kde.log_pdf(X)
        loss = torch.kl_div(
            flow_log_density,
            kde_log_density,
            log_target=True
        ).mean().item()
        losses[fname] = loss
    
    if save_file is not None:
        with open(save_file, 'w') as f:
            lines = [f'{fname}\t{loss}\n' for fname, loss in losses.items()]
            f.writelines(lines)
    
    return losses


def get_gw_event_pe_posterior_samples(event: str, method: Literal['2D', '3D']):
    posterior_file = GW_PE_POSTERIOR_FILES[event][method]
    
    samples = _read_prior_or_posterior_file(posterior_file, method)
    
    if method == '2D':
        return torch.tensor(samples['lambdat'], dtype=torch.float32), torch.tensor(samples['q'], dtype=torch.float32)
    elif method == '3D':
        return torch.tensor(samples['lambda1'], dtype=torch.float32), \
            torch.tensor(samples['q'], dtype=torch.float32), \
            torch.tensor(samples['lambda2'], dtype=torch.float32)
    else:
        raise NotImplementedError()


def get_gw_event_pe_posterior_normalizing_flows(event: str, method: str, transformed: bool = True):
    if transformed:
        native_flow = torch.load(GWXTREME_FLOW_FILES[event][method]['transformed']['native'], weights_only=False)
        flow_ensemble = _read_ensemble_flows(GWXTREME_FLOW_FILES[event][method]['transformed']['ensemble'])
    else:
        native_flow = torch.load(GWXTREME_FLOW_FILES[event][method]['whitened']['native'], weights_only=False)
        flow_ensemble = _read_ensemble_flows(GWXTREME_FLOW_FILES[event][method]['whitened']['ensemble'])
    return native_flow, flow_ensemble


def get_gw_event_pe_posterior_kde_grid(event: str, method: str):
    kde_grid = torch.load(GWXTREME_KDE_GRID_FILES[event][method])
    return kde_grid


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


def _read_ensemble_flows(ensemble_dir: str):
    flow_ensemble = []
    for file in pathlib.Path(ensemble_dir).iterdir():
        if file.suffix == '.pkl':
            flow_ensemble.append(torch.load(file, weights_only=False))
    return flow_ensemble


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


class NormalizingFlow:
    def __init__(
            self,
            event: str,
            method: Literal['2D', '3D'] = '2D'
    ):    
        assert method in ['2D', '3D'], "method must be one of ['2D', '3D']"
        self.method = method
        
        assert event in SUPPORTED_EVENTS, f"event must be one of {SUPPORTED_EVENTS}."
        self.event = event
        self.native_flow, self.flow_ensemble = get_gw_event_pe_posterior_normalizing_flows(event, method)
        # self.kde_prob_grid = get_gw_event_pe_posterior_kde_grid(event, method)
            
    def set_normalizing_flows(self, native_flow=None, flow_ensemble: list | None = None):
        if native_flow is not None: self.native_flow = native_flow 
        if flow_ensemble is not None: self.flow_ensemble = flow_ensemble

    def set_kde_grid(self, kde_grid: str | torch.Tensor):
        if type(kde_grid) is str:
            self.kde_prob_grid = torch.load(kde_grid, weights_only=False)
        else:
            assert type(kde_grid) is torch.Tensor
            self.kde_prob_grid = kde_grid

    def sample(self, size: int) -> torch.Tensor:
        assert self.native_flow is not None, "No native flow has been initialized."
        z = self.native_flow().sample([size])
        x = _to_data_space(z)
        return x
    
    def ensemble_sample(self, size: int) -> torch.Tensor:
        assert self.flow_ensemble is not None, "No flow ensemble has been initialized."
        samples = torch.zeros((len(self.flow_ensemble), size))
        
        for i in range(len(self.flow_ensemble)):
            z = self.flow_ensemble[i].sample([size])
            x = _to_data_space(z)
            samples[i] = x
        
        return samples # (N_ensemble, size)

    def log_pdf(self, x: torch.Tensor) -> torch.Tensor:
        assert self.native_flow is not None, "No normalizing flow has been set."
        # w = self._scale_down(x)
        z = _to_latent_space(x)
        ladj = _get_log_abs_det_jacobian(x)
        # ladj += torch.log(self._scale_jacobian())
        lp = self.native_flow().log_prob(z).detach() + ladj
        lp = torch.nan_to_num(lp, nan=-torch.inf)
        return lp
    
    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.log_pdf(x))
    
    def ensemble_log_pdf(self, x: torch.Tensor) -> torch.Tensor:
        assert self.flow_ensemble is not None, "No normalizing flow ensemble has been set."
        log_probs = torch.zeros((len(self.flow_ensemble), len(x)))
        
        for i in range(len(self.flow_ensemble)):
            z = _to_latent_space(x)
            ladj = _get_log_abs_det_jacobian(x)
            
            lp = self.flow_ensemble[i]().log_prob(z).detach() + ladj
            lp = torch.nan_to_num(lp, nan=-torch.inf)
            log_probs[i] = lp
        
        return log_probs # (N_flows, N_points)
    
    def ensemble_pdf(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.ensemble_log_pdf(x))
    
    def pointwise_error(self, x: torch.Tensor, N_grid: int = 10) -> torch.Tensor:
        error = torch.zeros(x.shape[0])

        for i in range(x.shape[0]):
            if self.method == '2D':
                # find indices of the grid where the box (lambdat, q) coords are less than
                # or equal to the needed (lambdat, q) point
                where_le = torch.logical_and(
                    self.kde_prob_grid[:, :, 0] <= x[i, 0], 
                    self.kde_prob_grid[:, :, 1] <= x[i, 1]
                )

                if not torch.any(where_le):
                    box = self.kde_prob_grid[0, 0]
                else:
                    box = self.kde_prob_grid[where_le][-1]
                
                kde_prob = box[2]

                box_width = self.kde_prob_grid[1, 1, 0] - self.kde_prob_grid[0, 0, 0]
                box_height = self.kde_prob_grid[1, 1, 1] - self.kde_prob_grid[0, 0, 1]
                
                flow_prob = self.integrate_pdf_over_box(
                    low=[box[0], box[1]],
                    high=[box[0] + box_width, box[1] + box_height],
                    N_grid=N_grid
                )

            else:
                raise NotImplementedError()
            
            # does not take absolute value
            error[i] = flow_prob - kde_prob

        return error

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
            eos_list: list = [], 
            mean_chirp_mass: float | None = None,
            min_mass: float | None = None,
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
            # ax.set_title(self.event)
            # ax.set_xlabel(r'$\tilde{\Lambda}$', fontsize=14)
            # ax.set_ylabel(r'$q$', fontsize=14)

            ax.set_xlabel('Effective Tidal Deformability', fontsize=14)
            ax.set_ylabel('Mass Ratio', fontsize=14)

            if len(eos_list) > 0: assert (mean_chirp_mass is not None and min_mass is not None), "Must pass mean_chirp_mass and min_mass to plot EOS curves."
            eos_q = np.linspace(0.01, 1.0, N_grid)
            m1, m2 = get_masses(eos_q, mean_chirp_mass)
            m1, m2, eos_q = apply_mass_constraint(m1, m2, eos_q, min_mass)

            colors = ["#97972E", "#378F53", "#B86C44", "#56B1C6"]
            for eos, color in zip(eos_list, colors):
                if type(eos) in [tuple, list, np.ndarray]:
                    s, _, max_mass_eos, min_mass = get_eos_interpolant_from_parameters(eos, parameterization='spectral', N_points=300, m_min=min_mass)

                    # label for parameterized EOS
                    eos = "Inferred Spectral Parameterization"

                else:
                    s, _, _, max_mass_eos = get_eos_interpolant(eos, m_min=min_mass, N_points=1000)

                eos_lambdat = get_lambdat_for_eos(m1, m2, max_mass_eos, s)
                plt.plot(eos_lambdat, eos_q, linewidth=1, label=eos, alpha=0.9, color=color)
            
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

    def plot_error(self, save_file: str | None = None):
        assert self.method == '2D', 'can not plot 3D distribution'
        N_boxes = len(self.kde_prob_grid)

        patches = []
        error_values = []
        width = self.kde_prob_grid[1, 1, 0] - self.kde_prob_grid[0, 0, 0]
        height = self.kde_prob_grid[1, 1, 1] - self.kde_prob_grid[0, 0, 1]
        
        for i in range(N_boxes):
            for j in range(N_boxes):
                box = self.kde_prob_grid[i, j]
                patches.append(
                    matplotlib.patches.Rectangle(
                        (box[0].item(), box[1].item()), 
                        width.item(), 
                        height.item()
                    )
                )
                error = self.pointwise_error(torch.unsqueeze(box[:2], 0)).item() # the (lambdat, q) point
                error_values.append(error)

        collection = matplotlib.collections.PatchCollection(
            patches, 
            cmap=plt.colormaps['inferno'],
            norm=matplotlib.colors.Normalize(vmin=min(error_values), vmax=max(error_values))
        )
        collection.set_array(np.array(error_values))

        fig, ax = plt.subplots()
        ax.add_collection(collection)
        fig.colorbar(collection, label='Error')
        ax.set_xlim(0., 5000.)
        ax.set_ylim(0., 1.)
        ax.set_xlabel(r'$\tilde{\Lambda}$')
        ax.set_ylabel(r'$q$')
        ax.set_title(f"{self.event} Error Against KDE")
        
        if save_file is not None:
            fig.savefig(save_file)
        else:
            plt.show()


class ReflectiveNormalizingFlow:
    def __init__(
            self,
            event: str,
            method: Literal['2D', '3D'],
    ):   
        assert method in ['2D', '3D'], "method must be one of ['2D', '3D']"
        self.method = method

        assert event in SUPPORTED_EVENTS, f"event must be one of {SUPPORTED_EVENTS}."
        self.event = event

        self.native_flow, self.flow_ensemble = get_gw_event_pe_posterior_normalizing_flows(event, method, transformed=False)
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
    
    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        """Return an estimate of the density evaluated at the given points."""
        w = self._scale_down(x)
        p = torch.exp(self.native_flow().log_prob(w).detach())
        
        for i, (low, high) in enumerate(zip(self.low, self.high)):                
            if torch.isfinite(low):
                reflect_w = w.clone()
                reflect_w[:, i] = 2.0 * low - w[:, i]

                p += torch.exp(self.native_flow().log_prob(reflect_w).detach())

            if torch.isfinite(high):
                reflect_w = w.clone()
                reflect_w[:, i] = 2.0 * high - w[:, i]

                p += torch.exp(self.native_flow().log_prob(reflect_w).detach())

        # jacobian of transformation to whitened space
        p *= self._scale_jacobian()

        return p

    def log_pdf(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self.pdf(x))
    
    def ensemble_pdf(self, x: torch.Tensor) -> torch.Tensor:
        assert self.flow_ensemble is not None, "No normalizing flow ensemble has been set."
        probs = torch.zeros((len(self.flow_ensemble), len(x)))

        w = self._scale_down(x)
        for i in range(len(self.flow_ensemble)):
            p = torch.exp(self.flow_ensemble[i]().log_prob(w).detach())
            
            for i, (low, high) in enumerate(zip(self.low, self.high)):                
                if torch.isfinite(low):
                    reflect_w = w.clone()
                    reflect_w[:, i] = 2.0 * low - w[:, i]

                    p += torch.exp(self.flow_ensemble[i]().log_prob(reflect_w).detach())

                if torch.isfinite(high):
                    reflect_w = w.clone()
                    reflect_w[:, i] = 2.0 * high - w[:, i]

                    p += torch.exp(self.flow_ensemble[i]().log_prob(reflect_w).detach())

            # jacobian of transformation to whitened space
            p *= self._scale_jacobian()

            probs[i] = p

        return probs
    
    def ensemble_log_pdf(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(self.ensemble_pdf(x))
    
    def plot_density(self, N_grid: int = 200, save_file: str | None = None):
        if self.method == '2D':        
            lambdat, q = get_gw_event_pe_posterior_samples(self.event, self.method) # type: ignore

            fig, ax = plt.subplots(figsize=(7, 7))
            lt_grid, q_grid = torch.meshgrid([torch.linspace(0., lambdat.max(), N_grid), torch.linspace(0.0, 1.0, N_grid)])
            points = torch.stack([lt_grid, q_grid], dim=-1).reshape((N_grid**2, 2))
            lp = self.log_pdf(points).reshape((N_grid, N_grid)).detach()
            p = torch.exp(lp)

            qcs = ax.contourf(lt_grid, q_grid, p, cmap='inferno')
            plt.colorbar(qcs, ax=ax)
            ax.scatter(x=lambdat, y=q, s=0.25, c='gray', alpha=0.40)
            ax.set_title(self.event)
            ax.set_xlabel(r'$\tilde{\Lambda}$', fontsize=14)
            ax.set_ylabel(r'$q$', fontsize=14)

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
            fig.savefig(save_file, bbox_inches='tight', dpi='figure')
        else:
            plt.show()
    
    def _scale_down(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, 'x should be an (N, D) shaped Tensor'
        assert x.shape[-1] in (2, 3), 'last dimension of x must be size 2 or 3'
        
        if x.shape[-1] == 2:
            # bring LambdaT to original range
            w = torch.stack((x[:, 0] / self.posterior_samples[:, 0].max(), x[:, 1]), dim=-1)
        else:
            # bring Lambda1 and Lambda2 to original range
            w = torch.stack(
                (
                    x[:, 0] / self.posterior_samples[:, 0].max(), 
                    x[:, 1], 
                    x[:, 2] / self.posterior_samples[:, 2].max()
                ),
                dim=-1
            )

        return w
    
    def _scale_up(self, w: torch.Tensor) -> torch.Tensor:
        assert w.ndim == 2, 'w should be an (N, D) shaped Tensor'
        assert w.shape[-1] in (2, 3), 'last dimension of w must be size 2 or 3'
        
        if w.shape[-1] == 2:
            # bring LambdaT to original range
            x = torch.stack((w[:, 0] * self.posterior_samples[:, 0].max(), w[:, 1]), dim=-1)
        else:
            # bring Lambda1 and Lambda2 to original range
            x = torch.stack(
                (
                    w[:, 0] * self.posterior_samples[:, 0].max(), 
                    w[:, 1], 
                    w[:, 2] * self.posterior_samples[:, 2].max()
                ),
                dim=-1
            )
        return x
    
    def _scale_jacobian(self):
        if self.method == '2D':
            # 1 / LambdaT_max
            return 1 / self.posterior_samples[:, 0].max()
        else:
            # 1 / (Lambda1_max * Lambda2_max)
            return 1 / (self.posterior_samples[:, 0].max() * self.posterior_samples[:, 2].max())


class BoundedKDE:
    def __init__(
            self,
            event: str,
            method: Literal['2D', '3D'] = '2D',
            Ns=None
    ):    
        assert method in ['2D', '3D'], "method must be one of ['2D', '3D']"
        self.method = method
        assert event in SUPPORTED_EVENTS, f"event must be one of {SUPPORTED_EVENTS}."
        self.event = event
        
        self.posterior_samples = torch.stack(
            get_gw_event_pe_posterior_samples(event, method),
            dim=-1
        )

        if Ns is not None:
            self.posterior_samples = self.posterior_samples[:Ns]

        whitened_samples = self._scale_down(self.posterior_samples)
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
            W = self._scale_down(X)
            kde = scipy.stats.gaussian_kde(W.T.numpy())
        else:
            kde = self.base_kde               
        
        w = self._scale_down(x)
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
        p *= self._scale_jacobian()

        return p
    
    def sample(self, size: int) -> torch.Tensor:
        samples = torch.tensor(self.base_kde.resample(size), dtype=torch.float32).T
        samples = self._scale_up(samples)

        for i, (low, high) in enumerate(zip(self.low, self.high)):
            if low is not None:
                samples[:, i][samples[:, i] < low] = 2. * low - samples[:, i][samples[:, i] < low]
                    
            if high is not None:
                samples[:, i][samples[:, i] > high] = 2. * high - samples[:, i][samples[:, i] > high]

        return samples

    def _scale_down(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 2, 'x should be an (N, D) shaped Tensor'
        assert x.shape[-1] in (2, 3), 'last dimension of x must be size 2 or 3'
        
        if x.shape[-1] == 2:
            # bring LambdaT to original range
            w = torch.stack((x[:, 0] / self.posterior_samples[:, 0].max(), x[:, 1]), dim=-1)
        else:
            # bring Lambda1 and Lambda2 to original range
            w = torch.stack(
                (
                    x[:, 0] / self.posterior_samples[:, 0].max(), 
                    x[:, 1], 
                    x[:, 2] / self.posterior_samples[:, 2].max()
                ),
                dim=-1
            )

        return w
    
    def _scale_up(self, w: torch.Tensor) -> torch.Tensor:
        assert w.ndim == 2, 'w should be an (N, D) shaped Tensor'
        assert w.shape[-1] in (2, 3), 'last dimension of w must be size 2 or 3'
        
        if w.shape[-1] == 2:
            # bring LambdaT to original range
            x = torch.stack((w[:, 0] * self.posterior_samples[:, 0].max(), w[:, 1]), dim=-1)
        else:
            # bring Lambda1 and Lambda2 to original range
            x = torch.stack(
                (
                    w[:, 0] * self.posterior_samples[:, 0].max(), 
                    w[:, 1], 
                    w[:, 2] * self.posterior_samples[:, 2].max()
                ),
                dim=-1
            )
        return x
    
    def _scale_jacobian(self):
        if self.method == '2D':
            # 1 / LambdaT_max
            return 1 / self.posterior_samples[:, 0].max()
        else:
            # 1 / (Lambda1_max * Lambda2_max)
            return 1 / (self.posterior_samples[:, 0].max() * self.posterior_samples[:, 2].max())
