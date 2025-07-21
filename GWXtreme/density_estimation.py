from typing import Literal
import pathlib
import time
import multiprocessing

import numpy as np
import torch
import pandas as pd
import scipy.stats

import matplotlib.colors
import matplotlib.patches
import matplotlib.collections
import matplotlib.pyplot as plt

from .config import SUPPORTED_EVENTS, GW_PE_POSTERIOR_FILES, GWXTREME_FLOW_FILES, GWXTREME_KDE_GRID_FILES


def learn_flow(
    data: torch.Tensor,
    flow,
    optimizer,
    N_epochs: int,
    batch_size: int,
    save_file: str
):  
    assert pathlib.Path(save_file).parent.exists(), "directory for given save_file doesn't exist"
    training_summary = f" \
        data.shape: {data.shape}\n \
        flow: {flow}\n \
        optimizer: {optimizer}\n \
        N_epochs: {N_epochs}\n \
        batch_size: {batch_size}\n"            

    train_loader = torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(data),
        batch_size=batch_size, 
        shuffle=True
    )

    start = time.perf_counter()
    for epoch in range(N_epochs + 1):
        losses = []

        for d in train_loader:
            # minimize expected KL divergence
            loss = -flow().log_prob(torch.stack(d)).mean() # -log p(x)
            if loss == torch.nan:
                print(f'Aborting: loss = nan at epoch {epoch}.')
                return
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach())
        
        losses = torch.stack(losses)

        if epoch % 10 == 0:
            progress = f"[{epoch:6d} / {N_epochs}]\tavg. loss = {losses.mean().item():3.4f} +- {losses.std().item():3.4f}"
            training_summary += f"{progress}\n"
            print(f'{progress}')
    
    end = time.perf_counter()
    torch.save(flow, save_file)

    training_summary += f"\ntrain time: {(end - start) / 60:.2f} minutes\n"
    
    model_save_file = pathlib.Path(save_file)
    summary_save_file = model_save_file.parent.joinpath(model_save_file.stem + '_train_summary.txt')
    summary_save_file.touch(exist_ok=True)
    summary_save_file.write_text(training_summary)


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

    kde = TransformKDE(event, method) # type: ignore

    flow_ensemble = _read_ensemble_flows(ensemble_dir)
    losses = []
    for flow in flow_ensemble:
        flow_log_density = flow().log_prob(Z).detach() + ladj
        kde_log_density = kde.log_pdf(X)
        loss = torch.kl_div(
            flow_log_density,
            kde_log_density,
            log_target=True
        ).mean().item()
        losses.append(loss)
    
    if save_file is not None:
        np.savetxt(
            save_file,
            np.stack((np.arange(len(losses)), losses), axis=-1),
            fmt="flow #%4d      %1.4e"
        )
    
    return losses


def get_gw_event_pe_posterior_samples(event: str, method: str):
    posterior_file = GW_PE_POSTERIOR_FILES[event][method]
    samples = pd.read_table(posterior_file)

    if method == '2D':
        return torch.tensor(samples['lambdat'], dtype=torch.float32), torch.tensor(samples['q'], dtype=torch.float32)
    elif method == '3D':
        return torch.tensor(samples['lambda_1'], dtype=torch.float32), \
            torch.tensor(samples['q'], dtype=torch.float32), \
            torch.tensor(samples['lambda_2'], dtype=torch.float32)
    else:
        raise NotImplementedError()


def get_gw_event_pe_posterior_normalizing_flows(event: str, method: str):
    native_flow = torch.load(GWXTREME_FLOW_FILES[event][method]['native'], weights_only=False)
    flow_ensemble = _read_ensemble_flows(GWXTREME_FLOW_FILES[event][method]['ensemble'])
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


class TransformKDE:
    def __init__(
            self,
            event: str,
            method: Literal['2D', '3D'] = '2D'
    ):
        assert method in ['2D', '3D'], "method must be one of ['2D', '3D']"
        self.method = method
        
        assert event in SUPPORTED_EVENTS, f"event must be one of {SUPPORTED_EVENTS}."
        self.event = event
        self.posterior_samples = torch.stack(
            get_gw_event_pe_posterior_samples(event, method),
            dim=-1
        )
        post_latent = _to_latent_space(self.posterior_samples)
        self.base_kde = scipy.stats.gaussian_kde(post_latent.T.numpy())

    def sample(self, size: int) -> torch.Tensor:
        z = torch.tensor(self.base_kde.resample(size).T, dtype=torch.float32)
        x = _to_data_space(z)
        return x

    def log_pdf(self, x: torch.Tensor, resample: bool = False) -> torch.Tensor:
        if resample:
            Z = self.base_kde.resample()
            kde = scipy.stats.gaussian_kde(Z)
        else:
            kde = self.base_kde
        
        z = _to_latent_space(x)
        ladj = _get_log_abs_det_jacobian(x)
        valid_i = ~torch.logical_or(torch.isnan(z[:, 0]), torch.isnan(z[:, 1]))
        lp = torch.empty(z.shape[0])
        lp[valid_i] = torch.log(torch.tensor(kde(z[valid_i].T.numpy()).T, dtype=torch.float32)) + ladj[valid_i]
        lp[~valid_i] = -torch.inf
        return lp

    def pdf(self, x: torch.Tensor, resample: bool = False) -> torch.Tensor:
        return torch.exp(self.log_pdf(x, resample=resample))

    def compute_probability_over_grid(
            self,
            N_boxes: int,
            save_file: str
        ):
        eps = 1e-5
        if self.method == '2D':
            z_grid = _to_latent_space(
                torch.stack(
                    torch.meshgrid(
                        torch.linspace(eps, 5000, N_boxes),
                        torch.linspace(eps, 1 - eps, N_boxes),
                        indexing='xy'
                    ),
                    dim=-1
                ).reshape((N_boxes**2, 2))
            ).reshape((N_boxes, N_boxes, 2))
            
            z_box_probs = torch.zeros((N_boxes - 1, N_boxes - 1, 3))

            for i in range(N_boxes - 1):
                for j in range(N_boxes - 1):
                    low = z_grid[i, j]
                    high = z_grid[i + 1, j + 1]

                    box_prob = self.base_kde.integrate_box(low, high)
                    z_box_probs[i, j] = torch.hstack((low, torch.tensor(box_prob)))

            x_box_probs = torch.stack(
                (
                    torch.exp(z_box_probs[:, :, 0]), 
                    torch.sigmoid(z_box_probs[:, :, 1]), 
                    z_box_probs[:, :, 2]
                ), 
                dim=-1
            )
        
        else:
            raise NotImplementedError()
        
        torch.save(x_box_probs, save_file)
    

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
        self.kde_prob_grid = get_gw_event_pe_posterior_kde_grid(event, method)
            
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
        z = _to_latent_space(x)
        ladj = _get_log_abs_det_jacobian(x)
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

    def plot_density(self, N_grid: int = 200, save_file: str | None = None):
        assert self.method == '2D', 'can not plot 3D distribution'
        
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
        if save_file is not None:
            fig.savefig(save_file)
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


class ReflectKDE:
    def __init__(
            self,
            event: str,
            method: Literal['2D', '3D'] = '2D'
    ):    
        assert method in ['2D', '3D'], "method must be one of ['2D', '3D']"
        self.method = method
        assert event in SUPPORTED_EVENTS, f"event must be one of {SUPPORTED_EVENTS}."
        self.event = event
        
        self.posterior_samples = torch.stack(
            get_gw_event_pe_posterior_samples(event, method),
            dim=-1
        )
        # self.posterior_samples = self.posterior_samples[torch.unique(torch.randint(0, len(self.posterior_samples), (4800,)))]
        # print(len(self.posterior_samples))
        self.base_kde = scipy.stats.gaussian_kde(self.posterior_samples.T.numpy())

        if method == '2D':
            self.low = torch.tensor([0., 0.], dtype=torch.float32)
            self.high = torch.tensor([torch.inf, 1.], dtype=torch.float32)
        elif method == '3D':
            self.low = torch.tensor([0., 0., 0.], dtype=torch.float32)
            self.high = torch.tensor([torch.inf, 1., torch.inf], dtype=torch.float32)
    
    def log_pdf(self, x: torch.Tensor, resample: bool = False):
        return torch.log(self.pdf(x, resample))

    def pdf(self, x: torch.Tensor, resample: bool = False):
        """Return an estimate of the density evaluated at the given points."""
        if resample:
            X = self.sample(len(self.posterior_samples))
            kde = scipy.stats.gaussian_kde(X.T)
        else:
            kde = self.base_kde               
        
        p = torch.tensor(kde(x.T.numpy()).T, dtype=torch.float32)
        
        for i, (low, high) in enumerate(zip(self.low, self.high)):                
            if torch.isfinite(low):
                reflect_x = x.clone()
                reflect_x[:, i] = 2.0 * low - x[:, i]

                p += torch.tensor(kde(reflect_x.T.numpy()).T, dtype=torch.float32)

            if torch.isfinite(high):
                reflect_x = x.clone()
                reflect_x[:, i] = 2.0 * high - x[:, i]
                p += torch.tensor(kde(reflect_x.T.numpy()).T, dtype=torch.float32)

        return p
    
    def sample(self, size: int):
        samples = torch.tensor(self.base_kde.resample(size), dtype=torch.float32)
        for i, (low, high) in enumerate(zip(self.low, self.high)):
            if low is not None:
                samples[i, :][samples[i, :] < low] = 2. * low - samples[i, :][samples[i, :] < low]
                    
            if high is not None:
                samples[i, :][samples[i, :] > high] = 2. * high- samples[i, :][samples[i, :] > high]

        return samples.T
    
    def compute_probability_over_grid(
            self,
            N_boxes: int,
            save_file: str
        ):
        eps = 1e-5
        if self.method == '2D':
            z_grid = _to_latent_space(
                torch.stack(
                    torch.meshgrid(
                        torch.linspace(eps, 5000, N_boxes),
                        torch.linspace(eps, 1 - eps, N_boxes),
                        indexing='xy'
                    ),
                    dim=-1
                ).reshape((N_boxes**2, 2))
            ).reshape((N_boxes, N_boxes, 2))
            
            z_box_probs = torch.zeros((N_boxes - 1, N_boxes - 1, 3))

            for i in range(N_boxes - 1):
                for j in range(N_boxes - 1):
                    low = z_grid[i, j]
                    high = z_grid[i + 1, j + 1]

                    box_prob = self.base_kde.integrate_box(low, high)
                    z_box_probs[i, j] = torch.hstack((low, torch.tensor(box_prob)))

            x_box_probs = torch.stack(
                (
                    torch.exp(z_box_probs[:, :, 0]), 
                    torch.sigmoid(z_box_probs[:, :, 1]), 
                    z_box_probs[:, :, 2]
                ), 
                dim=-1
            )
        
        else:
            raise NotImplementedError()
        
        torch.save(x_box_probs, save_file)


if __name__ == "__main__":
    flow = NormalizingFlow("GW170817", '2D')
    kde = TransformKDE("GW170817", '2D')
    reflectkde = ReflectKDE("GW170817", '2D')

    ng = 100

    LT = torch.linspace(-200., 1500., ng)
    Q = torch.linspace(-0.20, 1.20, ng)
    points = torch.stack(torch.meshgrid([LT, Q], indexing='xy'), dim=-1).reshape((ng*ng, 2))
    
    
    flow_pgrid = flow.pdf(points).reshape((ng, ng))
    kde_pgrid = kde.pdf(points).reshape((ng, ng))
    reflectkde_pgrid = reflectkde.pdf(points).reshape((ng, ng))

    flow_sample = flow.sample(1000)
    kde_sample = kde.sample(1000)
    reflectkde_sample = reflectkde.sample(1000)

    fig, ax = plt.subplots(1, 3, figsize=(22, 8))
    fig.tight_layout()

    qcs = ax[0].contourf(LT, Q, flow_pgrid, levels=50, cmap="inferno")
    ax[0].scatter(flow_sample[:,0], flow_sample[:,1], c='gray', s=1.5, alpha=0.70)
    ax[0].set_title('flow')
    ax[0].set_xlim((torch.min(LT), torch.max(LT)))
    plt.colorbar(qcs, ax=ax[0])

    qcs = ax[1].contourf(LT, Q, kde_pgrid, levels=50, cmap="inferno")
    ax[1].scatter(kde_sample[:,0], kde_sample[:,1], c='gray', s=1.5, alpha=0.70)
    ax[1].set_title('kde')
    ax[1].set_xlim((torch.min(LT), torch.max(LT)))
    plt.colorbar(qcs, ax=ax[1])

    qcs = ax[2].contourf(LT, Q, reflectkde_pgrid, levels=50, cmap="inferno")
    ax[2].scatter(reflectkde_sample[:,0], reflectkde_sample[:,1], c='gray', s=1.5, alpha=0.70)
    ax[2].set_title('reflectkde')
    ax[2].set_xlim((torch.min(LT), torch.max(LT)))
    plt.colorbar(qcs, ax=ax[2])

    plt.savefig("/home/joseph/LocalProjects/GWXtreme/systematics/compare_densities.png")
    

    # samp = rk.sample(1000)
    # plt.scatter(samp[:, 0], samp[:, 1])
    # plt.savefig("./test.png")

    # from .eos_inference import ModelSelector

    # ms = ModelSelector("GW170817", density_est_method='reflectkde')
    # bf, trials = ms.compute_eos_evidence_ratio('APR4_EPP', 'SLY', 100, N_trials=20, verbose=True)
    # print(bf, np.mean(trials), np.std(trials))
    # print(trials)

