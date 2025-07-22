import pathlib

import numpy as np
import torch
import zuko

from ..GWXtreme.density_estimation import NormalizingFlow, get_gw_event_pe_posterior_samples, learn_flow, _to_latent_space, learn_ensemble


def train_native_flow():
    event = 'GW230529'
    method = '3D'
    model_label = 'zuko_prebuilt_maf'
    
    out_dir = f"/home/joseph/LocalProjects/GWXtreme/systematics/src/gwxtreme/trained_density_estimators/{event}/{method}/{model_label}"
    
    flow_constructor = zuko.flows.MAF

    flow_kwargs = dict(
        features=3 if method == '3D' else 2,
        transforms=20
    )

    optimizer_constructor = torch.optim.Adam
    optimizer_kwargs = dict(
        lr=3e-5,
        betas=(0.9, 0.99)
    )

    X = torch.stack(get_gw_event_pe_posterior_samples(event, method), dim=-1)
    Z = _to_latent_space(X)

    print(Z.shape)

    flow = flow_constructor(**flow_kwargs)
    optimizer = optimizer_constructor(params=flow.parameters(), **optimizer_kwargs)

    learn_flow(
        data=Z,
        flow=flow,
        optimizer=optimizer,
        N_epochs=500,
        batch_size=1000,
        save_file=f"{out_dir}/native/{event}_{method}_flow.pkl"
    )


def train_flow_ensemble():
    event = 'GW230529'
    method = '3D'
    model_label = 'zuko_prebuilt_maf'
    
    out_dir = f"/home/joseph/LocalProjects/GWXtreme/systematics/src/gwxtreme/trained_density_estimators/{event}/{method}/{model_label}"
    
    flow_constructor = zuko.flows.MAF

    flow_kwargs = dict(
        features=3 if method == '3D' else 2,
        transforms=20
    )

    optimizer_constructor = torch.optim.Adam
    optimizer_kwargs = dict(
        lr=3e-5,
        betas=(0.9, 0.99)
    )

    learn_ensemble(
        event=event,
        method=method,
        flow_constructor=flow_constructor,
        optimizer_constructor=optimizer_constructor,
        flow_kwargs=flow_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        save_dir=f"{out_dir}/ensemble",
        N_epochs=500,
        N_ensemble=100,
        batch_size=1000,
        resample_size=None,
        N_processors=1
    )
   

if __name__ == '__main__':
    # event = 'GW230529'
    # method = '3D'
    # model_label = 'zuko_prebuilt_maf'
    
    # out_dir = f"/home/joseph/LocalProjects/GWXtreme/systematics/src/gwxtreme/trained_density_estimators/{event}/{method}/{model_label}"
    
    # X = torch.stack(get_gw_event_pe_posterior_samples(event, method), dim=-1)
    # Z = _to_latent_space(X)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(2, 3, figsize=(18, 10), width_ratios=[0.15, 0.15, 0.15])
    
    # ax[0, 0].scatter(X[:, 0], X[:, 1], s=1.5, c='b')
    # ax[0, 0].set_xlabel(r"$\Lambda_1$")
    # ax[0, 0].set_ylabel(r"$q$")

    # ax[0, 1].scatter(X[:, 2], X[:, 1], s=1.5, c='b')
    # ax[0, 1].set_xlabel(r"$\Lambda_2$")
    # ax[0, 1].set_ylabel(r"$q$")

    # ax[0, 2].scatter(X[:, 0], X[:, 2], s=1.5, c='b')
    # ax[0, 2].set_xlabel(r"$\Lambda_1$")
    # ax[0, 2].set_ylabel(r"$\Lambda_2$")

    # ax[1, 0].scatter(Z[:, 0], Z[:, 1], s=1.5, c='r')
    # ax[1, 0].set_xlabel(r"$\log{\Lambda_1}$")
    # ax[1, 0].set_ylabel(r"$\log{\frac{q}{1-q}}$")

    # ax[1, 1].scatter(Z[:, 2], Z[:, 1], s=1.5, c='r')
    # ax[1, 1].set_xlabel(r"$\log{\Lambda_2}$")
    # ax[1, 1].set_ylabel(r"$\log{\frac{q}{1-q}}$")

    # ax[1, 2].scatter(Z[:, 0], Z[:, 2], s=1.5, c='r')
    # ax[1, 2].set_xlabel(r"$\log{\Lambda_1}$")
    # ax[1, 2].set_ylabel(r"$\log{\Lambda_2}$")

    # plt.savefig(f"{out_dir}/{event}_3D_densities.png", bbox_inches='tight')
    
    # train_native_flow()
    train_flow_ensemble()
