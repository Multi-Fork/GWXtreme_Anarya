import pathlib

import numpy as np
import torch
import zuko

from ..GWXtreme.density_estimation import EnsembleDensityEstimator, get_gw_event_pe_posterior_samples, learn_flow


def train_native_flow():
    event = 'GW170817'
    method = '2D'
    model_label = 'zuko_prebuilt_maf'
    
    out_dir = f"/home/joseph/LocalProjects/GWXtreme/systematics/src/gwxtreme/GWXtreme/trained_density_estimators/{event}/{method}/{model_label}"
    
    flow_constructor = zuko.flows.MAF

    flow_kwargs = dict(
        features=2,
        transforms=20
    )

    optimizer_constructor = torch.optim.Adam
    optimizer_kwargs = dict(
        lr=1e-5,
        betas=(0.9, 0.98)
    )

    X = torch.stack(get_gw_event_pe_posterior_samples(event, method), dim=-1)
    Z = torch.stack((torch.log(X[:, 0]), torch.logit(X[:, 1])), dim=-1)

    print(Z.shape)

    flow = flow_constructor(**flow_kwargs)
    optimizer = optimizer_constructor(params=flow.parameters(), **optimizer_kwargs)

    learn_flow(
        data=Z,
        flow=flow,
        optimizer=optimizer,
        N_epochs=500,
        batch_size=1000,
        save_file=f"{out_dir}/native/{event}_{method}_flow___.pkl"
    )


def train_flow_ensemble():
    event = 'GW170817'
    method = '2D'
    model_label = 'zuko_prebuilt_maf'
    
    out_dir = f"/home/joseph/LocalProjects/GWXtreme/systematics/src/gwxtreme/trained_density_estimators/{event}/{method}/{model_label}"
    
    flow_constructor = zuko.flows.MAF

    flow_kwargs = dict(
        features=2,
        transforms=20
    )

    optimizer_constructor = torch.optim.Adam
    optimizer_kwargs = dict(
        lr=1e-5,
        betas=(0.9, 0.99)
    )

    ede = EnsembleDensityEstimator(event, method)
    ede.learn_ensemble(
        flow_constructor=flow_constructor,
        optimizer_constructor=optimizer_constructor,
        flow_kwargs=flow_kwargs,
        optimizer_kwargs=optimizer_kwargs,
        save_dir=f"{out_dir}/ensemble",
        N_epochs=200,
        N_ensemble=100,
        batch_size=1000,
        resample_size=None,
        N_processors=1
    )
   

if __name__ == '__main__':
    train_flow_ensemble()