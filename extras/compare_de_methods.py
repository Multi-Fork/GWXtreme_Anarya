import json
import time

import numpy as np
import matplotlib.pyplot as plt

from ..GWXtreme.config import EOS_LIST
from ..GWXtreme.density_estimation import NormalizingFlow
from .scripts import (
    compute_bayes_factors_from_nested_sampling_evidences, 
    compute_single_event_bayes_factors,
    sample_spectral_EoS_parameters,
    compute_EoS_constraints_from_spectral_samples
)
from .plotting_scripts import (
    plot_bayes_factors_bar_chart,
    plot_EoS_constraints,
    plot_parameterized_eos_posterior
)


outdir = "/home/joseph/LocalProjects/GWXtreme/systematics/comparing_density_estimators"


def plot_bf_dist_method_comparison(bf_file, method, waveform, save_dir):
    with open(bf_file) as f:
        data = json.load(f)[method][waveform]
    
    for eos in EOS_LIST:
        try:
            flow_bf = data['flow'][eos]['native']
            flow_bf_array = data['flow'][eos]['ensemble']
            
            kde_bf = data['kde'][eos]['native']
            kde_bf_array = data['kde'][eos]['resamples']
            
            reflectkde_bf = data['reflectkde'][eos]['native']
            reflectkde_bf_array = data['reflectkde'][eos]['resamples']
            
            lal_bf = data['lal'][eos]['bf']
    
            plt.clf()

            n1, _, _ = plt.hist(flow_bf_array, label='flow ensemble', histtype='step', density=True, color='blue')
            n2, _, _ = plt.hist(kde_bf_array, label='kde trials', histtype='step', density=True, color='red')
            n3, _, _ = plt.hist(reflectkde_bf_array, label='reflectkde trials', histtype='step', density=True, color='orange')
            ymax = 1.1 * np.max(np.stack((n1, n2, n3)))
            
            plt.vlines(flow_bf, 0., ymax, label='flow', color='blue')
            plt.vlines(kde_bf, 0., ymax, label='reflectkde', color='red')
            plt.vlines(reflectkde_bf, 0., ymax, label='reflectkde', color='orange')
            plt.vlines(lal_bf, 0., ymax, label='lal', color='green')

            plt.legend()
            plt.title(f"KDE vs. Flows for GW170817, {eos}")
            plt.savefig(f"{save_dir}/GW170817_{eos}_flows_vs_kde_bayes_factors.png")
        except Exception as e:
            print(e)


def compute_bayes_factors(save_file):
    event = "GW170817"
    method = '2D'
    waveform = 'TaylorF2'
    nested_method = "LALInference_NEST"

    N_trials = 2000
    
    all_results = {}

    start = time.perf_counter()
    flow_bfs = compute_single_event_bayes_factors(
        event,
        method,
        waveform,
        "flow",
        EOS_LIST,
        N_trials=N_trials,
        save_file=None
    )
    end = time.perf_counter()
    print(f"flow time = {end - start}")

    start = time.perf_counter()
    kde_bfs = compute_single_event_bayes_factors(
        event,
        method,
        waveform,
        "kde",
        EoS_names=EOS_LIST,
        N_trials=N_trials,
        save_file=None
    )
    end = time.perf_counter()
    print(f"kde time = {end - start}")

    start = time.perf_counter()
    reflectkde_bfs = compute_single_event_bayes_factors(
        event,
        method,
        waveform,
        "reflectkde",
        EOS_LIST,
        N_trials=N_trials,
        save_dir=None
    )
    end = time.perf_counter()
    print(f"reflectkde time = {end - start}")

    nested_bfs = compute_bayes_factors_from_nested_sampling_evidences(
        event,
        waveform
    )
      
    all_results = {
        method: {
            waveform: {
                "flow": flow_bfs[method][waveform]["flow"],
                "kde": kde_bfs[method][waveform]["kde"],
                "reflectkde": reflectkde_bfs[method][waveform]["reflectkde"],
                "lal": nested_bfs[nested_method][waveform]
            }
        }
    }

    with open(save_file, 'w') as f:
        json.dump(all_results, f, indent=4)


def plot_bar_chart(bf_file):
    event = "GW170817"
    save_dir = f"/home/joseph/LocalProjects/GWXtreme/systematics/comparing_KDE_and_normalizing_flows/{event}"
    method_sets = [
        ('2D', 'TaylorF2', 'flow'),
        ('2D', 'TaylorF2', 'kde'),
        ('2D', 'TaylorF2', 'reflectkde'),
        # ('2D', 'TaylorF2', 'lal')
    ]
    
    plot_bayes_factors_bar_chart(
        bf_file, 
        method_sets,
        yscale='linear',
        error_type='2std',
        save_file=f"{save_dir}/{event}_bfs.png"
    )


def add_old_kde_results():
    event = "GW170817"
    save_dir = f"/home/joseph/LocalProjects/GWXtreme/systematics/comparing_KDE_and_normalizing_flows/{event}"
    
    new_bfs_file = f"{save_dir}/all_bayes_factors4.json"
    old_bfs_file = "/home/joseph/LocalProjects/GWXtreme/systematics/data/BNS/BFs/GW170817_all_bayes_factors.json"
    
    with open(old_bfs_file) as f:
        old_bfs = json.load(f)['2D KDE TaylorF2']
    
    with open(new_bfs_file) as f:
        new_bfs = json.load(f)
    
    new_bfs['2D']['TaylorF2']['reflective_kde'] = {} 
    for eos in EOS_LIST:
        new_bfs['2D']['TaylorF2']['reflective_kde'][eos] = {
            'native': old_bfs[eos][0],
            'resamples': old_bfs[eos][1]
        }
    
    with open(f"{save_dir}/all_bayes_factors5.json", 'w+') as f:
        json.dump(new_bfs, f, indent=4)


def sample_spectral_params():
    event = 'GW170817'
    method = '2D'
    de_method = 'kde'

    sample_spectral_EoS_parameters(
        event=event,
        method=method,
        density_est_method=de_method,
        save_file=f"{outdir}/constraints/{event}/{event}_{method}_{de_method}_spectral_posterior_samples_10K.h5",
        N_pool=1,
        N_walkers=50,
        N_samples=10000
    )


def plot_constraints():
    flow_samples_file = f"{outdir}/constraints/GW170817/GW170817_2D_flow_spectral_posterior_samples_10K.h5"
    flow_constraints_file = f"{outdir}/constraints/GW170817/GW170817_2D_flow_spectral_constraints.txt"

    kde_samples_file = f"{outdir}/constraints/GW170817/GW170817_2D_kde_spectral_posterior_samples.h5"
    kde_constraints_file = f"{outdir}/constraints/GW170817/GW170817_2D_kde_spectral_constraints.txt"

    reflectkde_samples_file = f"{outdir}/constraints/GW170817/GW170817_2D_reflectkde_spectral_posterior_samples.h5"
    reflectkde_constraints_file = f"{outdir}/constraints/GW170817/GW170817_2D_reflectkde_spectral_constraints.txt"

    compute_EoS_constraints_from_spectral_samples(
        flow_samples_file, 
        save_file=flow_constraints_file
    )
    # compute_EoS_constraints_from_spectral_samples(
    #     kde_samples_file, 
    #     save_file=kde_constraints_file
    # )
    # compute_EoS_constraints_from_spectral_samples(
    #     reflectkde_samples_file, 
    #     save_file=reflectkde_constraints_file
    # )

    plot_EoS_constraints(
        [
            flow_constraints_file, 
            kde_constraints_file,
            reflectkde_constraints_file
        ],
        [
            '2D Flow', 
            '2D Transformed KDE', 
            '2D Reflective KDE'
        ],
        ['APR4_EPP'],
        save_file=f"{outdir}/constraints/GW170817/GW170817_2D_spectral_constraints_all.png"
    )


def plot_posterior():
    flow_samples_file = f"{outdir}/constraints/GW170817/GW170817_2D_flow_spectral_posterior_samples_10K.h5"
    kde_samples_file = f"{outdir}/constraints/GW170817/GW170817_2D_kde_spectral_posterior_samples.h5"
    reflectkde_samples_file = f"{outdir}/constraints/GW170817/GW170817_2D_reflectkde_spectral_posterior_samples.h5"

    plot_parameterized_eos_posterior(
        [
            flow_samples_file, 
            kde_samples_file, 
            reflectkde_samples_file
        ],
        [
            '2D Flow', 
            '2D Transformed KDE', 
            '2D Reflective KDE'
        ],
        save_file=f"{outdir}/constraints/GW170817/GW170817_2D_spectral_posterior_all.png",
    )



if __name__ == '__main__':
    event = 'GW230529'
    method = '3D'
    flow = NormalizingFlow(event, method)
    flow.plot_density(50, f'./{event}_{method}_flow_density.png')
    
    # bf_file = f"/home/joseph/LocalProjects/GWXtreme/systematics/comparing_KDE_and_normalizing_flows/GW170817/all_bayes_factors_2K_trials.json"
    # compute_bayes_factors(bf_file)
    # plot_bar_chart(bf_file)

    # sample_spectral_params()
    # plot_constraints()
    # plot_posterior()