import json

import numpy as np
import matplotlib.pyplot as plt

from ..GWXtreme.eos_inference import ModelSelector
from ..GWXtreme.config import EOS_LIST, LAL_NESTED_SAMPLING_TAYLORF2_EVIDENCES_FILE
from .plotting_scripts import plot_bayes_factors_bar_chart
from .scripts import compute_bayes_factors_from_nested_sampling_evidences, compute_single_event_bayes_factors


def compare_bayes_factors(eos: str, nested_method: str, nested_evidences_file: str, save_dir: str, use_new_kde: bool = True, old_kde_results: dict | None = None):
    # Normalizing Flow
    flow_ms = ModelSelector(
        'GW170817',
        method='2D',
        density_est_method='flow',
        parameterization='spectral'
    )

    flow_bf, flow_bf_array = flow_ms.compute_eos_evidence_ratio(
        EoS1=eos,
        EoS2="SLY",
        N_grid=1000,
        save_file=f'{save_dir}/GW170817_flow_{eos}_bfs.json',
        N_trials=1
    )

    # KDE
    if use_new_kde:
        # transformed KDE
        kde_ms = ModelSelector(
            'GW170817',
            method='2D',
            density_est_method='kde',
            parameterization='spectral'
        )

        kde_bf, kde_bf_array = kde_ms.compute_eos_evidence_ratio(
            EoS1="APR4_EPP",
            EoS2="SLY",
            N_grid=500,
            save_file=f'{save_dir}/GW170817_transformed_kde_{eos}_bfs.json',
            N_trials=100
        )
    else:
        # reflective KDE
        assert old_kde_results is not None
        kde_bf = old_kde_results['2D KDE TaylorF2'][eos][0]
        kde_bf_array = np.array(old_kde_results['2D KDE TaylorF2'][eos][1])

    # LAL Results
    nested_bf, nested_errors = compute_bayes_factors_from_nested_sampling_evidences(
        event='GW170817',
        method_label=nested_method,
        evidences_file=nested_evidences_file
    )

    print(flow_bf, flow_bf_array.shape)
    print(kde_bf, kde_bf_array.shape)
    print(nested_bf)    

    try:
        plt.clf()

        n1, _, _ = plt.hist(flow_bf_array, label='flow ensemble', histtype='step', density=True, color='blue')
        n2, _, _ = plt.hist(kde_bf_array, label='KDE trials', histtype='step', density=True, color='red')
        ymax = 1.1 * np.max(np.stack((n1, n2)))
        
        plt.vlines(flow_bf, 0., ymax, label='flow', color='blue')
        plt.vlines(kde_bf, 0., ymax, label='KDE', color='red')
        plt.vlines(nested_bf, 0., ymax, label=nested_method, color='green')

        plt.legend()
        plt.title(f"KDE vs. Flows for GW170817, {eos}")
        plt.savefig(f"{save_dir}/GW170817_{eos}_flows_vs_kde_bayes_factors.png")
    except Exception as e:
        print(e)

    return flow_bf, flow_bf_array, kde_bf, kde_bf_array, nested_bf, nested_errors


def plot_comparison_bar_chart(eos: str, nested_method: str, save_dir: str, use_new_kde: bool = True):
    # plot_bayes_factors_bar_chart(
    #     'GW170817',
    #     bayes_factors_file=
    # )
    pass


def main():
    event = "GW170817"
    method = '2D'
    waveform = 'TaylorF2'
    nested_method = "TaylorF2 LALInference_Nest"
    save_dir = f"/home/joseph/LocalProjects/GWXtreme/systematics/comparing_KDE_and_normalizing_flows/{event}"

    # all_bfs_file = "/home/joseph/LocalProjects/GWXtreme/systematics/data/BNS/BFs/GW170817_all_bayes_factors.json"
    # with open(all_bfs_file) as f:
    #     all_bfs = json.load(f)

    all_results = {}

    flow_bfs = compute_single_event_bayes_factors(
        event,
        method,
        waveform,
        "flow",
        EOS_LIST,
        N_trials=1,
        save_dir=None
    )

    kde_bfs = compute_single_event_bayes_factors(
        event,
        method,
        waveform,
        "kde",
        EOS_LIST,
        N_trials=100,
        save_dir=None
    )
    
    all_results = {
        method: {
            waveform: {
                "flow": flow_bfs[method][waveform]["flow"],
                "kde": kde_bfs[method][waveform]["kde"]
            }
        }
    }

    with open(f"{save_dir}/all_bayes_factors.json", 'w+') as f:
        json.dump(all_results, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    main()