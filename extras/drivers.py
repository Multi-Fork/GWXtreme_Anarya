import json
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

import lalsimulation as lalsim
import lal

from ..GWXtreme.eos_prior import compute_log_pressure_from_eos
from ..GWXtreme.eos_inference import ParameterizedEoSSampler
from ..GWXtreme.eos_inference import JointModelSelector
from ..GWXtreme.config import EOS_LIST, LAL_NESTED_SAMPLING_PHENOM_EVIDENCES_FILE, LAL_NESTED_SAMPLING_TAYLORF2_EVIDENCES_FILE



def compute_all_eos_bayes_factors(
        model_selector: JointModelSelector,
        EoS_names: list[str] = EOS_LIST,
        N_trials: int = 0
) -> dict:
    """
    Compute Bayes Factors for each EoS in EoS_names for a given event.
    """
    
    BFs = {}
    for EoS in EoS_names:
        result = model_selector.compute_joint_eos_evidence_ratio(
            target_eos=EoS,
            ref_eos="SLY",
            N_grid=1000,
            N_trials=N_trials
        )

        if type(result) != tuple: # no repeated trials are returned
            bf = result
            bf_trials = []
        else:
            bf, bf_trials = result
            bf_trials = bf_trials.tolist()
        
        BFs[EoS] = {
            "native": bf,
            "resamples": bf_trials
        }
    
    return BFs


def compute_bayes_factors_from_nested_sampling_evidences(waveform: str) -> dict:
    evidences_file = LAL_NESTED_SAMPLING_TAYLORF2_EVIDENCES_FILE if waveform == "TaylorF2" else LAL_NESTED_SAMPLING_PHENOM_EVIDENCES_FILE
    # Opens files that originate from a single file with GW170817's nested sampling
    # evidences for each EoS. We compute the BFs w.r.t. SLY and try out multiple 
    # variations on its "error": 
    # 1) quadrature sum
    # 2) "worst possible error"
    # 3) fractional error
    with open(evidences_file) as f:
        evidences = json.load(f)

    bayes_factors = {waveform: {}}
    # Compute BFs from evidences from nested sampling inference
    for EoS in EOS_LIST:
        EoS1 = evidences[EoS][0] # evidence of EoS1
        EoS2 = evidences['SLY'][0] # evidence of EoS2
        BF = EoS1/EoS2

        EoS1err = evidences[EoS][-1]
        EoS2err = evidences['SLY'][-1]

        # 1) quadrature sum
        err1 = ((EoS1err**2) + (EoS2err**2))**0.5

        # 2) "worst possible error"
        EoS1min, EoS1max = EoS1-EoS1err, EoS1+EoS1err
        EoS2min, EoS2max = EoS2-EoS2err, EoS2+EoS2err

        ErrMin = EoS1max/EoS2min
        ErrMax = EoS1min/EoS2max
        err2 = ErrMax - ErrMin

        # 3) fractional error
        err3 = BF * (((EoS1err/EoS1)**2) + ((EoS2err/EoS2)**2)) ** 0.5
        
        bayes_factors[waveform][EoS] = {
            "bf": BF, 
            "quad_error": err1,
            "worst_error": err2,
            "fractional_error": err3
        }
    
    return bayes_factors


def plot_bayes_factors_bar_chart(
        bayes_factors_file: str,      # .json
        method_sets: list[tuple],
        save_file: str,
        yscale: str = 'linear',
        error_type: Literal['std', '2std', 'range'] = 'range',
        EoS_list: list[str] = EOS_LIST
):
    with open(bayes_factors_file) as f:
        data = json.load(f)
    
    num_methods = len(method_sets)
    if num_methods > 3:
        raise UserWarning("More than 3 BFs methods not supported for plotting.")
        
    colors = ["#f94b42", "#c8c0ff", "#ffa551"]
    spacing = [-.10, .10] if num_methods == 2 else [-.20, .0, .20]
   
    plt.clf()
    plt.rcParams.update({"font.size":18})
    plt.figure(figsize=(15, 10))

    x_axis = np.arange(len(EoS_list))

    labels = []
    all_bars = []
    all_uncerts = []
    for i, method_set in enumerate(method_sets):
        method, waveform, density_est_method = method_set
        label = f"{method} {density_est_method} {waveform}"
        labels.append(label)

        BFs = []
        uncerts = []

        for eos in EoS_list:
            eos_bfs = data[method][waveform][density_est_method][eos]
            
            # normalizing flow results
            if density_est_method == 'flow':
                BFs.append(eos_bfs["native"])
                ensemble = np.array(eos_bfs["ensemble"])
                ensemble = ensemble[~np.isnan(ensemble)] 
                
                if error_type == '2std':                
                    uncerts.append(2 * np.std(ensemble))
                elif error_type == 'std':
                    uncerts.append(np.std(ensemble))
                elif error_type == 'range':
                    uncerts.append(np.max(ensemble) - np.min(ensemble))
            
            # nested sampling results
            elif density_est_method == 'lal':
                BFs.append(eos_bfs["bf"])
                uncerts.append(np.abs(eos_bfs["worst_error"]))
            
            # kde results
            else:
                BFs.append(eos_bfs["native"])
                resamples = eos_bfs["resamples"]
                
                if error_type == '2std':                
                    uncerts.append(2 * np.std(resamples))
                elif error_type == 'std':
                    uncerts.append(np.std(resamples))
                elif error_type == 'range':
                    uncerts.append(np.max(resamples) - np.min(resamples))

        plt.bar(
            x=x_axis + spacing[i],
            height=BFs,
            width=.20,
            label=label,
            color=colors[i]
        )
        plt.errorbar(
            x=x_axis + spacing[i],
            y=BFs,
            yerr=uncerts,
            ls="none",
            ecolor="black"
        )
        
        all_bars += BFs
        all_uncerts += uncerts
    
    if yscale == 'log':
        plt.yscale('log')
        plt.ylim(1.0e-5, 10.)
    else:
        # plt.ylim(0., 1.1 * np.add(all_bars, all_uncerts).max())
        plt.ylim(bottom=0.)
    
    plt.xticks(x_axis, EoS_list, rotation=90, ha="right")
    plt.axhline(1.0,color="k",linestyle="--",alpha=0.2)
    plt.ylabel("Bayes-factor w.r.t SLY")
    plt.legend(loc='upper right')
    plt.savefig(save_file, bbox_inches="tight")


def plot_eos_constraints(
        constraints_files: list[str],
        labels: list[str],
        EoS_list: list[str],
        save_file: str
):
    colors = ["#0E6316","#d62728","#251b9a"]
    hatches = ["","|","-"]

    plt.figure(figsize=(12,12))
    plt.rc('font', size=20)
    #plt.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    plt.rc('xtick', direction='out', color='black')
    plt.rc('ytick', direction='out', color='black')
    plt.rc('lines', linewidth=2)

    rho = 0 # just to define the var
    for file, label, color, hatch in zip(constraints_files, labels, colors, hatches):
        # Load the samples
        # nest result is named differently
        rho, lower_bound, median, upper_bound = np.loadtxt(file).T
        plt.fill_between(np.log10(rho), lower_bound, upper_bound, color=color, alpha=0.45, label=label, zorder=1., hatch=hatch)
    
    for EoS in EoS_list:
        logp = compute_log_pressure_from_eos(rho, lalsim.SimNeutronStarEOSByName(EoS))
        plt.plot(np.log10(rho), logp, 'k', linewidth=2.0, label=EoS, alpha=0.45)

    plt.xlim([min(np.log10(rho)), 18.25])
    plt.xlabel(r'$\log10{\frac{\rho}{g cm^-3}}$',fontsize=20)
    plt.ylabel(r'$log10(\frac{p}{dyne cm^{-2}})$',fontsize=20)
    plt.legend()
    plt.grid()
    plt.savefig(save_file, bbox_inches='tight')


def plot_parameterized_eos_posterior(
        sampler: ParameterizedEoSSampler,
        samples_files: list[str],
        labels: list[str],
        save_file: str,
        burn_in_frac: float = 0.5,
        thinning: int | None = None
):
    assert len(samples_files) <= 3
    colors = ["#2D199A","#33af37","#ea7164"]
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    
    for file, label, color in zip(samples_files, labels, colors):
        sampler.load_samples(samples_file=file)
        samples = sampler.parse_samples(burn_in_frac, thinning)
        
        means = np.mean(samples, axis=0)

        n_bins = None
        
        n, _, _ = ax[0, 0].hist(samples[:, 0], bins=n_bins, density=True, label=label, color=color, histtype='step')
        ax[0, 0].vlines(means[0], 0., np.max(n), color=color, linestyle='dashed')
        ax[0, 0].set_xlabel(r"$\gamma_1$")
        ax[0, 0].legend()
        
        n, _, _ = ax[0, 1].hist(samples[:, 1], bins=n_bins, density=True, label=label, color=color, histtype='step')
        ax[0, 1].vlines(means[1], 0., np.max(n), color=color, linestyle='dashed')
        ax[0, 1].set_xlabel(r"$\gamma_2$")
        ax[0, 1].legend()

        n, _, _ = ax[1, 0].hist(samples[:, 2], bins=n_bins, density=True, label=label, color=color, histtype='step')
        ax[1, 0].vlines(means[2], 0., np.max(n), color=color, linestyle='dashed')
        ax[1, 0].set_xlabel(r"$\gamma_3$")
        ax[1, 0].legend()

        n, _, _ = ax[1, 1].hist(samples[:, 3], bins=n_bins, density=True, label=label, color=color, histtype='step')
        ax[1, 1].vlines(means[3], 0., np.max(n), color=color, linestyle='dashed')
        ax[1, 1].set_xlabel(r"$\gamma_4$")
        ax[1, 1].legend()
    
    plt.savefig(save_file, bbox_inches='tight')


def plot_lambdas_from_spectral_eos_parameters(
        lambdas_samples_files: list[str],
        method_labels: list[str],
        colors: list[str],
        save_file: str,
        EoS: str = "APR4_EPP"
):
    m = 1.4
    eos = lalsim.SimNeutronStarEOSByName(eos_name)
    fam = lalsim.CreateSimNeutronStarFamily(eos)

    rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
    kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, fam)
    cc = m*lal.MRSUN_SI/rr
    eosLambda = (2/3)*kk/(cc**5)

    plt.figure(figsize=(12,12))
    plt.rc('font', size=20)
    plt.rc('xtick', direction='out', color='black')
    plt.rc('ytick', direction='out', color='black')
    plt.rc('lines', linewidth=2)

    for file, color, method_label in zip(lambdas_samples_files, colors, method_labels):
        Lambdas = np.loadtxt(file).T
        plt.hist(Lambdas, label=method_label, alpha=0.45, fill=True, density=True, color=color, histtype='step')

    plt.axvline(x=eosLambda, label=EoS, color="black")
    plt.xlabel(r"$\Lambda$(1.4)",fontsize=20)
    plt.yticks([])
    plt.legend()
    plt.savefig(save_file, bbox_inches='tight')


def plot_max_masses_from_spectral_eos_parameters(
        max_masses_samples_files: list[str],
        method_labels: list[str],
        colors: list[str],
        save_file: str,
        EoS: str = "APR4_EPP",
):
    m = 1.4
    eos = lalsim.SimNeutronStarEOSByName(eos_name)
    fam = lalsim.CreateSimNeutronStarFamily(eos)
    eosMaxMass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI

    plt.figure(figsize=(12,12))
    plt.rc('font', size=20)
    plt.rc('xtick', direction='out', color='black')
    plt.rc('ytick', direction='out', color='black')
    plt.rc('lines', linewidth=2)

    for file, method_label, color in zip(max_masses_samples_files, method_labels, colors):
        MaxMasses = np.loadtxt(file).T
        plt.hist(MaxMasses, label=method_label, alpha=0.45, fill=True, density=True, color=color, histtype='step')

    plt.axvline(x=eosMaxMass, label=EoS, color="black")

    plt.xlabel("Max NS Mass", fontsize=20)
    plt.yticks([])
    plt.legend()
    plt.savefig(save_file, bbox_inches='tight')