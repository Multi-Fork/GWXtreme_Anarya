import json
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import corner
import arviz as az

import lalsimulation as lalsim
import lal

from .eos_model_selection import get_lambda_for_eos
from .eos_prior import compute_log_pressure_from_eos

from shared_config import EOS_LIST


def plot_bayes_factors_bar_chart(
        event_label: str,
        bayes_factors_file: str,      # .json
        method_labels: list[str],
        EoS_list: list[str] = EOS_LIST,
        nested_uncert_method: Literal["quad", "worst", "frac"] = 'quad',
        save_dir: str = "plots/BNS/BFs",
):
    with open(bayes_factors_file) as f:
        data = json.load(f)
    
    num_methods = len(method_labels)
    if num_methods > 3:
        raise UserWarning("More than 3 BFs methods not supported for plotting.")
        
    colors = ["#f94b42", "#c8c0ff", "#ffa551"]

    # index of where to grab the uncert value in the bf_array for nested sampling method results.
    # this is defined implicitly based on how these uncert methods are computed and stored by the
    # compute_bayes_factors_from_nested_sampling_evidences() function.
    nested_uncert_method_map = {'quad': 0, 'worst': 1, 'frac': 2}
    
    # the different kinds of errors we considerd for nested BFs (cause we're given evidences)
    spacing = [-.10, .10] if num_methods == 2 else [-.20, .0, .20]
   
    plt.clf()
    plt.rcParams.update({"font.size":18})
    plt.figure(figsize=(15, 10))

    x_axis = np.arange(len(EoS_list))

    plt.clf()
    max_bf = 0
    max_uncert = 0
    
    for i, method_label in enumerate(method_labels):
        BFs = []
        uncerts = []

        for eos in EoS_list:
            BFs.append(data[method_label][eos][0]) 
            # Because the nested BFs have their errors computed already and there's 3 different ones
            if len(data[method_label][eos][1]) > 3: 
                trials = np.array(data[method_label][eos][1])
                uncerts.append(np.std(trials) * 2)
            else:
                uncerts.append(np.abs(data[method_label][eos][1][nested_uncert_method_map[nested_uncert_method]])) # abs because they could be negative

        plt.bar(
            x=x_axis + spacing[i],
            height=BFs,
            width=.20,
            label=method_label,
            color=colors[i]
        )
        plt.errorbar(
            x=x_axis + spacing[i],
            y=BFs,
            yerr=uncerts,
            ls="none",
            ecolor="black"
        )

        max_bf = max(max_bf, max(BFs))
        max_uncert = max(max_uncert, max(uncerts))
        
        plt.yscale("log")
        plt.xticks(x_axis, EoS_list, rotation=90, ha="right")
        plt.ylim(1.0e-5, max(10., max_bf + max_uncert * 10.))
        plt.axhline(1.0,color="k",linestyle="--",alpha=0.2)
        plt.ylabel("Bayes-factor w.r.t SLY")
        plt.legend(loc='lower left')
        methods_str = '_'.join([label.replace(" ", "-") for label in method_labels])
        plt.savefig(f"{save_dir}/{event_label}_{methods_str}_{nested_uncert_method}-error_bayes_factors.png", bbox_inches="tight")


def plot_BNS_parameter_corner(
        event_label: str,
        posterior_label: str,
        posterior_file: str, # .json
        save_dir: str,
        EoS: str = "APR4_EPP",
):
    with open(posterior_file) as f:
        data = json.load(f)['posterior']['content']
    
    try:
        m1, m2, q, mc, Lambda1, Lambda2 = (
            np.array(data['m1_source']),
            np.array(data['m2_source']),
            np.array(data['q']),
            np.array(data['mc_source']),
            np.array(data['lambda_1']),
            np.array(data['lambda_2'])
            )
    except KeyError as e:
        print(f"Posterior samples must contain 'lambda_1' and 'lambda_2'.\nError:{str(e)}")
        return

    # Obtain EoS curve
    eos = lalsim.SimNeutronStarEOSByName(EoS)
    fam = lalsim.CreateSimNeutronStarFamily(eos)
    m_min = 0.8
    max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI

    # This is necessary so that interpolant is computed over the full range
    # Keeping number upto 3 decimal places
    # Not rounding up, since that will lead to RuntimeError
    max_mass = int(max_mass*1000)/1000
    masses = np.linspace(m_min, max_mass, 1000)
    masses = masses[masses <= max_mass]
    Lambdas = []
    gravMass = []
    for m in masses:
        try:
            rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
            kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, fam)
            cc = m*lal.MRSUN_SI/rr
            Lambdas = np.append(Lambdas, (2/3)*kk/(cc**5))
            gravMass = np.append(gravMass, m)
        except RuntimeError:
            break
    Lambdas = np.array(Lambdas)
    gravMass = np.array(gravMass)
    eosfunc = interp1d(gravMass, Lambdas)

    M1, M2 = np.linspace(min(m1),max(m1),1000), np.linspace(min(m2),max(m2),1000)
    Q = M2/M1
    MC = ((M1*M2)**(3/5)) / ((M1+M2)**(1./5.))
    Lambda1, Lambda2 = get_lambda_for_eos(M1, max_mass, eosfunc), get_lambda_for_eos(M2, max_mass, eosfunc)

    # Format curve values
    EoS_values = np.array([Lambda1,Lambda2,M1,M2,MC,Q])

    # Construct corner & and overlay appropriate curves
    figure = corner.corner(az.from_dict(data))
    ndim = 6
    axes = np.array(figure.axes).reshape((ndim, ndim))
    for yi in range(ndim):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.plot(EoS_values[xi], EoS_values[yi], color="red")

    plt.savefig(f"{save_dir}/{event_label}_{posterior_label}_{EoS}_corner.png")


def plot_EoS_constraints(
        event_label: str,
        constraints_files: list[str],
        method_labels: list[str],
        EoS_list: list[str] = ["APR4_EPP"],
        save_dir: str = "plots/BNS/constraints"
):
    num_methods = len(constraints_files)

    colors = ['#beaed4','#fdc086'] if num_methods == 2 else ['#ffffb3','#bebada','#fb8072']
    hatches = ["","x"] if num_methods == 2 else ["|","-",""]

    plt.figure(figsize=(12,12))
    plt.rc('font', size=20)
    #plt.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    plt.rc('xtick', direction='out', color='black')
    plt.rc('ytick', direction='out', color='black')
    plt.rc('lines', linewidth=2)

    rho = 0 # just to define the var
    for file, method_label, color, hatch in zip(constraints_files, method_labels, colors, hatches):
        # Load the samples
        # nest result is named differently
        rho, lower_bound, median, upper_bound = np.loadtxt(file).T
        plt.fill_between(np.log10(rho), lower_bound, upper_bound, color=color, alpha=0.45, label=method_label, zorder=1., hatch=hatch)
    
    for EoS in EoS_list:
        logp = compute_log_pressure_from_eos(rho, lalsim.SimNeutronStarEOSByName(EoS))
        plt.plot(np.log10(rho), logp, linewidth=2.0, label=EoS, alpha=0.35)

    plt.xlim([min(np.log10(rho)), 18.25])
    plt.xlabel(r'$\log10{\frac{\rho}{g cm^-3}}$',fontsize=20)
    plt.ylabel(r'$log10(\frac{p}{dyne cm^{-2}})$',fontsize=20)
    plt.legend()
    methods_str = '_'.join([label.replace(" ", "-") for label in method_labels])
    plt.savefig(f"{save_dir}/{event_label}_{methods_str}_constraints.png", bbox_inches='tight')


def plot_lambdas_from_spectral_EoS_parameters(
        event_label: str,
        lambdas_samples_files: list[str],
        method_labels: list[str],
        colors: list[str],
        eos_name: str = "APR4_EPP",
        save_dir: str = "plots/NSBH/lambdaHists"
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
    #plt.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    plt.rc('xtick', direction='out', color='black')
    plt.rc('ytick', direction='out', color='black')
    plt.rc('lines', linewidth=2)

    for file, color, method_label in zip(lambdas_samples_files, colors, method_labels):
        Lambdas = np.loadtxt(file).T
        plt.hist(Lambdas, label=method_label, alpha=0.45, fill=True, density=True, color=color, histtype='step')

    plt.axvline(x=eosLambda, label=eos_name, color="black")
    plt.xlabel(r"$\Lambda$(1.4)",fontsize=20)
    plt.yticks([])
    plt.legend()
    methods_str = '_'.join([label.replace(" ", "-") for label in method_labels])
    plt.savefig(f"{save_dir}/{event_label}_{methods_str}_lambdas_samples.png", bbox_inches='tight')


def plot_max_masses_from_spectral_EoS_parameters(
        event_label: str,
        max_masses_samples_files: list[str],
        method_labels: list[str],
        colors: list[str],
        eos_name: str = "APR4_EPP",
        save_dir: str = "plots/BNS/massHists"
):
    m = 1.4
    eos = lalsim.SimNeutronStarEOSByName(eos_name)
    fam = lalsim.CreateSimNeutronStarFamily(eos)
    eosMaxMass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI

    plt.figure(figsize=(12,12))
    plt.rc('font', size=20)
    #plt.rc('axes', facecolor='#E6E6E6', edgecolor='black')
    plt.rc('xtick', direction='out', color='black')
    plt.rc('ytick', direction='out', color='black')
    plt.rc('lines', linewidth=2)

    for file, method_label, color in zip(max_masses_samples_files, method_labels, colors):
        MaxMasses = np.loadtxt(file).T
        plt.hist(MaxMasses, label=method_label, alpha=0.45, fill=True, density=True, color=color, histtype='step')

    plt.axvline(x=eosMaxMass, label=eos_name, color="black")

    plt.xlabel("Max NS Mass", fontsize=20)
    plt.yticks([])
    plt.legend()
    methods_str = '_'.join([label.replace(" ", "-") for label in method_labels])
    plt.savefig(f"{save_dir}/{event_label}_{methods_str}_max_masses_samples.png", bbox_inches='tight')