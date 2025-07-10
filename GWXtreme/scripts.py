import json
import pathlib
from typing import Literal

import numpy as np
import h5py

import lalsimulation as lalsim
import lal

from gwxtreme.GWXtreme.eos_inference import ModelSelector
from parametrized_eos_sampler import ParameterizedEoSSampler
from eos_prior import compute_log_pressure_from_eos, create_spectral_eos
from shared_config import *

##### composed files #####
# cornerPlots_EoScurves.py
# eventBFs.py
# eventConstraint.py
# indivLambdaHist.py
# indivSimulConstraint.py
# lambdaHist.py
# maxMassHist.py
# nestSamp_BFs.py
# simulBFs.py


def compute_single_event_bayes_factors(
        event_label: str,
        method_label: str,
        posterior_file: str,
        method: Literal['2D', '3D'],
        EoS_names: list[str] = EOS_LIST,
        N_trials: int = 10_000,
        save_dir: str = "data/BNS/BFs"
):
    """
    Compute Bayes Factors for each EoS in EoS_names from the event parameter posteriors in posterior_file, 
    re-computing N_trials number of times to be used later for error estimation.
    """
    model_selector = ModelSelector(
        posterior_file,
        method=method,
        N_samples=4000, 
    )
    BFs = []
    trials = []
    for EoS in EoS_names:
        if N_trials == 0:
            bf = model_selector.compute_eos_evidence_ratio(EoS1=EoS, EoS2="SLY", N_trials=N_trials)
            bf_trials = []
        else:
            bf, bf_trials = model_selector.compute_eos_evidence_ratio(EoS1=EoS, EoS2="SLY", trials=N_trials) # type: ignore
            bf_trials = bf_trials.tolist()
        BFs.append(bf)
        trials.append(bf_trials)

    out = {method_label : {EoS_names[i] : [BFs[i], trials[i]] for i in range(len(EoS_names))}}
    
    save_file = pathlib.Path(f"{save_dir}/{event_label}_{method_label.replace(" ", "-")}_bayes_factors.json")
    save_file.touch(exist_ok=True)
    
    with open(save_file, "w") as f:
        json.dump(out, f, indent=4, sort_keys=True)


def compute_bayes_factors_from_nested_sampling_evidences(
        event_label: str,
        method_label: str,
        evidences_file: str, # .json
        save_dir: str = "data/BNS/BFs",
):
    # Opens files that originate from a single file with GW170817's nested sampling
    # evidences for each EoS. We compute the BFs w.r.t. SLY and try out multiple 
    # variations on its "error": 
    # 1) quadrature sum
    # 2) "worst possible error"
    # 3) fractional error
    with open(evidences_file) as f:
        evidences = json.load(f)

    bayes_factors = {method_label: {}}
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
        
        bayes_factors[method_label][EoS] = [BF, [err1, err2, err3]]

    with open(f"{save_dir}/{event_label}_{method_label.replace(" ", "-")}_bayes_factors.json", "w+") as f:
        json.dump(bayes_factors, f, indent=4, sort_keys=True)


def combine_bayes_factors_files(
        event_label: str,
        bayes_factors_files: list[str],
        save_dir: str
) -> str:
    all_bfs = {}
    for file in bayes_factors_files:
        with open(file) as f:
            bfs = json.load(f)
        all_bfs.update(bfs)
    
    save_file = f"{save_dir}/{event_label}_all_bayes_factors.json"
    with open(save_file, "w") as f:
        json.dump(all_bfs, f, indent=4, sort_keys=False)

    return save_file


def sample_spectral_EoS_parameters(
        event_label: str,
        method_label: str,
        posterior_file: str, # .dat file
        method: Literal['2D', '3D'],
        N_walkers: int = 100,
        N_parameter_samples: int = 10_000,
        N_pool: int = 100,
        save_dir: str = "data/BNS/constraints",
):
    samples_save_file = f'{save_dir}/{event_label}_{method_label.replace(" ", "-")}_spectral_parameter_posterior_samples'

    #Initialize Sampler Object:
    sampler = ParameterizedEoSSampler(
        posterior_files=[posterior_file], 
        prior_bounds={
            'gamma1': {'params':{"min": 0.2, "max": 2.00}},
            'gamma2': {'params':{"min": -1.6, "max": 1.7}},
            'gamma3': {'params':{"min": -0.6, "max": 0.6}},
            'gamma4': {'params':{"min": -0.02, "max": 0.02}}
        },
        save_file=samples_save_file,
        methods=[method],
        N_walkers=N_walkers, 
        N_parameter_samples=N_parameter_samples, 
        N_dim=4, 
        parameterization='spectral',
        N_pool=N_pool,
    )

    #Run, Save , Plot
    sampler.initialize_walkers()
    sampler.run_sampler()
    sampler.save_data()


def compute_EoS_constraints_from_spectral_samples(
        event_label: str,
        method_label: str,
        spectral_samples_file: str,
        burn_in_frac: float = 0.5,
        thin_every: int = 5,
        save_dir: str = "data/BNS/constraints"
):
    # Load the samples   
    file_type = pathlib.Path(spectral_samples_file).suffix
    if file_type == '.h5':
        with h5py.File(spectral_samples_file) as f:
            samples = np.array(f['chains'], dtype=np.float32)
            samples = samples.reshape((samples.shape[0]*samples.shape[1], 4))
    elif file_type == '.txt':
        samples = np.loadtxt(spectral_samples_file, dtype=np.float32)
    else:
        print("Samples file type must be .h5 or .txt.")
        return
    
    # "Clean" the samples
    np.nan_to_num(samples, copy=False)
    burn_in = int(samples.shape[0]*burn_in_frac)
    samples = samples[burn_in::thin_every]
    
    # Turn into confidence interval data
    logp = []
    rho = np.logspace(17.1, 18.25, 1000)

    for s in samples:
        params = (s[0], s[1], s[2], s[3])
        p = compute_log_pressure_from_eos(rho, create_spectral_eos(params))
        logp.append(p)

    logp = np.array(logp)
    logp_CIup = np.array([np.quantile(logp[:,i], 0.95) for i in range(len(rho))])
    logp_CIlow = np.array([np.quantile(logp[:,i], 0.05) for i in range(len(rho))])
    logp_med = np.array([np.quantile(logp[:,i], 0.5) for i in range(len(rho))])

    # Save confidence interval data
    np.savetxt(f"{save_dir}/{event_label}_{method_label.replace(" ", "-")}_confidence_interval.txt", np.array([rho, logp_CIlow, logp_med, logp_CIup]).T)


def compute_lambdas_from_spectral_EoS_samples(
        event_label: str,
        method_label: str,
        spectral_samples_file: str, # .txt
        save_dir: str = "data/NSBH/lambdaHists"
):
    # Load the samples   
    file_type = pathlib.Path(spectral_samples_file).suffix
    if file_type == '.h5':
        with h5py.File(spectral_samples_file) as f:
            samples = np.array(f['chains'], dtype=np.float32)
            samples = samples.reshape((samples.shape[0]*samples.shape[1], 4))
    elif file_type == '.txt':
        samples = np.loadtxt(spectral_samples_file, dtype=np.float32)
    else:
        raise UserWarning("Samples file type must be .h5 or .txt.")
    
    lambdas = []
    m = 1.4
 
    for sample in samples:
        g0, g1, g2, g3 = sample
        EoS = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(g0, g1, g2, g3)
        fam = lalsim.CreateSimNeutronStarFamily(EoS)

        rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, fam)
        kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, fam)
        cc = m*lal.MRSUN_SI/rr
        lambda_ = (2/3)*kk/(cc**5)
        lambdas.append(lambda_)

    np.savetxt(f"{save_dir}/{event_label}_{method_label.replace(" ", "-")}_lambdas_samples.txt", np.array(lambdas).T)


def compute_max_masses_from_spectral_EoS_samples(
        event_label: str,
        method_label: str,
        spectral_samples_file: str, # .txt
        save_dir: str = "data/BNS/massHists"
):
    # Load the samples   
    file_type = pathlib.Path(spectral_samples_file).suffix
    if file_type == '.h5':
        with h5py.File(spectral_samples_file) as f:
            samples = np.array(f['chains'], dtype=np.float32)
    elif file_type == '.txt':
        samples = np.loadtxt(spectral_samples_file, dtype=np.float32)
    else:
        print("Samples file type must be .h5 or .txt.")
        return
    
    maxMasses = []
    m = 1.4
    
    for sample in samples:
        g0, g1, g2, g3 = sample
        EoS = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(g0, g1, g2, g3)
        fam = lalsim.CreateSimNeutronStarFamily(EoS)
        maxMass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
        maxMasses.append(maxMass)

    np.savetxt(f"{save_dir}/{event_label}_{method_label.replace(" ", "-")}_max_masses_samples.txt", np.array(maxMasses).T)