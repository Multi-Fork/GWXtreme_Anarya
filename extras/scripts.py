import json
import pathlib
from typing import Literal

import numpy as np
import h5py

import lalsimulation as lalsim
import lal

from ..GWXtreme.eos_inference import ModelSelector, ParameterizedEoSSampler
from ..GWXtreme.eos_prior import compute_log_pressure_from_eos, create_spectral_eos
from ..GWXtreme.config import EOS_LIST, SUPPORTED_WAVEFORMS, LAL_NESTED_SAMPLING_PHENOM_EVIDENCES_FILE, LAL_NESTED_SAMPLING_TAYLORF2_EVIDENCES_FILE


def compute_single_event_bayes_factors(
        event: str,
        method: Literal['2D', '3D'],
        waveform: Literal['TaylorF2', 'IMRPhenomD_NRTidalv2'],
        density_est_method: Literal['kde', 'flow', 'reflectflow'],
        EoS_names: list[str] = EOS_LIST,
        N_trials: int = 0,
        save_file: str | None = None
):
    """
    Compute Bayes Factors for each EoS in EoS_names for a given event.
    """
    model_selector = ModelSelector(
        event=event,
        method=method,
        density_est_method=density_est_method
    )
    
    BFs = {method: {waveform: {density_est_method: {}}}}
    for EoS in EoS_names:
        result = model_selector.compute_eos_evidence_ratio(
            EoS1=EoS,
            EoS2="SLY",
            N_grid=1000,
            N_trials=N_trials
        )

        if type(result) != tuple: # no repeated trials are returned
            bf = result
            bf_trials = []
        else:
            bf, bf_trials = result
            bf_trials = bf_trials.tolist()
        
        if density_est_method == 'kde':
            BFs[method][waveform][density_est_method][EoS] = {
                "native": bf,
                "resamples": bf_trials
            }
        else:
            BFs[method][waveform][density_est_method][EoS] = {
                "native": bf,
                "ensemble": bf_trials
            }
    
    if save_file is not None:
        with open(save_file, "w") as f:
            json.dump(BFs, f, indent=4)
    
    return BFs


def compute_bayes_factors_from_nested_sampling_evidences(
        event: str,
        waveform: str,
        save_file: str | None = None,
):
    assert event == "GW170817" and waveform in SUPPORTED_WAVEFORMS
    evidences_file = LAL_NESTED_SAMPLING_TAYLORF2_EVIDENCES_FILE if waveform == "TaylorF2" else LAL_NESTED_SAMPLING_PHENOM_EVIDENCES_FILE
    method = "LALInference_NEST"
    # Opens files that originate from a single file with GW170817's nested sampling
    # evidences for each EoS. We compute the BFs w.r.t. SLY and try out multiple 
    # variations on its "error": 
    # 1) quadrature sum
    # 2) "worst possible error"
    # 3) fractional error
    with open(evidences_file) as f:
        evidences = json.load(f)

    bayes_factors = {method: {waveform: {}}}
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
        
        bayes_factors[method][waveform][EoS] = {
            "bf": BF, 
            "quad_error": err1,
            "worst_error": err2,
            "fractional_error": err3
        }

    if save_file is not None:
        with open(save_file, "w") as f:
            json.dump(bayes_factors, f, indent=4, sort_keys=True)
    
    return bayes_factors


def combine_bayes_factors_files(
        bayes_factors_files: list[str],
        save_file: str
) -> str:
    all_bfs = {}
    for file in bayes_factors_files:
        with open(file) as f:
            bfs = json.load(f)
        all_bfs.update(bfs)
    
    with open(save_file, "w") as f:
        json.dump(all_bfs, f, indent=4, sort_keys=False)

    return save_file


def sample_spectral_EoS_parameters(
        event: str,
        method: Literal['2D', '3D'],
        density_est_method: Literal['kde', 'flow', 'reflectkde'],
        save_file: str,
        N_pool: int,
        N_walkers: int = 100,
        N_samples: int = 10_000,
):
    sampler = ParameterizedEoSSampler(
        events=[event],
        method=method,
        prior_bounds={
            'gamma1': {'params':{"min": 0.2, "max": 2.00}},
            'gamma2': {'params':{"min": -1.6, "max": 1.7}},
            'gamma3': {'params':{"min": -0.6, "max": 0.6}},
            'gamma4': {'params':{"min": -0.02, "max": 0.02}}
        },
        density_est_method=density_est_method,
        parameterization='spectral',
    )

    sampler.initialize_walkers(N_walkers)
    sampler.run_sampler(N_samples, N_pool=N_pool, N_grid=1000, save_file=save_file)


def compute_EoS_constraints_from_spectral_samples(
        spectral_samples_file: str,
        burn_in_frac: float = 0.5,
        thin_every: int = 5,
        save_file: str | None = None
):
    # Load the samples   
    file_type = pathlib.Path(spectral_samples_file).suffix
    if file_type == '.h5':
        with h5py.File(spectral_samples_file) as f:
            samples = np.array(f['samples'], dtype=np.float32)
    elif file_type == '.txt':
        samples = np.loadtxt(spectral_samples_file, dtype=np.float32)
    else:
        raise ValueError("Samples file type must be .h5 or .txt.")
    
    # "Clean" the samples
    np.nan_to_num(samples, copy=False)
    burn_in = int(samples.shape[0]*burn_in_frac)
    samples = samples[burn_in::thin_every]
    
    # Turn into confidence interval data
    logp = []
    rho = np.logspace(17.1, 18.25, 1000)

    for s in samples:
        logp.append(compute_log_pressure_from_eos(rho, create_spectral_eos(s)))

    logp = np.array(logp)
    logp_CIup =  np.quantile(logp, 0.95, axis=0)
    logp_CIlow = np.quantile(logp, 0.05, axis=0)
    logp_med =   np.quantile(logp, 0.50, axis=0)

    out = np.array([rho, logp_CIlow, logp_med, logp_CIup]).T

    if save_file is not None:
        # Save credible interval data
        np.savetxt(save_file, out)
    
    return out


def compute_lambdas_from_spectral_EoS_samples(
        spectral_samples_file: str, # .txt
        save_file: str | None = None
):
    # Load the samples   
    file_type = pathlib.Path(spectral_samples_file).suffix
    if file_type == '.h5':
        with h5py.File(spectral_samples_file) as f:
            samples = np.array(f['samples'], dtype=np.float32)
    elif file_type == '.txt':
        samples = np.loadtxt(spectral_samples_file, dtype=np.float32)
    else:
        raise ValueError("Samples file type must be .h5 or .txt.")
    
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
    
    lambdas = np.array(lambdas).T

    if save_file is not None:
        np.savetxt(save_file, lambdas)
    
    return lambdas


def compute_max_masses_from_spectral_EoS_samples(
        spectral_samples_file: str,
        save_file: str | None = None
):
    # Load the samples   
    file_type = pathlib.Path(spectral_samples_file).suffix
    if file_type == '.h5':
        with h5py.File(spectral_samples_file) as f:
            samples = np.array(f['samples'], dtype=np.float32)
    elif file_type == '.txt':
        samples = np.loadtxt(spectral_samples_file, dtype=np.float32)
    else:
        raise ValueError("Samples file type must be .h5 or .txt.")
    
    maxMasses = []
    m = 1.4
    
    for sample in samples:
        g0, g1, g2, g3 = sample
        EoS = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(g0, g1, g2, g3)
        fam = lalsim.CreateSimNeutronStarFamily(EoS)
        maxMass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
        maxMasses.append(maxMass)
    
    maxMasses = np.array(maxMasses).T

    if save_file is not None:
        np.savetxt(save_file, maxMasses)
    
    return maxMasses