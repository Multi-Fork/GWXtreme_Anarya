from typing import Literal
import json
import pathlib

import h5py
import numpy as np
import torch
import scipy.interpolate
import lal
import lalsimulation as lalsim

from .eos_prior import compute_log_pressure_from_eos, create_spectral_eos


def get_masses(q, mc):
    '''
    Given chirp-mass and mass ratio, compute the individual masses
    '''
    m1 = mc * (1 + q)**(1/5) * (q)**(-3/5)
    m2 = mc * (1 + q)**(1/5) * (q)**(2/5)
    return (m1, m2)


def get_lambdat(m1, m2, Lambda1, Lambda2):
    '''
    This function converts Lambda1, Lambda2, mass1, mass2
    to Lambda tilde.
    '''
    LambdaTilde = (2**5)/(26*(m1 + m2)**5)
    LambdaTilde *= (m1**5 + 12*m2*m1**4)*Lambda1 + (m2**5 + 12*m1*m2**4)*Lambda2
    return LambdaTilde


def get_lambdat_for_eos(m1, m2, max_mass_eos, eosfunc):
    '''
    This function accepts the masses and an equation of state interpolant
    with its maximum allowed mass, and return the values of LambdaT.
    '''
    kerr_cases_1 = m1 >= max_mass_eos
    kerr_cases_2 = m2 >= max_mass_eos

    Lambda1 = np.zeros_like(m1)
    Lambda2 = np.zeros_like(m2)

    # interpolate from known curves to obtain tidal
    # deformabilities as a function of mass for the
    # rest of the points
    Lambda1[~kerr_cases_1] = eosfunc(m1[~kerr_cases_1])
    Lambda2[~kerr_cases_2] = eosfunc(m2[~kerr_cases_2])

    # compute chirp tidal deformability
    LambdaT = get_lambdat(m1, m2, Lambda1, Lambda2)

    return LambdaT


def get_lambda_for_eos(m, max_mass_eos, eosfunc):
    '''
    This function accepts the mass and an equation of state interpolant
    with its maximum allowed mass, and return the values of Lambda.
    '''
    kerr_cases = m >= max_mass_eos

    Lambda = np.zeros_like(m)

    # interpolate from known curves to obtain tidal
    # deformabilities as a function of mass for the
    # rest of the points
    Lambda[~kerr_cases] = eosfunc(m[~kerr_cases])
    return Lambda


def apply_mass_constraint(m1, m2, q, minMass):
    '''
    Apply constraints on masses based on the prior or posterior sample
    spread.
    '''
    min_mass_violation_1 = m1 < minMass
    min_mass_violation_2 = m2 < minMass
    min_mass_violation = min_mass_violation_1 + min_mass_violation_2
    m1 = m1[~min_mass_violation]
    m2 = m2[~min_mass_violation]
    q = q[~min_mass_violation]
    return (m1, m2, q)


def get_eos_interpolant(EoS: str, m_min: float = 1.0, N_points: int = 100) -> tuple[scipy.interpolate.interp1d, float]:
    '''
    This method accepts one of the NS native equations of state
    and uses that to return  (s, max_mass) where
    s is the interpolation function for the mass and the tidal
    deformability.

    EoS     :: Equation of state native to LALsuite

    m_min       :: The minimum mass of the NS from which value
                    the interpolant will be constructed
                    (default = 1.0).

    N_points           :: Number of points that will be used for the
                    construction of the interpolant.
    '''

    assert EoS in list(lalsim.SimNeutronStarEOSNames), \
        'EoS family not available in lalsimulation\nAllowed EoS are :\n' + str(lalsim.SimNeutronStarEOSNames)

    eos = lalsim.SimNeutronStarEOSByName(EoS)
    fam = lalsim.CreateSimNeutronStarFamily(eos)
    max_mass = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI

    # This is necessary so that interpolant is computed over the full range
    # Keeping number upto 3 decimal places
    # Not rounding up, since that will lead to RuntimeError
    max_mass = int(max_mass*1000) / 1000

    # if max_mass of EoS is smaller than population's min mass, they're all BBHs
    # if definition of max_mass or m_min is changed in the future, adjust this logic accordingly
    if max_mass < m_min: return [None, np.nan]

    masses = np.linspace(m_min, max_mass, N_points)
    masses = masses[masses <= max_mass]
    
    grav_masses, Lambdas = get_eos_lambdas_from_masses(masses, fam)

    s = scipy.interpolate.interp1d(grav_masses, Lambdas)
    return s, max_mass


def get_eos_interpolant_from_parameters(
        params, 
        parameterization: Literal['spectral', 'polytrope'],
        N_points: int = 100,
        m_min: float = 0.8
    ) -> tuple[scipy.interpolate.interp1d, float, float]:
    '''
    This method accepts a four parameter description of the neutron star 
    equation of state, and returns (s, max_mass, min_mass) where s is 
    the interpolation function for the mass and the tidal deformability.

    params      :: Four parameter list.

    N           :: Number of points that will be used for the
                    construction of the interpolant.
    '''

    if parameterization == 'polytrope':
        # params are log_p1_SI, g1, g2, g3
        eos = lalsim.SimNeutronStarEOS4ParameterPiecewisePolytrope(*params)
    
    elif parameterization == 'spectral':
        # params are g0, g1, g2, g3
        eos = lalsim.SimNeutronStarEOS4ParameterSpectralDecomposition(*params)

    fam = lalsim.CreateSimNeutronStarFamily(eos)
    m_max = lalsim.SimNeutronStarMaximumMass(fam)/lal.MSUN_SI
    
    # This is necessary so that interpolant is computed over the full range
    # Keeping number upto 3 decimal places
    # Not rounding up, since that will lead to RuntimeError
    max_mass = int(m_max*1000    ) / 1000
    min_mass = int(m_min*1000 + 1) / 1000
    masses = np.linspace(max(m_min, min_mass), max_mass, N_points)
    masses = masses[masses <= max_mass]
    
    grav_masses, lambdas = get_eos_lambdas_from_masses(masses, fam)
    
    s = scipy.interpolate.interp1d(x=grav_masses, y=lambdas)
    
    return s, max_mass, max(m_min, min_mass)


def get_eos_interpolant_from_mass_tidal_file(mass_tidal_file: str) -> tuple[scipy.interpolate.interp1d, float]:
    '''
    This method accepts the data from a file that has the
    tidal deformability information in the following format:

    #mass    	λ

    ...	    	...

    ...		    ...

    max_mass	...

    The values of masses should be in units of solar masses. The
    tidal deformability λ should be supplied in SI unit.

    The method computes the dimensionless tidal deformabiliy Λ and
    returns (s, max_mass) where s is the interpolation
    function for the mass and the tidal deformability.
    '''
    masses, lambdas = np.loadtxt(mass_tidal_file, unpack=True)
    Lambdas = lal.G_SI*lambdas*(1/(lal.MRSUN_SI*masses)**5)
    s = scipy.interpolate.interp1d(masses, lambdas)
    max_mass = np.max(masses)
    return s, max_mass


def get_eos_interpolant_from_mass_radius_file(MRFile: str) -> tuple[scipy.interpolate.interp1d, float]:
    '''
    This method accepts the data from a file that have the
    mass-radius-love deformability information in the following format:

    #mass		Radius       love_num

    ...	    	...          ...

    ...		...          ...

    max_mass	...          ...

    The values of masses should be in units of solar masses. The
    tidal deformability radius should be supplied in meters.

    The method computes the dimensionless tidal deformabiliy Λ and
    returns (s, max_mass) where s is the interpolation
    function for the mass and the tidal deformability.
    '''
    masses, radius, kappa = np.loadtxt(MRFile, unpack=True)
    compactness = masses*lal.MRSUN_SI/radius
    Lambdas = (2/3)*kappa / (compactness**5)
    s = scipy.interpolate.interp1d(masses, Lambdas)
    max_mass = np.max(masses)
    return s, max_mass


def get_eos_lambdas_from_masses(masses: np.ndarray, eos_fam) -> tuple[np.ndarray, np.ndarray]:
    lambdas = []
    grav_masses = []
    for m in masses:
        try:
            rr = lalsim.SimNeutronStarRadius(m*lal.MSUN_SI, eos_fam)
            kk = lalsim.SimNeutronStarLoveNumberK2(m*lal.MSUN_SI, eos_fam)
            cc = m*lal.MRSUN_SI/rr
            lambdas.append((2/3)*kk/(cc**5))
            grav_masses.append(m)
        except RuntimeError:
            break
    
    lambdas = np.array(lambdas)
    grav_masses = np.array(grav_masses)
    return grav_masses, lambdas


def _read_prior_or_posterior_file(posterior_file: str, method: Literal['2D', '3D']) -> dict:
    posterior_file_ = pathlib.Path(posterior_file)
    ext = posterior_file_.suffix

    m1, m2, q, mc, lambda1, lambda2, lambdat = None, None, None, None, None, None, None

    if ext == '.h5':
        with h5py.File(posterior_file_) as f:
            data = np.array(f['posterior_samples'])
        
        if method == '2D':
            m1 = np.array(data['m1_source'])
            m2 = np.array(data['m2_source'])
            q = np.array(data['q'])
            mc = np.array(data['mc_source'])
            lambdat = np.array(data['lambdat'])
        
        elif method == '3D':
            m1 = np.array(data['m1_source'])
            m2 = np.array(data['m2_source'])
            q = np.array(data['q'])
            mc = np.array(data['mc_source'])
            lambda1 = np.array(data['lambda_1'])
            lambda2 = np.array(data['lambda_2'])

    elif ext == '.txt':
        data = np.loadtxt(posterior_file)
        if method == '2D':
            m1 = np.array(data[0])
            m2 = np.array(data[1])
            q = np.array(data[2])
            mc = np.array(data[3])
            lambdat = np.array(data[4])

        elif method == '3D':
            m1 = np.array(data[0])
            m2 = np.array(data[1])
            q = np.array(data[2])
            mc = np.array(data[3])
            lambda1 = np.array(data[4])
            lambda2 = np.array(data[5])

    elif ext == '.json':
        with open(posterior_file) as f:
            data = json.load(f)['posterior']['content']
        
        if method == '2D':
            m1 = np.array(data['m1_source'])
            m2 = np.array(data['m2_source'])
            q = np.array(data['q'])
            mc = np.array(data['mc_source'])
            lambdat = np.array(data['lambdat'])
        
        elif method == '3D':
            m1 = np.array(data['m1_source'])
            m2 = np.array(data['m2_source'])
            q = np.array(data['q'])
            mc = np.array(data['mc_source'])
            lambda1 = np.array(data['lambda_1'])
            lambda2 = np.array(data['lambda_2'])

    else:
        data = np.genfromtxt(posterior_file, names=True)
        
        if method == '2D':
            m1 = np.array(data['m1_source'])
            m2 = np.array(data['m2_source'])
            q = np.array(data['q'])
            mc = np.array(data['mc_source'])
            lambdat = np.array(data['lambdat'])
        
        elif method == '3D':
            m1 = np.array(data['m1_source'])
            m2 = np.array(data['m2_source'])
            q = np.array(data['q'])
            mc = np.array(data['mc_source'])
            lambda1 = np.array(data['lambda_1'])
            lambda2 = np.array(data['lambda_2'])
    
    return {
        'm1_source': m1,
        'm2_source': m2,
        'q': q,
        'mc_source': mc,
        'lambdat': lambdat,
        'lambda1': lambda1,
        'lambda2': lambda2
    }


def get_gw_event_pe_posterior_samples(posterior_file: str, method: Literal['2D', '3D']):    
    samples = _read_prior_or_posterior_file(posterior_file, method)
    
    if method == '2D':
        return torch.tensor(samples['lambdat'], dtype=torch.float32), torch.tensor(samples['q'], dtype=torch.float32)
    elif method == '3D':
        return torch.tensor(samples['lambda1'], dtype=torch.float32), \
            torch.tensor(samples['q'], dtype=torch.float32), \
            torch.tensor(samples['lambda2'], dtype=torch.float32)
    else:
        raise NotImplementedError()
    

def compute_eos_constraints_from_spectral_samples(
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


def compute_lambdas_from_spectral_eos_samples(
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


def compute_max_masses_from_spectral_eos_samples(
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