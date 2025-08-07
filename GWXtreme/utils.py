from typing import Literal
import json
import pathlib

import h5py
import numpy as np
import scipy.interpolate

import lal
import lalsimulation as lalsim


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


def get_eos_interpolant(EoS: str, m_min: float = 1.0, N_points: int = 100):
    '''
    This method accepts one of the NS native equations of state
    and uses that to return a list [s, mass, Λ, max_mass] where
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
    if max_mass < m_min: return [np.nan, np.nan, np.nan, np.nan]

    masses = np.linspace(m_min, max_mass, N_points)
    masses = masses[masses <= max_mass]
    
    grav_masses, Lambdas = get_eos_lambdas_from_masses(masses, fam)

    s = scipy.interpolate.interp1d(grav_masses, Lambdas)
    return s, grav_masses, Lambdas, max_mass


def get_eos_interpolant_from_parameters(
        params, 
        parameterization: Literal['spectral', 'polytrope'],
        N_points: int = 100,
        m_min: float = 0.8
    ):
    '''
    This method accepts a four parameter description of the neutron star 
    equation of state, and returns a list [s, masses, max_mass, min_mass] where s is 
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
    
    return s, grav_masses, max_mass, max(m_min, min_mass)


def get_eos_interpolant_from_mass_tidal_file(mass_tidal_file):
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
    returns a list [s, mass, Λ, max_mass] where s is the interpolation
    function for the mass and the tidal deformability.
    '''
    masses, lambdas = np.loadtxt(mass_tidal_file, unpack=True)
    Lambdas = lal.G_SI*lambdas*(1/(lal.MRSUN_SI*masses)**5)
    s = scipy.interpolate.interp1d(masses, lambdas)
    max_mass = np.max(masses)
    return [s, masses, Lambdas, max_mass]


def get_eos_interpolant_from_mass_radius_file(MRFile):
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
    returns a list [s, mass, Λ, max_mass] where s is the interpolation
    function for the mass and the tidal deformability.
    '''
    masses, radius, kappa = np.loadtxt(MRFile, unpack=True)
    compactness = masses*lal.MRSUN_SI/radius
    Lambdas = (2/3)*kappa / (compactness**5)
    s = scipy.interpolate.interp1d(masses, Lambdas)
    max_mass = np.max(masses)
    return [s, masses, Lambdas, max_mass]


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