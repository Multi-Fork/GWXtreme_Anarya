# Copyright (C) 2017 Daniel Wysocki
# Assimilated from Wysocki D., O'Shaughnessy R., Bayesian Parametric Population Models, 
# 2017–, bayesian-parametric-population-models.readthedocs.io
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


import typing
import numpy as np

import lalsimulation
import lal


LARGEST_NS_MASS = 1.97
_X_GRID = np.linspace(0., 12.3081, 500)


def compute_log_pressure_from_eos(rho, eos):
    '''
    Calculates log10 pressure as a function of density rho
    for any eos.
    
    rho     ::     An array of densities in g cm^-3.
    
    eos     ::     A lalsimulation.SimNeurtronStarEOS*
                   object.            
    '''

    # FAM = lalsimulation.CreateSimNeutronStarFamily(eos)
    p_max_i = min(6e35, lalsimulation.SimNeutronStarEOSMaxPressure(eos))
    log10_p_grid = np.linspace(*np.log10([5e31, p_max_i]), 128)
    p_grid = np.power(10.0, log10_p_grid)
    rho_grid = np.empty_like(p_grid)
    
    for j, p in np.ndenumerate(p_grid):
        h = lalsimulation.SimNeutronStarEOSPseudoEnthalpyOfPressure(p, eos)
        rho_grid[j] = (lalsimulation.SimNeutronStarEOSRestMassDensityOfPseudoEnthalpy(h, eos,))

    log10_rho_grid = np.log10(rho_grid)
    
    log10_p_out = np.interp(
        np.log10(rho), 
        log10_rho_grid, 
        log10_p_grid, 
        left=-np.inf, 
        right=-np.inf,
    )
    return log10_p_out


def create_spectral_eos(eos_parameters: tuple) -> typing.Any:
    '''
    Creates a lalsimulation.SimNeutronStarEOS* object
    for the 4 parameter spectral decomposition (spectral eos)
    , given a particular value of spectral parameters.
    
    eos_parameters  ::  (gamma_1,gamma_2,gamma3,gamma4),
                        a tuple of the four spectral parameters
                        describing the eos.
    
    '''
    
    gamma1, gamma2, gamma3, gamma4 = eos_parameters
    eos = lalsimulation.SimNeutronStarEOS4ParameterSpectralDecomposition(
        gamma1, gamma2, gamma3, gamma4,
    )
    return eos


def create_polytrope_eos(eos_parameters: tuple) -> typing.Any:
    '''
    Creates a lalsimulation.SimNeutronStarEOS* object
    for the piece-wise polytropic EoS,  given
    a particular value of the central pressure and 
    the adiabatic indices.
    
    eos_parameters  ::  (\log p_0,\Gamma1,\Gamma2,\Gamma3),
                        a tuple of the four parameters describing
                        the eos.
    
    '''

    logP1, gamma1, gamma2, gamma3 = eos_parameters

    return lalsimulation.SimNeutronStarEOS4ParameterPiecewisePolytrope(
        logP1, gamma1, gamma2, gamma3,
    )


def spectral_eos_adiabatic_index(x, spectral_parameters):
    '''
    returns the adiabatic index at a particular value of log pressure,
    using the spectral eos, given a particular choice of spectral
    parameters.
    
    x                  ::    An array of log10 pressure at which to 
                             evaluate the adiabatic index
                   
    spectral_parameters :: (gamma1,gamma2,gamma3,gamma4), spectral 
                            parameters describing the eos
    
    '''

    x_sq = x * x
    x_cu = x_sq * x

    gamma1, gamma2, gamma3, gamma4 = spectral_parameters

    log_gamma = (
        gamma1 +
        gamma2 * x +
        gamma3 * x_sq +
        gamma4 * x_cu
    )

    return np.exp(log_gamma)


def is_valid_adiabatic_index(spectral_parameters: tuple):
    '''
    Confirm that the adiabatic index is within a tolerable range
    for the spectral eos.
    
    spectral_parameters :: (gamma1,gamma2,gamma3,gamma4), spectral 
                            parameters describing the eos.
    '''
    
    adiabatic_index = spectral_eos_adiabatic_index(_X_GRID, spectral_parameters)
    return (adiabatic_index > 0.6).all() and (adiabatic_index < 4.5).all()


def has_enough_points(eos: typing.Any) -> bool:
    '''
    Confirm that the TOV solver returns increasing 
    masses for increasing pressure for at least 8 points
    in the pressure grid. If this is not true, lalsimulation
    might return interpolation errors.
    
    eos  ::   A lalsimulation.SimNeutronStarEOS* object.
    
    '''

    min_points = 8

    logpmin = 75.5
    logpmax = np.log(lalsimulation.SimNeutronStarEOSMaxPressure(eos))

    dlogp = (logpmax - logpmin) / 100

    m_prev = 0.0

    for i in range(min_points):
        p = np.exp(logpmin + i*dlogp)

        r, m, k = lalsimulation.SimNeutronStarTOVODEIntegrate(p, eos)

        if m <= m_prev:
            return False

        m_prev = m

    return True

def eos_max_sound_speed(eos: typing.Any, eos_fam: typing.Any) -> float:
    '''
    Compute the maximum sound speed for a given eos and eos family.
    
    eos          ::     A lalsimulation.SimNeutronStarEOS* object.
    
    eos_fam      ::     A lalsimulation.CreateSimNeutronStarFamily object.
    
    '''

    # Maximum allowed mass
    m_max_kg = lalsimulation.SimNeutronStarMaximumMass(eos_fam)
    # Central pressure
    p_max = lalsimulation.SimNeutronStarCentralPressure(m_max_kg, eos_fam)
    # Pseudo-enthalpy at the core.
    h_max = lalsimulation.SimNeutronStarEOSPseudoEnthalpyOfPressure(p_max, eos)
    # Maximum sound speed should occur at the max pseudo-enthalpy for a typical
    # TOV neutron star.
    return lalsimulation.SimNeutronStarEOSSpeedOfSoundGeometerized(h_max, eos)


def is_causal_eos(eos: typing.Any, eos_fam: typing.Any) -> bool:
    ''' 
    Check if the maximum sound speed of an eos is less 
    than the speed of light upto a buffer of 10%.
    
    eos          ::     A lalsimulation.SimNeutronStarEOS* object.
    
    eos_fam      ::     A lalsimulation.CreateSimNeutronStarFamily object.
    
    '''
    
    c_max = eos_max_sound_speed(eos, eos_fam)

    # Confirm that the sound speed is less than speed of light, with an added
    # buffer (10%) to account for imperfect modeling.
    return c_max < 1.10


def eos_mass_range(eos_fam: typing.Any) -> typing.Tuple[float,float]:
    '''
    Find the maximum and minimun mass of an eos family.
    
    eos_fam      ::     A lalsimulation.CreateSimNeutronStarFamily 
                        object.
    '''
    
    m_min = lalsimulation.SimNeutronStarFamMinimumMass(eos_fam)
    m_max = lalsimulation.SimNeutronStarMaximumMass(eos_fam)

    return m_min / lal.MSUN_SI, m_max / lal.MSUN_SI


#########################
#  Combined EoS Priors  #
#########################
def is_valid_eos(
        parameters, 
        prior_settings,
        spectral=True, 
        largest_ns_mass=LARGEST_NS_MASS,
        require_mass_ranges=None,
    ):
    
    '''
    Main eos prior. Checks if for a given value of eos parameters, a 
    range of allowed parameters and eos coordinates, all the priors of
    causality, observational consistency of maximum mass, thermal stability
    and enough points for interpolation are all satisfied simultaneously.
    
    parameters          ::   A choice of the four parameters,  (gamma1,gamma2,
                             gamma3,gamma4) if spectral=True , or
                             (\log p_0,\Gamma1,\Gamma2,\Gamma3) if spectral=
                             'polytropic'.
    prior_settings      ::   A dictionary containing the allowed range of eos parameters
                             . example for spectral=True : {'gamma1':{'params'
                             :{"min":0.2,"max":2.00}},'gamma2':{'params':{"min":-1.6,"max":
                             1.7}},'gamma3':{'params':{"min":-0.6,"max":0.6}},'gamma4':{
                             'params':{"min":-0.02,"max":0.02}}} .
                          
    spectral            ::    Specifies eos parametrization, if True then spectral eos is 
                              considered. if False, polytropic eos is considered.
                        
    
    largest_ns_mass     ::    Largest observed NS mass in solar units. default is 1.97 .
    
    require_mass_ranges ::    A range of masses inside which the maximum and minimum mass
                              of the eos must lie. default is None.
                              
    '''
                          
    if spectral:
        gamma1, gamma2, gamma3, gamma4 = parameters['gamma1'], parameters['gamma2'], parameters['gamma3'], parameters['gamma4']

        params_shape = gamma1.shape

        valid = (
            (gamma1 >= prior_settings["gamma1"]["params"]["min"]) &
            (gamma1 <= prior_settings["gamma1"]["params"]["max"]) &
            (gamma2 >= prior_settings["gamma2"]["params"]["min"]) &
            (gamma2 <= prior_settings["gamma2"]["params"]["max"]) &
            (gamma3 >= prior_settings["gamma3"]["params"]["min"]) &
            (gamma3 <= prior_settings["gamma3"]["params"]["max"]) &
            (gamma4 >= prior_settings["gamma4"]["params"]["min"]) &
            (gamma4 <= prior_settings["gamma4"]["params"]["max"])
        )

        for i in np.ndindex(*params_shape):
            # Skip already invalidated samples
            if not valid[i]:
                continue

            spectral_parameters_at_i = (gamma1[i], gamma2[i], gamma3[i], gamma4[i])

            try:
                if not is_valid_adiabatic_index(spectral_parameters_at_i):
                    valid[i] = False
                    continue

                eos_at_i = create_spectral_eos(spectral_parameters_at_i)

                if not has_enough_points(eos_at_i):
                    valid[i] = False
                    continue

                eos_fam_at_i = lalsimulation.CreateSimNeutronStarFamily(eos_at_i)
                
                if not is_causal_eos(eos_at_i, eos_fam_at_i):
                    valid[i] = False
                    continue

                eos_m_min, eos_m_max = eos_mass_range(eos_fam_at_i)

                if eos_m_max < largest_ns_mass:
                    valid[i] = False
                    continue

                # Check that a series of mass ranges are included within the EOS's
                # allowed mass range.
                if require_mass_ranges is not None:
                    for m_mins, m_maxs in require_mass_ranges:
                        if m_mins[i] >= eos_m_max:
                            valid[i] = False
                            continue
                        
                        if m_maxs[i] <= eos_m_min:    
                            valid[i] = False
                            continue

            except RuntimeError as e:
                if str(e) != "Generic failure":
                    raise
                else:
                    valid[i] = False
    else:
        logP, gamma1, gamma2, gamma3 =  parameters["logP"], parameters['gamma1'], parameters['gamma2'], parameters['gamma3']

        params_shape = logP.shape

        valid = (
            (logP >= prior_settings["logP"]["params"]["min"]) &
            (logP <= prior_settings["logP"]["params"]["max"]) &
            (gamma1 >= prior_settings["gamma1"]["params"]["min"]) &
            (gamma1 <= prior_settings["gamma1"]["params"]["max"]) &
            (gamma2 >= prior_settings["gamma2"]["params"]["min"]) &
            (gamma2 <= prior_settings["gamma2"]["params"]["max"]) &
            (gamma3 >= prior_settings["gamma3"]["params"]["min"]) &
            (gamma3 <= prior_settings["gamma3"]["params"]["max"])
        )

        for i in np.ndindex(*params_shape):
            # Skip already invalidated samples
            if not valid[i]:
                continue

            polytropic_parameters_at_i = (logP[i], gamma1[i], gamma2[i], gamma3[i])

            try:
                eos_at_i = create_polytrope_eos(polytropic_parameters_at_i)

                if not has_enough_points(eos_at_i):
                    valid[i] = False
                    continue

                eos_fam_at_i = lalsimulation.CreateSimNeutronStarFamily(eos_at_i)

                if not is_causal_eos(eos_at_i, eos_fam_at_i):
                    valid[i] = False
                    continue

                eos_m_min, eos_m_max = eos_mass_range(eos_fam_at_i)

                if eos_m_max < largest_ns_mass:
                    valid[i] = False
                    continue

                # Check that a series of mass ranges are included within the EOS's
                # allowed mass range.
                if require_mass_ranges is not None:
                    for m_mins, m_maxs in require_mass_ranges:
                        if m_mins[i] >= eos_m_max:
                            valid[i] = False
                            continue
                        
                        if m_maxs[i] <= eos_m_min:    
                            valid[i] = False
                            continue

            except RuntimeError as e:
                if str(e) != "Generic failure":
                    raise
                else:
                    valid[i] = False
    return valid