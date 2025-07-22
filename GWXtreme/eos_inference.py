# Copyright (C) 2022 Shaon Ghosh, Michael Camilo, Xiaoshu Liu
# Copyright (C) 2021 Anarya Ray
#
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

import os
import json
import multiprocessing
from typing import Literal
import pathlib

import numpy as np
import torch
import h5py
import emcee

from .eos_prior import is_valid_eos, create_spectral_eos, create_polytrope_eos
from .utils import (
    get_eos_interpolant_from_mass_radius_file,
    get_eos_interpolant,
    get_eos_interpolant_from_mass_tidal_file,
    get_eos_interpolant_from_parameters,
    get_masses,
    get_lambdat,
    get_lambda_for_eos,
    get_lambdat_for_eos,
    apply_mass_constraint,
    _read_posterior_file
)
from .density_estimation import NormalizingFlow, TransformKDE, ReflectKDE
from .config import SUPPORTED_EVENTS, GW_PE_POSTERIOR_FILES


class ModelSelector:
    def __init__(
            self,
            event: str,
            prior_file: str | None = None,
            method: Literal['2D', '3D'] = '2D',
            density_est_method: Literal['kde', 'flow', 'reflectkde'] = 'flow',
            parameterization: Literal['spectral', 'polytrope'] = 'spectral'
        ):
        '''
        Initiates the Bayes factor calculator with the posterior
        samples from the uniform lambdat, dlambdat parameter
        estimation runs.

        prior_file     :: The full path to the priors file (optional).
                         If the prior file is supplied, the mass
                         boundaries for the KDE computation is
                         obtained from the prior file. If this is not
                         supplied, the posterior samples will be used
                         to determine the bounds.
        '''
        assert method in ['2D', '3D']
        assert density_est_method in ['kde', 'flow', 'reflectkde']
        assert parameterization in ['spectral', 'polytrope']
        assert event in SUPPORTED_EVENTS, f'event must be one of {SUPPORTED_EVENTS}'
        
        self.method = method
        self.density_est_method = density_est_method
        self.parameterization = parameterization
        self.event = event
        
        posterior_file = GW_PE_POSTERIOR_FILES[event][method]
        m1, m2, q, mc, lambda1, lambda2, lambdat = _read_posterior_file(posterior_file, method)
        data = {
            'm1_source': m1,
            'm2_source': m2,
            'q': q,
            'mc_source': mc,
            'lambdat': lambdat,
            'lambda1': lambda1,
            'lambda2': lambda2
        }
        self.data = {k:v for k, v in data.items() if v is not None}
        
        if prior_file is not None:
            self.prior = np.genfromtxt(prior_file, names=True)
            self.min_mass = np.min(self.prior['m2_source'])
            self.max_mass = np.max(self.prior['m1_source'])
            self.q_max = np.max(self.prior['q'])
            self.q_min = np.min(self.prior['q'])
        else:
            self.prior = None
            self.min_mass = np.min(self.data['m2_source'])  # min posterior mass
            self.max_mass = np.max(self.data['m1_source'])  # max posterior mass
            self.q_max = np.max(self.data['q'])
            self.q_min = np.min(self.data['q'])
        
        self.m_min = 0.8
        
        if self.method == '2D':
            self.marginal_posterior = np.stack(
                (
                    self.data['lambdat'],
                    self.data['q']
                ),
                axis=-1
            )
                
        elif self.method == '3D':
            self.marginal_posterior = np.stack(
                (
                    self.data['lambda1'],
                    self.data['q'],
                    self.data['lambda2']
                ),
                axis=-1
            )
        
        if density_est_method == "flow":
            self.density_estimator = NormalizingFlow(event=event, method=method)
        elif density_est_method == "kde":
            self.density_estimator = TransformKDE(event, method)
        elif density_est_method == "reflectkde":
            self.density_estimator = ReflectKDE(event, method)

    def compute_eos_evidence_ratio(
            self,
            EoS1: str,
            EoS2: str,
            N_grid: int = 1000,
            save_file: str | None = None, 
            N_trials: int = 0,
            verbose: bool = False
        ):
        '''
        This method computes the ratio of evidences for two
        tabulated EoS. It first checks if a file exists with
        the name associated with the strings EoS1 and EoS2.
        If it does, then use method getEoSInterpFromFile,
        else use method getEoSInterp. This computation is
        conducted for multiple trials to get an estimation of
        the uncertainty.

        EoS1    :: The name of the first tabulated equation of
                   state or the name of the file from which the
                   EoS data is to be read.
        EoS2    :: The name of the second tabulated equation of
                   state or the name of the file from which the
                   EoS data is to be read.
        N_grid   :: Number of grid points over which the
                   line-integral is computed. (Default = 1000)
        N_trials  :: Number of trials for estimating the
                   uncertainty in the Bayes-factor.
        '''

        # generate interpolators for both EOS
        min_mass1, min_mass2 = 0., 0.

        if type(EoS2) == list:
            [s2, _, max_mass_eos2, min_mass2] = get_eos_interpolant_from_parameters(EoS2, N=1000)

        elif os.path.exists(EoS2):
            if verbose: print('Trying m-R-k file to compute EoS interpolant')
            try:
                [s2, _, _, max_mass_eos2] = get_eos_interpolant_from_mass_radius_file(EoS2)
            except ValueError:
                if verbose: print('Trying m-λ file to compute EoS interpolant')
                [s2, _, _, max_mass_eos2] = get_eos_interpolant_from_mass_tidal_file(EoS2)
        else:
            [s2, _, _, max_mass_eos2] = get_eos_interpolant(EoS=EoS2, m_min=self.min_mass, N_points=100)

        # Check the return values from get_eos_interpolant(), which may return [nan, nan, nan, nan] if the
        # system is a BBH.
        assert [s2, max_mass_eos2] != [np.nan, np.nan], "The system being studied is a binary black hole system."

        if type(EoS1) is list:
            [s1, _, max_mass_eos1,min_mass1] = get_eos_interpolant_from_parameters(EoS1, N=1000)

        elif os.path.exists(EoS1):
            if verbose: print('Trying m-R-k file to compute EoS interpolant')
            try:
                [s1, _, _, max_mass_eos1] = get_eos_interpolant_from_mass_radius_file(EoS1)
            except ValueError:
                if verbose: print('Trying m-λ file to compute EoS interpolant')
                [s1, _, _, max_mass_eos1] = get_eos_interpolant_from_mass_tidal_file(EoS1)
        else:
            [s1, _, _, max_mass_eos1] = get_eos_interpolant(EoS1, m_min=self.min_mass)

        # Check the return values from get_eos_interpolant(), which may return [nan, nan, nan, nan] if the
        # system is a BBH.
        assert [s1, max_mass_eos1] != [np.nan, np.nan], "The system being studied is a binary black hole system."

        # compute support
        evidence_1, evidences_1 = self._integrate_posterior_for_eos_support(
            s1, 
            max_mass_eos1,
            N_grid=N_grid,
            do_ensemble=(N_trials > 0),
            min_mass=max(self.min_mass, min_mass1),
            N_kde_trials=N_trials,
        )

        evidence_2, evidences_2 = self._integrate_posterior_for_eos_support(
            s2, 
            max_mass_eos2,
            N_grid=N_grid,
            do_ensemble=(N_trials > 0),
            min_mass=max(self.min_mass, min_mass2),
            N_kde_trials=N_trials,
        )

        bf = evidence_1 / evidence_2
        
        if N_trials == 0:
            return bf
        
        bf_array = evidences_1 / evidences_2
    
        if save_file is not None:
            results = {
                'ref_eos': EoS1,
                'target_eos': EoS1,
                'bf': bf,
                'bf_array': bf_array.tolist()
            }
            
            with open(save_file, 'x') as f:
                json.dump(results, f, indent=4)
        
        return bf, bf_array

    def compute_parameterized_eos_evidence(self, params, N_grid=1000):
        '''
        This method computes the evidence for a parametrized EoS.

        params      :: Four parameter list.
        N_grid       :: Number of grid points over which the
                       line-integral is computed. (Default = 100)
        '''

        # generate interpolator for eos
        s, _, max_mass_eos, min_mass = get_eos_interpolant_from_parameters(
            params,
            parameterization=self.parameterization, #type: ignore
            N_points=100,
            m_min=self.m_min
        )

        # compute support
        evidence, evidences = self._integrate_posterior_for_eos_support(
            s, 
            max_mass_eos,
            N_grid=N_grid,
            min_mass=min_mass
        )

        return evidence

    def _integrate_posterior_for_eos_support(
            self,
            eos_interpolant,
            max_mass_eos: float,
            N_grid: int = 1000,
            min_mass: float = 0.1,
            do_ensemble: bool = False,
            N_kde_trials: int = 0,
        ):
        '''
        This function numerically integrates the KDE along the
        EoS curve.

        eos_interpolant	 :: interpolation function of Λ = eos_interpolant(m)

        max_mass_eos :: Maximum mass allowed by the EoS.

        N_grid  :: Number of steps for the integration (default=1K)

        min_mass :: The value of the minimum mass for the lines integration

        If for the choice of mass-ratio and mc, the masses of one
        or both the object goes above the maximum mass of NS
        allowed by the EoS, then the object(s) is(are) treated as
        BH (Λ=0). If the masses are below the minimum mass, the
        points are excludeds from the integral.

        '''
        # get values for line integral
        q = np.linspace(self.q_min, self.q_max, N_grid)
        m1, m2 = get_masses(q, np.mean(self.data['mc_source']))

        m1, m2, q = apply_mass_constraint(m1, m2, q, min_mass)
        
        if self.method == '2D':
            lambdat = get_lambdat_for_eos(m1, m2, max_mass_eos, eos_interpolant)
            points = np.stack((lambdat, q), axis=-1)
        else:
            lambda1, lambda2 = get_lambda_for_eos(m1, max_mass_eos, eos_interpolant), get_lambda_for_eos(m2, max_mass_eos, eos_interpolant)
            lambdat = get_lambdat(m1, m2, lambda1, lambda2)
            points = np.stack((lambda1, q, lambda2), axis=-1)

        points = torch.tensor(points, dtype=torch.float32)

        evidences = np.array([])

        # perform integration via trapezoidal approximation
        # use normalizing flow(s)
        if isinstance(self.density_estimator, NormalizingFlow):
            prob_density = self.density_estimator.pdf(points).numpy()
            evidence = np.trapezoid(prob_density, q)
            
            if do_ensemble:
                ensemble_prob_density = self.density_estimator.ensemble_pdf(points).numpy()
                evidences = np.trapezoid(ensemble_prob_density, q, axis=1)
                
        # use KDE (bounded using transformation or reflection)
        else:
            prob_density = self.density_estimator.pdf(points).numpy()
            evidence = np.trapezoid(prob_density, q)

            if N_kde_trials > 0:
                evidences = np.empty(N_kde_trials)
                for i in range(N_kde_trials):
                    resampled_prob_density = self.density_estimator.pdf(points, resample=True).numpy()
                    evidences[i] = np.trapezoid(resampled_prob_density, q)

        return evidence, evidences


class JointModelSelector:
    def __init__(
            self,
            events: list[str],
            method: Literal['2D', '3D'] = '2D',
            prior_files: list[str] | None = None,
            density_est_method: Literal['kde', 'flow', 'reflectkde'] = 'flow',
            parameterization: Literal['spectral', 'polytrope'] = 'spectral'
        ):
        '''
        This class takes as input a list of posterior-samples files for
        various events. Optionally, prior samples files can also be
        supplied and allows us to compute the various quantities related
        to each of the posterior samples. N_samples is the Number of samples to which the single event q and 
        lambda_tilde posteriors are downsampled and is only required for speeding up the parametric eos
        analysis
        '''

        if prior_files is not None:
            for file in prior_files:
                if not os.path.exists(file):
                    raise FileNotFoundError(f"prior file {file} does not exist")
            self.event_priors = prior_files
        else:
            self.event_priors = [None] * len(events)

        # Right now the method demands a unique prior file for each event
        # This may be changed later.
        assert len(self.event_priors) == len(events), 'Number of prior and posterior files should be same'
        
        self.model_selectors = []
        for event, prior_file in zip(events, self.event_priors):
            self.model_selectors.append(
                ModelSelector(
                    event=event,
                    prior_file=prior_file,
                    method=method,
                    density_est_method=density_est_method,
                    parameterization=parameterization,
                )
            )
        
        self.method = method
        self.parameterization = parameterization
        self.N_events = len(self.model_selectors)
    
    def compute_joint_eos_evidence_ratio(
            self,
            EoS1,
            EoS2,
            N_trials: int = 0,
            N_grid: int = 1000,
            verbose: bool = False,
            save_file=None
        ):
        '''
        Loop through each event and compute the joint Bayes-factor.
        Each individual event's Bayes-factor can be accessed from the JointModelSelector
        object, using JointModelSelector.all_bayes_factors. Uncertainty for each case can
        be accessed by using JointModelSelector.all_bayes_factors_errors.

        EoS1 :: The name of the first equation of state model. This can be
                either one of the named equation of state models from LALSuite,
                or a file containing the information of the equation of state,
                (m, λ) or (m, r, κ).

        EoS2 :: The name of the second equation of state model. This can be
                either one of the named equation of state models from LALSuite,
                or a file containing the information of the equation of state,
                (m, λ) or (m, r, κ).

        N_trials :: Number of trials to be used to computed the uncertainty in
                  the Bayes-factor.

        N_grid :: Number of grid points over which the line-integral is
                 computed (Default = 1000).

        save_file :: Use this option to save the results into json files. If nothing
                is provided, then output will not be saved. If a name is
                provided then the output will be saved to a file with that
                name.
        '''
        joint_bf = 1.0
        self.all_bayes_factors = []  # To be populated by B.Fs from all events
        
        joint_bf_array = np.ones(N_trials)
        self.all_bayes_factors_errors = []
        
        for model_selector in self.model_selectors:
            '''NOTE:
            It seems to be the logical thing to parallelize the run of the
            individual events on different CPUs using ray. However, it does not
            seem to be the right thing to do if we want to preserve the
            scalability of the infrastructure. If the user wants to run this
            on HTCondor, this is the sequence of events that will follow:
            1. A condor DAG will be generated to submit accross multiple nodes
               the multiple jobs such that the number of trials will be
               distributed accross them.
            2. In each node then ray will launch parallel processes across
               various available CPU for the different events.
            3. Each of these processes will now launch multiple ray processes
               within the available CPUs in the same node to run trials that are
               scheduled for this jobs on this Node.

            This will not scale with large number of trials and events. The
            ideal situation would be to first distribute the individual events
            across different Nodes, and then from each node multiple jobs will
            be spawned to multiple nodes that will distribute the trials
            internally using ray. But it is not obvious to me how this can be
            done in Condor. Also, running each event on a unique node will
            require that Condor distributes each event. That will mean that
            this code should have no way of computing the joint-Bayes-factor.
            Which would mean that the joint-Bayes-factor computation will only
            be possible on Condor. Thus, we have decided to keep this part of
            the computation serial. We will be processing each event
            sequentially. Thus, upon running the code, for each event ray will
            spawn multiple processes across available cores and then upon
            completion will move on to the next event. 
            '''
            
            result = model_selector.compute_eos_evidence_ratio(
                EoS1,
                EoS2,
                N_grid=N_grid,
                N_trials=N_trials,
                verbose=verbose
            )

            if N_trials > 0:
                bf, bf_trials = result
                error = 2*np.std(bf_trials)
                self.all_bayes_factors_errors.append(error)
                joint_bf_array *= bf_trials
            else:
                bf = result
            
            joint_bf *= bf
            self.all_bayes_factors.append(bf)                    

        if save_file is not None:
            results = {
                'ref_eos': EoS1,
                'target_eos': EoS1,
                'joint_bf': joint_bf,
                'joint_bf_array': joint_bf_array.tolist() if N_trials > 0 else [],
                'all_bf': self.all_bayes_factors,
                'all_bf_err': self.all_bayes_factors_errors if N_trials > 0 else []
            }

            with open(save_file, 'w+') as f:
                json.dump(results, f, indent=4, sort_keys=True)

        return [joint_bf, joint_bf_array] if N_trials > 0 else joint_bf

    def compute_parameterized_eos_joint_evidence(self, EoS, N_grid: int = 1000):
        '''
        Loop through each event and compute the joint evidence.

        EoS :: The list of parameters that characterise the equation of state. 
               This can be in the form of either one of the two supported 
               parametrized equations of state: Spectral Decomposition and 
               Piecewise Polytrope.

        N_grid :: Number of grid points over which the line-integral is
                 computed (Default = 1000).
        '''
        all_evidences = []  # To be populated by B.Fs from all events

        for model_selector in self.model_selectors:
            all_evidences.append(model_selector.compute_parameterized_eos_evidence(EoS, N_grid=N_grid))

        joint_evidence = np.prod(all_evidences)
        return joint_evidence, all_evidences


class ParameterizedEoSSampler:
    def __init__(
            self, 
            events: list[str], 
            method: Literal['2D', '3D'],
            prior_bounds: dict[str, dict[str, dict]],
            density_est_method: Literal['kde', 'flow', 'reflectkde'] = 'flow',
            parameterization: Literal['spectral', 'polytrope'] = 'spectral',
        ):
        '''
        Parametric EoS MCMC sampler class that stacks multiple events from 
        uniform in (LambdaT, dLambdaT) parameter estimation runs. 
        Parametrization chosen: 4 parameter spectal decomposition of adiabatic index in terms of
        pressure or 4 parameter piecewise polytrope.
            
        prior_bounds :: dictionary containining prior bounds
                        of parameters. example for spectral:
                        {
                            'gamma1': {'params': {"min":  0.2, "max":  2.00}},
                            'gamma2': {'params': {"min": -1.6, "max":  1.70}},
                            'gamma3': {'params': {"min": -0.6, "max":  0.60}},
                            'gamma4': {'params': {"min": -0.02, "max": 0.02}}
                        }
                                          
        '''
        
        self.prior_bounds = prior_bounds
        self.parameterization = parameterization
        
        self.joint_selector = JointModelSelector(
            events=events,
            method=method,
            density_est_method=density_est_method,
            parameterization=parameterization,
        )
        
        if parameterization == 'spectral':
            self.keys = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
            self.eos = create_spectral_eos
        elif parameterization == 'polytrope':
            self.keys = ['logP', 'gamma1', 'gamma2', 'gamma3']
            self.eos = create_polytrope_eos
        
    def log_post(self, parameters, N_grid: int):
        '''
        This method accepts an array of spectral parameters
        and returns their log posterior given gw data from all
        the events provided while initializing the class.
        
        p  :: array of spectral parameters.
        
        N_grid :: Number of grid points to use to perform integral
                over q
        '''
        
        params = {k: np.array([par]) for k, par in zip(self.keys, parameters)}
        
        if not is_valid_eos(params, self.prior_bounds, spectral=self.parameterization == 'spectral'):
            return -np.inf
        
        joint_evidence, _ = self.joint_selector.compute_parameterized_eos_joint_evidence(parameters, N_grid=N_grid)
        log_evidence = np.log(joint_evidence)
        return log_evidence
    
    def initialize_walkers(self, N_walkers: int):
        '''
        This method initializes the walkers for mcmc 
        (to be run on the spectral parameters posterior
        given GW data from all the events)inside the prior 
        region.
        '''
        n_valid_walkers = 0
        self.p0 = []
        while n_valid_walkers < N_walkers:
            gammas = np.array(
                [
                    np.random.uniform(
                        self.prior_bounds[k]["params"]["min"], 
                        self.prior_bounds[k]["params"]["max"]
                    ) for k in self.keys
                ]
            )
            params = {k:np.array([gammas[i]]) for i, k in enumerate(self.keys)}
    
            if is_valid_eos(params, self.prior_bounds, spectral=self.parameterization == 'spectral'):
                if self.log_post(gammas, N_grid=10) == -np.inf:
                    continue

                self.p0.append(gammas)
                n_valid_walkers += 1
                
    def run_sampler(self, N_samples: int, N_pool: int, N_grid: int, save_file: str):
        '''
        runs mcmc sampler to draw samples of 
        the spectral parameters from their
        posterior given GW data from all the 
        events.
        '''
        if N_pool > 1:
            # MP code isn't really working yet
            ctx = multiprocessing.get_context('fork')
            with ctx.Pool(min(multiprocessing.cpu_count(), N_pool)) as pool:
                sampler = emcee.EnsembleSampler(
                    nwalkers=len(self.p0), 
                    ndim=4, 
                    log_prob_fn=self.log_post,
                    args=[N_grid],
                    pool=pool
                )
                sampler.run_mcmc(self.p0, N_samples, progress=True)
        
                self.samples = sampler.get_chain(flat=True)
                self.logp = sampler.get_log_prob(flat=True)
        else:
              sampler = emcee.EnsembleSampler(
                    nwalkers=len(self.p0), 
                    ndim=4, 
                    log_prob_fn=self.log_post,
                    args=[N_grid]
                )
              sampler.run_mcmc(self.p0, N_samples, progress=True)
        
              self.samples = sampler.get_chain(flat=True)
              self.logp = sampler.get_log_prob(flat=True)
        
        with h5py.File(save_file, 'w') as f:
            f.create_dataset('samples', data=np.array(self.samples))
            f.create_dataset('logp', data=np.array(self.logp))        
        
    def parse_samples(self, burn_in_frac: float = 0.5, thinning: int | None = None):
        '''
        This methods parses the MCMC samples of
        EoS hyper-parameters.
        see https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
        for some documentation on choosing thinning and burn-in
        
        burn_in_frac  :: fraction of samples to discard from each chain
                         for MCMC burn in. Default corresponds to discarding 
                         half the samples in each chain.
        
        thinning      :: Number of samples to skip in each chain post
                         burn in. "None" implements the default value 
                         which is either (length of chain)/50 or half of
                         the maximum integrated autocorrelation time. The 
                         former is used in case autocorrelation analysis 
                         throws non-convergence error. For no thinning 
                         set thinning=1
        
        '''
        assert self.samples is not None, 'No samples attributed to this ParameterizedEoSSampler object to parse.'
        
        burn_in = int(self.samples.shape[0] * burn_in_frac)
        
        if thinning is None:
            thinning = int(self.samples.shape[0] / 50)
            try: 
                thinning = int(max(np.array(emcee.autocorr.integrated_time(self.samples))) / 2.)
            except emcee.autocorr.AutocorrError as e:
                print(e)
        
        thinning = max(thinning, 1)
        return self.samples[burn_in::thinning]
    
    def load_samples(self, samples_file):
        '''
        If the plotting function is called in post-
        processing i.e. as part of a different script 
        than the one that ran the sampling, then this 
        function needs be called to load the EoS hyper-parameter
        samples. In addition, if one wishes to make their own
        plots using the parsed samples, they can do so by first
        calling this method and then extracting the parsed samples
        using parse_samples() method of this class.
        
        samples_file :: h5py file containing MCMC samples of 
                    EoS hyper-parameters
                    
        Example:
        
        >>> sampler_spectral=mcmc_sampler([],  
        {'gamma1':{'params':{"min":0.2,"max":2.00}},
        'gamma2':{'params':{"min":-1.6,"max":1.7}},
        'gamma3':{'params':{"min":-0.6,"max":0.6}},
        'gamma4':{'params':{"min":-0.02,"max":0.02}}},
        out, N_walkers=100, N_parameter_samples=10000, N_dim=4,
        spectral=True,N_pool=16)
        >>> sampler_spectral.load_samples('file/containing/EoS/hyperparameter/samples')
        >>> figures = sampler_spectral.plot(p_vs_rho={'plot':True,'true_eos': None}) #for plotting using this classes plot() function. This step can be skipped if one wishes to manually the extracted samples.
        >>> samples = sampler_spectral.parse_samples() # to extract parsed samples for manual plotting if desired
        
        '''
        
        with h5py.File(samples_file) as f:
            self.samples = np.array(f['samples'])
            self.logp = np.array(f['logp'])


if __name__ == "__main__":
    ems = ModelSelector(
        event='GW170817'
    )

    bf = ems.compute_eos_evidence_ratio('APR4_EPP', 'SLY')
    print(bf)

    sp = (6.768730840689067829e-01, 1.849793950121006447e-01, -1.545969552248221621e-02, -9.786142132722361537e-05)
    evi = ems.compute_parameterized_eos_evidence(sp)
    print(evi)

    jms = JointModelSelector(
        events=["GW170817", "GW190425"]
    )

    joint_bf = jms.compute_joint_eos_evidence_ratio('APR4_EPP', 'SLY', verbose=True)
    print(joint_bf)