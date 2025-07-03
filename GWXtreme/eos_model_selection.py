# Copyright (C) 2022 Shaon Ghosh, Michael Camilo, Xiaoshu Liu
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


from __future__ import division, print_function

import os
import json
import multiprocessing
from typing import Literal, Sequence
import pathlib

import ray
import numpy as np
import h5py

from .density_estimation import BoundedKDE
from .utils import *


class ModelSelector:
    def __init__(
            self, 
            posterior_file: str, 
            prior_file: str | None = None,
            method: Literal['2D', '3D'] = '2D',
            density_est_method: Literal['kde', 'flow'] = 'kde',
            N_samples: int | None = None,
            parameterization: Literal['spectral', 'polytrope'] = 'spectral'
        ):
        '''
        Initiates the Bayes factor calculator with the posterior
        samples from the uniform lambdat, dlambdat parameter
        estimation runs.

        posterior_file :: The full path to the posterior_samples.dat
                         file

        prior_file     :: The full path to the priors file (optional).
                         If the prior file is supplied, the mass
                         boundaries for the KDE computation is
                         obtained from the prior file. If this is not
                         supplied, the posterior samples will be used
                         to determine the bounds.
        '''
        self.method = method
        self.density_est_method = density_est_method
        self.parameterization = parameterization
        self.isBBH = False
        
        m1, m2, q, mc, lambda1, lambda2, lambdat = self._read_posterior_file(posterior_file)
        data = {
            'm1_source': m1,
            'm2_source': m2,
            'q': q,
            'mc_source': mc,
            'lambdat': lambdat,
            'lambda1': lambda1,
            'lambda2': lambda2
        }
        data = {k:v for k, v in data.items() if v is not None}
        
        N_samples_original = len(self.data['q'])
        if N_samples is None or N_samples > N_samples_original:
            N_samples = N_samples_original # By default we use all the posterior samples without thinning
        
        # Thin samples
        self.data = {k:data[k][::int(N_samples_original/N_samples)] for k in list(data.keys())}
        
        if prior_file:
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
        
        self.m_min=0.8
        
        if self.method == '2D':
            self.marginal_posterior = np.vstack(
                (
                    self.data['lambdat'],
                    self.data['q']
                )
            ).T
                
            if self.density_est_method == 'kde':
                self.density_estimator = BoundedKDE(
                    self.marginal_posterior,
                    low=[0.,      self.q_min],
                    high=[np.inf, 1.        ]
                )
            
            elif self.density_est_method == 'flow':
                # TODO
                self.density_estimator = None
                
        elif self.method == '3D':
            self.marginal_posterior = np.vstack(
                (
                    self.data['lambda1'],
                    self.data['q'],
                    self.data['lambda2']
                )
            ).T

            if self.density_est_method == 'kde':
                self.density_estimator = BoundedKDE(
                    self.marginal_posterior,
                    low=[0.,      self.q_min , 0.    ], # q_min or 0?
                    high=[np.inf, 1. ,         np.inf]
                )
            elif self.density_est_method == 'flow':
                # TODO
                self.density_estimator = None

    def integrate_posterior_for_eos_support(
            self,
            eosfunc,
            max_mass_eos,
            N_grid=1000,
            min_mass=0.1,
            density_estimator=None
        ):
        '''
        This function numerically integrates the KDE along the
        EoS curve.

        eosfunc	 :: interpolation function of Λ = eosfunc(m)

        max_mass_eos :: Maximum mass allowed by the EoS.

        N_grid  :: Number of steps for the integration (default=1K)

        var_lambdat :: Standard deviation of the LambdT

        var_q  :: Standard deviation of the mass-ratio

        min_mass :: The value of the minimum mass for the lines integration

        If for the choice of mass-ratio and mc, the masses of one
        or both the object goes above the maximum mass of NS
        allowed by the EoS, then the object(s) is(are) treated as
        BH (Λ=0). If the masses are below the minimum mass, the
        points are excludeds from the integral.

        '''
        if density_estimator is None:
            density_estimator = self.density_estimator

        # get values for line integral
        q = np.linspace(self.q_min, self.q_max, N_grid)
        m1, m2 = get_masses(q, self.data['mc_source'])

        m1, m2, q = apply_mass_constraint(m1, m2, q, min_mass)
        
        if self.method == '2D':
            lambdat = get_lambdat_for_eos(m1, m2, max_mass_eos, eosfunc)

            # perform integration via trapezoidal approximation
            dq = np.diff(q)
            f = self.density_estimator.evaluate(np.vstack((lambdat, q)).T)
            f_centers = 0.5*(f[1:] + f[:-1])
            int_element = f_centers * dq 

            return [lambdat, q, np.sum(int_element)]
        
        elif self.method == '3D':
            lambda1, lambda2 = get_lambda_for_eos(m1, max_mass_eos, eosfunc), get_lambda_for_eos(m2, max_mass_eos, eosfunc)

            # perform integration via trapezoidal approximation
            dq = np.diff(q)
            f = self.density_estimator.evaluate(np.vstack((lambda1, q, lambda2)).T)
            f_centers = 0.5*(f[1:]+f[:-1])
            int_element = f_centers * dq
            lambdat = get_lambdat(m1, m2, lambda1, lambda2)
            
            return [lambdat, q ,np.sum(int_element)]
        
        else:
            raise ValueError(f'Invalid: self.method = {self.method} is not in ["2D", "3D"]')

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
        if [s2, max_mass_eos2] == [np.nan, np.nan]:
            print("The system being studied is a binary black hole system.")
            if N_trials == 0:
                return np.nan
            else:
                return [np.nan, np.repeat(np.nan, N_trials)]

        if type(EoS1) is list:
            [s1, _, max_mass_eos1,min_mass1] = get_eos_interpolant_from_parameters(EoS1, N=1000)

        elif os.path.exists(EoS1):
            if verbose:
                print('Trying m-R-k file to compute EoS interpolant')
            try:
                [s1, _, _, max_mass_eos1] = get_eos_interpolant_from_mass_radius_file(EoS1)
            except ValueError:
                if verbose:
                    print('Trying m-λ file to compute EoS interpolant')
                [s1, _, _, max_mass_eos1] = get_eos_interpolant_from_mass_tidal_file(EoS1)
        else:
            [s1, _, _, max_mass_eos1] = get_eos_interpolant(EoS1, m_min=self.min_mass)

        # Check the return values from get_eos_interpolant(), which may return [nan, nan, nan, nan] if the
        # system is a BBH.
        if [s1, max_mass_eos1] == [np.nan, np.nan]:
            print("The system being studied is a binary black hole system.")
            if N_trials == 0:
                return np.nan
            else:
                return [np.nan, np.repeat(np.nan, N_trials)]

        # compute support
        [_, _, support_1] = self.integrate_posterior_for_eos_support(
            s1, 
            max_mass_eos1,
            N_grid=N_grid,
            min_mass=max(self.min_mass, min_mass1),
        )

        [_, _, support_2] = self.integrate_posterior_for_eos_support(
            s2, 
            max_mass_eos2,
            N_grid=N_grid,
            min_mass=max(self.min_mass, min_mass2),
        )

        # Iterate to determine uncertainty via re-drawing from
        # smoothed distribution:
        # NOTE: This is known to introduce a bias into the mean
        # and variance estimate.
        if N_trials == 0:
            return (support_1 / support_2)

        ray.init(logging_level=1) if verbose else ray.init(logging_level=40)

        N_cores = multiprocessing.cpu_count()
        if verbose:
            print("Total number of cores in this machine: {}".format(N_cores))

        # Splitting (nearly) equally the # of trials over the # of workers
        if N_trials < N_cores:
            workers = N_trials
            trials_per_worker = np.ones(workers, dtype=int)
        else:
            workers = N_cores
            split = np.array_split(np.arange(N_trials), workers)
            trials_per_worker = [len(split[i]) for i in range(N_cores)]

        futures = []
        for i, worker_trials, in enumerate(trials_per_worker):
            future_dict = {
                "marginal_posterior": self.marginal_posterior,
                "s1": s1,
                "s2": s2,
                "max_mass_eos1": max_mass_eos1,
                "max_mass_eos2": max_mass_eos2, 
                "N_grid": N_grid,
                "min_mass": self.min_mass, 
                'N_trials': worker_trials,
            }

            futures.append(self._compute_eos_evidence_ratios_over_trials.remote(self, future_dict))
            if verbose:
                print("Submitted task in core: {}".format(i+1))
        
        ray.get(futures)
        supports = np.array([ray.get(future) for future in futures])
        ray.shutdown()

        if save_file is not None:
            bf_dict = {}
            bf_dict['ref_eos'] = EoS2
            bf_dict['target_eos'] = EoS1
            bf_dict['bf'] = support_1 / support_2
            bf_dict['bf_array'] = supports.tolist()
            
            with open(save_file, 'w') as f:
                json.dump(bf_dict, f, indent=2, sort_keys=True)
            if verbose:
                print(f"Result saved in: {save_file}")
        
        return [support_1 / support_2, supports]

    def compute_parameterized_eos_evidence(self, params, N_grid=1000):
        '''
        This method computes the evidence for a parametrized EoS.

        params      :: Four parameter list.
        N_grid       :: Number of grid points over which the
                       line-integral is computed. (Default = 100)
        '''

        # generate interpolator for eos
        [s, _, max_mass_eos, min_mass] = get_eos_interpolant_from_parameters(
            params,
            parameterization=self.parameterization, #type: ignore
            N_points=100,
            m_min=self.m_min
        )

        # compute support
        [_, _, support] = self.integrate_posterior_for_eos_support(
            s, 
            max_mass_eos,
            N_grid=N_grid,
            min_mass=min_mass
        )

        return support

    @ray.remote
    def _compute_eos_evidence_ratios_over_trials(self, fd):
        supports_1 = []
        supports_2 = []

        for _ in range(fd['trials']):
            # generate new (synthetic) data
            new_marginal_posterior = np.array([])
            counter = 0
            while len(new_marginal_posterior) < len(fd['marginal_posterior']):
                prune_adjust_factor = 1.1 + counter / 10.
                N_resample = int(len(fd['marginal_posterior']) * prune_adjust_factor)
                
                new_marginal_posterior = fd['kde'].resample(size=N_resample).T
                
                unphysical = [new_marginal_posterior[:, 0] < 0.0] + \
                            [new_marginal_posterior[:, 1] > 1.0] + \
                            [new_marginal_posterior[:, 1] < 0.0]
                
                new_marginal_posterior = new_marginal_posterior[~unphysical]
                
                print("Count: {}".format(counter))
                counter += 1
            
            indices = np.arange(len(new_marginal_posterior))
            chosen = np.random.choice(indices, len(fd['marginal_posterior']))
            new_marginal_posterior = new_marginal_posterior[chosen]

            # generate a new kde
            if self.method == '2D':
                new_kde = BoundedKDE(
                    new_marginal_posterior,
                    low= [0.0,  0.0],
                    high=[None, 1.0]
                )
            elif self.method == '3D':
                new_kde = BoundedKDE(
                    new_marginal_posterior,
                    low= [0.0,    0.0, 0.0   ],
                    high=[np.inf, 1.0, np.inf]
                )
            else:
                raise ValueError(f'Invalid: self.method = {self.method} is not in ["2D", "3D"]')

            # integrate to get support
            [_, _, support_1] = self.integrate_posterior_for_eos_support(
                fd['s1'],
                fd['max_mass_eos1'],
                N_grid=fd['N_grid'],                
                min_mass=fd['min_mass'],
                density_estimator=new_kde
            )
            [_, _, support_2] = self.integrate_posterior_for_eos_support(
                fd['s2'],
                fd['max_mass_eos2'],
                N_grid=fd['N_grid'],                
                min_mass=fd['min_mass'],
                density_estimator=new_kde
            )

            # store the result
            supports_1.append(support_1)
            supports_2.append(support_2)

        return np.array(supports_1) / np.array(supports_2)

    def _read_posterior_file(self, posterior_file: str):
        posterior_file_ = pathlib.Path(posterior_file)
        ext = posterior_file_.suffix

        m1, m2, q, mc, lambda1, lambda2, lambdat = None, None, None, None, None, None, None

        if ext == '.h5':
            with h5py.File(posterior_file_) as f:
                data = np.array(f['posterior_samples'])
            
            if self.method == '2D':
                m1 = np.array(data['m1_source'])
                m2 = np.array(data['m2_source'])
                q = np.array(data['q'])
                mc = np.array(data['mc_source'])
                lambdat = np.array(data['lambdat'])
            
            elif self.method == '3D':
                m1 = np.array(data['m1_source'])
                m2 = np.array(data['m2_source'])
                q = np.array(data['q'])
                mc = np.array(data['mc_source'])
                lambda1 = np.array(data['lambda_1'])
                lambda2 = np.array(data['lambda_2'])

        elif ext == '.txt':
            data = np.loadtxt(posterior_file)
            if self.method == '2D':
                m1 = np.array(data[0])
                m2 = np.array(data[1])
                q = np.array(data[2])
                mc = np.array(data[3])
                lambdat = np.array(data[4])

            elif self.method == '3D':
                m1 = np.array(data[0])
                m2 = np.array(data[1])
                q = np.array(data[2])
                mc = np.array(data[3])
                lambda1 = np.array(data[4])
                lambda2 = np.array(data[5])

        elif ext == '.json':
            with open(posterior_file) as f:
                data = json.load(f)['posterior']['content']
            
            if self.method == '2D':
                m1 = np.array(data['m1_source'])
                m2 = np.array(data['m2_source'])
                q = np.array(data['q'])
                mc = np.array(data['mc_source'])
                lambdat = np.array(data['lambdat'])
            
            elif self.method == '3D':
                m1 = np.array(data['m1_source'])
                m2 = np.array(data['m2_source'])
                q = np.array(data['q'])
                mc = np.array(data['mc_source'])
                lambda1 = np.array(data['lambda_1'])
                lambda2 = np.array(data['lambda_2'])

        else:
            data = np.genfromtxt(posterior_file, names=True)
            
            if self.method == '2D':
                m1 = np.array(data['m1_source'])
                m2 = np.array(data['m2_source'])
                q = np.array(data['q'])
                mc = np.array(data['mc_source'])
                lambdat = np.array(data['lambdat'])
            
            elif self.method == '3D':
                m1 = np.array(data['m1_source'])
                m2 = np.array(data['m2_source'])
                q = np.array(data['q'])
                mc = np.array(data['mc_source'])
                lambda1 = np.array(data['lambda_1'])
                lambda2 = np.array(data['lambda_2'])
        
        return m1, m2, q, mc, lambda1, lambda2, lambdat


class JointModelSelector():
    def __init__(
            self,
            posterior_files: Sequence[str],
            methods: Sequence[Literal['2D', '3D']],
            prior_files: Sequence[str] | None = None,
            event_labels: Sequence[str | None] | None = None,
            density_est_method: Literal['kde', 'flow'] = 'kde',
            N_samples: int | None = None,
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
        if event_labels is None:
            event_labels = [None] * len(posterior_files)

        # Loop over the list and make sure all the paths exists.
        # Keep only those events whose file exits.

        sanitized_event_list = []
        self.event_labels = []
        for event, label in zip(posterior_files, event_labels):
            if os.path.exists(event):
                sanitized_event_list.append(event)
                self.event_labels.append(label)
            else:
                print('Could not file {}. Skipping event'.format(event))

        self.event_list = sanitized_event_list

        if prior_files:
            sanitized_event_priors = []
            for event_prior in prior_files:
                if os.path.exists(event_prior):
                    sanitized_event_priors.append(event_prior)
                else:
                    print('Could not file {}. Skipping'.format(event_prior))

            self.event_priors = sanitized_event_priors

        else:
            self.event_priors = [None] * len(self.event_list)

        # Right now the method demands a unique prior file for each event
        # This may be changed later.
        assert len(self.event_priors) == len(self.event_list), 'Number of prior and posterior files should be same'
        
        self.methods = methods
        self.parameterization = parameterization
        
        self.model_selectors = []
        for prior_file, event_file, method in zip(self.event_priors, self.event_list, self.methods):
            self.model_selectors.append(
                ModelSelector(
                    posterior_file=event_file,
                    prior_file=prior_file,
                    method=method,
                    density_est_method=density_est_method,
                    parameterization=self.parameterization,
                    N_samples=N_samples,
                )
            )
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
        
        if N_trials > 0:
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
            
            bayes_factor = model_selector.compute_eos_evidence_ratio(
                EoS1,
                EoS2,
                N_grid=N_grid,
                N_trials=N_trials,
                verbose=verbose
            )

            if N_trials > 0:
                if type(bayes_factor[0]) == np.float64:
                    joint_bf *= bayes_factor[0]
                    self.all_bayes_factors.append(bayes_factor[0])
                    
                    this_event_trials = bayes_factor[-1]
                    this_event_error = 2*np.std(this_event_trials)
                    self.all_bayes_factors_errors.append(this_event_error)
                    
                    joint_bf_array *= bayes_factor[-1]
            
            else:
                joint_bf *= bayes_factor
                self.all_bayes_factors.append(bayes_factor)
                           

        if save_file is not None:
            stack_dict = {}
            stack_dict['ref_eos'] = EoS2
            stack_dict['target_eos'] = EoS1

            stack_dict['joint_bf'] = joint_bf
            if N_trials > 0:
                stack_dict['joint_bf_array'] = joint_bf_array.tolist()
            else:
                stack_dict['joint_bf_array'] = None

            stack_dict['all_bf'] = self.all_bayes_factors
            if N_trials > 0:
                stack_dict['all_bf_err'] = self.all_bayes_factors_errors
            else:
                stack_dict['all_bf_err'] = None

            with open(save_file, 'w+') as f:
                json.dump(stack_dict, f, indent=4, sort_keys=True)

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
