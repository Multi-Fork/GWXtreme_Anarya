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


from multiprocessing import cpu_count, Pool
from typing import Literal

import numpy as np
import h5py
import emcee

from .eos_prior import is_valid_eos, create_spectral_eos, create_polytrope_eos
from .eos_model_selection import JointModelSelector


class ParameterizedEoSSampler():
    def __init__(
            self, 
            posterior_files, 
            prior_bounds,
            methods,
            save_file,
            N_grid: int = 100, 
            N_walkers: int = 100,
		    N_parameter_samples: int = 10000, 
            N_dim: int = 4, 
            density_est_method: Literal['kde', 'flow'] = 'kde',
            N_posterior_samples: int | None = None,
            parameterization: Literal['spectral', 'polytrope'] = 'spectral',
            N_pool: int = 1,
        ):
        '''
        Initiates Parametric EoS mcmc Sampler Class
        that also stacks over multiple events,from the
        single event, uniform in LambdaT, dLambdaT parameter
        estimation runs. Parametrization chosen: 4 parameter 
        Spectal decomposition of Adiabatic Index in terms of
        Pressure or 4 parameter piecewise polytrope.
            
        prior_bounds :: dictionary containining prior bounds
                        of parameters. example for spectral:
                        {'gamma1':{'params':{"min":0.2,"max":2.00}},
                        'gamma2':{'params':{"min":-1.6,"max":1.7}},
                        'gamma3':{'params':{"min":-0.6,"max":0.6}},
                        'gamma4':{'params':{"min":-0.02,"max":0.02}}}
                              
        posterior_files :: Array containing full paths to 
                                single event PE posterior files
                                    
        N_walkers  ::  Number of walkers to use for mcmc
            
        N_parameter_samples  ::   Numper of samples to draw using mcmc
                        to infer the joint posterior of spectral
                        parameters
                           
            
        save_file  ::   .h5 File name that will store samples
            
        
        N_pool     ::    Number of cpu's to use for parallelization
        
        N_grid     ::    Number of grid points to evaluate evidence integral
                        over, while evaluationg log_prob
                        
        N_posterior_samples        ::    Number of samples to which the single event q and 
                        lambda_tilde posteriors are downsampled
                        
        '''
        
        self.prior_bounds = prior_bounds
        self.save_file = save_file
        self.N_walkers = N_walkers
        self.N_parameter_samples = N_parameter_samples
        self.N_dim = N_dim
        self.N_pool = N_pool
        self.N_grid = N_grid
        self.parameterization = parameterization
        
        self.joint_selector = JointModelSelector(
            posterior_files,
            methods=methods,
            density_est_method=density_est_method,
            parameterization=parameterization,
            N_samples=N_posterior_samples,
        )
        
        if parameterization == 'spectral':
            self.keys = ['gamma1', 'gamma2', 'gamma3', 'gamma4']
            self.eos = create_spectral_eos
        
        elif parameterization == 'polytrope':
            self.keys = ['logP', 'gamma1', 'gamma2', 'gamma3']
            self.eos = create_polytrope_eos
        
    def log_post(self,p):
        '''
        This mathod accepts an array of spectral parameters
        and returns their log posterior given gw data from all
        the events provided while initializing the class.
        
        p  :: array of spectral parameters.
        
        N_grid :: Number of grid points to use to perform integral
                over q
        '''
        
        params = {k:np.array([par]) for k, par in zip(self.keys, p)}
        
        if not is_valid_eos(params, self.prior_bounds, spectral=self.parameterization == 'spectral'):
            return -np.inf
        
        return np.nan_to_num(np.log(self.joint_selector.compute_parameterized_eos_joint_evidence(p, N_grid=self.N_grid)))
    
    def initialize_walkers(self):
        
        '''
        This method initializes the walkers for mcmc 
        (to be run on the spectral parameters posterior
        given GW data from all the events)inside the prior 
        region.
        '''
        
        n_walkers = 0
        self.p0 = []
        while n_walkers < self.N_walkers:
            g = np.array(
                [
                    np.random.uniform(
                        self.prior_bounds[k]["params"]["min"], 
                        self.prior_bounds[k]["params"]["max"]
                    ) for k in self.keys
                ]
            )
            params = {k:np.array([g[i]]) for i, k in enumerate(self.keys)}
    
            if is_valid_eos(params,self.prior_bounds,spectral=self.parameterization == 'spectral'):
                try:
                    post = self.log_post(g)
                except ValueError as e:
                    print(e, '\n', g, n_walkers)
                    continue
                if post == -np.inf:
                    continue

                self.p0.append(g)
                n_walkers += 1
                
    def run_sampler(self):
        '''
        runs mcmc sampler to draw samples of 
        the spectral parameters from their
        posterior given GW data from all the 
        events.
        '''
        
        if self.N_pool > 1:
            with Pool(min(cpu_count(), self.N_pool)) as pool:
                sampler = emcee.EnsembleSampler(
                    self.N_walkers, 
                    self.N_dim, 
                    self.log_post,
                    pool=pool
                )
                sampler.run_mcmc(self.p0, self.N_parameter_samples, progress=True)
        
                self.samples = sampler.get_chain()
                self.logp = sampler.get_log_prob()
        else:
              sampler = emcee.EnsembleSampler(
                  self.N_walkers,
                  self.N_dim,
                  self.log_post
                )
              sampler.run_mcmc(self.p0, self.N_parameter_samples, progress=True)
        
              self.samples = sampler.get_chain()
              self.logp = sampler.get_log_prob() 
    
    def save_data(self):
        '''
        This method saves the posterior samples 
        of the spectral parameters and there corresponding
        log probablities
        '''
        with h5py.File(self.save_file, 'w') as f:
            f.create_dataset('chains', data=np.array(self.samples))
            f.create_dataset('logp', data=np.array(self.logp))
        
    def parse_samples(self, burn_in_frac: float = 0.5, thinning=None):
        '''
        This methods parses the MCMC samples of
        EoS hyper-parameters and for use by the plotting 
        functions.
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
        if self.samples is None: return
        
        burn_in = int(self.samples.shape[0] * burn_in_frac)
        samples = []
        
        if thinning is None:
            thinning = int(self.samples.shape[0] / 50)

            try:
                thinning = int(max(emcee.autocorr.integrated_time(self.samples)) / 2.)
            except emcee.autocorr.AutocorrError as e:
                print(e)
        
        for i in range(burn_in, self.samples.shape[0], thinning):
            for j in range(self.samples.shape[1]):
                samples.append(self.samples[i, j, :])

        return np.array(samples)
    
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
            self.samples = np.array(f['chains'])
            self.logp = np.array(f['logp'])