import pathlib


PROJECT_DIR = pathlib.Path(__file__).parent.parent

LAL_NESTED_SAMPLING_TAYLORF2_EVIDENCES_FILE = f"{PROJECT_DIR}/lal_nested_sampling_eos_evidences/TaylorF2_broad-prior_eos-evidences.json"
LAL_NESTED_SAMPLING_PHENOM_EVIDENCES_FILE = f"{PROJECT_DIR}/lal_nested_sampling_eos_evidences/PhenomNRT_broad-prior_eos-evidences.json"

EOS_LIST = ["APR4_EPP","HQC18","SKOP","MPA1","SKI4","SKI6","SKMP","SK272","SK255","RS","SKI3","SKI2","SKI5","H4","MS1B_PP","MS1_PP"]

SUPPORTED_GW_EVENTS = ['GW170817', 'GW190425', 'GW230529']
SUPPORTED_WAVEFORMS = ['TaylorF2', 'PhenomNRT']

CBC_PE_POSTERIOR_FILES = {
    'GW170817': {
        '2D': fr"{PROJECT_DIR}/cbc_pe_samples/GW170817/posterior_samples/GW170817_posterior-samples_broad-spin-prior.dat",
        '3D': fr"{PROJECT_DIR}/cbc_pe_samples/GW170817/posterior_samples/GW170817_posterior-samples_PhenomNRT.json"
    },
    'GW190425': {
        '2D': fr"{PROJECT_DIR}/cbc_pe_samples/GW190425/posterior_samples/GW190425_posterior_samples_broad_spin_prior.dat",
        '3D': ""
    },
    'GW230529': { # same file for both
        '2D': fr"{PROJECT_DIR}/cbc_pe_samples/GW230529/posterior_samples/GW230529_posterior-samples_PhenomNRT.json",
        '3D': fr"{PROJECT_DIR}/cbc_pe_samples/GW230529/posterior_samples/GW230529_posterior-samples_PhenomNRT.json"
    }
}

NICER_PULSAR_SAMPLE_FILES = {
    'J0030': fr"{PROJECT_DIR}/nicer_pulsar_samples/N50k_J0030_3spot_CM.txt",
    'J0740': fr"{PROJECT_DIR}/nicer_pulsar_samples/N50k_J0740_XMM_CM.txt"
}