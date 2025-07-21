import pathlib


GW170817_POSTERIOR_ULTS_PRIOR = "/home/michael/projects/eos/GWXtreme_Tasks/year2/bilby_runs/simulations/outdir/real/uniformP_LTs/GW170817/simplified_result.json"
GW170817_POSTERIOR_ULS_PRIOR = "/home/michael/projects/eos/GWXtreme_Tasks/year3/GW170817_prior_L1L2/CIT_attempt_successful/outdir/simplified_result.json"
GW170817_POSTERIOR_ULS_PHENOM_PRIOR = "/home/michael/projects/eos/GWXtreme_Tasks/year3/lastStretch/files/BNS/GW170817phenom.json"

GW230529_PHENOM_POSTERIOR = "/home/michael/projects/eos/GWXtreme_Tasks/year3/lastStretch/files/NSBH/gw230529_phenom_lowSpin.json"

LAL_NESTED_SAMPLING_TAYLORF2_EVIDENCES_FILE = "/home/michael/projects/eos/GWXtreme_Tasks/year3/lastStretch/files/BNS/TaylorF2_eos_prior_narrow_evidences.json"
LAL_NESTED_SAMPLING_PHENOM_EVIDENCES_FILE = "/home/michael/projects/eos/GWXtreme_Tasks/year3/lastStretch/files/BNS/IMRphenom_eos_prior_narrow_evidences.json"

EOS_LIST = ["APR4_EPP","HQC18","SKOP","MPA1","SKI4","SKI6","SKMP","SK272","SK255","RS","SKI3","SKI2","SKI5","H4","MS1B_PP","MS1_PP"]

SUPPORTED_EVENTS = ['GW170817', 'GW190425']
SUPPORTED_WAVEFORMS = ['TaylorF2', 'IMRPhenomD_NRTidalv2']

PROJECT_DIR = pathlib.Path(__file__).parent.parent

GW_PE_POSTERIOR_FILES = {
    'GW170817': {
        '2D': fr"{PROJECT_DIR}/GW_event_PE_samples/GW170817/posterior_samples/GW170817_posterior_samples_broad_spin_prior.dat",
        '3D': ""
    },
    'GW190425': {
        '2D': fr"{PROJECT_DIR}/GW_event_PE_samples/GW190425/posterior_samples/GW190425_posterior_samples_broad_spin_prior.dat",
        '3D': ""
    }
}

GWXTREME_FLOW_FILES = {
    'GW170817': {
        '2D': {
            'native': fr"{PROJECT_DIR}/trained_density_estimators/GW170817/2D/zuko_prebuilt_maf/native/GW170817_2D_flow.pkl",
            'ensemble': fr"{PROJECT_DIR}/trained_density_estimators/GW170817/2D/zuko_prebuilt_maf/ensemble"
        },
        '3D': {
            'native': "",
            'ensemble': ""
        }
    },
    'GW190425': {
        '2D': {
            'native': fr"{PROJECT_DIR}/trained_density_estimators/GW190425/2D/zuko_prebuilt_maf/native/GW190425_2D_flow.pkl",
            'ensemble': fr"{PROJECT_DIR}/trained_density_estimators/GW190425/2D/zuko_prebuilt_maf/ensemble"
        },
        '3D': {
            'native': "",
            'ensemble': ""
        }
    }
}

GWXTREME_KDE_GRID_FILES = {
    'GW170817': {
        '2D': fr"{PROJECT_DIR}/trained_density_estimators/GW170817/2D/GW170817_2D_kde_probability_grid_250K.pt",
        '3D': ""
    },
    'GW190425': {
        '2D': fr"{PROJECT_DIR}/trained_density_estimators/GW170817/2D/GW170817_2D_kde_probability_grid_250K.pt",
        '3D': ""
    }
}