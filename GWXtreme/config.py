import pathlib


LAL_NESTED_SAMPLING_TAYLORF2_EVIDENCES_FILE = "/home/michael/projects/eos/GWXtreme_Tasks/year3/lastStretch/files/BNS/TaylorF2_eos_prior_broad_evidences.json"
LAL_NESTED_SAMPLING_PHENOM_EVIDENCES_FILE = "/home/michael/projects/eos/GWXtreme_Tasks/year3/lastStretch/files/BNS/IMRphenom_eos_prior_broad_evidences.json"

EOS_LIST = ["APR4_EPP","HQC18","SKOP","MPA1","SKI4","SKI6","SKMP","SK272","SK255","RS","SKI3","SKI2","SKI5","H4","MS1B_PP","MS1_PP"]

SUPPORTED_EVENTS = ['GW170817', 'GW190425', 'GW230529']
SUPPORTED_WAVEFORMS = ['TaylorF2', 'PhenomNRT']

PROJECT_DIR = pathlib.Path(__file__).parent.parent

GW_PE_POSTERIOR_FILES = {
    'GW170817': {
        '2D': fr"{PROJECT_DIR}/pe_samples/GW170817/posterior_samples/GW170817_posterior_samples_broad_spin_prior.dat",
        '3D': fr"{PROJECT_DIR}/pe_samples/GW170817/posterior_samples/GW170817phenom.json"
    },
    'GW190425': {
        '2D': fr"{PROJECT_DIR}/pe_samples/GW190425/posterior_samples/GW190425_posterior_samples_broad_spin_prior.dat",
        '3D': ""
    },
    'GW230529': { # same file for both
        '2D': fr"{PROJECT_DIR}/pe_samples/GW230529/posterior_samples/GW230529_posterior_samples_phenom_lowspin.json",
        '3D': fr"{PROJECT_DIR}/pe_samples/GW230529/posterior_samples/GW230529_posterior_samples_phenom_lowspin.json"
    }
}

GWXTREME_FLOW_FILES = {
    'GW170817': {
        '2D': {
            'whitened': {
                'native': fr"{PROJECT_DIR}/density_estimators/GW170817/2D/zuko_prebuilt_maf_whitened/native/GW170817_2D_flow.pkl",
                'ensemble': fr"{PROJECT_DIR}/density_estimators/GW170817/2D/zuko_prebuilt_maf_whitened/ensemble"
            },
            'transformed': {
                'native': fr"{PROJECT_DIR}/density_estimators/GW170817/2D/zuko_prebuilt_maf_transformed/native/GW170817_2D_flow.pkl",
                'ensemble': fr"{PROJECT_DIR}/density_estimators/GW170817/2D/zuko_prebuilt_maf_transformed/ensemble"
            }
        },
        '3D': {
            'whitened': {
                'native': fr"{PROJECT_DIR}/density_estimators/GW170817/3D/zuko_prebuilt_maf_whitened/native/GW170817_3D_flow.pkl",
                'ensemble': fr"{PROJECT_DIR}/density_estimators/GW170817/3D/zuko_prebuilt_maf_whitened/ensemble",
            },
            'transformed': {
                'native': fr"{PROJECT_DIR}/density_estimators/GW170817/3D/zuko_prebuilt_maf_transformed/native/GW170817_3D_flow.pkl",
                'ensemble': fr"{PROJECT_DIR}/density_estimators/GW170817/3D/zuko_prebuilt_maf_transformed/ensemble",
            }
        }
    },
    'GW190425': {
        '2D': {
            'whitened': {
                'native': fr"{PROJECT_DIR}/density_estimators/GW190425/2D/zuko_prebuilt_maf_whitened/native/GW190425_2D_flow.pkl",
                'ensemble': fr"{PROJECT_DIR}/density_estimators/GW190425/2D/zuko_prebuilt_maf_whitened/ensemble",
            },
            'transformed': {
                'native': fr"{PROJECT_DIR}/density_estimators/GW190425/2D/zuko_prebuilt_maf_transformed/native/GW190425_2D_flow.pkl",
                'ensemble': fr"{PROJECT_DIR}/density_estimators/GW190425/2D/zuko_prebuilt_maf_transformed/ensemble"
            }
        },
        '3D': {
            'whitened': {
                'native': "",
                'ensemble': ""
            },
            'transformed': {
                'native': "",
                'ensemble': ""
            }
        }
    },
    'GW230529': {
        '2D': {
            'whitened': {
                'native': fr"{PROJECT_DIR}/density_estimators/GW230529/2D/zuko_prebuilt_maf_whitened/native/GW230529_2D_flow.pkl",
                'ensemble': fr"{PROJECT_DIR}/density_estimators/GW230529/2D/zuko_prebuilt_maf_whitened/ensemble"
            },
            'transformed': {
                'native': fr"{PROJECT_DIR}/density_estimators/GW230529/2D/zuko_prebuilt_maf_transformed/native/GW230529_2D_flow.pkl",
                'ensemble': fr"{PROJECT_DIR}/density_estimators/GW230529/2D/zuko_prebuilt_maf_transformed/ensemble"
            }
        },
        '3D': {
            'whitened': {
                'native': fr"{PROJECT_DIR}/density_estimators/GW230529/3D/zuko_prebuilt_maf_whitened/native/GW230529_3D_flow.pkl",
                'ensemble': fr"{PROJECT_DIR}/density_estimators/GW230529/3D/zuko_prebuilt_maf_whitened/ensemble"
            },
            'transformed': {
                'native': fr"{PROJECT_DIR}/density_estimators/GW230529/3D/zuko_prebuilt_maf_transformed/native/GW230529_3D_flow.pkl",
                'ensemble': fr"{PROJECT_DIR}/density_estimators/GW230529/3D/zuko_prebuilt_maf_transformed/ensemble"
            }
        }
    },
}

GWXTREME_KDE_GRID_FILES = {
    'GW170817': {
        '2D': "",
        '3D': ""
    },
    'GW190425': {
        '2D': "",
        '3D': ""
    },
    'GW230529': {
        '2D': "",
        '3D': ""
    }, 
}