import pathlib


PROJECT_DIR = pathlib.Path(__file__).parent.parent

LAL_NESTED_SAMPLING_TAYLORF2_EVIDENCES_FILE = f"{PROJECT_DIR}/lal_nested_sampling_eos_evidences/TaylorF2_broad-prior_eos-evidences.json"
LAL_NESTED_SAMPLING_PHENOM_EVIDENCES_FILE = f"{PROJECT_DIR}/lal_nested_sampling_eos_evidences/PhenomNRT_broad-prior_eos-evidences.json"

EOS_LIST = ["APR4_EPP","HQC18","SKOP","MPA1","SKI4","SKI6","SKMP","SK272","SK255","RS","SKI3","SKI2","SKI5","H4","MS1B_PP","MS1_PP"]