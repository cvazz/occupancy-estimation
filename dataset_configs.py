import os
from configuration import get_file_config
from configuration import load_homepath

def apply_config_rsEGFP2() -> dict:
    homepath = load_homepath()
    folderloc = homepath + "meteor/test/data/"
    dataloc_dark = folderloc + "scaled-test-data.mtz"
    dataloc_light = folderloc + "scaled-test-data.mtz"
    pdbloc_dark = folderloc + "8a6g.pdb"
    name_machine = "rsEGFP2"
    columns_dark = {
        "amplitude_column": "F_off",
        "phase_column": "PHIC_nochrom",
        "uncertainty_column": "SIGF_off",
    }
    columns_triggered = {
        "amplitude_column": "F_on",
        "phase_column": "PHIC_chrom",
        "uncertainty_column": "SIGF_on",
    }
    high_resolution_limit = 1.6

    return get_file_config(
        dataloc_dark=dataloc_dark,
        dataloc_light=dataloc_light,
        pdbloc_dark=pdbloc_dark,
        columns_dark=columns_dark,
        columns_triggered=columns_triggered,
        high_resolution_limit=high_resolution_limit,
        name_machine=name_machine,
    )

def load_all_PL(add_light = False) -> None:
    homepath = load_homepath()
    folderloc = homepath + "data/photolyase/"
    dataloc_dark = folderloc + "1_superdark/superdark_deposit.mtz"
    pdbloc_dark = folderloc + "1_superdark/superdark_deposit.pdb"
    datalocs_light = []
    folders = (os.listdir(folderloc))

    for ii, f in enumerate(folders):
        if not os.path.isdir(os.path.join(folderloc, f)):
            continue
        if f[:2] == "1_":
            continue
        if not f[0].isdigit():
            continue
        f = os.listdir(folderloc)[ii]
        final = f.split("_")[-1]
        changing_bit = f + "/" +  final
        dataloc_light = folderloc + changing_bit + "_deposit.mtz"
        datalocs_light.append(dataloc_light)
        print(ii, changing_bit, f)

    # dataloc_light = datalocs_light[idx]  # Just use the first

def apply_config_PL_general(name_ending: str, add_light = False) -> dict:
    homepath = load_homepath()
    folderloc = homepath + "data/photolyase/"
    dataloc_dark = folderloc + "1_superdark/superdark_deposit.mtz"
    pdbloc_dark = folderloc + "1_superdark/superdark_deposit.pdb"
    folders = os.listdir(folderloc)
    out = None
    for f in folders:
        if f[-len(name_ending):] == name_ending:
            out = f
    if out is None:
        print(folders)
        raise ValueError(f"No folder starting with {name_ending} found in {folderloc}")
    final = out.split("_")[-1]
    changing_bit = out + "/" +  final
    dataloc_light = folderloc + changing_bit + "_deposit.mtz"
    high_resolution_limit = 2.6

    name_machine = f"PL_{final}"
    name_human = f"Photolyase {final}"
    columns_dark = dict(
        amplitude_column="F-obs-filtered",
        uncertainty_column="SIGF-obs-filtered",
        phase_column="PHIF-model",
    )
    columns_triggered = dict(
        amplitude_column="F", uncertainty_column="SIGF", phase_column="PHIF-model"
    )
    config = get_file_config(
        dataloc_dark=dataloc_dark,
        dataloc_light=dataloc_light,
        pdbloc_dark=pdbloc_dark,
        columns_dark=columns_dark,
        columns_triggered=columns_triggered,
        high_resolution_limit=high_resolution_limit,
        name_machine=name_machine,
        name_human=name_human,
    )
    if add_light:
        config["input_files"]["pdbloc_triggered"] = config["input_files"]["map_triggered"][:-4] + ".pdb"
    # config["masking"]["dar"]
    return config

def apply_config_OLVPR1() -> dict:
    homepath = load_homepath()
    folderloc = homepath + "data/OLVPR1/"
    dataloc_dark = folderloc + f"ground_massif3.mtz"
    dataloc_light = folderloc + f"0-37p5ms_massif3.mtz"
    pdbloc_dark = folderloc + "OLPVR1_id30a3_ground_refine_8.pdb"
    # diffmap_loc = folderloc + "FoFoPHFc.mtz"
    high_resolution_limit = 1.7
    name_machine = "OLPVR1"
    dark_columns = dict(amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF")
    light_columns = dict(amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF")
    # diffmap_columns = dict(amplitude_column="FoFo", phase_column="PHFc", uncertainty_column="")
    config = get_file_config(
        dataloc_dark=dataloc_dark,
        dataloc_light=dataloc_light,
        pdbloc_dark=pdbloc_dark,
        columns_dark=dark_columns,
        columns_triggered=light_columns,
        high_resolution_limit=high_resolution_limit,
        name_machine=name_machine,
    )
    return config

def apply_config_CAN(use_dimple=True) -> dict:
    homepath = load_homepath()
    folderloc = homepath + "data/MAXIV_CAN_new/"
    add_dimple = "_dimple" if use_dimple else ""
    dataloc_dark = folderloc + f"can-full_dark{add_dimple}.mtz"
    dataloc_light = folderloc + f"can-laser{add_dimple}.mtz"
    pdbloc_dark = folderloc + "CAN_MAXIV_dark_prefin.pdb"
    # diffmap_loc = folderloc + "FoFoPHFc.mtz"

    dark_columns = dict(amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF")
    light_columns = dict(amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF")
    # diffmap_columns = dict(amplitude_column="FoFo", phase_column="PHFc", uncertainty_column="")
    
    name_human = "Canthaxanthine"
    name_machine = "CAN"

    config = get_file_config(
        dataloc_dark=dataloc_dark,
        dataloc_light=dataloc_light,
        pdbloc_dark=pdbloc_dark,
        columns_dark=dark_columns,
        columns_triggered=light_columns,
        high_resolution_limit=1.7,
        name_machine=name_machine,
        name_human=name_human,
    )
    return config

def apply_config_OCP() -> dict:
    homepath = load_homepath()
    dataloc = homepath + "data/MAXIV_ECH_new/"
    dataloc_dark = dataloc + "updated/ech-full_dark_dimple.mtz"
    dataloc_light = dataloc + "updated/ech-laser_dimple.mtz"
    pdbloc_light = dataloc + "models/ECH_xtrapol8_extrapolated_prefin.pdb"
    pdbloc_dark = dataloc + "models/ECH_MAXIV_dark_model.pdb"
    name_human = "MAX IV OCP data 2"
    name_machine= "OCP"
    high_resolution_limit = 1.7
    FreeR_col = "FreeR_flag"
    dark_columns = dict(amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF")
    light_columns = dict(amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF")

    config = get_file_config(
        dataloc_dark=dataloc_dark,
        dataloc_light=dataloc_light,
        pdbloc_dark=pdbloc_dark,
        columns_dark=dark_columns,
        columns_triggered=light_columns,
        high_resolution_limit=high_resolution_limit,
        name_machine=name_machine,
        name_human=name_human,
    )
    return config


def apply_config_ECH() -> dict:
    homepath = load_homepath()
    dataloc = homepath + "data/ECH2/"
    dataloc_dark = dataloc + "ech-dark_dimple.mtz"
    dataloc_light = dataloc + "ech-laser_dimple.mtz"
    # pdbloc_light = dataloc + "models/ECH_xtrapol8_extrapolated_prefin.pdb"
    pdbloc_dark = dataloc + "ECH_MAXIV_dark_new_prefin.pdb"
    FreeR_col= "FreeR_flag"

    dark_columns = dict(amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF")
    light_columns = dict(amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF")
    
    tname = "ECH data"
    name_machine = "ECH"
    high_resolution_limit=1.6

    config = get_file_config(
        dataloc_dark=dataloc_dark,
        dataloc_light=dataloc_light,
        pdbloc_dark=pdbloc_dark,
        columns_dark=dark_columns,
        columns_triggered=light_columns,
        high_resolution_limit=high_resolution_limit,
        name_machine=name_machine,
        name_human=tname,
    )
    return config

#################################################################################
#################################################################################
#################################################################################

def load_photolyase_paths() -> list[dict]:
    logger.info("Loading Photolyase paths")
    homepath = load_homepath()
    folderloc = homepath + "../data/photolyase/"
    dataloc_dark = folderloc + "1_superdark/superdark_deposit.mtz"
    pdbloc_dark = folderloc + "1_superdark/superdark_deposit.pdb"

    dark_columns = dict(
            amplitude_column="F-obs-filtered",
            uncertainty_column="SIGF-obs-filtered",
            phase_column="PHIF-model",
        )

    light_columns = dict(
        amplitude_column="F", uncertainty_column="SIGF", phase_column="PHIF-model"
    )
    diffmap_columns = dict(
        amplitude_column="KFOFOWT",  phase_column="PHIKFOFOWT"

    )

    mask_loc_sphere = (
        homepath + "../photolyase_masks/photolyase_TT_chainA_spherical_mask.ccp4"
    )
    mask_loc_handcraft = (
        homepath + "../photolyase_masks/photolyase_chainA_spherical_masks.ccp4"
    )
    # logger.info(f"Loading dark data from {dataloc_dark}")
    # loads all folders in the folderloc
    info_containers = []
    for f in os.listdir(folderloc):
        if not os.path.isdir(os.path.join(folderloc, f)):
            continue
        if f[:2] == "1_":
            continue
        if not f[0].isdigit():
            continue
        changing_bit = f + "/" + f.split("_")[-1]
        dataloc_light = folderloc + changing_bit + "_deposit.mtz"
        pdbloc_light = folderloc + changing_bit + "_deposit.pdb"
        diffmap_loc = folderloc + changing_bit + "-dark_kwt_ded.mtz"


        timepoint = changing_bit.split("/")[-1]
        fname = "photolyase_" + timepoint
        tname = "Photolyase " + timepoint
        info_container = {
            "dataloc_dark": dataloc_dark,
            "pdbloc_dark": pdbloc_dark,
            "dataloc_light": dataloc_light,
            "pdbloc_light": pdbloc_light,
            "diffmap_loc": diffmap_loc,
            "dark_cols": dark_columns,
            "light_cols": light_columns,
            "diffmap_cols": diffmap_columns,
            "mask_loc_sphere": mask_loc_sphere,
            "mask_hand": mask_loc_handcraft,
            "fname": fname,
            "tname": tname,
            "fshort": "PL" + timepoint,
            "hs_limit": 2.0,
            "map_sampling": 3,
            "datatype": "photolyase",
            "FreeR_col": "R-free-flags"
        }
        info_containers.append(info_container)
    for info_container in info_containers:
        logger.info(f"Loaded {info_container['tname']} with {info_container['fname']}")
    return info_containers

def load_cistrans_paths() -> list[dict]:
    logger.info("Loading CisTrans paths")
    dataloc = homepath + "../synthetic_cistrans/"
    pdbloc_dark = dataloc + "trans.pdb"
    pdbloc_light = dataloc + "100ps.pdb"
    info_container = {
        "pdbloc_dark": pdbloc_dark,
        "pdbloc_light": pdbloc_light,
        "tname": "CisTrans",
        "fname": "cistrans",
        "fshort": "CT",
        "datatype": "cistrans",
        "hs_limit": 1.8,
        "map_sampling": 3,
        "mid_xtr_factor": 4,
        "max_xtr_factor": 14,
        "FreeR_col": "",
    }
    return [info_container]




def load_mpro_paths() -> list[dict]:
    logger.info("Loading MPro paths")
    homepath = load_homepath()
    folderloc = homepath + "../meteor/test/data/"
    dataloc_dark = folderloc + "scaled-test-data.mtz"
    dataloc_light = folderloc + "scaled-test-data.mtz"
    pdbloc_dark = folderloc + "8a6g.pdb"
    fname = "rseGPF2"

    tname = "rseGPF2 (cistrans)"
    dark_columns = dict(amplitude_column="F_off", phase_column="PHIC_nochrom", uncertainty_column="SIGF_off")
    light_columns = dict(amplitude_column="F_on", phase_column="PHIC_nochrom", uncertainty_column="SIGF_on")
    info_container = {
        "dataloc_dark": dataloc_dark,
        "dataloc_light": dataloc_light,
        "pdbloc_dark": pdbloc_dark,
        "hs_limit": 1.6,
        "dark_cols": dark_columns,
        "light_cols": light_columns,
        "map_sampling": 3,
        "fname": fname,
        "tname": tname,
        "fshort": fname,
        "FreeR_col": "",
        "datatype": "mpro",
    }
    return [info_container]


def load_maxiv_paths() -> list[dict]:
    homepath = load_homepath()
    dataloc = homepath + "../data/MAXIV_ECH_new/"
    dataloc_dark = dataloc + "initial/ech-full_dark_dimple.mtz"
    pdbloc_light = dataloc + "initial/ech-light_dimple.mtz"
    info_container = {
        "dataloc_dark": dataloc_dark,
        "pdbloc_dark": None,
        "dataloc_light": pdbloc_light,
        "tname": "MAX IV OCP data",
        "fname": "maxiv",
        "fshort": "M4",
        "datatype": "maxiv",
        "hs_limit": 1.7,
        "map_sampling": 3,
        "mid_xtr_factor": 5,
        "max_xtr_factor": 30,

    }
    return [info_container]

def load_ECH_paths() -> list[dict]:
    homepath = load_homepath()
    dataloc = homepath + "../data/ECH2/"
    dataloc_dark = dataloc + "ech-dark_dimple.mtz"
    dataloc_light = dataloc + "ech-laser_dimple.mtz"
    # pdbloc_light = dataloc + "models/ECH_xtrapol8_extrapolated_prefin.pdb"
    pdbloc_dark = dataloc + "ECH_MAXIV_dark_new_prefin.pdb"
    info_container = {
        "dataloc_dark": dataloc_dark,
        "pdbloc_dark":  pdbloc_dark,
        "dataloc_light": dataloc_light,
        # "pdbloc_light":  pdbloc_light,
        "tname": "ECH data",
        "fname": "ECH",
        "fshort": "ECH",
        "datatype": "OCP",
        "hs_limit": 1.58,
        "FreeR_col": "FreeR_flag"
    }
    return [info_container]



def load_doeke_paths() -> list[dict]:
    logger.info("Loading Doeke paths")
    homepath = load_homepath()
    folderloc = homepath + "../data/Precognition_hkls/"
    dataloc_dark = folderloc + "processing_me/combined_200_OFF.mtz"
    dataloc_light = (
        folderloc
    + "small_P1_v2/e35cdef_200ns_HD_2sig_varEll.hkl_small_P1_no_sys_abs.mtz"
    )
    pdbloc_dark = None
    pdbloc_light = folderloc + "PDBs/5e11.pdb"
    tname = "Doeke"
    fname = "doeke"
    # logger.info(f"Loading dark data from {dataloc_dark}")
    info_container = {
        "dataloc_dark": dataloc_dark,
        "pdbloc_dark": pdbloc_dark,
        "dataloc_light": dataloc_light,
        "pdbloc_light": pdbloc_light,
        "hs_limit": 1.8,
        "map_sampling": 3,
        "fname": fname,
        "tname": tname,
        "fshort": fname,
        "dark_phases": True,
        "datatype": fname,
    }
    return [info_container]

apply_config_PL_30ns = lambda: apply_config_PL_general("30ns")
apply_config_PL_100us = lambda: apply_config_PL_general("100us")
apply_config_PL_10ns = lambda: apply_config_PL_general("10ns")
apply_config_PL_3ns = lambda: apply_config_PL_general("3ns")
apply_config_PL_3ps = lambda: apply_config_PL_general("3ps")

# apply_config_PL_30us = lambda: apply_config_PL_general(6)
# apply_config_PL_10us = lambda: apply_config_PL_general(13)
# apply_config_PL_3us = lambda: apply_config_PL_general(14)

# from collections.abc import Callable
def get_some_configs() -> list[dict]:
    configs = []
    configs.append(apply_config_OCP())
    # configs.append(apply_config_ECH())
    configs.append(apply_config_PL_30ns())
    configs.append(apply_config_rsEGFP2())
    # configs.append(apply_config_CAN())
    # configs.append(apply_config_PL_3ns())
    # configs.append(apply_config_PL_30ns())
    # configs.append(apply_config_PL_3ps())
    return configs
def get_all_configs() -> list[dict]:
    configs = []
    configs.append(apply_config_OCP())
    configs.append(apply_config_ECH())
    configs.append(apply_config_PL_10ns())
    configs.append(apply_config_rsEGFP2())
    configs.append(apply_config_CAN())
    configs.append(apply_config_PL_3ns())
    configs.append(apply_config_PL_30ns())
    configs.append(apply_config_PL_3ps())
    return configs