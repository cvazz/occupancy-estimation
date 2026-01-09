from meteorize import load_homepath
import os
from logger import setup_logger
logger = setup_logger()
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



def load_ocp_paths() -> list[dict]:
    homepath = load_homepath()
    dataloc = homepath + "../data/MAXIV_ECH_new/"
    dataloc_dark = dataloc + "updated/ech-full_dark_dimple.mtz"
    dataloc_light = dataloc + "updated/ech-laser_dimple.mtz"
    pdbloc_light = dataloc + "models/ECH_xtrapol8_extrapolated_prefin.pdb"
    pdbloc_dark = dataloc + "models/ECH_MAXIV_dark_model.pdb"
    info_container = {
        "dataloc_dark": dataloc_dark,
        "pdbloc_dark":  pdbloc_dark,
        "pdbloc_light":  pdbloc_light,
        "dataloc_light": dataloc_light,
        "tname": "MAX IV OCP data 2",
        "fname": "OCP",
        "fshort": "OCP",
        "datatype": "OCP",
        "hs_limit": 1.7,
        "map_sampling": 3,
        "mid_xtr_factor": 8,
        "max_xtr_factor": 40,
        "FreeR_col": "FreeR_flag"
    }
    return [info_container]

def load_OLVPR1_paths() -> list[dict]:
    logger.info("Loading OLVPR1 paths")
    homepath = load_homepath()
    folderloc = homepath + "../data/OLVPR1/"
    dataloc_dark = folderloc + f"ground_massif3.mtz"
    dataloc_light = folderloc + f"0-37p5ms_massif3.mtz"
    pdbloc_dark = folderloc + "OLPVR1_id30a3_ground_refine_8.pdb"
    diffmap_loc = folderloc + "FoFoPHFc.mtz"

    dark_columns = dict(amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF")
    light_columns = dict(amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF")
    diffmap_columns = dict(amplitude_column="FoFo", phase_column="PHFc", uncertainty_column="")
    
    tname = "Canthaxanthine"
    fname = "CAN"
    # logger.info(f"Loading dark data from {dataloc_dark}")
    info_container = {
        "dataloc_dark": dataloc_dark,
        "pdbloc_dark": pdbloc_dark,
        "dataloc_light": dataloc_light,
        "diffmap_loc": diffmap_loc,
        "dark_cols": dark_columns,
        "light_cols": light_columns,
        "diffmap_cols": diffmap_columns,
        "hs_limit": 1.7,
        "map_sampling": 3,
        "fname": fname,
        "tname": tname,
        "fshort": fname,
        "dark_phases": True,
        "datatype": fname,
        "FreeR_col": "FreeR_flag"
    }
    return None
def load_CAN_paths(use_dimple=True) -> list[dict]:
    logger.info("Loading CAN paths")
    homepath = load_homepath()
    folderloc = homepath + "../data/MAXIV_CAN_new/"
    add_dimple = "_dimple" if use_dimple else ""
    dataloc_dark = folderloc + f"can-full_dark{add_dimple}.mtz"
    dataloc_light = folderloc + f"can-laser{add_dimple}.mtz"
    pdbloc_dark = folderloc + "CAN_MAXIV_dark_prefin.pdb"
    diffmap_loc = folderloc + "FoFoPHFc.mtz"

    dark_columns = dict(amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF")
    light_columns = dict(amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF")
    diffmap_columns = dict(amplitude_column="FoFo", phase_column="PHFc", uncertainty_column="")
    
    tname = "Canthaxanthine"
    fname = "CAN"
    # logger.info(f"Loading dark data from {dataloc_dark}")
    info_container = {
        "dataloc_dark": dataloc_dark,
        "pdbloc_dark": pdbloc_dark,
        "dataloc_light": dataloc_light,
        "diffmap_loc": diffmap_loc,
        "dark_cols": dark_columns,
        "light_cols": light_columns,
        "diffmap_cols": diffmap_columns,
        "hs_limit": 1.7,
        "map_sampling": 3,
        "fname": fname,
        "tname": tname,
        "fshort": fname,
        "dark_phases": True,
        "datatype": fname,
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
