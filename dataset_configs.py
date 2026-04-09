import os
from configuration import get_file_config, get_file_config_diff_only
from configuration import load_homepath
import numpy as np


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


def load_all_PL_paths(add_light=False) -> list[str]:
    homepath = load_homepath()
    folderloc = homepath + "data/photolyase/"
    datalocs_light = []
    folders = os.listdir(folderloc)
    changing_bits = []
    for ii, f in enumerate(folders):
        if not os.path.isdir(os.path.join(folderloc, f)):
            continue
        if f[:2] == "1_":
            continue
        if not f[0].isdigit():
            continue
        f = os.listdir(folderloc)[ii]
        final = f.split("_")[-1]
        changing_bit = f + "/" + final
        dataloc_light = folderloc + changing_bit + "_deposit.mtz"
        datalocs_light.append(dataloc_light)
        changing_bits.append(changing_bit)
        print(ii, changing_bit, f)
    return changing_bits
    # dataloc_light = datalocs_light[idx]  # Just use the first


def apply_config_PL_general(name_ending: str | int, add_light=False) -> dict:
    homepath = load_homepath()
    folderloc = homepath + "data/photolyase/"
    dataloc_dark = folderloc + "1_superdark/superdark_deposit.mtz"
    pdbloc_dark = folderloc + "1_superdark/superdark_deposit.pdb"
    folders = os.listdir(folderloc)
    folders = [f for f in folders if f[:2] != "1_"]
    out = None
    for ii, f in enumerate(folders):
        if isinstance(name_ending, str):
            if f[-len(name_ending) :] == name_ending:
                out = f
        elif isinstance(name_ending, int):
            if ii == name_ending:
                out = f
        else:
            raise ValueError(
                f"name_ending should be str or int, not {type(name_ending)}"
            )
    if out is None:
        print(folders)
        raise ValueError(f"No folder starting with {name_ending} found in {folderloc}")
    final = out.split("_")[-1]
    changing_bit = out + "/" + final
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
        config["input_files"]["pdbloc_triggered"] = (
            config["input_files"]["map_triggered"][:-4] + ".pdb"
        )
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
    dark_columns = dict(
        amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF"
    )
    light_columns = dict(
        amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF"
    )
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

    dark_columns = dict(
        amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF"
    )
    light_columns = dict(
        amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF"
    )
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
    name_machine = "OCP"
    high_resolution_limit = 1.7
    FreeR_col = "FreeR_flag"
    dark_columns = dict(
        amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF"
    )
    light_columns = dict(
        amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF"
    )

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
    FreeR_col = "FreeR_flag"

    dark_columns = dict(
        amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF"
    )
    light_columns = dict(
        amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF"
    )

    tname = "ECH data"
    name_machine = "ECH"
    high_resolution_limit = 1.6

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


def get_myo_trig(print_it=False):
    return {
        "-0.1ps": "5cn4",
        "0.0ps": "5cn5",
        "0.1ps": "5cn6",
        "0.2ps": "5cn7",
        "0.3ps": "5cn8",
        "0.4ps": "5cn9",
        "0.5ps": "5cnb",
        "0.6ps": "5cnc",
        "3ps": "5cnd",
        "10ps": "5cne",
        "50ps": "5cnf",
        "150ps": "5cng",
    }


def apply_config_myoglobin_general(idx: str | int) -> dict:
    homepath = load_homepath()
    dataloc = homepath + "data/myoglobin/"
    if True:
        dataloc_dark = dataloc + "5cmv.mtz"
        pdbloc_dark = dataloc + "5CMV.pdb"
    else:
        dataloc_dark = dataloc + "5cn4.mtz"
        pdbloc_dark = dataloc + "5CN4.pdb"

    high_resolution_limit = 2.3
    delay, mtzname = None, None

    for ii, (key, value) in enumerate(get_myo_trig().items()):
        if ii == idx or idx == key:
            delay, mtzname = key, value
    if delay is None or mtzname is None:
        raise ValueError(f"Could not find matching delay and mtzname for index {idx}")

    dataloc_light = dataloc + mtzname + ".mtz"
    name_machine = f"myo_{delay}"
    name_human = f"Myoglobin {delay}"
    default_columns = dict(
        amplitude_column="F", phase_column="PHIC", uncertainty_column="SIGF"
    )
    ints_columns = dict(ints_column="IMEAN", int_uncertainty_column="SIGIMEAN")
    config = get_file_config(
        dataloc_dark=dataloc_dark,
        dataloc_light=dataloc_light,
        pdbloc_dark=pdbloc_dark,
        columns_dark=default_columns | ints_columns,
        columns_triggered=default_columns | ints_columns,
        high_resolution_limit=high_resolution_limit,
        name_machine=name_machine,
        name_human=name_human,
        outpath=None,
    )
    config["input_files"]["columns_are_ints"] = True
    return config


#################################################################################
#################################################################################
#################################################################################


def get_folders_B12(print_it=False):
    homepath = load_homepath()
    folderloc = homepath + "data/b12_sacla/"
    diffmap_locs = os.listdir(folderloc)
    diffmap_locs = [f for f in diffmap_locs if "qFoFo" in f]
    diffmap_locs = np.sort(diffmap_locs)
    if print_it:
        for idx, f in enumerate(diffmap_locs):
            print(idx, f)
    return diffmap_locs


def get_folders_B12_diff_with_pdb():
    return {
        "100us_30mJ.cm-2_SACLA_qFoFo.mtz": "9S0B",
        "10ns_12mJ.cm-2_SACLA_qFoFo.mtz": "9S0C",
        "10ns_30mJ.cm-2_SACLA_qFoFo.mtz": "9S08",
        "300ns_30mJ.cm-2_SACLA_qFoFo.mtz": "9S09",
        "3ms_30mJ.cm-2_SACLA_qFoFo.mtz": "9S0B",
        "3us_120mJ.cm-2_SACLA_qFoFo.mtz": None,
        "3us_12mJ.cm-2_SACLA_qFoFo.mtz": "9S0D",
        "3us_30mJ.cm-2_SACLA_qFoFo.mtz": "9S0E",
        "3us_60mJ.cm-2_SACLA_qFoFo.mtz": None,
    }


def get_folders_B12_light_with_pdb():
    return {
        "100us_30mJ.cm-2_light_FPFree.mtz": "9S0B",
        "10ns_12mJ.cm-2_light_FPFree.mtz": "9S0C",
        "10ns_30mJ.cm-2_light_FPFree.mtz": "9S08",
        "300ns_30mJ.cm-2_light_FPFree.mtz": "9S09",
        "3ms_30mJ.cm-2_light_FPFree.mtz": "9S0B",
        "3us_120mJ.cm-2_light_FPFree.mtz": None,
        "3us_12mJ.cm-2_light_FPFree.mtz": "9S0D",
        "3us_30mJ.cm-2_light_FPFree.mtz": "9S0E",
        "3us_60mJ.cm-2_light_FPFree.mtz": None,
    }


from logger import setup_logger

logger = setup_logger()


def apply_config_B12_general_light(idx: int) -> dict:
    homepath = load_homepath()
    folderloc = homepath + "data/b12_sacla/"
    dataloc_dark = folderloc + "dark_ref_FPFREE.mtz"
    pdbloc_dark = folderloc + "9S06.pdb"

    lightmap_locs = os.listdir(folderloc)
    lightmap_locs = [f for f in lightmap_locs if "light_FPFree" in f]
    lightmap_locs = np.sort(lightmap_locs)
    lightmap_locs = np.delete(lightmap_locs, [1])
    lightmap_name = str(lightmap_locs[idx])
    logger.info(f"Selected lightmap: {lightmap_name}")

    b12_xtr_pdb = get_folders_B12_light_with_pdb()[lightmap_name]
    if b12_xtr_pdb is not None:
        pdbloc_light = folderloc + b12_xtr_pdb + ".pdb"
        logger.info(f"Selected light PDB: {pdbloc_light}")

    else:
        logger.info(f"No PDB found for {lightmap_name}, proceeding without light PDB.")
        pdbloc_light = None
    logger.info("Proceeding with configuration...")
    lightmap_loc = folderloc + lightmap_name
    high_resolution_limit = 2.2

    final = lightmap_name.split("mJ")[0]
    name_machine = f"B12_{final}"
    name_human = f"B12 {final}"
    columns_dark = dict(
        amplitude_column="F", phase_column="MODEL", uncertainty_column="SIGF"
    )
    columns_light = dict(
        amplitude_column="F", phase_column="MODEL", uncertainty_column="SIGF"
    )
    config = get_file_config(
        dataloc_dark=dataloc_dark,
        dataloc_light=lightmap_loc,
        pdbloc_dark=pdbloc_dark,
        columns_dark=columns_dark,
        columns_triggered=columns_light,
        pdbloc_triggered=pdbloc_light,
        high_resolution_limit=high_resolution_limit,
        name_machine=name_machine,
        name_human=name_human,
    )
    return config


def apply_config_B12_general(idx: int) -> dict:
    homepath = load_homepath()
    folderloc = homepath + "data/b12_sacla/"
    dataloc_dark = folderloc + "9S06_dark.mtz"
    pdbloc_dark = folderloc + "9S06.pdb"

    diffmap_locs = os.listdir(folderloc)
    diffmap_locs = [f for f in diffmap_locs if "qFoFo" in f]
    diffmap_locs = np.sort(diffmap_locs)
    diffmap_locs = np.delete(diffmap_locs, [1])
    diffmap_name = str(diffmap_locs[idx])
    logger.info(f"Selected diffmap: {diffmap_name}")

    b12_xtr_pdb = get_folders_B12_diff_with_pdb()[diffmap_name]
    if b12_xtr_pdb is not None:
        pdbloc_light = folderloc + b12_xtr_pdb + ".pdb"
        logger.info(f"Selected light PDB: {pdbloc_light}")

    else:
        logger.info(f"No PDB found for {diffmap_name}, proceeding without light PDB.")
        pdbloc_light = None
    logger.info("Proceeding with configuration...")
    diffmap_loc = folderloc + diffmap_name
    high_resolution_limit = 2.3

    final = diffmap_name.split("mJ")[0]
    name_machine = f"B12_{final}"
    name_human = f"B12 {final}"
    columns_dark = dict(
        amplitude_column="FP", phase_column="PHIF-model", uncertainty_column="SIGFP"
    )
    columns_diff = dict(
        amplitude_column="QFOFOWT", uncertainty_column="SIGF", phase_column="PHIQFOFOWT"
    )
    config = get_file_config_diff_only(
        dataloc_dark=dataloc_dark,
        dataloc_diff=diffmap_loc,
        pdbloc_dark=pdbloc_dark,
        columns_dark=columns_dark,
        columns_diff=columns_diff,
        pdbloc_light=pdbloc_light,
        high_resolution_limit=high_resolution_limit,
        name_machine=name_machine,
        name_human=name_human,
    )
    return config


def apply_config_PL_general_diff(name_ending: str | int, add_light=False) -> dict:
    homepath = load_homepath()
    folderloc = homepath + "data/photolyase/"
    dataloc_dark = folderloc + "1_superdark/superdark_deposit.mtz"
    pdbloc_dark = folderloc + "1_superdark/superdark_deposit.pdb"
    folders = os.listdir(folderloc)
    folders = [f for f in folders if f[:2] != "1_"]
    out = None
    for ii, f in enumerate(folders):
        if isinstance(name_ending, str):
            if f[-len(name_ending) :] == name_ending:
                out = f
        elif isinstance(name_ending, int):
            if ii == name_ending:
                out = f
        else:
            raise ValueError(
                f"name_ending should be str or int, not {type(name_ending)}"
            )
    if out is None:
        print(folders)
        raise ValueError(f"No folder starting with {name_ending} found in {folderloc}")
    final = out.split("_")[-1]
    changing_bit = out + "/" + final
    diffmap_loc = folderloc + changing_bit + "-dark_kwt_ded.mtz"
    high_resolution_limit = 2.6

    name_machine = f"PL_{final}"
    name_human = f"Photolyase {final}"
    columns_dark = dict(
        amplitude_column="F-obs-filtered",
        uncertainty_column="SIGF-obs-filtered",
        phase_column="PHIF-model",
    )
    columns_diff = dict(
        amplitude_column="KFOFOWT", uncertainty_column="SIGF", phase_column="PHIKFOFOWT"
    )
    config = get_file_config_diff_only(
        dataloc_dark=dataloc_dark,
        dataloc_diff=diffmap_loc,
        pdbloc_dark=pdbloc_dark,
        columns_dark=columns_dark,
        columns_diff=columns_diff,
        high_resolution_limit=high_resolution_limit,
        name_machine=name_machine,
        name_human=name_human,
    )
    return config


#################################################################################
#################################################################################
#################################################################################


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
    configs.append(apply_config_rsEGFP2())
    configs.append(apply_config_CAN())
    configs.append(apply_config_PL_3ns())
    configs.append(apply_config_PL_30ns())
    configs.append(apply_config_PL_10ns())
    configs.append(apply_config_PL_3ps())
    return configs


def get_all_PL_configs(get_diff=False) -> list[dict]:
    bits = load_all_PL_paths()
    configs = []
    import numpy as np

    numbers = [bit.split("_")[0] for bit in bits]
    numbers = [int(n) for n in numbers]
    arg_idx = np.argsort(np.array(numbers, int))
    bits = np.array(bits)[arg_idx]
    for bit in bits:
        if get_diff:
            config = apply_config_PL_general_diff(bit.split("/")[-1], add_light=False)
        else:
            config = apply_config_PL_general(bit.split("/")[-1])
        configs.append(config)
    return configs


print(get_folders_B12_diff_with_pdb())
print(get_folders_B12())
