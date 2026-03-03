import numpy as np
from meteor import rsmap
import pandas as pd
import reciprocalspaceship as rs
from logger import setup_logger
from reciprocalspaceship.dtypes import StandardDeviationDtype  # Q
from pathlib import Path
import shutil, os

logger = setup_logger()


def adding_maps(
    map1: rsmap.Map, map2: rsmap.Map, *, factor1=1, factor2=1, suppress_warnings=False
) -> rsmap.Map:
    finite_mask1 = np.isfinite(map1.amplitudes)
    finite_mask2 = np.isfinite(map2.amplitudes)
    map_1 = map1[finite_mask1]
    map_2 = map2[finite_mask2]
    if (~finite_mask1).any() or (~finite_mask2).any() and not suppress_warnings:
        logger.warning(
            f"Some Maps have non-overlapping finite amplitudes: map1 {np.sum(~finite_mask1)} NaNs, map2 {np.sum(~finite_mask2)} NaNs"
        )

    common_indices = map_1.index.intersection(map_2.index)

    structure_factors1 = map_1.loc[common_indices].to_structurefactor()
    structure_factors2 = map_2.loc[common_indices].to_structurefactor()
    added_structure_factors = (
        factor1 * structure_factors1 + factor2 * structure_factors2
    )

    sum_of_map = rsmap.Map.from_structurefactor(
        added_structure_factors,
        index=common_indices,
        cell=map1.cell,
        spacegroup=map1.spacegroup,
    )
    logger.debug(
        f"High resolution limit after addition: {np.min(sum_of_map.compute_dHKL()):.2f}."
    )

    if map1.has_uncertainties and map2.has_uncertainties:
        sigmaF = np.sqrt(
            (factor1 * map_1.uncertainties[common_indices]) ** 2
            + (factor2 * map_2.uncertainties[common_indices]) ** 2
        )
        sum_of_map.set_uncertainties(pd.Series(sigmaF, index=common_indices))
    return sum_of_map


def save_extrapolated_map(
    info_container, xtr_factor, map_dark, diffmap, folder, name_prefix="", file_loc_diff=""
):
    xtr_map = adding_maps(map_dark, diffmap, factor2=xtr_factor)
    logger.info(f"Columns of xtr: {xtr_map.columns}")
    file_loc = str(folder / (name_prefix + f"_xtr{xtr_factor:.2f}.mtz"))
    # file_loc_dark_again = folder / (name_prefix + "_dark_again.mtz")
    # file_loc = str(folder / (name_prefix + f"_xtr{xtr_factor:.2f}_straight.mtz"))
    if file_loc_diff:
        diffmap.write_mtz(file_loc_diff)
    xtr_map.write_mtz(file_loc)
    # map_dark.write_mtz(file_loc_dark_again)
    # logger.info(f"Saving xtr map: {xtr_factor:.2f}, to {file_loc}")
    # ds_temp = rs.read_mtz(info_container["map_dark"])
    # if not diffmap.has_uncertainties:
    #     logger.warning("Diffmap has no uncertainties, adding fake uncertainties of 1.0")
    #     sigf = rs.DataSeries(np.ones(len(diffmap)), dtype=StandardDeviationDtype)
    #     diffmap.set_uncertainties(sigf, "Fake_uncertainty")

    # col_order = np.concatenate(
    #     [xtr_map.columns, [col for col in ds_temp.columns if "free" in col]]
    # )
    # for col in xtr_map.columns:
    #     ds_temp[col] = xtr_map[col]
    # ds_temp = ds_temp[col_order]
    # logger.info(f"Columns of xtr_map: {xtr_map.columns}")
    # logger.info(f"Columns of ds_temp: {ds_temp.columns}")

    # mask = np.logical_or(
    #     (~ds_temp["F"].isna() & ds_temp["SIGF"].isna()),
    #     (ds_temp["F"].isna() & ~ds_temp["SIGF"].isna()),
    # )
    # non_matching_indices = np.sum(np.array(mask))
    # if non_matching_indices > 0:
    #     logger.warning(
    #         f"Number of rows with not shared NaNs in F and SIGF: {non_matching_indices}"
    #     )
    #     # ds_temp.drop(mask, inplace=True) # drop rows where only one of F or SIGF is NaN
    #     ds_temp.loc[mask, xtr_map.columns] = np.nan

    # ds_temp.write_mtz(file_loc)

def save_to_folder(diffmap, map_dark, parameters, info_container, save_dict: dict):
    # copy file from pdbloc_dark to folder
    folder = Path(parameters["folder"])
    try:
        folder = folder.resolve()
        print(f"Absolute folder path: {folder}")
        if folder.exists() and not folder.is_dir():
            raise NotADirectoryError(f"Path exists and is not a directory: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured folder exists: {folder}")
    except Exception as e:
        logger.error(f"Failed to create folder {folder}: {e}")
        raise
    for key in ["pdb_dark", "pdb_triggered", "map_dark", "map_triggered"]:
        print(f"Checking for \n{key} copying to {folder}...")
        if key not in info_container:
            logger.warning(f"{key} not found in info_container, skipping copy.")
        try:
            shutil.copy(info_container[key], folder)
        except (PermissionError) as e:
            logger.warning(
                f"Could not copy {info_container[key]} to {folder}: {e}"
            )
        except KeyError as e:
            logger.warning(f"{key} not found in info_container, skipping copy: {e}")
    xtr_name = parameters["xtr_prefix"]
    for name_prefix, xtr_value in save_dict.items():
        prefix = xtr_name + "_" + name_prefix
        save_extrapolated_map(
            info_container, xtr_value, map_dark, diffmap, folder, name_prefix=prefix, file_loc_diff=parameters['diffmap_prefix']
        )

