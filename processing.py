import numpy as np
import gemmi
import reciprocalspaceship as rs
import matplotlib.pyplot as plt


import pickle
import os

from meteor import compute_meteor_difference_map
from meteor import rsmap
from meteor.utils import cut_resolution
from meteor.scale import scale_maps
from meteor.sfcalc import gemmi_structure_to_calculated_map
from meteor.diffmaps import compute_difference_map
from meteor.scripts.common import (
    DiffMapSet,
    WeightMode,
    kweight_diffmap_according_to_mode,
)

from masking import support_from_masker
from logger import setup_logger

logger = setup_logger()


def get_maps(input_files_dict):
    high_res_limit = input_files_dict["general"]["high_resolution_limit"]

    dataloc_dark = input_files_dict["input_files"]["map_dark"]
    dataloc_triggered = input_files_dict["input_files"]["map_triggered"]

    ds_triggered = rs.read_mtz(dataloc_triggered)
    ds_dark = rs.read_mtz(dataloc_dark)
    if high_res_limit:
        logger.info(f"Imposing high_resolution_limit: {high_res_limit}")
        ds_dark = cut_resolution(ds_dark, high_resolution_limit=high_res_limit)
        ds_triggered = cut_resolution(
            ds_triggered, high_resolution_limit=high_res_limit
        )

    dark_columns = input_files_dict["input_files"]["columns_dark"]
    triggered_columns = input_files_dict["input_files"]["columns_triggered"]
    if input_files_dict["input_files"]["impose_dark_phases"]:
        ds_triggered[triggered_columns["phase_column"]] = ds_dark[
            dark_columns["phase_column"]
        ]
    unscaled_dark = rsmap.Map(ds_dark, **dark_columns)
    unscaled_triggered = rsmap.Map(ds_triggered, **triggered_columns)

    return unscaled_dark, unscaled_triggered


def check_highres_limit(
    map_dark: rsmap.Map, map_triggered: rsmap.Map, info_container: dict
):
    dmin_dark = map_dark.compute_dHKL().min()
    dmin_triggered = map_triggered.compute_dHKL().min()

    if not np.isclose(dmin_dark, dmin_triggered):
        high_res_limit = np.round(max(dmin_dark, dmin_triggered), 1)
        logger.warning(
            f"Different resolution limits in dark and triggered maps: {dmin_dark:.2f} A vs {dmin_triggered:.2f} A"
        )
        logger.warning(f"Changing high-resolution limit to {high_res_limit:.2f} A")
        info_container["high_resolution_limit"] = high_res_limit
        map_dark = cut_resolution(map_dark, high_resolution_limit=high_res_limit)
        map_triggered = cut_resolution(
            map_triggered, high_resolution_limit=high_res_limit
        )
    return map_dark, map_triggered


def calculate_diffmaps(
    map_dark: rsmap.Map,
    map_triggered: rsmap.Map,
    map_dark_comp: rsmap.Map,
    meta_loc: str = "",
    only_kweighted: bool = False,
    parameters: dict={},
):
    overwrite_solution = parameters.get("overwrite_solution", False)
    calculate_again = bool(parameters.get("k_weight", False)) or (
        os.path.exists(meta_loc) and not overwrite_solution
    )

    logger.info(f"this is calculate_again {calculate_again}")
    map_set = DiffMapSet(map_dark, map_triggered, map_dark_comp)
    if calculate_again:
        if parameters.get("k_weight", False):
            opt_k = parameters.get("k_weight")
            opt_tv = parameters.get("tv_weight", None)
            only_kweighted = opt_tv is None
            logger.info(
                f"Using provided k_weight: {opt_k}, tv_weight: {opt_tv}, only_kweighted: {only_kweighted}"
            )

        elif os.path.exists(meta_loc) and not overwrite_solution:
            with open(meta_loc, "rb") as f:
                meta = pickle.load(f)
            # Extract the optimal parameters
            opt_k = (
                meta.k_parameter_optimization.optimal_parameter_value
                if meta.k_parameter_optimization
                else None
            )
            opt_tv = meta.tv_weight_optmization.optimal_parameter_value
            logger.info(
                f"loading: {opt_k}, tv_weight: {opt_tv}, only_kweighted: {only_kweighted}"
            )
        else:
            raise ValueError("No parameters provided and no meta file found.")

        if only_kweighted:
            diffmap, kparameter_metadata = kweight_diffmap_according_to_mode(
                kweight_mode=WeightMode.fixed,
                kweight_parameter=opt_k,
                mapset=map_set,
            )
            return diffmap
        else:
            # 2) Rerun with fixed parameters (no iteration/scan)
            final_map, _ = compute_meteor_difference_map(
                map_set,
                kweight_mode=WeightMode.fixed,
                kweight_parameter=(
                    opt_k if opt_k is not None else 0.0
                ),  # or omit if you don't want k-weighting
                tv_denoise_mode=WeightMode.fixed,
                tv_weight=opt_tv,
            )
    elif only_kweighted:
        logger.warning("Meta file not saved, will calculate again")
        diffmap, kparameter_metadata = kweight_diffmap_according_to_mode(
            kweight_mode=WeightMode.optimize,
            # kweight_parameter=opt_k,
            mapset=map_set,
        )
        return diffmap
    else:
        final_map, meta = compute_meteor_difference_map(
            map_set,
            kweight_mode=WeightMode.optimize,
            tv_denoise_mode=WeightMode.optimize,
        )

        if meta_loc != "":
            with open(meta_loc, "wb") as f:
                pickle.dump(meta, f)
    logger.info(f"diffmap has uncertainties: {final_map.has_uncertainties}")
    return final_map


def get_meta_loc(general_config):
    output_folder = general_config.get("output_folder", "./results/")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    evaluation_path_basis = output_folder + general_config["name_machine"] + "/"
    os.makedirs(evaluation_path_basis, exist_ok=True)
    name = f"diffmap_config_{general_config['high_resolution_limit']*10}.pkl"
    meta_loc = evaluation_path_basis + name
    return meta_loc


def combined_diffmap_calc(
    map_dark, map_triggered, map_dark_comp, diffmap_type="vanilla", general_config=None
):
    match diffmap_type:
        case "kweighted":
            meta_loc = get_meta_loc(general_config)
            diffmap = calculate_diffmaps(
                map_dark, map_triggered, map_dark_comp, meta_loc, only_kweighted=True
            )
        case "tv":
            meta_loc = get_meta_loc(general_config)
            diffmap = calculate_diffmaps(
                map_dark, map_triggered, map_dark_comp, meta_loc, only_kweighted=False
            )
        case "vanilla":
            diffmap = compute_difference_map(derivative=map_triggered, native=map_dark)
        case _:
            logger.warning(
                f"Unknown or unset diffmap_type: {diffmap_type}, defaulting to vanilla"
            )
            diffmap = compute_difference_map(derivative=map_triggered, native=map_dark)
    return diffmap


def autoshift_rsmap(
    map_in: rsmap.Map,
    map_dark_comp: rsmap.Map,
    config: dict,
    ignore_mask: np.ndarray | bool = False,
    diagnostic_plots: bool = False,
):
    map_sampling = config["map_sampling"]
    pdbloc_dark = config["pdbloc_dark"]

    rsmap_np = map_in.to_3d_numpy_map(map_sampling=map_sampling)
    map_dark_comp_np = map_dark_comp.to_3d_numpy_map(map_sampling=map_sampling)

    only_solvent = support_from_masker(pdbloc_dark, map_dark_comp_np.shape)
    ignore_mask = np.logical_or(only_solvent, ignore_mask)
    logger.info(f"ignore_mask voxel count: {np.sum(ignore_mask) / ignore_mask.size}")

    include_mask = ~ignore_mask
    if ignore_mask.all():
        logger.warning("All voxels are ignored in autoshift; no shift applied.")
        return map_in
    shifts = map_dark_comp_np[include_mask] - rsmap_np[include_mask]
    if diagnostic_plots:
        plt.figure()
        plt.plot(map_dark_comp_np[include_mask], shifts)
        plt.show()
    mean_shift = np.mean(shifts)

    zero_freq = mean_shift * map_in.cell.volume  # type: ignore
    logger.info(
        f"Shift value {mean_shift:.5f} corresponds to zero frequency {zero_freq:.3f}"
    )

    map_in.loc[(0, 0, 0)] = {
        map_in.amplitude_column_name: zero_freq,
        map_in.phase_column_name: 0,
        map_in.uncertainties_column_name: zero_freq / 10,
    }
    # map_in.infer_mtz_dtypes(inplace=True)
    # map_in.sort_index()
    map_in.write_mtz("autoshifted_map.mtz")

    return map_in, zero_freq


def prepare_maps(unscaled_dark, unscaled_triggered, config):

    struc = gemmi.read_pdb(config["input_files"]["pdb_dark"])
    check_highres_limit(unscaled_dark, unscaled_triggered, config["input_files"])
    map_dark_comp = gemmi_structure_to_calculated_map(
        struc, high_resolution_limit=config["general"]["high_resolution_limit"]
    )

    map_dark = scale_maps(reference_map=map_dark_comp, map_to_scale=unscaled_dark)
    map_triggered = scale_maps(
        reference_map=map_dark_comp, map_to_scale=unscaled_triggered
    )

    if config["map_processing"]["dark_mean_correction"]:
        diffmap_temp = combined_diffmap_calc(
            map_dark,
            map_triggered,
            map_dark_comp,
            diffmap_type=config["map_processing"]["diffmap_type"],
            general_config=config["general"],
        )
        diffmap_temp_np = diffmap_temp.to_3d_numpy_map(
            map_sampling=config["general"]["map_sampling"]
        )
        diffmap_larger = np.abs(diffmap_temp_np) > 1 * diffmap_temp_np.std()
        logger.info(f"Diffmap std: {diffmap_temp_np.std():.3f}")
        logger.info(
            f"diffmap larger voxel count: {np.sum(diffmap_larger)/diffmap_larger.size}"
        )

        map_dark, zero_freq_dark = autoshift_rsmap(
            map_dark,
            map_dark_comp,
            config["general"],
        )
        logger.info("calculating autoshift for triggered map... with extra mask")
        map_triggered, zero_freq_triggered = autoshift_rsmap(
            map_triggered, map_dark_comp, config["general"], diffmap_larger
        )
        logger.info("calculating autoshift for triggered map... done")

    if not config["map_processing"]["diffmap_v2_correction"]:
        diffmap = combined_diffmap_calc(
            map_dark,
            map_triggered,
            map_dark_comp,
            diffmap_type=config["map_processing"]["diffmap_type"],
            general_config=config["general"],
        )
    else:
        diffmap = diffmap_temp
    logger.info(f"Diffmap zero frequency: {diffmap.loc[(0,0,0)]['F']}")
    if config["map_processing"]["diffmap_mean_correction"]:
        zero_freq_diff = zero_freq_triggered - zero_freq_dark
        zero_uncertainty = np.sqrt(
            (zero_freq_dark * 0.1) ** 2 + (zero_freq_triggered * 0.1) ** 2
        )

        diffmap.loc[(0, 0, 0)] = {
            diffmap.amplitude_column_name: zero_freq_diff,
            diffmap.phase_column_name: 0,
            diffmap.uncertainties_column_name: zero_uncertainty,
        }
        logger.warning(f"Diffmap zero frequency: {zero_freq_diff}")

    return diffmap, map_dark, map_triggered
