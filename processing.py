import numpy as np
import gemmi
import reciprocalspaceship as rs
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from pathlib import Path
import copy


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


def convert_ints_to_sf2(
    ds: rs.DataSet, cols: dict, map_dark_comp: rsmap.Map
) -> rs.DataSet:
    ds_ints = ds[cols["ints_column"]]
    ds_sigi = ds[cols["int_uncertainty_column"]]
    ds[cols["amplitude_column"]] = np.sqrt(np.abs(ds_ints)) * np.sign(ds_ints)
    ds[cols["uncertainty_column"]] = np.sqrt(ds_sigi / (2 * np.abs(ds_ints)))
    ds[cols["phase_column"]] = map_dark_comp.phases
    cols.pop("ints_column")
    cols.pop("int_uncertainty_column")
    return ds


def get_maps(input_files_dict: dict) -> tuple[rsmap.Map, rsmap.Map]:
    dataloc_dark = input_files_dict["input_files"]["map_dark"]
    dataloc_triggered = input_files_dict["input_files"]["map_triggered"]
    ds_triggered = rs.read_mtz(dataloc_triggered)
    ds_dark = rs.read_mtz(dataloc_dark)

    if input_files_dict["input_files"]["columns_are_ints"]:
        dark_cols = input_files_dict["input_files"]["columns_dark"]
        triggered_cols = input_files_dict["input_files"]["columns_triggered"]
        struc = gemmi.read_pdb(input_files_dict["input_files"]["pdb_dark"])
        map_dark_comp = gemmi_structure_to_calculated_map(
            struc,
            high_resolution_limit=input_files_dict["general"]["high_resolution_limit"],
        )
        ds_dark = convert_ints_to_sf2(ds_dark, dark_cols, map_dark_comp)
        ds_triggered = convert_ints_to_sf2(ds_triggered, triggered_cols, map_dark_comp)
    return get_maps_sf(ds_dark, ds_triggered, input_files_dict)


def get_maps_sf(
    ds_dark, ds_triggered, input_files_dict: dict
) -> tuple[rsmap.Map, rsmap.Map]:
    high_res_limit = input_files_dict["general"]["high_resolution_limit"]

    if high_res_limit:
        logger.info(f"Imposing high_resolution_limit: {high_res_limit}")
        ds_dark = cut_resolution(ds_dark, high_resolution_limit=high_res_limit)
        ds_triggered = cut_resolution(
            ds_triggered, high_resolution_limit=high_res_limit
        )

    dark_columns = input_files_dict["input_files"]["columns_dark"]
    triggered_columns = input_files_dict["input_files"]["columns_triggered"]
    if input_files_dict["input_files"]["impose_dark_phases"]:
        ds_triggered.loc[:, triggered_columns["phase_column"]] = ds_dark[
            dark_columns["phase_column"]
        ]
    unscaled_dark = rsmap.Map(ds_dark, **dark_columns)
    unscaled_triggered = rsmap.Map(ds_triggered, **triggered_columns)
    return unscaled_dark, unscaled_triggered


def check_highres_limit(
    map_dark: rsmap.Map, map_triggered: rsmap.Map, general_config: dict
):
    dmin_dark = map_dark.compute_dHKL().min()
    dmin_triggered = map_triggered.compute_dHKL().min()
    high_res_limit = np.round(max(dmin_dark, dmin_triggered), 1)

    if not np.isclose(dmin_dark, dmin_triggered):
        logger.warning(
            f"Different resolution limits in dark and triggered maps: {dmin_dark:.2f} A vs {dmin_triggered:.2f} A"
        )
        general_config["high_resolution_limit"] = high_res_limit
        map_dark = cut_resolution(map_dark, high_resolution_limit=high_res_limit)  # type: ignore
        map_triggered = cut_resolution(
            map_triggered, high_resolution_limit=high_res_limit
        )  # type: ignore

    if not np.isclose(high_res_limit, general_config["high_resolution_limit"]):
        prev_dmin = general_config["high_resolution_limit"]
        logger.warning(
            f"Changing high-resolution limit from {prev_dmin:.2f} A to {high_res_limit:.2f} A"
        )
        general_config["high_resolution_limit"] = high_res_limit
    return map_dark, map_triggered


def calculate_diffmaps(
    map_dark: rsmap.Map,
    map_triggered: rsmap.Map,
    map_dark_comp: rsmap.Map,
    meta_loc: str = "",
    only_kweighted: bool = False,
    parameters: dict = {},
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
                logger.warning("Loaded meta from file:")
            # Extract the optimal parameters
            opt_k = (
                meta.k_parameter_optimization.optimal_parameter_value
                if meta.k_parameter_optimization
                else None
            )
            opt_tv = meta.tv_weight_optimization.optimal_parameter_value
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
    name = f"{general_config['high_resolution_limit']*10}"
    return evaluation_path_basis, name


def get_meta_loc_diffmap(general_config, processing_config):
    binary_string = processing_dict_2_binary(processing_config)
    evaluation_path_basis, name = get_meta_loc(general_config)
    name = f"diffmap_config_{general_config['high_resolution_limit']*10}_{binary_string}.pkl"
    meta_loc = evaluation_path_basis + name
    return meta_loc


def combined_diffmap_calc(
    map_dark,
    map_triggered,
    map_dark_comp,
    processing_config: dict,
    general_config=None,
    allow_saving=False,
) -> rsmap.Map:
    diffmap_type = processing_config["diffmap_type"]
    filepath = Path(diffmap_file_name(processing_config, general_config))
    if (
        filepath.exists()
        and (Path().stat().st_mtime - filepath.stat().st_mtime) < 24 * 3600
    ):
        logger.info(f"Loading preprocessed maps from {filepath}")
        diffmap = rsmap.Map.read_mtz_file(filepath)
        return diffmap
    else:
        logger.info(
            f"No recent preprocessed diffmap found at {filepath}, calculating diffmap..."
        )

    meta_loc = get_meta_loc_diffmap(general_config, processing_config)

    match diffmap_type:
        case "kweighted":
            diffmap = calculate_diffmaps(
                map_dark, map_triggered, map_dark_comp, meta_loc, only_kweighted=True
            )
        case "tv":
            diffmap = calculate_diffmaps(
                map_dark, map_triggered, map_dark_comp, meta_loc, only_kweighted=False
            )
        case "vanilla":
            diffmap = compute_difference_map(derivative=map_triggered, native=map_dark)
        case _:
            raise ValueError(f"Unknown diffmap_type: {diffmap_type}")
            # logger.warning(
            #     f"Unknown or unset diffmap_type: {diffmap_type}, defaulting to vanilla"
            # )
            diffmap = compute_difference_map(derivative=map_triggered, native=map_dark)
    if allow_saving:
        diffmap.write_mtz(filepath)
        logger.info(f"Saved diffmap to {filepath}")
    return diffmap


def autoshift_rsmap_old(
    map_in: rsmap.Map,
    general_config: dict,
    map_dark_comp: rsmap.Map | None = None,
    ignore_mask: np.ndarray | bool = False,
    diagnostic_plots: bool = False,
) -> tuple[rsmap.Map, float]:
    map_sampling = general_config["map_sampling"]
    pdbloc_dark = general_config["pdbloc_dark"]

    if map_dark_comp is None:
        struc = gemmi.read_pdb(pdbloc_dark)
        map_dark_comp = gemmi_structure_to_calculated_map(
            struc, high_resolution_limit=general_config["high_resolution_limit"]
        )

    rsmap_np = map_in.to_3d_numpy_map(map_sampling=map_sampling)
    map_dark_comp_np = map_dark_comp.to_3d_numpy_map(map_sampling=map_sampling)

    only_solvent = support_from_masker(pdbloc_dark, map_dark_comp_np.shape)
    ignore_mask = np.logical_or(only_solvent, ignore_mask)
    logger.info(f"ignore_mask voxel count: {np.sum(ignore_mask) / ignore_mask.size}")

    include_mask = ~ignore_mask
    if ignore_mask.all():
        logger.warning("All voxels are ignored in autoshift; no shift applied.")
        return map_in, 0
    shifts = map_dark_comp_np[include_mask] - rsmap_np[include_mask]
    if diagnostic_plots:
        plt.figure()
        plt.plot(map_dark_comp_np[include_mask], shifts, ".", alpha=0.5)
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


# ==========================================
# HELPER FUNCTIONS
# ==========================================


def generate_masks(
    pdb_file: str, grid_shape: tuple, cell: gemmi.UnitCell, spacegroup: gemmi.SpaceGroup
):
    """
    Generates binary masks for the protein and the bulk solvent using gemmi.
    Returns:
        protein_mask (bool array): True inside the macromolecule.
        solvent_mask (float32 array): 1.0 in the solvent, 0.0 inside the protein.
    """
    st = gemmi.read_structure(pdb_file)
    st.setup_entities()
    model = st[0]
    # model.remove_waters()

    mask_grid = gemmi.Int8Grid()
    mask_grid.set_unit_cell(cell)
    mask_grid.spacegroup = spacegroup

    nx, ny, nz = map(int, grid_shape)
    mask_grid.set_size(nx, ny, nz)

    masker = gemmi.SolventMasker(gemmi.AtomicRadiiSet.VanDerWaals)
    masker.rprobe = 1.4
    masker.put_mask_on_int8_grid(mask_grid, model)

    # Gemmi returns 1 for protein, 0 for solvent.
    protein_mask = ~np.array(mask_grid, copy=False).astype(bool)

    # Invert for the solvent mask (1.0 in solvent)
    solvent_mask = (~protein_mask).astype(np.float32)

    return protein_mask, solvent_mask


def error_metric_for_scaling(
    map_temp: rsmap.Map, map_exp: rsmap.Map, dmin: float = 3.0
):
    """
    Calculates the mean absolute error between the scaled model and
    experimental amplitudes at low resolution (where bulk solvent is active).
    """
    d_spacings = map_temp.compute_dHKL()
    low_res_idx = d_spacings > dmin

    # Calculate optimal scaling using meteor's internal function
    map_scaled = scale_maps(reference_map=map_temp, map_to_scale=map_exp)

    # Compute Mean Absolute Error
    absdiff = np.abs((map_scaled.amplitudes - map_temp.amplitudes)[low_res_idx])
    # absdiff = np.abs(f_obs - f_model_scaled)
    return np.mean(absdiff)


# ==========================================
# CORE CALCULATION FUNCTIONS
# ==========================================


def calculate_rho_atom(
    map_exp_np: np.ndarray, map_model_np: np.ndarray, protein_mask: np.ndarray
):
    """
    Calculates the mean density difference strictly within the protein region.
    This effectively calculates the 'rho_atom' absolute zero-frequency offset.
    """
    # Isolate voxels inside the protein mask
    shifts = map_model_np[protein_mask] - map_exp_np[protein_mask]

    mean_shift = np.mean(shifts)
    return mean_shift


def calculate_rho_bulk(
    map_exp: rsmap.Map,
    map_model_np: np.ndarray,
    solvent_mask: np.ndarray,
    cell: gemmi.UnitCell,
    spacegroup: gemmi.SpaceGroup,
    hs_limit: float,
    plot: bool = False,
):
    """
    Optimizes rho_bulk to minimize low-resolution differences between the model
    (plus the flat bulk solvent mask) and the experimental data.
    """

    def target_scaling_function(rho_bulk):
        # 1. Add the scaled solvent mask to the base protein model
        temp_data = map_model_np + (rho_bulk * solvent_mask)

        # 2. Convert to reciprocal map
        map_temp = rsmap.Map.from_3d_numpy_map(
            temp_data, cell=cell, spacegroup=spacegroup, high_resolution_limit=hs_limit
        ) # type: ignore

        # 3. Align indices to experimental map
        shared_indices = map_temp.index.intersection(map_exp.index)
        map_temp = map_temp.loc[shared_indices]

        # 4. Return the low-resolution scaling error
        return error_metric_for_scaling(map_temp, map_exp)

    print("Running 1D bounded optimization for rho_bulk...")

    # Use minimize_scalar for robust 1D valley-finding
    result = minimize_scalar(
        target_scaling_function,
        bounds=(0.2, 0.5),
        method="bounded",
        options={"xatol": 1e-4},
    )

    best_rho_bulk = result.x
    print(f"Optimal rho_bulk: {best_rho_bulk:.4f} e-/Å³")

    # Optional plotting of the optimization landscape
    if plot:
        rho_bulks = np.linspace(0.2, 0.5, 30)
        errors = [target_scaling_function(b) for b in rho_bulks]

        plt.figure(figsize=(8, 5))
        plt.plot(rho_bulks, errors, marker="o", linestyle="-")
        plt.xlabel("Bulk Density (e-/Å³)")
        plt.ylabel("Mean Absolute Difference")
        plt.title("Error Metric vs. Bulk Density")
        plt.grid(True, alpha=0.3)
        plt.axvline(
            best_rho_bulk,
            color="red",
            linestyle="--",
            label=f"Min: {best_rho_bulk:.3f}",
        )
        plt.legend()
        plt.show()

    return best_rho_bulk


# ==========================================
# MASTER PIPELINE
# ==========================================


def estimate_absolute_densities(
    map_exp: rsmap.Map,
    map_model: rsmap.Map,
    pdb_file: str,
    hs_limit: float,
    plot: bool = True,
):
    """
    Master function orchestrating the calculation of both rho_atom (protein offset)
    and rho_bulk (bulk solvent density).
    """
    print(f"--- Calibrating Absolute Densities for {pdb_file} ---")

    # 1. Extract metadata and numpy representations
    cell: gemmi.UnitCell = map_model.cell  # type: ignore
    spacegroup: gemmi.SpaceGroup = map_model.spacegroup  # type: ignore
    map_model_np = map_model.to_3d_numpy_map(map_sampling=3)
    map_exp_np = map_exp.to_3d_numpy_map(map_sampling=3)
    grid_shape = map_model_np.shape

    # 2. Generate Masks
    print("Generating protein and solvent masks...")
    protein_mask, solvent_mask = generate_masks(pdb_file, grid_shape, cell, spacegroup)
    # protein_mask = ~support_from_masker(pdb_file, map_dark_comp_np.shape)

    # 3. Calculate rho_atom (Mean shift in the protein region)
    rho_atom_shift = calculate_rho_atom(map_exp_np, map_model_np, protein_mask)

    print(
        f"Estimated rho_atom offset (mean shift inside protein): {rho_atom_shift:.5f}"
    )
    rho_atom = map_model_np.mean()

    # 4. Calculate rho_bulk (Optimized solvent density)
    rho_bulk = calculate_rho_bulk(
        map_exp=map_exp,
        map_model_np=map_model_np,
        solvent_mask=solvent_mask,
        cell=cell,
        spacegroup=spacegroup,
        hs_limit=hs_limit,
        plot=plot,
    )
    share_solvent = np.mean(solvent_mask)
    rho_comb = rho_atom + share_solvent * rho_bulk
    f000 = rho_comb * cell.volume

    logger.info("\n--- Calibration Complete ---")
    return {
        "rho_abs_shift": rho_atom_shift,
        "rho_atom": rho_atom,
        "rho_bulk": rho_bulk,
        "rho_comb": rho_comb,
        "share_solvent": share_solvent,
        "f000": f000,
    }


def autoshift_rsmap(
    map_in: rsmap.Map,
    general_config: dict,
    map_dark_comp: rsmap.Map | None = None,
    ignore_mask: np.ndarray | bool = False,
    diagnostic_plots: bool = False,
) -> tuple[rsmap.Map, float]:
    if map_dark_comp is None:
        struc = gemmi.read_pdb(general_config["pdbloc_dark"])
        map_dark_comp = gemmi_structure_to_calculated_map(
            struc,
            high_resolution_limit=general_config["high_resolution_limit"],
            map_sampling=general_config["map_sampling"],
        )
    estimates = estimate_absolute_densities(
        map_in,
        map_dark_comp,
        general_config["pdbloc_dark"],
        general_config["high_resolution_limit"],
        plot=diagnostic_plots,
    )
    if np.abs(estimates["rho_abs_shift"] - estimates["rho_comb"]) > 0.01:
        log_txt = f"Estimated rho_atom shift ({estimates['rho_abs_shift']:.4f}) "
        log_txt += f"and combined rho (rho_atom + share_solvent * rho_bulk) ({estimates['rho_comb']:.4f} "
        log_txt += f"{estimates['rho_atom']:.4f}+{estimates['share_solvent']:.4f}*{estimates['rho_bulk']:.4f}) "
        log_txt += "differ by more than 0.01 e-/Å³. This may indicate an issue with the estimation or the maps."
        logger.warning(log_txt)
    else:
        log_txt = f"Estimated rho_atom shift ({estimates['rho_abs_shift']:.4f}) "
        log_txt += f"and combined rho (rho_atom + share_solvent * rho_bulk) ({estimates['rho_comb']:.4f} "
        log_txt += f"{estimates['rho_atom']:.4f}+{estimates['share_solvent']:.4f}*{estimates['rho_bulk']:.4f}) "
        logger.info(log_txt)

    zero_freq = estimates["f000"]
    map_in.loc[(0, 0, 0)] = {
        map_in.amplitude_column_name: zero_freq,
        map_in.phase_column_name: 0,
        map_in.uncertainties_column_name: zero_freq / 10,
    }
    map_in.write_mtz("autoshifted_map.mtz")
    return map_in, zero_freq


def processing_dict_2_binary(processing_dict) -> str:
    key_names = [k for k in processing_dict.keys() if k not in ["diffmap_type"]]
    key_names = sorted(key_names)
    binary_string = ""

    for key in key_names:
        binary_string += "1" if processing_dict[key] else "0"
    if len(binary_string) != 4:
        raise ValueError(
            f"Expected 4 entries, got {len(binary_string)}. Keys were: {key_names}"
        )
    return binary_string


def diffmap_file_name(processing_config, general_config):
    binary_string = processing_dict_2_binary(processing_config)
    evaluation_path_basis, name = get_meta_loc(general_config)
    diffmap_type = processing_config["diffmap_type"]
    return evaluation_path_basis + f"diffmap_{name}_{diffmap_type}_{binary_string}.mtz"


def prepare_maps(
    unscaled_dark: rsmap.Map, unscaled_triggered: rsmap.Map, config: dict
) -> tuple[rsmap.Map, rsmap.Map, rsmap.Map]:

    struc = gemmi.read_pdb(config["input_files"]["pdb_dark"])
    check_highres_limit(unscaled_dark, unscaled_triggered, config["general"])
    map_dark_comp = gemmi_structure_to_calculated_map(
        struc,
        high_resolution_limit=config["general"]["high_resolution_limit"],
        map_sampling=config["general"]["map_sampling"],
    )

    map_dark = scale_maps(reference_map=map_dark_comp, map_to_scale=unscaled_dark)
    map_triggered = scale_maps(
        reference_map=map_dark_comp, map_to_scale=unscaled_triggered
    )

    if config["map_processing"]["dark_mean_correction"]:
        if True:
            diffmap_temp = combined_diffmap_calc(
                map_dark,
                map_triggered,
                map_dark_comp,
                processing_config=config["map_processing"] | {"preprocessing": True},
                general_config=config["general"],
                allow_saving=False,
            )
            diffmap_temp_np = diffmap_temp.to_3d_numpy_map(
                map_sampling=config["general"]["map_sampling"]
            )
            diffmap_larger = np.abs(diffmap_temp_np) > 1 * diffmap_temp_np.std()
            logger.info(f"Diffmap std: {diffmap_temp_np.std():.3f}")
            logger.info(
                f"diffmap larger voxel count: {np.sum(diffmap_larger)/diffmap_larger.size}"
            )
            map_dark, zero_freq_dark = autoshift_rsmap_old(
                map_dark, config["general"], map_dark_comp
            )
            logger.info("calculating autoshift for triggered map... with extra mask")
            map_triggered, zero_freq_triggered = autoshift_rsmap_old(
                map_triggered, config["general"], map_dark_comp, diffmap_larger
            )
            logger.info("calculating autoshift for triggered map... done")
        else:
            map_dark, zero_freq_dark = autoshift_rsmap(
                map_dark, config["general"], map_dark_comp
            )
            logger.info("calculating autoshift for triggered map... with extra mask")
            map_triggered, zero_freq_triggered = autoshift_rsmap(
                map_triggered,
                config["general"],
                map_dark_comp,  # diffmap_larger
            )
            logger.info("calculating autoshift for triggered map... done")
            diffmap_temp = None
    else:
        zero_freq_dark = None
        zero_freq_triggered = None
        diffmap_temp = None

    if not config["map_processing"]["diffmap_v2_correction"]:
        diffmap = combined_diffmap_calc(
            map_dark,
            map_triggered,
            map_dark_comp,
            processing_config=config["map_processing"],
            general_config=config["general"],
            allow_saving=True,
        )
    else:
        if config["map_processing"]["dark_mean_correction"]:
            if diffmap_temp is None:
                raise ValueError("Diffmap temp not calculated but needed for dark mean correction")
            diffmap = diffmap_temp  
        else:
            raise ValueError("Diffmap not defined")
    try:
        logger.info(f"Diffmap zero frequency: {diffmap.loc[(0,0,0)]['F']}")  # type: ignore
    except KeyError:
        logger.info("Diffmap zero frequency was not set")
    if config["map_processing"]["diffmap_mean_correction"]:
        if config["map_processing"]["dark_mean_correction"]:
            zero_freq_diff = zero_freq_triggered - zero_freq_dark  # type: ignore
            zero_uncertainty = np.sqrt(
                (zero_freq_dark * 0.1) ** 2 + (zero_freq_triggered * 0.1) ** 2  # type: ignore
            )
        else:
            raise ValueError(
                "Diffmap Correction can only be done if dark correction is done"
            )

        diffmap.loc[(0, 0, 0)] = {
            diffmap.amplitude_column_name: zero_freq_diff,
            diffmap.phase_column_name: 0,
            diffmap.uncertainties_column_name: zero_uncertainty,
        }
        logger.warning(f"Diffmap zero frequency: {zero_freq_diff}")

    return diffmap, map_dark, map_triggered


def prepare_maps_v2(
    unscaled_dark: rsmap.Map, unscaled_triggered: rsmap.Map, config: dict
) -> tuple[rsmap.Map, rsmap.Map, rsmap.Map]:

    general_config = config["general"]
    struc = gemmi.read_pdb(config["input_files"]["pdb_dark"])
    check_highres_limit(unscaled_dark, unscaled_triggered, general_config)
    map_dark_comp = gemmi_structure_to_calculated_map(
        struc,
        high_resolution_limit=general_config["high_resolution_limit"],
        map_sampling=general_config["map_sampling"],
    )

    map_dark = scale_maps(reference_map=map_dark_comp, map_to_scale=unscaled_dark)
    map_triggered = scale_maps(
        reference_map=map_dark_comp, map_to_scale=unscaled_triggered
    )

    diffmap_temp = combined_diffmap_calc(
        map_dark,
        map_triggered,
        map_dark_comp,
        processing_config=config["map_processing"] | {"preprocessing": True},
        general_config=config["general"],
        allow_saving=False,
    )
    diffmap_temp_np = diffmap_temp.to_3d_numpy_map(
        map_sampling=general_config["map_sampling"]
    )
    diffmap_larger = np.abs(diffmap_temp_np) > 1 * diffmap_temp_np.std()
    logger.info(f"Diffmap std: {diffmap_temp_np.std():.3f}")
    lgtxt = f"diffmap larger voxel count: {np.sum(diffmap_larger)/diffmap_larger.size}"
    logger.info(lgtxt)
    map_dark, zero_freq_dark = autoshift_rsmap_old(
        map_dark, config["general"], map_dark_comp
    )
    logger.info("calculating autoshift for triggered map... with extra mask")
    map_triggered, zero_freq_triggered = autoshift_rsmap_old(
        map_triggered, config["general"], map_dark_comp, diffmap_larger
    )
    logger.info("calculating autoshift for triggered map... done")

    if not config["map_processing"]["diffmap_v2_correction"]:
        diffmap = combined_diffmap_calc(
            map_dark,
            map_triggered,
            map_dark_comp,
            processing_config=config["map_processing"],
            general_config=config["general"],
            allow_saving=True,
        )
    else:
        if config["map_processing"]["dark_mean_correction"]:
            diffmap = diffmap_temp  # type: ignore
        else:
            raise ValueError("Diffmap not defined")

    if config["map_processing"]["diffmap_mean_correction"]:
        if config["map_processing"]["dark_mean_correction"]:
            zero_freq_diff = zero_freq_triggered - zero_freq_dark  # type: ignore
            zero_uncertainty = np.sqrt(
                (zero_freq_dark * 0.1) ** 2 + (zero_freq_triggered * 0.1) ** 2  # type: ignore
            )
        else:
            raise ValueError(
                "Diffmap Correction can only be done if dark correction is done"
            )

        diffmap.loc[(0, 0, 0)] = {
            diffmap.amplitude_column_name: zero_freq_diff,
            diffmap.phase_column_name: 0,
            diffmap.uncertainties_column_name: zero_uncertainty,
        }
        logger.warning(f"Diffmap zero frequency: {zero_freq_diff}")

    return diffmap, map_dark, map_triggered


def get_map_dark(config, old=False):
    ds_dark = rs.read_mtz(config["input_files"]["map_dark"])
    print(np.min(ds_dark.compute_dHKL()["dHKL"]))
    struc_dark = gemmi.read_pdb(config["input_files"]["pdb_dark"])
    map_dark_comp = gemmi_structure_to_calculated_map(
        struc_dark,
        high_resolution_limit=config["general"]["high_resolution_limit"],
    )
    if config["input_files"]["columns_dark"]["phase_column"] not in ds_dark.columns:
        phase_col_name = config["input_files"]["columns_dark"]["phase_column"]
        warning_msg = f"Phase column {phase_col_name} not found in dark dataset. "
        warning_msg += "Using calculated phases from the PDB structure."
        logger.warning(warning_msg)
        ds_dark[phase_col_name] = map_dark_comp.phases
    map_dark = rsmap.Map(ds_dark, **config["input_files"]["columns_dark"])
    map_dark.canonicalize_amplitudes()
    map_dark = cut_resolution(
        map_dark, high_resolution_limit=config["general"]["high_resolution_limit"]
    )
    map_dark = scale_maps(reference_map=map_dark_comp, map_to_scale=map_dark)
    if old:
        map_dark, shift2 = autoshift_rsmap_old(
            copy.deepcopy(map_dark),
            config["general"] | {"pdbloc_dark": config["input_files"]["pdb_dark"]},
            map_dark_comp,
        )
    else:
        map_dark, shift1 = autoshift_rsmap(
            copy.deepcopy(map_dark),
            config["general"] | {"pdbloc_dark": config["input_files"]["pdb_dark"]},
            map_dark_comp,
        )
    return map_dark


def get_maps_diff(config, map_dark=None, old=False):
    if map_dark is None:
        map_dark = get_map_dark(config, old)
    else:
        map_dark = copy.deepcopy(map_dark)

    ds_diff = rs.read_mtz(config["input_files"]["map_diff"])
    diffmap = rsmap.Map(ds_diff, **config["input_files"]["columns_diff"])
    compute_dHKL = ds_diff.compute_dHKL()
    dmin = np.min(compute_dHKL["dHKL"])
    map_dark = cut_resolution(map_dark, high_resolution_limit=dmin)
    return map_dark, diffmap
