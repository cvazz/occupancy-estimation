from systematic_plots import load_inputs, load_homepath, calculate_objects
from systematic_plots import run_plots, rescaling_diffmaps, load_cistrans_paths
import numpy as np
import gemmi
import time
import reciprocalspaceship as rs

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from scipy import stats
from scipy.signal import correlate

from logger import setup_logger

logger = setup_logger()


##################### relevant if in meteor-dev mode ###########################
import os
import sys

current_path = os.getcwd()
path_parts = current_path.split(os.sep)
idx = path_parts.index("occupancy-estimation")
homepath = os.sep.join(path_parts[: idx + 1]) + "/"
path = homepath + "../meteor/"
sys.path.append(path)
################################ end if ########################################

import meteor
from meteor import rsmap
from meteor.utils import cut_resolution
from meteor.scale import compute_scale_factors

from photolyase import get_hists, find_wasserstein_dip
from plotting3d import slice_3d, add_fit

from photolyase import load_mask_config, load_masks, make_diffmap_config
from meteorize import (
    find_most_positive_blobs_np,
    find_most_positive_blobs_rmsd,
    find_most_positive_blobs_fixed_basis,
    make_k_space_xtr,
    find_largest_blobs2,
    negsum_meteor,
    many_negsum,
    get_photolyase_maps,
    adding_maps,
)


def load_defaults_cistrans():
    filename_dict = {
        "save_fig": True,
        "display": False,
        "rerun": True,
        "rerun_old_only": False,
        "rescaling_diffmaps": False,
    }

    defaults_to_overwrite = {
        "mid_xtr_factor": 8,
        "max_xtr_factor": 40,
        "minimum_threshold": 0.25,
        "step_threshold": 0.2,
    }

    function_selection = [
        "single_negsum_overview",
        "many_negsum_best_guess",
        "sigma_test",
        "single_negsum_model",
        "many_negsum_many_thresh",
        "show_comparison",
    ]

    info_container = load_cistrans_paths()[0]
    info_container = defaults_to_overwrite | info_container
    return filename_dict, function_selection, info_container


def shift_offset(map_conv, map_sampling=3, set_val=None):
    map_conv.to_3d_numpy_map(map_sampling=map_sampling)

    if set_val is None:
        map_np = map_conv.to_3d_numpy_map(map_sampling=map_sampling)
        set_val = -map_conv.cell.volume * np.min(map_np)
        # set_val = np.nan
        print(set_val, map_np.shape)
    zero_freq = {
        map_conv._amplitude_column: set_val,
        map_conv._phase_column: 0,
    }
    if map_conv.has_uncertainties:
        zero_freq[map_conv._uncertainty_column] = 1
    # map_sf = map_conv.to_structure_factors()

    # Or loop through all index levels
    for i, name in enumerate(map_conv.index.names):
        print(f"Index level {i} ({name}): {map_conv.index.get_level_values(i).dtype}")
    idx_loc = (0, 0, 0)
    map_conv.loc[idx_loc] = zero_freq
    print(zero_freq)
    # sort index after adding new A
    map_conv.sort_index(inplace=True)
    from reciprocalspaceship.dtypes.integer import HKLIndexDtype

    map_conv.index = map_conv.index.set_levels(
        [
            map_conv.index.levels[0].astype(HKLIndexDtype),
            map_conv.index.levels[1].astype(HKLIndexDtype),
            map_conv.index.levels[2].astype(HKLIndexDtype),
        ]
    )
    print(map_conv.head(5))

    return map_conv


import pandas as pd
# def shift_offset(map_conv, map_sampling=3, set_val=None):
#     if set_val is None:
#         map_np = map_conv.to_3d_numpy_map(map_sampling=map_sampling)
#         set_val = np.nan

#     # Create a single-entry DataSet for (0,0,0)
#     zero_data = {
#         map_conv._amplitude_column: [set_val],
#         map_conv._phase_column: [0.0]
#     }

#     if map_conv.has_uncertainties:
#         zero_data[map_conv._uncertainty_column] = [1.0]

#     # Create with proper HKL index
#     hkl_index = pd.MultiIndex.from_tuples(
#         [(0, 0, 0)],
#         names=['H', 'K', 'L']
#     )

#     zero_dataset = rs.DataSet(zero_data, index=hkl_index)

#     # Combine with existing dataset
#     combined = pd.concat([map_conv, zero_dataset])
#     combined.sort_index(inplace=True)

#     return combined


# def loading_diffmaps(
#     map_light: rsmap.Map,
#     map_dark: rsmap.Map,
#     map_sampling: float,
#     path: str,
#     hs_limit: float,
#     force_compute=False,
# ):
#     """
#     Loads the difference maps for the dark and light datasets.
#     """
#     diffmap_config = make_diffmap_config(map_sampling)
#     map_types = [
#         # "direct_realspace",
#         "vanilla_diffmap",
#         "kweighted",
#         "tv",
#     ]

#     mtz_name = f"{path}diffmaps{hs_limit*10:.0f}.mtz"
#     diffmaps = {}
#     if not os.path.exists(mtz_name) or force_compute:
#         logger.info(f"Calculating from maps")
#         for key in map_types:
#             config = diffmap_config[key]
#             loader = config["loader"]
#             kwargs = config.get("kwargs", {})
#             diffmaps[key] = loader(map_light, map_dark, **kwargs)

#         diffmaps_mtz = rs.DataSet(diffmaps["tv"])
#         diffmaps_mtz = diffmaps_mtz.rename(
#             columns=lambda x: f"{x}_tv" if x != "index" else x
#         )
#         for key in diffmaps.keys():
#             if key != "tv":
#                 for col in diffmaps[key].columns:
#                     if col != "index":
#                         diffmaps_mtz[f"{col}_{key}"] = diffmaps[key][col]
#         rs.DataSet(diffmaps_mtz).write_mtz(mtz_name)

#     else:
#         logger.info(f"Reading from {mtz_name}")
#         diffmaps_mtz = rs.read_mtz(mtz_name)
#         for key in map_types:
#             diffmap = rsmap.Map(
#                 diffmaps_mtz,
#                 amplitude_column=f"F_{key}",
#                 phase_column=f"PHI_{key}",
#                 uncertainty_column=f"SigF_{key}",
#             )
#             diffmaps[key] = diffmap
#     return diffmaps, diffmap_config
from systematic_plots import loading_diffmaps

from generate_objects import generate_obj_cistrans_v3


def main(
    occupancies=[0.2, 0.5, 0.8],
    true_phases_list=[True, False],
    offsets=[True, False],
    # occupancies = [0.2],
    # true_phases_list = [True],
    # offsets = [True],
    f_noises=[
        0.0,
    ],
    phi_noises=[0.0],
    percentiles=[0.1, 0.01],
):
    filename_dict, function_selection, info_container = load_defaults_cistrans()
    constants = {"high_resolution_limit": 2}
    rescale_key = "vanilla_diffmap"
    blob_selection_func = find_most_positive_blobs_fixed_basis

    evaluation_path = load_homepath() + "../evaluation/cistrans/"
    if not os.path.exists(evaluation_path):
        os.makedirs(evaluation_path)

    # iterate over carthesian produc of parameters
    import itertools

    iter_product = itertools.product(
        occupancies, true_phases_list, offsets, f_noises, phi_noises, percentiles
    )
    value_list = []
    for occupancy, true_phases, offset, f_noise, phi_noise, percentile in iter_product:
        values = {
            "f_noise": f_noise,
            "phi_noise": phi_noise,
            "alpha": occupancy,
            "offset": offset,
            "true_phases": true_phases,
            "percentile": percentile,
        }
        info_container = values | info_container
        info_container["mid_xtr_factor"] = 1 / occupancy + 0.3

        map_dark, map_light = generate_obj_cistrans_v3(
            occupancy,
            f_noise=f_noise,
            phi_noise=phi_noise,
            hs_limit=constants["high_resolution_limit"],
        )
        if offset:
            map_dark = shift_offset(map_dark)
            map_light = shift_offset(
                map_light
            )  # map_dark.loc[(0,0,0),map_dark._amplitude_column])
        print(map_dark.head(4))
        print(map_light.head(4))
        from meteor.diffmaps import compute_difference_map

        if not true_phases:
            map_light["PHI"] = map_dark["PHI"]

        phase_str = "true_phases" if true_phases else "dark_phases"
        offset_str = "offset" if offset else "nooffset"
        identifier = f"occ_{occupancy * 100:.0f}_{phase_str}_fnoise_{f_noise * 100:.0f}_phinoise_{phi_noise * 100:.0f}_{offset_str}_percentile_{percentile:.4f}"
        tname = f"Occupancy : {occupancy:.2f}, {'True' if true_phases else 'Dark'} phases, {'With' if offset else 'No'} offset"
        info_container["tname"] = tname
        evaluation_path_basis = evaluation_path + identifier + "/"

        diffmap = compute_difference_map(map_light, map_dark, check_isomorphous=False)
        # diffmap = rsmap.Map.from_3d_numpy_map(
        #     map_light.to_3d_numpy_map(map_sampling=3) - map_dark.to_3d_numpy_map(map_sampling=3),
        #     cell = map_light.cell,
        #     spacegroup = map_light.spacegroup,
        #     high_resolution_limit = constants["high_resolution_limit"],
        # )
        if not true_phases:
            diffmap.F *= 2

        info_container["diffmap_config"] = {"vanilla_diffmap": {"title": ""}}

        # convert percentile to sigma
        diffmap_np = diffmap.to_3d_numpy_map(map_sampling=3)
        thresh = np.nanpercentile((diffmap_np), percentile)
        sigma = -(thresh - np.mean(diffmap_np)) / np.std(diffmap_np)
        print(sigma)

        info_container["sigma"] = sigma
        values["sigma"] = sigma

        logger.info(f"\nrunning {info_container['tname']}\n")
        diffmap_path = evaluation_path_basis
        filestart = f"{info_container['fshort']}_"
        os.makedirs(diffmap_path, exist_ok=True)
        logger.info(f"diffmap_path: {diffmap_path}")
        filename_dict2 = {
            "diffmap_path": diffmap_path,
            "filestart": filestart,
        } | filename_dict

        outcomes = run_plots(
            diffmap,
            map_dark,
            info_container,
            "vanilla_diffmap",
            filename_dict2,
            function_selection=function_selection,
            blob_selection_func=blob_selection_func,
        )
        values = values
        for key, outcome in outcomes.items():
            values[key + "_best_guess"] = outcome["best_guess"][-1]
            if outcome.get("uncertainty", False):
                values[key + "_uncertainty"] = outcome["uncertainty"][-1]

        value_list.append(values)

    import pandas as pd

    df = pd.DataFrame(value_list)
    summary_path = evaluation_path + "summary2.csv"
    df2 = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
    df = pd.concat([df2, df])
    logger.info(f"Writing summary to {summary_path}")
    df.to_csv(summary_path)


if __name__ == "__main__":
    main(occupancies=np.arange(0.1, 1, 0.1), percentiles=[0.1, 0.01])
