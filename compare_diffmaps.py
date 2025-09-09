import reciprocalspaceship as rs
from systematic_cistrans import load_defaults_cistrans, shift_offset

from generate_objects import generate_obj_cistrans_v3
from systematic_plots import (
    calculate_scaled_maps,
    load_defaults,
    load_doeke_paths,
    load_homepath,
    load_ocp_paths,
    load_photolyase_paths,
)


def get_ds_from_data_real(conds):
    evaluation_path = load_homepath() + "../evaluation/"
    if "ocp":
        info_container = load_ocp_paths()[0]
    elif "doeke":
        info_container = load_doeke_paths()[0]
    elif "photolyase":
        info_containers = load_photolyase_paths()
        for info_container in info_containers:
            if (
                info_container["fshort"] == "PL30ns"
                and info_container["datatype"] == "photolyase"
            ):
                break

    filename_dict, defaults_to_overwrite, function_selection, blob_selection_func = (
        load_defaults()
    )

    info_container = defaults_to_overwrite | info_container

    dataloc_dark = info_container["dataloc_dark"]
    dataloc_light = info_container["dataloc_light"]

    hs_limit = info_container["hs_limit"]
    map_sampling = info_container["map_sampling"]

    ds_light = rs.read_mtz(dataloc_light)
    ds_dark = rs.read_mtz(dataloc_dark)
    return ds_dark, ds_light, info_container


def get_real_diffmap(ds_dark, ds_light, info_container, conds):
    info_container = info_container.copy() | conds
    map_dark, map_light = calculate_scaled_maps(ds_dark, ds_light, info_container)
    if conds["offset"]:
        map_dark = shift_offset(map_dark.copy())
        map_light = shift_offset(map_light.copy())
    if conds["diffmap_func"] == "vanilla":
        diffmap = compute_difference_map(map_light, map_dark)
    elif conds["diffmap_func"] == "kweighted":
        diffmap = max_negentropy_kweighted_difference_map(map_light, map_dark)
    else:
        raise ValueError("Unknown diffmap function")
    return diffmap


def get_synthetic_diffmap(dark_phases=False, offset=False):
    constants = {"high_resolution_limit": 2}
    f_noise, phi_noise = 0.0, 0.0
    occupancy = 0.3
    map_dark, map_light = generate_obj_cistrans_v3(
        occupancy,
        f_noise=f_noise,
        phi_noise=phi_noise,
        hs_limit=constants["high_resolution_limit"],
    )
    if offset:
        map_dark = shift_offset(map_dark.copy())
        map_light = shift_offset(map_light.copy())
    if not dark_phases:
        map_light["PHI"] = map_dark["PHI"]

    diffmap = compute_difference_map(map_light, map_dark, check_isomorphous=False)
    return diffmap


def get_diffmap(conds):
    if conds["data"] == "cistrans":
        filename_dict, function_selection, info_container = load_defaults_cistrans()
        diffmap = get_synthetic_diffmap(conds["dark_phases"], conds["offset"])
    else:
        ds_dark, ds_light, info_container = get_ds_from_data_real(conds)
        diffmap = get_real_diffmap(ds_dark, ds_light, info_container, conds)
    return diffmap, info_container


#
#
# # ds_dark = cut_resolution(ds_dark, high_resolution_limit=hs_limit)
# # ds_light = cut_resolution(ds_light, high_resolution_limit=hs_limit)
# conds = {
#     "data": "cistrans"
#     "dark_phases": True,
#     "offset": True,
#     "diffmap_func": "kweighted",
# }
# from systematic_cistrans import shift_offset
# from meteor.diffmaps import (
#     compute_difference_map,
#     max_negentropy_kweighted_difference_map,
# )
#
#
#
#
# dark_phases = [False, True]
# offsets = [False, True]
# diffmap_funcs = ["vanilla", "kweighted"]
# # carthe
# import itertools
#
# carthesian_product = itertools.product(dark_phases, offsets, diffmap_funcs)
# diffmap_list = []
# for dark_phase, offset, diffmap_func in carthesian_product:
#     conds = {
#         "dark_phases": dark_phase,
#         "offset": offset,
#         "diffmap_func": diffmap_func,
#     }
#
#     diffmap = get_diffmap(ds_dark, ds_light, info_container, conds)
#     diffmap_list.append((diffmap, conds))
