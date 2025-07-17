import numpy as np
import pandas as pd
import gemmi
import matplotlib.pyplot as plt
import meteor
import reciprocalspaceship as rs



from photolyase import get_exp_structure_factors

from meteor.diffmaps import compute_difference_map, max_negentropy_kweighted_difference_map
from meteor.tv import tv_denoise_difference_map

from meteor.utils import cut_resolution


from meteor import rsmap

def get_scaled_maps(ds_dark, ds_light):
    dark_f = "F-obs-filtered"
    dark_phi = "PHIF-model"
    light_f = "F"
    light_sig = "SIGF"
    dark_sig = "SIGF-obs-filtered"

    ds_comb = get_exp_structure_factors(ds_dark, ds_light, return_ds=True)

    map_dark = rsmap.Map(ds_comb, amplitude_column=dark_f, phase_column=dark_phi, uncertainty_column=dark_sig)
    map_light = rsmap.Map(ds_comb, amplitude_column=light_f, phase_column=dark_phi, uncertainty_column=light_sig)
    return map_dark, map_light

def negsum_meteor(
    rho_xtrs: list[rsmap.Map],
    *,
    map_sampling: float,
    mask: None | np.ndarray = None,
):
    rho_shape = rho_xtrs[0].to_3d_numpy_map(map_sampling=map_sampling).shape
    mask = np.ones(rho_shape, bool) if mask is None else mask
    arrlen = len(rho_xtrs)
    neg_sum = np.empty((arrlen))
    for ii, dens in enumerate(rho_xtrs):
        dens = dens.to_3d_numpy_map(map_sampling=map_sampling)[mask]
        neg_sum[ii] = np.sum(dens[dens < 0])
    return neg_sum
################################################################################
################################################################################
################################################################################
def fetch_map2numpy_args(rsmap:rsmap.Map):
    return (lambda **kwargs:kwargs)(
        spacegroup=rsmap.spacegroup,
        cell=rsmap.cell,
        high_resolution_limit = rsmap.resolution_limits[1],
    )
    
def calc_direct_difference(map_light, map_dark, map_sampling):
    direct_diff = (
    map_light.to_3d_numpy_map(map_sampling=map_sampling)
        -
    map_dark.to_3d_numpy_map(map_sampling=map_sampling)
    )
    
    return rsmap.Map.from_3d_numpy_map(direct_diff,**fetch_map2numpy_args(map_dark))
                                       
def fetch_without_meta(map_light, map_dark, diffmap_maker):
    k_weighted_diffmap, kparameter_metadata = diffmap_maker(
        map_light, map_dark
    )
    return k_weighted_diffmap
    
def fetch_tv_denoised(map_light, map_dark):
    k_weighted_diffmap, kparameter_metadata = max_negentropy_kweighted_difference_map(
        map_light, map_dark
    )
    tv_denoised_map, metadata = tv_denoise_difference_map(k_weighted_diffmap, full_output=True)
    return tv_denoised_map

################################################################################
################################################################################
################################################################################

def adding_maps(map1: rsmap.Map, map2: rsmap.Map, *, factor1=1, factor2=1):
    common_indices = map1.index.intersection(map2.index)
    structure_factors1 = map1.loc[common_indices].to_structurefactor()
    structure_factors2 = map2.loc[common_indices].to_structurefactor()
    added_structure_factors = (
        factor1 * structure_factors1 + factor2 * structure_factors2
    )
    return rsmap.Map.from_structurefactor(
        added_structure_factors,
        index=common_indices,
        cell=map1.cell,
        spacegroup=map1.spacegroup,
    )
def make_k_space_xtr(map_dark,
                        diffmap,
                        extrapolation_factors,
                       ):

    map_xtrs = []
    for xtr_factor in extrapolation_factors:
        
        map_xtr = adding_maps(diffmap , map_dark, factor1=xtr_factor )
        map_xtrs.append(map_xtr)
    return map_xtrs

def make_real_space_xtr(map_dark,
                        diffmap,
                        extrapolation_factors,
                        map_sampling,
                       ):
    rho_xtrs = []
    for xtr_factor in extrapolation_factors:

        xtr = (diffmap.to_3d_numpy_map(map_sampling=map_sampling)
        * xtr_factor 
        + map_dark.to_3d_numpy_map(map_sampling=map_sampling)
        )
        rho_xtr = rsmap.Map.from_3d_numpy_map(
            xtr,
            spacegroup = map_dark.spacegroup,
            cell = map_dark.cell,
            high_resolution_limit = map_dark.resolution_limits[1],
        )
        rho_xtrs.append(rho_xtr)
    return rho_xtrs