import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import ndimage
import sys, os

################################################################################
######################### Data preparation #####################################
################################################################################

def make_objs(
    nx=256,
    ny=256,
    n_pos=600,
    dx=50,
    dy=50,
    delta_x=3,
    delta_y=3,
    blur_by=2,
    seed=0,
):

    # Initialize arrays
    obj0 = np.zeros((nx, ny))

    # Generate random positions
    np.random.seed(seed)  # Replace with `seed` if needed
    x_pos = (np.random.rand(n_pos) * nx).astype(int)
    y_pos = (np.random.rand(n_pos) * ny).astype(int)

    # Set initial object values
    obj0[x_pos, y_pos] = 1.0

    # Identify positions for modification
    i_diff = np.where(
        (x_pos < nx / 2)
        & (x_pos > nx / 2 - dx)
        & (y_pos < ny / 2)
        & (y_pos > ny / 2 - dy)
    )[0]
    # Modify object to create obj1
    obj1 = obj0.copy()
    obj1[x_pos[i_diff], y_pos[i_diff]] = 0.0
    obj1[x_pos[i_diff] + delta_x, y_pos[i_diff] + delta_y] = 1.0

    if blur_by:
        obj0 = ndimage.gaussian_filter(obj0, blur_by)
        obj1 = ndimage.gaussian_filter(obj1, blur_by)

    # Calculate delta_obj
    return obj1, obj0


def make_working_vars(obj1, obj0, alpha, old_version=False):
    delta_obj = obj1 - obj0
    f0 = np.fft.fftn(obj0)
    f0_abs = np.abs(f0)
    f0_cphase = np.exp(
        1j * np.angle(f0)
    )  # Equivalent to complex(cos(f0_phase), sin(f0_phase))

    if old_version:
        obj1a = (1 - alpha) * obj0 + alpha * delta_obj
        f1a = np.fft.fftn(obj1a)
        delta_fa_abs = np.abs(f1a) - f0_abs
        delta_fa_abs_ph0 = delta_fa_abs * f0_cphase
    else:
        # Calculate obj1a and its Fourier transform
        obj1a = (1 - alpha) * obj0 + alpha * delta_obj
        obj1a = (1 - 2 * alpha) * obj0 + alpha * obj1
        obj1a = obj0 + alpha * delta_obj
        obj1a = (1 - alpha) * obj0 + alpha * obj1
        # obj1a = obj0 + alpha * delta_obj
        f1a = np.fft.fftn(obj1a)
        # Calculate delta_fa_abs and delta_fa_abs_ph0
        delta_fa_abs = np.abs(f1a) - f0_abs
        delta_fa_abs_ph0 = delta_fa_abs * f0_cphase

    return f0, f1a, delta_fa_abs



def make_f_xtr(
    alphas, f_dark, f_light, f_angle, version, noise_level=0, poisson_noise=-1
):
    """
    alphas: 1d list of float
    f_dark: 2d complex array
    f_light: 2d complex array
    f_angle: 2d array of float

    only use version 1
    """

    f_dark_abs = np.abs(f_dark)
    f_light_abs = np.abs(f_light)
    noise = np.random.normal(size=f_dark_abs.shape) * np.mean(f_dark_abs) * noise_level
    delta_f = (f_light_abs - f_dark_abs) + noise
    delta_f_with_phase = delta_f * np.exp(1j * f_angle)
    many_none = (None,) * f_dark.ndim
    f_xtr_abs = np.abs(
        2 / alphas[(slice(None), *many_none)] * (delta_f)[None, ...]
        + f_dark_abs[None, ...]
    )
    if poisson_noise > 0:
        pixel_count = poisson_noise / f_xtr_abs.sum()
        f_xtr_abs = np.random.poisson(f_xtr_abs * pixel_count) / pixel_count

    match version:
        case 1:
            f_xtr = (
                2 / alphas[(slice(None), *many_none)] * (delta_f_with_phase[None, ...])
                + f_dark[None, ...]
            )
        case 2:
            f_xtr = f_xtr_abs * np.exp(1j * f_angle)[None, ...]
        case 3:
            f_xtr = 2 / alphas * f_light
        case 4:
            pass

    return f_xtr

################################################################################
############################## Processing ######################################
################################################################################


def comp_cc(f_xtra, f_dark, f_light):
    delta_meas = f_light - f_dark
    delta_xtra = f_xtra - f_dark
    out = pearsonr(delta_meas.flatten(), delta_xtra.flatten())
    return out[0]

def x8_inspired(
    f_dark,
    f_light,
    f_xtrs,
):
    corrs = np.empty(len(f_xtrs))
    for ii, f_xtr in enumerate(f_xtrs):
        corrs[ii] = comp_cc(f_xtr, f_dark, f_light)
    return corrs

################################ PanDDA ########################################

def pandda(
    f_dark,
    f_xtrs,
    mask_pks,
):
    rho_dark = np.fft.ifftn(f_dark).real
    mean_global = np.empty(len(f_xtrs))
    mean_local = np.empty(len(f_xtrs))
    for ii, f_xtr in enumerate(f_xtrs):
        rho_xtr = np.fft.ifftn(f_xtr).real
        mean_global[ii] = pearsonr(
            rho_xtr[~mask_pks].flatten(), rho_dark[~mask_pks].flatten()
        )[0]
        mean_global[ii] = pearsonr(rho_xtr.flatten(), rho_dark.flatten())[0]
        mean_local[ii] = pearsonr(
            rho_xtr[mask_pks].flatten(), rho_dark[mask_pks].flatten()
        )[0]
    return mean_local, mean_global


######################## Negative Sum Explosion ################################

def marius(f_xtrs):
    arrlen = len(f_xtrs)
    neg_sum = np.empty((arrlen))
    pos_sum = np.empty((arrlen))
    densities = np.empty(f_xtrs.shape)
    for ii, f_xtr in enumerate(f_xtrs):
        dens = np.fft.ifftn(f_xtr).real
        densities[ii] = dens
        neg_sum[ii] = np.sum(dens[dens < 0])
        pos_sum[ii] = np.sum(dens[dens > 0])
    return densities, neg_sum


############################### Xtrapol8 #######################################

def x8_density_map_f1(f_xtrs, mask_pks, fofo, obj0):
    arrlen = len(f_xtrs)
    peak_sum = np.empty((arrlen))
    real_CC = np.empty((arrlen))
    for ii, f_xtr in enumerate(f_xtrs):
        dens = np.fft.ifftn(f_xtr).real
        real_CC[ii] = pearsonr(dens.flatten(), obj0.flatten())[0]
        peak_sum[ii] = np.abs(dens[mask_pks]).sum() / np.abs(dens).sum()
        # peak_sum[ii] = np.abs(dens[mask_pks]).sum()/np.abs(fofo[mask_pks]).sum()
    return peak_sum, real_CC


def x8_density_map_fdiff(f_xtrs, mask_pks, obj0, fofo):
    arrlen = len(f_xtrs)
    peak_sum = np.empty((arrlen))
    real_CC = np.empty((arrlen))
    for ii, f_xtr in enumerate(f_xtrs):
        dens = np.fft.ifftn(f_xtr).real
        real_CC[ii] = pearsonr((dens - obj0).flatten(), fofo.flatten())[0]
        peak_sum[ii] = np.abs((dens - obj0)[mask_pks]).sum() / np.abs(dens - obj0).sum()

    return peak_sum, real_CC


def x8_density_map_fdiff_norm(f_xtrs, mask_pks, obj0, fofo):
    arrlen = len(f_xtrs)
    peak_sum = np.empty((arrlen))
    real_CC = np.empty((arrlen))
    obj0 = obj0 / np.max(obj0)
    for ii, f_xtr in enumerate(f_xtrs):
        dens = np.fft.ifftn(f_xtr).real
        dens = dens / np.max(dens)
        real_CC[ii] = pearsonr((dens - obj0).flatten(), fofo.flatten())[0]
        peak_sum[ii] = np.abs((dens - obj0)[mask_pks]).sum() / np.abs(dens - obj0).sum()

    return peak_sum, real_CC
