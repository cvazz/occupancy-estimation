import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import ndimage
from scipy import stats
import sys, os

# get fig folder
from inspect import getsourcefile
from os.path import abspath, dirname

try:
    import gemmi 
except ImportError:
    print("gemmi not available")

try:
    import reciprocalspaceship as rs
except ImportError:
    print("reciprocalspaceship not available")

try:
    import meteor 
    from meteor import sfcalc  
except ImportError:
    print("meteor not available")
    
############################## Folder Magic ######################################    

def get_base_folder():
    """returns current folder location (of this file)

    Returns:
        str: location of folder
    """
    return dirname(abspath(getsourcefile(lambda:0)))

def get_fig_folders():
    """retuns folders for figures

    Returns:
        list: strings of folders to save to
    """

    base_folder = get_base_folder()
    return [
        base_folder + "/plots/",
        base_folder + "/../tex/figs/",
    ]

############################## 2d Objects ######################################    
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
    """
    Generate a pair of similar objects with a small region of differences.

    This function creates two 2D arrays (objects) with random dots, where the second object
    has a specific region where dots are shifted by a small delta compared to the first object.
    Both objects can optionally be blurred with a Gaussian filter.

    Parameters
    ----------
    nx : int, optional
        Width of the output arrays, by default 256
    ny : int, optional
        Height of the output arrays, by default 256
    n_pos : int, optional
        Number of random positions (dots) to generate, by default 600
    dx : int, optional
        Width of the region where objects will differ, by default 50
    dy : int, optional
        Height of the region where objects will differ, by default 50
    delta_x : int, optional
        Horizontal shift applied to dots in the difference region for obj1, by default 3
    delta_y : int, optional
        Vertical shift applied to dots in the difference region for obj1, by default 3
    blur_by : int, optional
        Standard deviation for Gaussian kernel used for blurring, by default 2.
        If 0, no blurring is applied.
    seed : int, optional
        Random seed for reproducibility, by default 0

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing (obj1, obj0), where:
        - obj1: Array with shifted dots in the difference region
        - obj0: Initial array with random dots
    """

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

############################## 2d Objects ######################################    

def generate_obj_cistrans(imagetype, mean_value_offset=0):
    dataloc = "../synthetic_cistrans/"
    match imagetype:
        case "cistrans_nonoise":
            name_dark = "trans_sf.mtz"
            name_light = "100ps_sf.mtz"
            alpha = 0.27
        case "cistrans_noise":
            name_light = "100ps_withSIGFC_amplitudenoise.mtz"
            name_dark = "trans_withSIGFC_amplitudenoise.mtz"
            alpha = 0.27
        case "cistrans_little":
            name_light = "100ps_noise3.mtz"
            name_dark = "trans_noise3.mtz"
            alpha = 0.27
    ds_light = rs.read_mtz(dataloc + name_light)
    ds_dark = rs.read_mtz(dataloc + name_dark)

    if imagetype in ["cistrans_little"]:
        ds_light["sf"] = ds_light.to_structurefactor("F_on", "PHI_on")
        ds_dark["sf"] = ds_dark.to_structurefactor("F_k", "PHI_k")
    else:
        ds_light["sf"] = ds_light.to_structurefactor("FC", "PHIC")
        ds_dark["sf"] = ds_dark.to_structurefactor("FC", "PHIC")

    f_light = ds_light.to_reciprocal_grid("sf")
    f_dark = ds_dark.to_reciprocal_grid("sf")
    delta_fa_abs = np.abs(f_light) - np.abs(f_dark)

    if mean_value_offset:
        obj1 = np.real(np.fft.ifftn(f_light))
        obj0 = np.real(np.fft.ifftn(f_dark))
        minobj1 = np.min(obj1)
        if minobj1<0:
            obj0-=minobj1
            obj1-=minobj1
        f_dark = np.fft.fftn(obj0)
        f_light = np.fft.fftn(obj1)

    else:
        obj1 = np.real(np.fft.ifftn(f_light))
        obj0 = np.real(np.fft.ifftn(f_dark))
    if False:
        f_light[0, 0, 0] += mean_value_offset
        f_dark[0, 0, 0] += mean_value_offset

    return obj0, obj1, f_dark, f_light, delta_fa_abs, alpha


def generate_obj(imagetype, kwargs={}):
    """
    selector between different generator functions
    
    Parameters
    ----------
    imagetype : str
        defines type of generated object

    Returns 
    ----------
    values    
    """
    if imagetype == "2d":
        alpha = 0.3 if not "alpha" in kwargs.keys() else kwargs["alpha"]
        obj1, obj0 = make_objs(
            n_pos=50, blur_by=5, delta_x=-7, delta_y=7, seed=4
        )  # simpler image
        f_dark, f_light, delta_fa_abs = make_working_vars(
            obj1, obj0, alpha, old_version=False
        )
        return obj0, obj1, f_dark, f_light, delta_fa_abs, alpha
    elif "cistrans" in imagetype:
        mean_value_offset = (
            0.3
            if not "mean_value_offset" in kwargs.keys()
            else kwargs["mean_value_offset"]
        )
        if "offset" in imagetype:
            mean_value_offset= 90_000 
            imagevariant = imagetype[:-7]
        else:
            mean_value_offset = 0
            imagevariant = imagetype
        return generate_obj_cistrans(imagevariant, mean_value_offset)
    

def generate_obj_v2(imagetype, kwargs={}):
    """generate objects

    Args:
        imagetype (str): describes which objects to generate
        kwargs (dict, optional): _description_. Defaults to {}.

    Returns:
        _type_: _description_
    """

    mean_value_offset = (
        0.3
        if not "mean_value_offset" in kwargs.keys()
        else kwargs["mean_value_offset"]
    )
    if "offset" in imagetype:
        mean_value_offset= 90_000 
        imagevariant = imagetype[:-7]
    else:
        mean_value_offset = 0
        imagevariant = imagetype
    return generate_obj_cistrans(imagevariant, mean_value_offset)

def struc2realspace(struc, hs_limit, noise_level):
    """
    Convert a molecular structure to real space electron density map with optional noise.

    This function calculates structure factors from a given molecular structure,
    adds Gaussian noise to the amplitudes, and performs an inverse Fourier transform
    to obtain a real space electron density map.

    Parameters
    ----------
    struc : gemmi.Structure or similar
        The molecular structure to convert to real space.
        
    hs_limit : float
        High resolution limit (in Angstroms) for the structure factor calculation.
        
    noise_level : float
        Scale factor for the noise to be added to the structure factor amplitudes.
        The noise is drawn from a normal distribution with mean 0 and
        standard deviation equal to noise_level * |F|.

    Returns
    -------
    numpy.ndarray
        Real space electron density map as a 3D numpy array.
    """
    map_vals = meteor.sfcalc.gemmi_structure_to_calculated_map(struc, high_resolution_limit=hs_limit)
    struc_vals=rs.DataSet(map_vals)
    noise = np.random.normal(loc=0.0, scale=noise_level * np.abs(struc_vals["F"]))
    struc_vals["F"] = struc_vals["F"] + noise
    struc_vals["sf"] = struc_vals.to_structurefactor("F","PHI")
    struc_vals_grid = struc_vals.to_reciprocal_grid("sf")
    struc_vals_real = np.fft.ifftn(struc_vals_grid).real
    return struc_vals_real


def overwrite_occupancy(struc, new_occ):
    target_occ = 0.27
    for x in struc[0].all():
        if np.round(x.atom.occ,2)==  target_occ:
            x.atom.occ=new_occ
        elif np.round(x.atom.occ,2) == 1-target_occ:
            x.atom.occ=1-new_occ
        else:
            pass
    return struc
    
def generate_obj_cistrans_v2(occupancy, noise_level, no_negs = False):
    hs_limit = 1.6
    identifier = f"ct_occ_{occupancy*100:.0f}_noise_{noise_level*100:.0f}"
    dataloc = "../synthetic_cistrans/"
    pdbname_dark = "trans.pdb"
    pdbname_light = "100ps.pdb"

    struc_dark = gemmi.read_structure(dataloc + pdbname_dark)
    struc_light = gemmi.read_structure(dataloc + pdbname_light)

    struc_light = overwrite_occupancy(struc_light, occupancy)

    obj0 = struc2realspace(struc_dark, hs_limit, noise_level)
    obj1 = struc2realspace(struc_light, hs_limit, noise_level)

    if no_negs:
        minobj1 = np.min(obj1)
        if minobj1<0:
            obj0-=minobj1
            obj1-=minobj1

    f_dark = np.fft.fftn(obj0)
    f_light = np.fft.fftn(obj1)

    delta_fa_abs = np.abs(f_light) - np.abs(f_dark)

    return obj0, obj1, f_dark, f_light, delta_fa_abs, identifier
    