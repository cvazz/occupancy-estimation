import pandas as pd
import numpy as np
import reciprocalspaceship as rs
import gemmi
from meteor import rsmap
from meteor.sfcalc import gemmi_structure_to_calculated_map
from configuration import load_homepath


def apply_noise_poisson(map_pdb: rsmap.Map, snr_factor: float) -> rsmap.Map:
    dhkl = map_pdb.compute_dHKL()
    mask = np.logical_and(dhkl < 2.1, dhkl >= 1.9)

    F_2angstrom = float(np.mean(map_pdb["F"][mask]))
    sigI = map_pdb["F"] * F_2angstrom / snr_factor
    noisy_I = map_pdb["F"] ** 2 + sigI * np.random.normal(0, 1, map_pdb["F"].shape)

    map_pdb["F"] = np.sqrt(np.abs(noisy_I))

    map_pdb["PHI"] += (1 - np.sign(noisy_I)) * 90
    map_pdb["PHI"] = map_pdb["PHI"] % 360
    sigF = np.sqrt(sigI)
    map_pdb.set_uncertainties(sigF)  # type: ignore
    return map_pdb[map_pdb["F"] > 1]


def apply_noise_gaussian(map_pdb: rsmap.Map, snr_factor: float) -> rsmap.Map:
    sigF = map_pdb["F"] / snr_factor
    noisy_F = map_pdb["F"] + sigF * np.random.normal(0, 1, map_pdb["F"].shape)
    map_pdb["F"] = noisy_F

    map_pdb["PHI"] += (1 - np.sign(noisy_F)) * 90
    map_pdb["PHI"] = map_pdb["PHI"] % 360
    map_pdb.set_uncertainties((sigF))  # type: ignore

    mask = np.logical_and(True, map_pdb["F"] > 1)
    return map_pdb[mask]


def apply_noise_gaussian_snr_floor(
    map_pdb: rsmap.Map,
    snr_factor: float,
    floor_range: tuple[float, float],
    floor_snr: float,
) -> rsmap.Map:
    def rand():
        return np.random.normal(0, 1, map_pdb["F"].shape)
    dhkl = map_pdb.compute_dHKL()
    mask = np.logical_and((dhkl >= floor_range[0]), (dhkl <= floor_range[1]))
    if mask.sum() == 0:
        raise ValueError("No reflections found in the specified floor range.")
    sigF_floor = float(np.mean(map_pdb["F"][mask])) / floor_snr

    sigF_gaussian = map_pdb["F"] / snr_factor
    noisy_F = map_pdb["F"] + sigF_gaussian * rand() + sigF_floor * rand()
    map_pdb["F"] = noisy_F

    map_pdb["PHI"] += (1 - np.sign(noisy_F)) * 90
    map_pdb["PHI"] = map_pdb["PHI"] % 360
    map_pdb.set_uncertainties(np.sqrt(sigF_gaussian**2+sigF_floor**2))  # type: ignore

    mask = np.logical_and(True, map_pdb["F"] > 1)
    return map_pdb[mask]

def pdb2noisy(pdbloc, noise_type, high_resolution_limit=1.5, snr_factor=10, save2file=False):
    struc = gemmi.read_pdb(pdbloc)
    map_pdb = gemmi_structure_to_calculated_map(
        struc, high_resolution_limit=high_resolution_limit
    )
    match noise_type:
        case "poisson":
            map_noisy = apply_noise_poisson(map_pdb.copy(), snr_factor)
            noise_name = "snr"
        case "gaussian":
            map_noisy = apply_noise_gaussian(map_pdb.copy(), snr_factor)
            noise_name = "gaussian"
        case "gaussian_flat":
            floor_range = (1.55, 1.65)
            floor_snr = 1.0
            map_noisy = apply_noise_gaussian_snr_floor(map_pdb.copy(), snr_factor, floor_range, floor_snr)
            noise_name = "gaussian_flat"
        case _:
            raise ValueError(f"Unknown noise type: {noise_type}")

    map_noisy = map_noisy.drop((0, 0, 0)) # type: ignore

    output_mtz_path = pdbloc.replace(
        ".pdb", f"_{noise_name}_{snr_factor}_dmin_{high_resolution_limit*10:.0f}.mtz"
    )
        
    ds_noisy = rs.DataSet(map_noisy)
    ds_noisy.drop(columns=["PHI"], inplace=True, errors="ignore")
    if save2file:
        ds_noisy.write_mtz(output_mtz_path)
        print("saved to ", output_mtz_path)
    else:
        print("Test run - not saving file")






def make_clean_maps(
    pdbloc_dark,
    pdbloc_light,
    high_resolution_limit=1.5,
):
    struc_dark = gemmi.read_pdb(pdbloc_dark)
    map_dark = gemmi_structure_to_calculated_map(
        struc_dark, high_resolution_limit=high_resolution_limit
    )

    struc_light = gemmi.read_pdb(pdbloc_light)
    map_light = gemmi_structure_to_calculated_map(
        struc_light, high_resolution_limit=high_resolution_limit
    )
    return map_dark, map_light


def apply_noise_to_maps(
    map_dark,
    map_light,
    dark_phases=True,
    snr_factor=10,
    noise_type: str = "poisson",
):
    if snr_factor:
        if noise_type == "half":
            map_light = apply_noise_gaussian(map_light, snr_factor)
        elif noise_type == "floor":
            floor_range = (1.45, 1.55)
            floor_snr = 1.0
            map_light = apply_noise_gaussian_snr_floor(map_light, snr_factor, floor_range, floor_snr)
        elif noise_type == "gaussian":
            map_dark = apply_noise_gaussian(map_dark, snr_factor)
            map_light = apply_noise_gaussian(map_light, snr_factor)
        elif noise_type == "poisson":
            map_dark = apply_noise_poisson(map_dark, snr_factor)
            map_light = apply_noise_poisson(map_light, snr_factor)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
    if dark_phases:
        diffmap_amps = map_light.amplitudes - map_dark.amplitudes
        diffmap_ds = pd.DataFrame({"F": diffmap_amps, "PHI": map_dark.phases})
        diffmap = rsmap.Map(
            diffmap_ds,
            amplitude_column="F",
            phase_column="PHI",
            cell=map_dark.cell,
            spacegroup=map_dark.spacegroup,
        )
    else:
        map_dark_sf = map_dark.to_structurefactor()
        map_light_sf = map_light.to_structurefactor()
        diffmap_sf = map_light_sf - map_dark_sf
        diffmap = rsmap.Map.from_structurefactor(
            diffmap_sf,
            cell=map_dark.cell,
            spacegroup=map_dark.spacegroup,
        )
    return diffmap, map_dark, map_light


def make_clean_synthetic_data(
    pdbloc_dark,
    pdbloc_light,
    high_resolution_limit=1.5,
    dark_phases=True,
    snr_factor=0,
    noise_type: str = "poisson",
):
    map_dark, map_light = make_clean_maps(
        pdbloc_dark, pdbloc_light, high_resolution_limit
    )

    diffmap, map_dark, map_light = apply_noise_to_maps(
        map_dark,
        map_light,
        dark_phases,
        snr_factor,
        noise_type,
    )
    return diffmap, map_dark, map_light


def mix_maps(
    pdbloc_dark,
    pdbloc_light,
    high_resolution_limit=1.5,
    dark_phases=True,
    snr_factor=0,
    noise_type: str = "poisson",
    true_occupancy: float = 0.3,
):
    map_dark, map_target = make_clean_maps(
        pdbloc_dark, pdbloc_light, high_resolution_limit
    )
    map_dark_sf = map_dark.to_structurefactor()
    map_target_sf = map_target.to_structurefactor()
    map_light_sf = (1 - true_occupancy) * map_dark_sf + true_occupancy * map_target_sf
    map_light = rsmap.Map.from_structurefactor(
        map_light_sf,
        cell=map_dark.cell,
        spacegroup=map_dark.spacegroup,
    )

    diffmap, map_dark, map_light = apply_noise_to_maps(
        map_dark,
        map_light,
        dark_phases,
        snr_factor,
        noise_type,
    )
    return diffmap, map_dark, map_light
