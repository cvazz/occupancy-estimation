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

    F_2angstrom = np.mean(map_pdb["F"][mask])  # type : ignore
    sigI = map_pdb["F"] * F_2angstrom / snr_factor
    noisy_I = map_pdb["F"] ** 2 + sigI * np.random.normal(0, 1, map_pdb["F"].shape)

    map_pdb["F"] = np.sqrt(np.abs(noisy_I))

    map_pdb["PHI"] += (1 - np.sign(noisy_I)) * 90
    map_pdb["PHI"] = map_pdb["PHI"] % 360
    sigF = np.sqrt(sigI)
    map_pdb.set_uncertainties(sigF)  # type: ignore
    return map_pdb[map_pdb["F"] > 1]


def apply_noise_gaussian(map_pdb, snr_factor):
    sigF = map_pdb["F"] / snr_factor
    noisy_F = map_pdb["F"] + sigF * np.random.normal(0, 1, map_pdb["F"].shape)
    mask_one = None
    # mask_one = np.ones(len(map_pdb))
    if mask_one is None:
        map_pdb["F"] = noisy_F
    else:
        map_pdb.loc[mask_one, "F"] = noisy_F[mask_one]

    # map_pdb["PHI"] += (1 - np.sign(noisy_F)) * 90
    # map_pdb["PHI"] = map_pdb["PHI"] % 360
    map_pdb.set_uncertainties(np.sqrt(sigF))

    mask = np.logical_and(True, map_pdb["F"] > 1)
    return map_pdb[mask]


def pdb2noisy_mtz(pdbloc, high_resolution_limit=1.5, snr_factor=10, save2file=False):
    file_loc = synthloc + "100ps_noise3.mtz"
    ds_temp = rs.read_mtz(file_loc)
    struc = gemmi.read_pdb(pdbloc)
    map_pdb = gemmi_structure_to_calculated_map(
        struc, high_resolution_limit=high_resolution_limit
    )
    map_noisy = apply_noise_poisson(map_pdb.copy(), snr_factor)
    print(f"Zero frequency: {map_noisy.loc[(0,0,0), 'F']:.1f}")
    map_noisy = map_noisy.drop((0, 0, 0))
    print(map_noisy)
    output_mtz = pdbloc.replace(
        ".pdb", f"_snr_{snr_factor}_dmin_{high_resolution_limit*10:.0f}.mtz"
    )
    print("saved to ", output_mtz)
    ds_temp["F_on"] = map_noisy["F"]
    # ds_temp["PHIC"] = map_noisy["PHI"]
    ds_temp["SIGF_on"] = map_noisy["SIGF"]
    # ds_temp.drop(columns=["F_k", "PHI_k", "SIGF_k"], inplace=True, errors='ignore')
    ds_temp.drop(columns=["SIGFC"], inplace=True, errors="ignore")
    print(ds_temp.columns)
    if save2file:
        ds_temp.write_mtz(output_mtz)
    else:
        print("Test run - not saving file")


def pdb2noisy_gaussian(
    pdbloc, high_resolution_limit=1.5, snr_factor=10, save2file=False, synthloc=""
):
    synthloc = synthloc if synthloc else load_homepath() + "synthetic_cistrans/"
    file_loc = synthloc + "100ps_noise3.mtz"
    ds_temp = rs.read_mtz(file_loc)
    struc = gemmi.read_pdb(pdbloc)
    map_pdb = gemmi_structure_to_calculated_map(
        struc, high_resolution_limit=high_resolution_limit
    )

    map_noisy = apply_noise_gaussian(map_pdb.copy(), snr_factor)
    map_noisy = map_noisy.drop((0, 0, 0))
    output_mtz = pdbloc.replace(
        ".pdb", f"_gaussian_{snr_factor}_dmin_{high_resolution_limit*10:.0f}.mtz"
    )

    ds_temp["F_on"] = map_noisy["F"]
    ds_temp["SIGF_on"] = map_noisy["SIGF"]
    ds_temp.drop(columns=["SIGFC"], inplace=True, errors="ignore")

    print(ds_temp.columns)
    print("saving ", output_mtz)
    if save2file:
        ds_temp.write_mtz(output_mtz)
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
