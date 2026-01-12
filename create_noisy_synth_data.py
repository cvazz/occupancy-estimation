import numpy as np
import reciprocalspaceship as rs
import gemmi
from meteor import rsmap
from meteor.sfcalc import gemmi_structure_to_calculated_map
from configuration import load_homepath



def apply_noise_poisson(map_pdb: rsmap.Map, snr_factor: float) -> rsmap.Map:
    dhkl = map_pdb.compute_dHKL()
    mask = np.logical_and(dhkl < 2.1, dhkl>= 1.9)

    F_2angstrom = np.mean(map_pdb["F"][mask], dtype=np.float64)
    sigI = map_pdb["F"] * F_2angstrom / snr_factor
    noisy_I =  map_pdb["F"]**2 + sigI * np.random.normal(0, 1, map_pdb["F"].shape)

    map_pdb["F"] = np.sqrt(np.abs(noisy_I))

    map_pdb["PHI"] += (1 - np.sign(noisy_I))*90
    map_pdb["PHI"] = map_pdb["PHI"] % 360
    sigF = np.sqrt(sigI)
    map_pdb.set_uncertainties(sigF) # type: ignore
    return map_pdb[map_pdb["F"] > 1]

def apply_noise_gaussian(map_pdb, snr_factor):
    sigF = map_pdb["F"] / snr_factor
    noisy_F =  map_pdb["F"] + sigF * np.random.normal(0, 1, map_pdb["F"].shape)

    map_pdb["F"] = np.abs(noisy_F)

    map_pdb["PHI"] += (1 - np.sign(noisy_F))*90
    map_pdb["PHI"] = map_pdb["PHI"] % 360
    map_pdb.set_uncertainties(np.sqrt(sigF))

    return map_pdb[map_pdb["F"] > 1]

def pdb2noisy_mtz(pdbloc, high_resolution_limit=1.5, snr_factor=10, save2file=False):
    file_loc = synthloc + "100ps_noise3.mtz"
    ds_temp = rs.read_mtz(file_loc)
    struc = gemmi.read_pdb(pdbloc)
    map_pdb = gemmi_structure_to_calculated_map(
        struc, high_resolution_limit=high_resolution_limit
    )
    map_noisy = apply_noise_poisson(map_pdb.copy(), snr_factor)
    print(f"Zero frequency: {map_noisy.loc[(0,0,0), 'F']:.1f}")
    map_noisy = map_noisy.drop((0,0,0))
    print(map_noisy)
    output_mtz = pdbloc.replace(".pdb", f"_snr_{snr_factor}_dmin_{high_resolution_limit*10:.0f}.mtz")
    print("saved to ", output_mtz)
    ds_temp["F_on"] = map_noisy["F"]
    # ds_temp["PHIC"] = map_noisy["PHI"]
    ds_temp["SIGF_on"] = map_noisy["SIGF"]
    # ds_temp.drop(columns=["F_k", "PHI_k", "SIGF_k"], inplace=True, errors='ignore')
    ds_temp.drop(columns=["SIGFC"], inplace=True, errors='ignore')
    print(ds_temp.columns)
    if save2file:
        ds_temp.write_mtz(output_mtz)
    else:
        print("Test run - not saving file")

def pdb2noisy_gaussian(pdbloc, high_resolution_limit=1.5, snr_factor=10, save2file=False, synthloc=""):
    synthloc = synthloc if synthloc else load_homepath() + "synthetic_cistrans/"
    file_loc = synthloc + "100ps_noise3.mtz"
    ds_temp = rs.read_mtz(file_loc)
    struc = gemmi.read_pdb(pdbloc)
    map_pdb = gemmi_structure_to_calculated_map(
        struc, high_resolution_limit=high_resolution_limit
    )

    map_noisy = apply_noise_gaussian(map_pdb.copy(), snr_factor)
    map_noisy = map_noisy.drop((0,0,0))
    output_mtz = pdbloc.replace(".pdb", f"_gaussian_{snr_factor}_dmin_{high_resolution_limit*10:.0f}.mtz")

    ds_temp["F_on"] = map_noisy["F"]
    ds_temp["SIGF_on"] = map_noisy["SIGF"]
    ds_temp.drop(columns=["SIGFC"], inplace=True, errors='ignore')

    print(ds_temp.columns)
    print("saving ", output_mtz)
    if save2file:
        ds_temp.write_mtz(output_mtz)
    else:
        print("Test run - not saving file")

