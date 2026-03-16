import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import reciprocalspaceship as rs
import gemmi
import shutil
import subprocess
from pathlib import Path

from meteor import rsmap
from meteor.sfcalc import gemmi_structure_to_calculated_map
from configuration import load_homepath
from generate_objects import overwrite_occupancy


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
    map_pdb.set_uncertainties(np.sqrt(sigF_gaussian**2 + sigF_floor**2))  # type: ignore

    mask = np.logical_and(True, map_pdb["F"] > 1)
    return map_pdb[mask]

def pdb2noisy(
    pdbloc, noise_type, high_resolution_limit=1.5, snr_factor=10, save2file=False
):
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
            map_noisy = apply_noise_gaussian_snr_floor(
                map_pdb.copy(), snr_factor, floor_range, floor_snr
            )
            noise_name = "gaussian_flat"
        case _:
            raise ValueError(f"Unknown noise type: {noise_type}")

    map_noisy = map_noisy.drop((0, 0, 0))  # type: ignore

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
    pdbloc_dark: str | Path,
    pdbloc_light: str | Path,
    high_resolution_limit: float = 1.5,
    new_occupancy_level: float | None = None,
):
    struc_dark = gemmi.read_pdb(pdbloc_dark)  # type: ignore
    map_dark = gemmi_structure_to_calculated_map(
        struc_dark, high_resolution_limit=high_resolution_limit
    )

    struc_light = gemmi.read_pdb(pdbloc_light)  # type: ignore
    if new_occupancy_level is not None:
        struc_light = overwrite_occupancy(struc_light, new_occupancy_level)
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
            map_light = apply_noise_gaussian_snr_floor(
                map_light, snr_factor, floor_range, floor_snr
            )
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
    occupancy_level: float | None = None,
    **kwargs,
):
    map_dark, map_light = make_clean_maps(
        pdbloc_dark, pdbloc_light, high_resolution_limit, occupancy_level
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



def diagnosis_plot(list_of_dicts, options, folder):
    xmin = 0
    ymin = 0
    xmax = 0.5

    fig, axs = plt.subplots(4, 4, figsize=(10, 12), constrained_layout=True)
    print(len(list_of_dicts))
    for i, (dl, ax) in enumerate(zip(list_of_dicts, axs.flat)):
        diffmap = -dl["diffmap"]
        map_dark = -dl["map_dark"]
        map_dark_np = map_dark.to_3d_numpy_map(map_sampling=3)
        diffmap_np = diffmap.to_3d_numpy_map(map_sampling=3)
        pseudo = -diffmap_np / map_dark_np
        diff = diffmap_np

        # 2. Calculate stats LOCALLY for this specific subplot
        current_mean = np.mean(diff)
        current_std = np.std(diff)
        # current_mean_peak = np.mean(diff[clean_mask]) - default_mean
        print(i, current_mean)

        # 3. Define the transformation functions with local closure
        # We use default arguments to "snap" the current mean/std into the function
        def abs_to_std(y, m=current_mean, s=current_std):
            return (y - m) / s

        def std_to_abs(z, m=current_mean, s=current_std):
            return (z * s) + m

        sig3 = std_to_abs(0.2)
        mask = np.logical_and.reduce([diff > ymin, pseudo > xmin, pseudo < xmax])
        mask_main = np.logical_and(mask, diff < sig3)
        mask_outlier = np.logical_and(mask, diff >= sig3)

        # 1. Plot the data
        ax.plot(pseudo[mask_outlier], diff[mask_outlier], ".", color="green", alpha=0.1)
        # ax.plot(pseudo[mask_main], diff[mask_main], ',', color='green', alpha=0.1)

        # 4. Vertical lines and limits
        ax.axvline(0.135, color="red", linestyle="--", alpha=0.5)
        ax.axvline(0.27, color="green", linestyle="--", alpha=0.5)
        ax.set_ylim(0, None)
        ax.set_xlim(xmin, xmax)
        title_ax = f"Dataset i={i}"
        # title_ax += f"\n Mean: {current_mean*100:.4f}"
        # title_ax += f"\n Mean Peak: {current_mean_peak*100:.4f}"
        ax.set_title(title_ax)
        secax = ax.secondary_yaxis("right", functions=(abs_to_std, std_to_abs))
        if (i + 1) % 4 == 0:
            secax.set_ylabel(r"$\sigma$ Level", fontsize=10)
        # secax.set_ylabel(r"Standard Deviations ($\sigma$)")
    for ax in axs[-1, :]:
        ax.set_xlabel("Pseudo-Occupancy")
    for ax in axs[:, 0]:
        ax.set_ylabel("Difference Map Value")

    title = "Simulated rsEGFP2 Data\n"
    title += f"Vacuum Matching \n{options['noise_type']} noise with SNR of {options['snr_factor']}\n"
    title += f"using {'dark' if options['dark_phases'] else 'true'} phases"
    _ = fig.suptitle(title)
    fig.savefig(folder + "diagnosis_plot.png")


def make_string(options):
    return f"{options['noise_type']}/snr_{options['snr_factor']}_occ_{options['occupancy_level']}_no{options['attempt_no']}/"


def save_to_file(rmap, loc):
    rmap.drop((0, 0, 0), inplace=True)
    ds_noisy = rs.DataSet(rmap)
    ds_noisy.drop(columns=["PHI"], inplace=True, errors="ignore")
    ds_noisy.write_mtz(loc)


def slurm_command(x8_instructions_loc):
    return f"""#!/bin/bash
#SBATCH --partition=allcpu
#SBATCH -t 2-12:00
#SBATCH -o slurm_%j.out
#SBATCH -e slurm_%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=sebastian.bielfeldt@desy.de

phenix.python /Users/sbielfel/Nextcloud2/time_resolved/Xtrapol8/Fextr.py {x8_instructions_loc}
"""


def get_pipeline_configs(is_it_poisson=True):
    number_of_repeats = 10
    occupancy_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.8]
    occupancy_levels = [0.05, 0.1, 0.15, 0.2]
    occupancy_levels = [0.25, 0.3, 0.35, 0.4, 0.8]
    poisson_snr = [0.1, 0.5, 1, 2, 5, 10, 20]
    flat_snr = [1, 2.5, 5, 7.5, 10, 15, 20, 30]
    configs = {
        "number_of_repeats": number_of_repeats,
        "occupancy_levels": occupancy_levels,
        "poisson_snr": poisson_snr,
        "flat_snr": flat_snr,
        "is_it_poisson": is_it_poisson,
    }

    options = dict(
        high_resolution_limit=1.5,
        dark_phases=True,
        snr_factor=1,
        noise_type="poisson",
        occupancy_level=0.3,
        attempt_no=-1,
    )
    return configs, options


def elaborate_x8_pipeline():
    # list_occ = 0.01 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9

    configs, options = get_pipeline_configs()
    number_of_repeats = configs["number_of_repeats"]
    occupancy_levels = configs["occupancy_levels"]
    poisson_snr = configs["poisson_snr"]
    flat_snr = configs["flat_snr"]

    synthloc = load_homepath() + "synthetic_cistrans/"
    pdbloc_dark = synthloc + "trans.pdb"
    pdbloc_light = synthloc + "100ps.pdb"

    dict_of_folders = {}
    x8_instructions_loc = "x8_command.phil"
    do_it_yourself = True
    skip = False
    is_it_poisson = True
    snr_it = poisson_snr if is_it_poisson else flat_snr
    for occupancy in occupancy_levels:
        options["occupancy_level"] = occupancy
        for snr in snr_it:
            options["noise_type"] = "poisson" if is_it_poisson else "gaussian_flat"
            options["snr_factor"] = snr
            list_of_dicts = []
            for attempt_no in range(number_of_repeats):
                options["attempt_no"] = attempt_no
                folder = make_string(options)
                dict_of_folders[folder] = False
                diffmap, map_dark, map_light = make_clean_synthetic_data(
                    pdbloc_dark, pdbloc_light, **options
                )
                # ensure folder exists
                Path(folder).mkdir(parents=True, exist_ok=True)
                save_to_file(map_dark, folder + "map_dark.mtz")
                save_to_file(map_light, folder + "map_light.mtz")
                # copy x8 instructions to folder
                shutil.copy(x8_instructions_loc, folder)
                try:
                    if skip:
                        pass
                    else:
                        if do_it_yourself:
                            subprocess.run(
                                [
                                    "phenix.python",
                                    "/Users/sbielfel/Nextcloud2/time_resolved/Xtrapol8/Fextr.py",
                                    x8_instructions_loc,
                                ],
                                cwd=folder,
                            )
                        else:
                            script_path = Path(folder) / "submit_job.sh"
                            with open(script_path, "w") as f:
                                slurm_script_content = slurm_command(
                                    x8_instructions_loc
                                )
                                f.write(slurm_script_content)
                            subprocess.run(["sbatch", "submit_job.sh"], cwd=folder)
                except Exception as e:
                    dict_of_folders[folder] = f"Error: {e}"

                list_of_dicts.append(
                    {
                        "diffmap": diffmap,
                        "map_dark": map_dark,
                    }
                )
            diagnosis_plot(list_of_dicts, options, folder[:-2])


def get_paths():
    configs, options = get_pipeline_configs()
    number_of_repeats = configs["number_of_repeats"]
    occupancy_levels = configs["occupancy_levels"]
    poisson_snr = configs["poisson_snr"]
    flat_snr = configs["flat_snr"]
    snr_it = poisson_snr if configs["is_it_poisson"] else flat_snr
    folder_locs = []
    for occupancy in occupancy_levels:
        options["occupancy_level"] = occupancy
        for snr in snr_it:
            options["noise_type"] = (
                "poisson" if configs["is_it_poisson"] else "gaussian_flat"
            )
            options["snr_factor"] = snr
            for attempt_no in range(number_of_repeats):
                options["attempt_no"] = attempt_no
                folder = make_string(options)
                folder_locs.append({folder: options.copy()})
    return folder_locs


if __name__ == "__main__":
    elaborate_x8_pipeline()
    # get_paths()
