import os


def load_homepath():
    # This function returns the path in which t
    current_path = os.getcwd()
    path_parts = current_path.split(os.sep)
    idx = path_parts.index("time_resolved")
    homepath = os.sep.join(path_parts[: idx + 1]) + "/"
    return homepath

def load_figurepath():
    return r"/Users/sbielfel/Dropbox/Apps/Overleaf/Occupancy Determination/figs/"

def minimal_masking_config():
    return {
            "sigma": 3,
            "min_blob_size": 0.3,  # in A^3
            "blocking_radius": 0.1,
            "blocking_percentile": 95,
            "exclude_solvent": False,
            "dark_size_threshold": 0.,
            "exclude_large_occupancy_outliers": False,
    }

def get_file_config(
    dataloc_dark: str,
    dataloc_light: str,
    pdbloc_dark: str,
    columns_dark: dict,
    columns_triggered: dict,
    high_resolution_limit: float = 0.1,
    name_machine: str = "unnamed_experiment",
    name_human: str | None = None,
    outpath: str | None = None,
):
    config = {
        "general": {
            "name_human": name_human if name_human else name_machine,
            "name_machine": name_machine,
            "output_base_folder": outpath if outpath else load_homepath() +"tmp/diffmap_data/",
            "map_sampling": 3,
            "high_resolution_limit": high_resolution_limit,
        },
        "input_files": {
            "map_dark": dataloc_dark,
            "map_triggered": dataloc_light,
            "pdb_dark": pdbloc_dark,
            "columns_dark": columns_dark,
            "columns_triggered": columns_triggered,
            "impose_dark_phases": True,
        },
        "masking": {
            "sigma": 3,
            "min_blob_size": 3,  # in A^3
            "blocking_radius": 1.5,
            "blocking_percentile": 95,
            "exclude_solvent": True,
            "dark_size_threshold": 0.1,
            "exclude_large_occupancy_outliers": False,
        },
        "map_processing": {
            "diffmap_type": "tv", # "kweighted", "tv", or "vanilla"
            "dark_mean_correction": True,
            "diffmap_mean_correction": True,
            "diffmap_v2_correction": False,
        },
        "plot": {
            "show_ignored_voxels": True,
            "set_ylim": False,
            "is_composite": False,
            "std_cutoff": 3.0,
            "solvent_density": 0.4,
            "minimum_datapoints": 10,
        },
    }

    output_folder = config["general"]["output_base_folder"] + "/" 
    config["general"]["output_folder"] = output_folder
    config["general"]["pdbloc_dark"] = pdbloc_dark
    return config
