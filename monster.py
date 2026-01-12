import matplotlib.pyplot as plt
import os


from masking import make_inclusion_mask
from processing import get_maps, prepare_maps
from configuration import get_file_config
from configuration import load_homepath
from estimation import plot_extrapolation_estimate
from logger import setup_logger


logger = setup_logger()


def apply_config_rsEGFP2() -> dict:
    homepath = load_homepath()
    folderloc = homepath + "meteor/test/data/"
    dataloc_dark = folderloc + "scaled-test-data.mtz"
    dataloc_light = folderloc + "scaled-test-data.mtz"
    pdbloc_dark = folderloc + "8a6g.pdb"
    name_machine = "rsEGFP2"
    columns_dark = {
        "amplitude_column": "F_off",
        "phase_column": "PHIC_nochrom",
        "uncertainty_column": "SIGF_off",
    }
    columns_triggered = {
        "amplitude_column": "F_on",
        "phase_column": "PHIC_chrom",
        "uncertainty_column": "SIGF_on",
    }
    high_resolution_limit = 1.6

    return get_file_config(
        dataloc_dark=dataloc_dark,
        dataloc_light=dataloc_light,
        pdbloc_dark=pdbloc_dark,
        columns_dark=columns_dark,
        columns_triggered=columns_triggered,
        high_resolution_limit=high_resolution_limit,
        name_machine=name_machine,
    )


def apply_config() -> dict:
    homepath = load_homepath()
    folderloc = homepath + "data/photolyase/"
    dataloc_dark = folderloc + "1_superdark/superdark_deposit.mtz"
    pdbloc_dark = folderloc + "1_superdark/superdark_deposit.pdb"
    datalocs_light = []
    for ii, f in enumerate(os.listdir(folderloc)):
        if not os.path.isdir(os.path.join(folderloc, f)):
            continue
        if f[:2] == "1_":
            continue
        if not f[0].isdigit():
            continue
        changing_bit = f + "/" + f.split("_")[-1]
        dataloc_light = folderloc + changing_bit + "_deposit.mtz"
        print(ii, changing_bit)

        datalocs_light.append(dataloc_light)
    dataloc_light = datalocs_light[2]  # Just use the first
    high_resolution_limit = 2.6

    name_machine = "Photolyase"
    columns_dark = dict(
        amplitude_column="F-obs-filtered",
        uncertainty_column="SIGF-obs-filtered",
        phase_column="PHIF-model",
    )
    columns_triggered = dict(
        amplitude_column="F", uncertainty_column="SIGF", phase_column="PHIF-model"
    )
    print(dataloc_light)

    return get_file_config(
        dataloc_dark=dataloc_dark,
        dataloc_light=dataloc_light,
        pdbloc_dark=pdbloc_dark,
        columns_dark=columns_dark,
        columns_triggered=columns_triggered,
        high_resolution_limit=high_resolution_limit,
        name_machine=name_machine,
    )


def main() -> None:
    config = apply_config_rsEGFP2()
    unscaled_dark, unscaled_triggered = get_maps(config)
    diffmap, map_dark, _ = prepare_maps(unscaled_dark, unscaled_triggered, config)

    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=config["general"]["map_sampling"])
    logger.warning(
        f"Diffmap Mean: {diffmap_np.mean():.6f}, Std: {diffmap_np.std():.6f}"
    )
    inclusion_mask = make_inclusion_mask(diffmap, map_dark, config)
    # diffmap_np, map_dark_np = _prepare_maps(
    #     diffmap, map_dark, map_dark_comp, info_container, params
    # )
    fig, ax = plot_extrapolation_estimate(diffmap, map_dark, inclusion_mask, config)
    plt.show()


if __name__ == "__main__":
    main()
