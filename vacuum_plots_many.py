import matplotlib.pyplot as plt
import os


from masking import make_inclusion_mask
from processing import get_maps, prepare_maps
from configuration import get_file_config
from configuration import load_homepath
from estimation import plot_extrapolation_estimate, plot_extrapolation_estimate_new
from logger import setup_logger


logger = setup_logger()


def loader_double(config:dict, ax: plt.Axes, ax_old) -> None:
    unscaled_dark, unscaled_triggered = get_maps(config)
    diffmap, map_dark, _ = prepare_maps(unscaled_dark, unscaled_triggered, config)

    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=config["general"]["map_sampling"])
    logger.warning(
        f"Diffmap Mean: {diffmap_np.mean():.6f}, Std: {diffmap_np.std():.6f}"
        
    )
    inclusion_mask = make_inclusion_mask(diffmap, map_dark, config)
    _ = plot_extrapolation_estimate(diffmap, map_dark, inclusion_mask, config, ax=ax_old)
    _ = plot_extrapolation_estimate_new(diffmap, map_dark, inclusion_mask, config, ax=ax)
    ax.set_title(config["general"]["name_human"])
    ax_old.set_title(config["general"]["name_human"])

def loader(config:dict, ax: plt.Axes, ) -> None:
    unscaled_dark, unscaled_triggered = get_maps(config)
    diffmap, map_dark, _ = prepare_maps(unscaled_dark, unscaled_triggered, config)

    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=config["general"]["map_sampling"])
    logger.warning(
        f"Diffmap Mean: {diffmap_np.mean():.6f}, Std: {diffmap_np.std():.6f}"
        
    )
    inclusion_mask = make_inclusion_mask(diffmap, map_dark, config)
    _ = plot_extrapolation_estimate_new(diffmap, map_dark, inclusion_mask, config, ax=ax)
    ax.set_title(config["general"]["name_human"])


from dataset_configs import get_all_configs

def main() -> None:
    config_list = get_all_configs()
    cols = 2
    rows = len(config_list)//cols+len(config_list)%cols
    kwargs = dict(nrows=rows,ncols=cols, figsize=(10,10), tight_layout=True)
    fig, axs = plt.subplots(**kwargs) # type: ignore
    fig_old, axs_old = plt.subplots(**kwargs) # type: ignore
    for ax,ax_old, config in zip(axs.flatten(), axs_old.flatten(), config_list):
        loader_double(config, ax, ax_old)
    plt.show()

if __name__ == "__main__":
    main()
