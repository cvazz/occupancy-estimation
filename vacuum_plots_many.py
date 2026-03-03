import matplotlib.pyplot as plt
import os


from masking import make_inclusion_mask
from processing import get_maps, prepare_maps
from configuration import get_file_config
from configuration import load_homepath
from estimation import plot_extrapolation_estimate
from logger import setup_logger


logger = setup_logger()



def loader(config:dict, ax: plt.Axes) -> None:
    unscaled_dark, unscaled_triggered = get_maps(config)
    diffmap, map_dark, _ = prepare_maps(unscaled_dark, unscaled_triggered, config)

    diffmap_np = diffmap.to_3d_numpy_map(map_sampling=config["general"]["map_sampling"])
    logger.warning(
        f"Diffmap Mean: {diffmap_np.mean():.6f}, Std: {diffmap_np.std():.6f}"
    )
    inclusion_mask = make_inclusion_mask(diffmap, map_dark, config)
    fig, ax = plot_extrapolation_estimate(diffmap, map_dark, inclusion_mask, config, ax=ax)

from dataset_configs import get_all_configs

def main() -> None:
    config_list = get_all_configs()
    fig, axs = plt.subplots(len(config_list)//2+len(config_list)%2,2)
    for ax, config in zip(axs.flatten(), config_list):
        loader(config, ax)
    plt.show()

if __name__ == "__main__":
    main()
