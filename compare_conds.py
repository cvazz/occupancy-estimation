import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd
import logging
# my imports
from occupancy import *
from plotting3d import *
from generate_objects import *


def get_intersect_and_angle(
    alpha_invs,
    neg_sum,
    n_largest,
    print_it=False,
):
    a_sorted = np.argsort(alpha_invs)
    m1 = a_sorted <= n_largest
    m2 = a_sorted >= len(a_sorted) - n_largest
    res_1 = stats.linregress(alpha_invs[m1], -neg_sum[m1])
    res_2 = stats.linregress(alpha_invs[m2], -neg_sum[m2])
    intersection = (res_2.intercept - res_1.intercept) / (res_1.slope - res_2.slope)
    # print("intersection slopes", res_1.slope,  res_2.slope, intersection, 2/intersection)
    logging.debug(f"intersection point, {2/intersection:.3f} ({intersection:.2f})")

    angle_raw = (res_1.slope - res_2.slope) / (1 + res_1.slope * res_2.slope)
    angle = np.degrees(np.arctan(np.abs(angle_raw)))
    return intersection, angle


def default_alpha_xtrs():
    alpha_xtrs = np.linspace(1e-2, 1, 150)
    alpha_xtrs = np.linspace(1e-2, 1, 40)
    alpha_invs = np.linspace(1, 2, 4)
    alpha_invs2 = np.array((50, 100, 150))
    alpha_xtrs_other = 2 / alpha_invs
    alpha_xtrs_other2 = 2 / alpha_invs2
    alpha_xtrs = np.sort(
        np.concatenate([alpha_xtrs, alpha_xtrs_other, alpha_xtrs_other2])
    )[::-1]
    return alpha_xtrs


def main(alpha_xtrs, alpha_trues, noise_levels):
    alpha_invs = 2 / alpha_xtrs

    alpha_0 = 0.27
    obj0, obj1, _, _, _, _ = generate_obj_cistrans_v2(alpha_0, noise_level=0)

    delta_obj = obj1 - obj0
    mask_thresh_neg = 0.03
    mask_pks_neg = (delta_obj) < -mask_thresh_neg
    mask_thresh = 0.05
    n_largest = 4
    mask_pks = np.abs(delta_obj) > mask_thresh
    # plt.figure()
    # plt.plot(alpha_xtrs, "x")
    # plt.show()
    cols = ["true", "noise", "neg_sum", "pandda"]

    rows = []
    for noise_level in noise_levels:
        for alpha in alpha_trues:
            obj0, obj1, f_dark, f_light, delta_fa_abs, imagetype = (
                generate_obj_cistrans_v2(alpha, noise_level)
            )
            config = Config(imagetype, alpha)
            delta_obj = obj1 - obj0
            f_xtrs = make_f_xtr(
                alpha_xtrs, f_dark, f_light, np.angle(f_dark), version=1, noise_level=0
            )

            mean_local_strict, mean_global_strict = pandda(f_dark, f_xtrs, mask_pks)
            panda_res = alpha_xtrs[np.argmax(mean_global_strict - mean_local_strict)]

            _, neg_sum = marius(f_xtrs, mask_pks_neg)
            intersection, angle = get_intersect_and_angle(
                alpha_invs, neg_sum, n_largest
            )
            marius_res = 2 / intersection
            row = [
                alpha,
                noise_level,
                marius_res,
                panda_res,
            ]
            rows.append(row)
            # print("hi")
            # print(panda_res, marius_res)
            # neg_sum_explosion(alpha_invs, neg_sum, config, 7, 3)
            # plt.show()
            # fig,axs = plt.subplots(1,2)
            # pandda_actual_plot(alpha_xtrs, mean_global_strict, mean_local_strict, axs, "", config)
            # plt.show()

    res_log = pd.DataFrame(data=rows, columns=cols)
    return res_log, rows
    plt.figure()
    plt.plot(res_log.true, res_log.neg_sum)
    plt.plot(res_log.true, res_log.pandda)
    plt.show()


if __name__ == "__main__":
    main()
