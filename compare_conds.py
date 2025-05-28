import numpy as np
from scipy.ndimage import gaussian_filter

# my imports
from occupancy import *
from plotting3d import *
from generate_objects import *

def get_intersect_and_angle(alpha_invs, neg_sum):
    n_largest = 4
    a_sorted = np.argsort(alpha_invs)
    m1 = a_sorted <= n_largest
    m2 = a_sorted >= len(a_sorted) - n_largest
    res_1 = stats.linregress(alpha_invs[m1], -neg_sum[m1])
    res_2 = stats.linregress(alpha_invs[m2], -neg_sum[m2])
    print(res_1.slope,  res_2.slope)
    intersection = (res_2.intercept-res_1.intercept) / (res_1.slope-res_2.slope)
    
    angle_raw = (res_1.slope-res_2.slope) / (1 + res_1.slope*res_2.slope)
    angle = np.degrees(np.arctan(np.abs(angle_raw)))
    return intersection, angle

def main():
    alpha_xtrs = np.linspace(1e-2,1, 150)
    alpha_xtrs = np.linspace(1e-2,1, 50)
    alpha_invs = (np.arange(0,20)+1e-10)
    alpha_xtrs_other = 2/alpha_invs
    alpha_xtrs = np.sort(np.concatenate([alpha_xtrs, alpha_xtrs_other]))[::-1]
    alpha_invs = 2/alpha_xtrs

    alpha_0 = 0.27
    obj0, obj1, _, _, _, _ =  generate_obj_cistrans_v2(alpha_0, noise_level=0)


    delta_obj = obj1-obj0
    mask_thresh_neg = 0.03
    mask_pks_neg = (delta_obj)<-mask_thresh_neg
    mask_thresh = 0.05
    mask_pks = np.abs(delta_obj)<mask_thresh


    config = Config(imagetype, alpha)
    alphas = [0.54]
    noise_levels = [0.3]
    for noise_level in noise_levels:
        for alpha in alphas: 
            obj0, obj1, f_dark, f_light, delta_fa_abs, imagetype =  generate_obj_cistrans_v2(alpha, noise_level)
            delta_obj = obj1-obj0
            f_xtrs = make_f_xtr(alpha_xtrs, f_dark, f_light, np.angle(f_dark),  version=1, noise_level = 0)

            mean_local_strict, mean_global_strict = pandda(f_dark, f_xtrs, mask_pks)
            panda_res = alpha_xtrs[np.argmax(mean_global_strict-mean_local_strict)]

            _, neg_sum = marius(f_xtrs, mask_pks_neg )
            intersection, angle = get_intersect_and_angle(alpha_invs, neg_sum)
            marius_res = 2/intersection
