import numpy as np
import matplotlib.pyplot as plt
from occupancy import *

try:
    import ipywidgets as widgets
except ImportError:
    print("no interactive widgets for you")
try:
    import gemmi
except ImportError:
    print("no gemmi support for you")
import matplotlib.animation as animation


def get_pos_from_pdb(struc: gemmi.Structure, search_occ=None):
    try:
        gemmi
    except ImportError:
        print("install gemmi to use this function")
        raise ImportError
    prefactors = np.array(
        [
            [1, 1, 1],
            [-1, -1, 1],
            [1, -1, -1],
            [-1, 1, -1],
        ]
    )
    translation = (
        np.array(
            [
                [0, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [0, 1, 1],
            ]
        )
        / 2
    )

    pos_list = np.array(
        [struc.cell.fractionalize(rca.atom.pos).tolist() for rca in struc[0].all()]
    )
    frac_list = pos_list % 1

    frac_add = [translation[ii] + prefactors[ii] * frac_list for ii in range(4)]

    frac_list = np.concatenate(frac_add, axis=0)
    frac_list = frac_list % 1
    if search_occ:
        occurences = np.array([np.round(x.atom.occ, 2) for x in struc[0].all()])
        occ_mask = 1 * (occurences == search_occ) + 2 * (occurences == 1 - search_occ)
        occ_mask = np.tile(occ_mask, (4))
        return frac_list, occ_mask
    return frac_list


def make_points2(frac_list, base_idx, imlen, masks=None):

    delta = 1 / imlen
    other_idcs = [[1, 2], [0, 2], [0, 1]][base_idx]
    wk_list = frac_list
    idx_list = np.array(np.round(wk_list[:, base_idx] / delta), int)
    list_2d = []
    for i in range(imlen):
        if (idx_list == i).any():
            indices = wk_list[idx_list == i][:, other_idcs].T
        else:
            indices = [[], []]
        # print(indices)
        list_2d.append(indices)
    if masks is None:
        return list_2d
    mask_slices = []
    for i in range(imlen):
        mask_slices.append(masks[idx_list == i])
    return list_2d, mask_slices


def make_extent(arr_shape, origin, delta):
    opposite = arr_shape * delta + origin
    d1 = delta[1] / 2
    d2 = delta[2] / 2
    extent = (origin[2] - d2, opposite[2] - d2, opposite[1] - d1, origin[1] - d1)
    extent = (opposite[2] - d2, origin[2] - d2, origin[1] - d1, opposite[1] - d1)
    return extent


def mtz_comp(
    frac_list,
    mask,
    mtzdata,
    gif_name="",
    extent=None,
):
    startval = 10
    kk = 0
    d2_points, d2_mask = make_points2(frac_list, kk, len(mtzdata), mask)
    xlen, ylen, zlen = mtzdata.shape
    zline = np.linspace(0, 1, zlen)
    xline = np.linspace(0, 1, xlen)
    plines = []
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot()
    ax.set_title(xline[0])
    mvals = np.unique(mask)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    extent = [1, 0, 0, 1] if extent is None else extent
    # im.set_extent(extent)
    im = plt.imshow(mtzdata[startval], extent=extent)
    labels = ["main", "Cis", "Trans"]
    marks = [".", "o", "o"]
    color = ["cyan", "red", "orange"]

    for boo in mvals:
        (pline,) = plt.plot(
            d2_points[startval][1][boo == d2_mask[startval]],
            d2_points[startval][0][boo == d2_mask[startval]],
            label=labels[int(boo)],
            linestyle="",
            marker=marks[int(boo)],
            c=color[int(boo)],
            alpha=0.8,
        )
        plines.append(pline)

    nline = np.linspace(0, 1, len(xline))
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    @widgets.interact(f0=(0, len(xline) - 1, 1))
    def update(f0=0):
        # divline.set_ydata(zloc[:,xline ==xline[f0]])
        # divline2.set_ydata(zloc2[:,xline ==xline[f0]])
        im.set_data(mtzdata[f0])
        ax.set_title(f"z={xline[f0]:.3f}")
        for i, boo in enumerate(mvals):
            plines[i].set_data(
                d2_points[f0][1][boo == d2_mask[f0]],
                d2_points[f0][0][boo == d2_mask[f0]],
            )

    if gif_name != "":
        anim = animation.FuncAnimation(fig, update, frames=len(xline), interval=500)
        anim.save(gif_name)
        plt.show()
        return anim
    return fig


def slice_3d(
    mtzdata, gif_name="", extent=None, startval=10, is_diff=False, imkwargs={}, fig=None
):
    xlen, ylen, zlen = mtzdata.shape
    xline = np.linspace(0, 1, xlen)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.gca()
    ax.set_title(xline[0])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    extent = [1, 0, 0, 1] if extent is None else extent
    if is_diff:
        vmax = np.max(np.abs(mtzdata))
        imkwargs = imkwargs | {"cmap": "bwr", "vmax": vmax, "vmin": -vmax}
    im = plt.imshow(mtzdata[startval], extent=extent, **imkwargs)
    plt.colorbar(im)

    @widgets.interact(f0=(0, len(xline) - 1, 1))
    def update(f0=0):
        im.set_data(mtzdata[f0])
        ax.set_title(f"z={xline[f0]:.3f}")

    if gif_name != "":
        anim = animation.FuncAnimation(fig, update, frames=len(xline), interval=500)
        anim.save(gif_name)
        plt.show()
        return anim
    return fig


def fname_variant(variant):
    match variant:
        case "basic" | "cistrans_nonoise":
            fname = "cistrans"
        case "basic-alt":
            fname = "cistrans_alt"
        case "noise" | "cistrans_noise":
            fname = "cistrans_noise"
        case "realmeteor":
            fname = "realmeteor"
        case _:
            print("using the following for plot name:", end="")
            print(variant)
            fname = variant
    return fname


def show_xtrs(
    dens_xtrs,
    alpha_xtrs_vis,
    obj0,
    alpha=None,
    version="only",
    variant="basic",
    make_gif=False,
):
    fig, axs = plt.subplots(2, 3, figsize=(8, 4), constrained_layout=True)
    imlen = dens_xtrs.shape[1]
    ims = []
    fname = fname_variant(variant)
    match version:
        case "only":
            obj_mod = obj0
            dens_mod = dens_xtrs
            title = r"$F_{xtr}$"
            fname += "_fxtr.gif"
        case "diff":
            obj_mod = obj0
            dens_mod = dens_xtrs - obj0
            title = r"$F_{xtr} -  F_0$"
            fname += "_diff.gif"
        case "diffnorm":
            obj_mod = obj0 / np.max(obj0)
            dens_mod = [dens / np.max(dens) - obj_mod for dens in dens_xtrs]
            title = r"$\frac{F_{xtr}}{max(F_{xtr})} -  \frac{F_0}{max(F_0)}$"
            fname += "_diffnorm.gif"
        case "diffnorm2":
            obj_mod = obj0
            dens_mod = [dens / np.max(dens) - obj_mod for dens in dens_xtrs]
            title = r"$\frac{F_{xtr}}{max(F_{xtr}} -  F_0$"
            fname += "_diffnorm2.gif"
        case "diffxtr":
            obj_mod = obj0
            dens_mod = [dens / np.max(dens) - obj_mod for dens in dens_xtrs]
            dens_mod = [(dens + obj_mod) / 2 for dens in dens_mod]
            title = r"modified F_{xtr}"
            fname += "_diffxtr.gif"
        case _:
            print(
                "use only the following: only, diff, diffnorm, diffnorm2 or diffxtr\n"
            )

            raise ValueError
    for ax, alpha_xtr, arr in zip(axs.flat, alpha_xtrs_vis, dens_xtrs):
        raw = [r"$\alpha_{xtr}$", r"$\alpha_t$"]
        tit = f"{raw[0]}: {alpha_xtr:.2f}"
        ax.set_title(tit)
        idx = 15
        vmax = 0.2
        im = ax.imshow((arr[idx]), cmap="bwr", vmin=-vmax, vmax=vmax)
        ims.append(im)
        plt.colorbar(im)
    fig.suptitle(title + f" z={idx}/{imlen-1}", fontsize=16)
    plt.show()

    @widgets.interact(f0=(0, dens_xtrs.shape[1] - 1, 1))
    def update(
        f0=idx,
    ):
        for dens, imo in zip(dens_mod, ims):
            imo.set_data(dens[f0])
        alpha_info = "\t " + r"$\alpha_{true}" + f"={alpha}$" if alpha else ""
        fig.suptitle(title + alpha_info + f"\t z={f0}/{imlen-1}", fontsize=16)

    if make_gif:
        interval = make_gif if not isinstance(make_gif, int) else 1000
        anim = animation.FuncAnimation(
            fig, update, frames=np.arange(0, dens_xtrs.shape[1]), interval=interval
        )
        loc = "gifs/" + fname
        print(loc)
        anim.save(loc)
        return anim


def direct_comp(
    dens_xtrs,
    alpha_xtrs_vis,
    obj0,
    obj1,
    version="only",
    variant="basic",
    make_gif=False,
    idx=15,
):
    titles = [r"$\alpha_{xtr}$" + f": {alpha_xtr:.2f}" for alpha_xtr in alpha_xtrs_vis]
    titles = ["obj0", *titles, "obj1-obj0"]

    fname = fname_variant(variant) + "_dcomp_"
    match version:
        case "only":
            obj_mod = obj0
            dens_mod = dens_xtrs
            title = r"$F_{xtr}$"
            fname += "_fxtr.gif"
        case "diff":
            obj_mod = obj0
            dens_mod = dens_xtrs - obj0
            title = r"$F_{xtr} -  F_0$"
            fname += "_diff.gif"
        case "diffnorm":
            obj_mod = obj0 / np.max(obj0)
            obj1 = obj1 / np.max(obj1)
            dens_mod = [dens / np.max(dens) - obj_mod for dens in dens_xtrs]
            title = r"$\frac{F_{xtr}}{max(F_{xtr})} -  \frac{F_0}{max(F_0)}$"
            fname += "_diffnorm.gif"
        case "diffnorm2":
            obj_mod = obj0
            dens_mod = [dens / np.max(dens) - obj_mod for dens in dens_xtrs]
            title = r"$\frac{F_{xtr}}{max(F_{xtr}} -  F_0$"
            fname += "_diffnorm2.gif"
        case "diffxtr":
            obj_mod = obj0
            dens_mod = [dens / np.max(dens) - obj_mod for dens in dens_xtrs]
            dens_mod = [(dens + obj_mod) / 2 for dens in dens_mod]
            title = r"modified $F_{xtr}$"
            fname += "_diffxtr.gif"
        case _:
            print(
                "use only the following: only, diff, diffnorm, diffnorm2 or diffxtr\n"
            )
            raise ValueError
    # dens_mod = np.flip(dens_mod, (1,2,3))[:,::-1]

    dens_mod = np.array([obj0, *dens_mod, obj1 - obj0])

    fig, axso = plt.subplots(
        2, len(titles), figsize=(len(dens_mod) * 2 + 1, 4), constrained_layout=True
    )
    imlen = dens_xtrs.shape[1]
    ims = []

    for axs in axso:
        for ax, arr, tit in zip(axs, dens_mod, titles):
            ax.set_title(tit)
            vmax = np.max(arr)
            im = ax.imshow((arr[idx]), cmap="bwr", vmin=-vmax, vmax=vmax)
            ims.append(im)
            plt.colorbar(im)
    for ax in axso[0]:
        ax.set_xlim(50, 76)
        ax.set_ylim(40, 65)
    fig.suptitle(title + f" z={idx}/{imlen-1}", fontsize=16)
    plt.show()

    denslen = len(dens_mod)

    @widgets.interact(f0=(0, dens_xtrs.shape[1] - 1, 1))
    def update(
        f0=idx,
    ):
        for dens, imo in zip(dens_mod, ims):
            imo.set_data(dens[f0])
        for dens, imo in zip(dens_mod, ims[denslen:]):
            imo.set_data(dens[f0])
        fig.suptitle(title + f"\t z={f0}/{imlen-1}", fontsize=16)

    if make_gif:
        interval = make_gif if not isinstance(make_gif, int) else 1000
        anim = animation.FuncAnimation(
            fig, update, frames=np.arange(0, dens_xtrs.shape[1]), interval=interval
        )
        loc = "gifs/" + fname
        print(loc)
        anim.save(loc)
        return anim


########################### Master Section #####################################


def add_fit(neg_sum, alpha_invs, n_largest, kwargs={}):
    kwargs = {"linestyle": "--", "alpha": 0.5} | {}
    alpha_line, fit_biggest1, fit_lowest1 = get_fits(neg_sum, alpha_invs, n_largest)
    plt.plot(
        alpha_line, fit_lowest1, **kwargs, c="red", label=f"Fit (largest {n_largest})"
    )
    plt.plot(
        alpha_line, fit_biggest1, **kwargs, c="g", label=f"Fit (smallest {n_largest})"
    )


def neg_sum_explosion(alpha_invs, neg_sum, config, n_largest=0, n_more=0, title=""):
    # alpha_line, fit_biggest2, fit_lowest2 = get_fits(neg_sum, alpha_invs, n_more)

    fig = plt.figure()
    plt.axvline(2 / config.alpha, c="k", linestyle="-.", label="2/alpha_true")
    if n_largest:
        add_fit(neg_sum, alpha_invs, n_largest)
    if n_more:
        add_fit(neg_sum, alpha_invs, n_more, kwargs={"linestyle": ".-"})

    plt.plot(alpha_invs, neg_sum * -1, "x")  # label="Datapoint")
    plt.legend()
    plt.xlabel("Inverse Occupancy")
    plt.ylabel(r"$\sum$ $|$neg. density$|$")
    plt.title(title)
    fname = "negsumexplosion"
    savefig(fig, config, fname)


def diffmap_versions(display_tup, alpha_xtrs, f_xtrs, mask_pks, obj0, delta_obj):
    display = {}

    dname = "fxtr"
    if dname in display_tup:
        display["fxtr"] = {"label": r"$F_{xtr}$ ", "color": "green"}
        out = x8_density_map_f1(f_xtrs, mask_pks, obj0, delta_obj)
        display[dname]["peak_sum"], display[dname]["real_CC"] = out

    dname = "fdiff"
    if dname in display_tup:
        display["fdiff"] = {"label": r"$F_{xtr}-F_0$", "color": "brown"}
        out = x8_density_map_fdiff(f_xtrs, mask_pks, obj0, delta_obj)
        display[dname]["peak_sum"], display[dname]["real_CC"] = out

    dname = "fconst"
    if dname in display_tup:
        display["fconst"] = {"label": r"$F_{xtr}-F_{0}+\mathcal N$", "color": "orange"}
        obj0_star = obj0 + np.random.normal(0, 0.1, size=obj0.shape) * obj0
        out = x8_density_map_fdiff(f_xtrs, mask_pks, obj0_star, delta_obj)
        display[dname]["peak_sum"], display[dname]["real_CC"] = out

    dname = "fmax"
    if dname in display_tup:
        display["fmax"] = {"label": r"norm($F_{xtr}$)-norm($F_0$)", "color": "blue"}
        out = x8_density_map_fdiff_norm(f_xtrs, mask_pks, obj0, delta_obj)
        display[dname]["peak_sum"], display[dname]["real_CC"] = out

    dname = "diffalpha"
    if dname in display_tup:
        display["diffalpha"] = {"label": r"$F_{xtr}-\mu(\alpha)F_0$", "color": "teal"}
        out = x8_density_map_fdiff_alpha(f_xtrs, mask_pks, obj0, delta_obj, alpha_xtrs)
        display[dname]["peak_sum"], display[dname]["real_CC"] = out
    return display


def difference_map_plot(
    alpha_xtrs,
    f_xtrs,
    mask_pks,
    obj0,
    delta_obj,
    config,
    display_tup=None,
    save_output=False,
):
    default_tup = ("fxtr", "fdiff", "diffalpha", "fconst", "fmax")
    display_tup = default_tup if display_tup is None else display_tup
    display = diffmap_versions(
        display_tup, alpha_xtrs, f_xtrs, mask_pks, obj0, delta_obj
    )

    fig, axs = plt.subplots(2, 2, figsize=(8, 8), tight_layout=True)
    for ax in axs[0]:
        ax.set_ylabel("Ratio prominent over all peaks")
        ax.set_title("Difference Map Method")

    for ax in axs[:, 0]:
        ax.axvline(1 / config.alpha, c="k", linestyle="-.", label="alpha_true")

    for ax in axs[:, 1]:
        ax.axvline(config.alpha, c="k", linestyle="-.", label="alpha_true")

    for ax in axs[1]:
        ax.axhline( 0, c="k", linewidth=0.5)
        ax.set_ylabel("Cross Correlation")
        ax.set_xlabel("Alphas")
        ax.set_title("Difference Map (CC) Method")

    alpha_inv = 1 / alpha_xtrs
    for disp in display.values():
        ax = axs[0, 0]
        ax.plot(alpha_inv, disp["peak_sum"], label=disp["label"], color=disp["color"])
        ax = axs[0, 1]
        ax.plot(alpha_xtrs, disp["peak_sum"], label=disp["label"], color=disp["color"])
        ax = axs[1, 0]
        ax.plot(alpha_inv, disp["real_CC"], label=disp["label"], color=disp["color"])
        ax = axs[1, 1]
        ax.plot(alpha_xtrs, disp["real_CC"], label=disp["label"], color=disp["color"])

    ax = axs[0, 0]
    ax.legend()

    ax = axs[1, 0]
    ax.set_xlabel("1/occupancy")

    ax = axs[1, 1]
    ax.set_xlabel("occupancy")

    fname = "differencemap"
    if save_output:
        savefig(fig, config, fname)


def pandda_bin_comp(delta_obj, strict, lax, config):
    bins = np.logspace(-6, 0, 100)
    bins = np.concatenate([[0], bins])
    fig = plt.figure()
    plt.hist(delta_obj.flatten(), bins=bins)
    plt.axvline(strict, label="narrow", color="r")
    plt.axvline(lax, label="wide", color="g")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"Value Distribution $\Delta \rho$")
    plt.ylabel("Frequency")
    fname = "pandda_bins"
    savefig(fig, config, fname)


def pandda_plot(alpha_xtrs, f_xtrs, f_dark, mask_pks_lax, mask_pks_strict, config):
    mean_local_strict, mean_global_strict = pandda(f_dark, f_xtrs, mask_pks_strict)
    mean_local_lax, mean_global_lax = pandda(f_dark, f_xtrs, mask_pks_lax)
    ml = [mean_local_strict, mean_local_lax]
    mg = [mean_global_strict, mean_global_lax]
    titles = ["narrow mask", "wide mask"]

    fig, axso = plt.subplots(2, 2, sharex=True, tight_layout=True)
    for axs, mean_local, mean_global, title in zip(axso.T, ml, mg, titles):
        ax = axs[0]
        ax.set_title(title)
        ax.axvline(
            config.alpha,
            c="k",
            linestyle="-.",
        )
        ax.plot(alpha_xtrs, +mean_global - mean_local, label="global-local")
        ax.set_ylabel("$\\Delta$ Cross correlation")
        ax.legend()
        ax = axs[1]
        ax.axvline(
            config.alpha,
            c="k",
            linestyle="-.",
        )
        ax.axhline(0, c="k", linestyle="--")
        ax.plot(alpha_xtrs, mean_local, label="local")
        ax.plot(alpha_xtrs, mean_global, label="global")
        ax.set_xlabel("Occupancy")
        ax.set_ylabel("Cross correlation with $F_0$")
        ax.legend()
        # ax = axs[2]
        # # ax.axvline(alpha/2,c="k", linestyle="-", label="1/2*alpha_true")
        # ax.axvline(alpha,c="k", linestyle="-.", label="alpha_true")
        # # ax.axvline(alpha*2,c="k", linestyle="--", label="2*alpha_true")
        # ax.plot(alpha_xtrs,+np.gradient(mean_global-mean_local), label="global-local")
        # ax.legend()

    fig.suptitle("PanDDA method")
    fname = "pandda"
    savefig(fig, config, fname)


def plot_density_match(ou1, config, title, fname):
    fig = plt.figure()
    bins = np.arange(0, 1, 0.01)
    vals, _, _ = plt.hist(ou1, alpha=0.5, density=True, bins=bins)
    plt.axvline(config.alpha, color="black", linestyle="--", label="True Occupancy")
    plt.axvline(np.mean(ou1), color="green", linestyle="-.", label="Mean")
    plt.axvline(np.median(ou1), color="blue", linestyle="--", label="Median")
    plt.legend()
    plt.ylabel("Frequency")
    plt.xlabel("Occupancy")
    plt.title(title)
    savefig(fig, config, fname)


def savefig(fig, config, fname):
    for ending, folder in zip([".png", ".pdf"], get_fig_folders()):
        loc = folder + fname_variant(config.imagetype)
        final_file_name = loc + "_" + fname + ending
        print(final_file_name)
        fig.savefig(final_file_name, bbox_inches="tight")


def density_matching(f_xtrs, alpha_xtrs, mask_pks_neg, config):
    root_blobs = root_finding_blobs(f_xtrs, alpha_xtrs, mask_pks_neg)
    root_voxel_3d = root_finding(f_xtrs, alpha_xtrs)
    root_voxel = root_voxel_3d[mask_pks_neg]
    # plt.figure()
    # plt.plot(1/alpha_xtrs,np.sum(f_xtrs[mask_pks_neg],axis=1))
    # plt.show()
    plot_density_match(
        root_blobs, config, "Density Matching (Blobs)", "density_matching_blobs"
    )
    plot_density_match(
        root_voxel, config, "Density Matching (Voxel)", "density_matching_voxel"
    )


from dataclasses import dataclass


@dataclass
class Config:
    imagetype: str
    alpha: float


def make_the_plots():
    imagetype = "cistrans_noise"

    obj0, obj1, f_dark, f_light, delta_fa_abs, alpha = generate_obj(
        imagetype, kwargs={}
    )
    config = Config(imagetype, alpha)
    if False:
        alpha_xtrs_vis = np.array([1, alpha * 2, alpha, alpha / 2, 0.19, 0.01])
        f_xtrs_vis = make_f_xtr(
            alpha_xtrs_vis, f_dark, f_light, np.angle(f_dark), version=1, noise_level=0
        )
        dens_xtrs_vis = [np.fft.ifftn(f_xtr).real for f_xtr in f_xtrs_vis]
        anim = show_xtrs(
            dens_xtrs_vis,
            alpha_xtrs_vis,
            obj0,
            alpha,
            version="only",
            variant=imagetype,
            make_gif=False,
        )
    title = "Negative Sum Explosion"

    alpha_invs = np.arange(0, 20) + 1e-10
    alpha_xtrs = 2 / alpha_invs
    f_xtrs = make_f_xtr(
        alpha_xtrs, f_dark, f_light, np.angle(f_dark), version=1, noise_level=0
    )
    _, neg_sum = marius(f_xtrs)
    n_largest = 9
    n_more = 3
    neg_sum_explosion(alpha_invs, neg_sum, config, n_largest, n_more, title=title)


def main():
    make_the_plots()


if __name__ == "__main__":
    main()
