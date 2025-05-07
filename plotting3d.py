import numpy as np
import matplotlib.pyplot as plt

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

    # return fig
    if gif_name != "":
        anim = animation.FuncAnimation(fig, update, frames=len(xline), interval=500)
        anim.save(gif_name)
        plt.show()
        return anim


def slice_3d(mtzdata, gif_name="", extent=None, startval=10, imkwargs={}):
    xlen, ylen, zlen = mtzdata.shape
    xline = np.linspace(0, 1, xlen)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot()
    ax.set_title(xline[0])
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    extent = [1, 0, 0, 1] if extent is None else extent
    imkwargs = {"cmap": "bwr", **imkwargs}
    im = plt.imshow(mtzdata[startval], extent=extent, **imkwargs)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    @widgets.interact(f0=(0, len(xline) - 1, 1))
    def update(f0=0):
        im.set_data(mtzdata[f0])
        ax.set_title(f"z={xline[f0]:.3f}")

    if gif_name != "":
        anim = animation.FuncAnimation(fig, update, frames=len(xline), interval=500)
        anim.save(gif_name)
        plt.show()
        return anim


def fname_variant(variant):
    match variant:
        case "basic":
            fname = "cistrans"
        case "basic-alt":
            fname = "cistrans_alt"
        case "noise":
            fname = "cistrans_noise"
        case "realmeteor":
            fname = "realmeteor"
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
        fig.suptitle(title + alpha_info + "\t z={f0}/{imlen-1}", fontsize=16)

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


def main():
    pass
