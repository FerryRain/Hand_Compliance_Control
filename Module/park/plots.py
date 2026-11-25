"""Offline density and planning figures. All distances in metres unless labelled."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


from .kde import HandKDE
from .settings import FINGERS


def kde_figure(directory):
    directory = Path(directory)
    hand = HandKDE.load(directory)
    report = json.loads((directory / "kde-validation.json").read_bytes())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    colors = ["#3779b9", "#00866d", "#aa7523", "#ac4977"]
    for i, (m, name) in enumerate(zip(hand.models, FINGERS)):
        ax = axes.flat[i]
        p = m.centers * 1000
        # Integrate out Z analytically for an XY marginal of the fitted isotropic 3-D KDE.
        xy = [np.linspace(p[:, j].min() - 6, p[:, j].max() + 6, 80) for j in range(2)]
        x, y = np.meshgrid(*xy)
        h = 1000 * m.bandwidth
        dx = np.exp(-.5 * ((xy[0][:, None] - p[:, 0]) / h)**2)
        dy = np.exp(-.5 * ((xy[1][:, None] - p[:, 1]) / h)**2)
        z = (dy * m.weights) @ dx.T / (2 * np.pi * h**2)
        ax.contourf(x, y, z, levels=14, cmap="viridis")
        ax.set(title=name.title() + "  ·  XY marginal density", xlabel="palm X (mm)", ylabel="palm Y (mm)", aspect="equal")
        axes.flat[4].contour(x, y, z, levels=[z.max() * .12], colors=[colors[i]])
        axes.flat[4].plot([], [], color=colors[i], label=name)
        bins = report["fingers"][name]["calibration_bins"]
        axes.flat[5].plot([b["predicted"] for b in bins], [b["observed"] for b in bins],
                          ".-", color=colors[i], label=name)
    axes.flat[4].set(title="Four fingers in the same palm frame", xlabel="palm X (mm)", ylabel="palm Y (mm)", aspect="equal")
    axes.flat[4].legend(fontsize=8)
    ax = axes.flat[5]
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "--", color=".5")
    ax.set(title="Held-out ovoid  ·  10 mm ball mass", xlabel="predicted probability", ylabel="observed frequency")
    fig.savefig(directory / "reachability.png", dpi=150)
    plt.close(fig)


def scatter_axes(ax, points):
    if not len(points):
        return
    low, high = points.min(0), points.max(0)
    span = max(float((high - low).max()), .012) * .53
    center = (low + high) / 2
    for set_lim, v in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        set_lim(v - span, v + span)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel="X (m)", ylabel="Y (m)", zlabel="Z (m)")
    ax.tick_params(labelsize=7)


def paint_mesh(ax, vertices, faces, scalar=None):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    subset = faces[::max(1, int(np.ceil(len(faces) / 6000)))]
    polygons = Poly3DCollection(vertices[subset], linewidth=0., alpha=.86)
    if scalar is None:
        polygons.set_facecolor("#c3d2d9")
    else:
        polygons.set_cmap("viridis")
        polygons.set_array(scalar[subset].mean(1))
        polygons.set_clim(0., 1.)
    ax.add_collection3d(polygons)
    return polygons


def posterior_mesh(gp, contacts):
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes
    margin = .022
    origin = contacts.min(0) - margin
    extent = contacts.max(0) + margin - origin
    spacing = max(.0045, extent.max() / 74)
    axes = [np.arange(int(np.ceil(v / spacing)) + 1) for v in extent]
    xyz = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1) * spacing + origin
    flat = xyz.reshape(-1, 3)
    tree = cKDTree(contacts)
    mask = np.isfinite(tree.query(flat, distance_upper_bound=margin)[0])
    field = np.full(len(flat), gp.cfg.prior_m)
    ids = np.flatnonzero(mask)
    for i in range(0, len(ids), 512):
        ix = ids[i:i + 512]
        field[ix] = gp.query(flat[ix])[0]
    if not (field[mask].min() < 0 < field[mask].max()):
        return None
    shape = xyz.shape[:-1]
    vertices, faces, _, _ = marching_cubes(field.reshape(shape), 0, spacing=(spacing,)*3,
                                            mask=mask.reshape(shape), allow_degenerate=False)
    vertices += origin
    means, variances = [], []
    for i in range(0, len(vertices), 512):
        mean, var, _ = gp.query(vertices[i:i + 512])
        means.extend(mean)
        variances.extend(var)
    # Strip artificial mask-boundary faces whose vertices are off the inferred zero level.
    keep = np.max(np.abs(np.asarray(means)[faces]), axis=1) <= .0025
    return vertices, faces[keep], np.asarray(variances) / gp.cfg.output_scale_m**2


def trial_figure(out, meta, q, contacts, mesh, quality, series, example, gp):
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    ax = fig.add_subplot(2, 3, 1, projection="3d")
    if mesh is not None:
        paint_mesh(ax, mesh.vertices, mesh.faces)
    path = q[::max(1, len(q) // 1500), :3]
    ax.plot(*path.T, color="#bd441b", lw=1.1, label="measured wrist")
    ax.scatter(*path[0], marker="o", color="black", s=12)
    ax.set_title("Reconstruction and measured wrist")
    scatter_axes(ax, np.vstack((contacts, path)))
    ax.legend(fontsize=7)
    ax = fig.add_subplot(2, 3, 2, projection="3d")
    inferred = posterior_mesh(gp, contacts)
    if inferred is not None and len(inferred[1]):
        polygons = paint_mesh(ax, *inferred)
        fig.colorbar(polygons, ax=ax, shrink=.6, label="variance / prior variance")
    else:
        ax.scatter(*contacts.T, s=2)
    scatter_axes(ax, contacts)
    ax.set_title("Signed GPIS zero surface")
    ax = fig.add_subplot(2, 3, 3)
    secs = [r["seconds"] for r in series]
    ax.plot(secs, [r["c25"] for r in series], label="C25", color="#007d73")
    ax.plot(secs, [r["c10"] for r in series], label="C10", color="#314d77")
    ax.set(xlabel="exploration time (s)", ylabel="geodesic surface coverage", ylim=(0, 1.02))
    ax.legend()
    ax = fig.add_subplot(2, 3, 4)
    clock = [r["seconds"] for r in quality]
    f5 = [r["f5"] for r in quality]
    ax.plot(clock, f5, "o-", color="#314d77")
    ax.set(xlabel="exploration time (s)", ylabel="reconstruction F@5", ylim=(0, 1.02))
    ax = fig.add_subplot(2, 3, 5)
    ax.plot([r["path_m"] for r in quality], f5, "o-", color="#314d77")
    ax.set(xlabel="measured wrist path length (m)", ylabel="reconstruction F@5", ylim=(0, 1.02))
    ax = fig.add_subplot(2, 3, 6)
    ax.plot(clock, [r["mean_reference_variance_m2"] * 1e6 for r in quality], "o-", color="#945619")
    ax.set(xlabel="exploration time (s)", ylabel="mean GPIS variance (mm²)", title="Fixed offline surface probes")
    fig.suptitle(f"{meta['object']}   /   {meta['method']}   /   {meta['seconds']} s   /   seed {meta['seed']}")
    fig.savefig(out / "exploration.png", dpi=130)
    plt.close(fig)
    if example is not None:
        selection_figure(out, example)


def selection_figure(out, example):
    from .settings import rotation
    fig = plt.figure(figsize=(12, 8.5), constrained_layout=True)
    fig.get_layout_engine().set(hspace=.16)
    surf = np.asarray(example["surface"])
    poses = np.asarray(example["poses"])
    j = example["selected"]
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.scatter(*surf.T, c=example["variance_m2"], s=10, cmap="viridis")
    rep = surf[example["regions"]]
    ax.scatter(*rep.T, color="#be4319", marker="x", label="target regions")
    ax.quiver(*poses[:, :3].T, *(-rotation(poses).as_matrix()[:, :, 2]).T,
              length=.011, linewidth=.5, alpha=.40)
    ax.scatter(*poses[j, :3], marker="*", s=90, color="#be4319", label="selected wrist")
    nominal = np.array(example["trajectory"])
    ax.plot(*nominal.T, color="#be4319", lw=2, label="published segment")
    scatter_axes(ax, np.vstack((surf, poses[:, :3], nominal)))
    ax.set_title("Uncertain targets and palm-facing axes")
    ax.legend(fontsize=7)
    ax = fig.add_subplot(2, 2, 2, projection="3d")
    local = np.asarray(example["palm_points"])
    ax.scatter(*local.T, c=example["variance_m2"], s=18, cmap="viridis")
    ax.scatter(0, 0, 0, marker="+", color="black")
    scatter_axes(ax, np.vstack((local, [0, 0, 0])))
    ax.set_title("Uncertain surface in the selected palm frame")
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(example["independent"], label="independent", color="#006e65")
    ax.plot(example["maximum"], label="max finger", color="#ab6222")
    ax.scatter(j, example["reach"][j], marker="*", s=75, color="#be4319")
    ax.set(xlabel="candidate index", ylabel="uncertainty-weighted ball probability")
    ax.legend(fontsize=8)
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(example["scores"], ".-", color="#314d77", ms=2)
    ax.scatter(j, example["scores"][j], marker="*", s=80, color="#be4319",
               label=f"selected at {example['tick'] / 100:g} s")
    ax.set(xlabel="candidate index", ylabel="selection score")
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(out / "wrist-selection.png", dpi=130)
    plt.close(fig)

