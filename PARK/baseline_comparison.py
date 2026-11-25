"""Summarize matched PARK, PalmReach-6 and AUDE trials and render their shared-clock replays."""
import os

import numpy as np
import trimesh

from Module.palmreach6.__main__ import FIXTURES
from PARK.comparison import METHODS, collect
from PARK.core.common import table
from PARK.replays import NAMES, join_movies, load_view, replay


def reference_view(vertices):
    """Choose a three-quarter view from the reference shape alone, with its long axis upright."""
    centered = vertices - vertices.mean(axis=0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    up, _, front = axes
    height = centered @ up
    radius2 = np.square(centered).sum(1) - height**2
    lo, hi = np.quantile(height, [.15, .85])
    if radius2[height >= hi].mean() > radius2[height <= lo].mean():
        up = -up
    if front[np.argmax(abs(front))] < 0:
        front = -front
    eye = front + .40*np.cross(up,front) + .22*up
    eye /= np.linalg.norm(eye)
    horizontal = np.cross(up,eye)
    horizontal /= np.linalg.norm(horizontal)
    return np.column_stack((horizontal, np.cross(eye,horizontal), eye))


def draw_mesh(ax, mesh, vertices, basis, color):
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgb

    triangles = vertices[mesh.faces]
    order = np.argsort(triangles[:,:,2].mean(1), kind="stable")
    normals = mesh.face_normals @ basis
    normals[normals[:,2] < 0] *= -1
    light = np.clip(normals @ [-.42,.56,.71], 0, 1)
    shade = (.46+.54*light[:,None])*np.array(to_rgb(color))
    faces = PolyCollection(triangles[order,:,:2], facecolors=shade[order], edgecolors="face",
                           linewidths=.20, antialiaseds=False)
    ax.add_collection(faces)


def mesh_panel(path, objects, seconds, trials, mesh_files):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    colors = ("#949DA5", "#5FB7A5", "#7CA6D7", "#DDAD6C")
    titles = ("参考真值", *METHODS)
    height = 1.28+2.05*len(objects)
    with plt.rc_context({"font.family":"Microsoft YaHei"}):
        fig,axes = plt.subplots(len(objects),4,squeeze=False,figsize=(10.4,height))
        fig.subplots_adjust(left=.102,right=.985,top=1-.99/height,bottom=.25/height,wspace=.035,hspace=.24)
        fig.text(.025,1-.15/height,f"{seconds} 秒时的表面重建",ha="left",va="top",fontsize=18,weight="bold")
        fig.text(.025,1-.52/height,"每行采用相同视角和比例。C25 为实触覆盖率，F@5 为 5 mm 容差下的重建 F 分数。",
                 ha="left",va="top",fontsize=10,color="#48576A")
        for row,obj in enumerate(objects):
            truth = trimesh.load_mesh(FIXTURES/obj/"surface.obj",process=False)
            meshes = [truth]
            for method in METHODS:
                source = mesh_files[obj,method]
                meshes.append(trimesh.load_mesh(source,process=False) if source.is_file() else None)
            basis = reference_view(np.asarray(truth.vertices))
            projected = [np.asarray(m.vertices) @ basis if m is not None else None for m in meshes]
            xy = np.concatenate([v[:,:2] for v in projected if v is not None])
            low,high = xy.min(0),xy.max(0)
            center,half = .5*(low+high),.56*max(high-low)
            for col,ax in enumerate(axes[row]):
                ax.set_xlim(center[0]-half,center[0]+half)
                ax.set_ylim(center[1]-half,center[1]+half)
                ax.set_aspect("equal")
                ax.set_axis_off()
                if row == 0:
                    ax.set_title(titles[col],fontsize=13,weight="semibold",pad=3)
                if col == 0:
                    ax.text(-.055,.5,NAMES[obj],transform=ax.transAxes,ha="right",va="center",fontsize=12)
                if meshes[col] is not None:
                    draw_mesh(ax,meshes[col],projected[col],basis,colors[col])
                else:
                    ax.text(.5,.5,"无有效网格",transform=ax.transAxes,ha="center")
                if col:
                    result = trials[obj,METHODS[col-1]]
                    ax.text(.5,-.02,f"C25 {result['c25']:.1%}   ·   F@5 {result['f5']:.1%}",
                            transform=ax.transAxes,ha="center",va="top",fontsize=9.5,color="#344253")
        fig.savefig(path,dpi=160,facecolor="white")
        plt.close(fig)


def compare(root, seconds, objects, repeat=0):
    if seconds not in (240, 600):
        raise ValueError(f"No matched historical budget of {seconds} seconds")
    rows,_,_,sources = collect(root,[seconds],objects=objects,repeats=[repeat])
    output = root/"comparisons"/f"{seconds}s"/f"repeat-{repeat+1}"
    output.mkdir(parents=True,exist_ok=True)
    group = "-".join(objects)
    trials = {(r["object"],r["method"]):r for r in rows}
    mesh_files,details = {},[]
    for source in sources:
        obj,method = source["object"],source["method"]
        row = trials[obj,method]
        score = root/source["score"]
        mesh = score.with_name(f"signed-{seconds}s.ply")
        mesh_files[obj,method] = mesh
        details.append(dict(object=obj,seed=row["seed"],budget_s=seconds,method=method,
                            c25_pct=100*row["c25"],f5_pct=100*row["f5"],surface_error_mm=row["surface_error_mm"],
                            chamfer_mm2=row["chamfer_mm2"],score_file=os.path.relpath(score,output),
                            mesh_file=os.path.relpath(mesh,output) if mesh.is_file() else "",trace_sha256=source["trace"]))
    table(output/f"{group}.csv",details)
    mesh_panel(output/f"{group}.png",objects,seconds,trials,mesh_files)
    view_class = load_view()
    movies = [replay(root,obj,seconds,view_class,repeat) for obj in objects]
    if len(movies)>1:
        join_movies(movies,output/f"{group}.mp4")
    print(f"Verified {len(rows)} paired-method results and {len(movies)} synchronized replays at {output}",flush=True)
