"""Score a finished tactile rollout against its paired offline reference surface."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.linalg import cholesky
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist
import trimesh

from .active_gpis import NOISE_VARIANCE, VOXEL_M
from .experiment import digest, read_json, write_json
from .full_signed import signed_surface_for_output
from ..i04_geometric_traversal import fingertip_r_gpis as ridge
from ..i04_geometric_traversal.contracts import ExperimentConfig
from ..i04_geometric_traversal.fingertip_surface_report import mesh_accuracy, MESH_SAMPLES, METRIC_SEED
from ..i04_geometric_traversal.fixtures import SurfaceGraph
from ..i04_geometric_traversal.oracle import MeasuredCoverage


def score(folder, reference):
    folder, reference = Path(folder), Path(reference)
    run = read_json(folder/'run.json')
    if digest(folder/'trace.npz')!=run['trace_sha256'] or digest(reference)!=run['offline_mesh_sha256']:
        raise ValueError('Trace or evaluation mesh failed source verification')
    with np.load(folder/'trace.npz') as tr:
        loads,live = tr['region_load_n'],tr['pad_valid']
        qualified = live & (loads.max(1)<8)[:,None]
        ticks,fingers = np.nonzero(qualified)
        points = tr['points_m'][ticks,fingers]
        normals = tr['normals_outward'][ticks,fingers]
    end=len(qualified)-1
    mesh=trimesh.load_mesh(reference,process=False)
    graph=SurfaceGraph(mesh,coverage_radius_m=.025)
    areas=MeasuredCoverage(graph,ExperimentConfig()).vertex_area
    first=np.full(len(areas),end+1,dtype=np.int32)
    for idx in range(0,len(ticks),10000):
        sl=slice(idx,idx+10000)
        verts,error=graph.oriented_nearest_vertices(points[sl],normals[sl])
        ok=(error<=.005)&(np.einsum('ij,ij->i',graph.normals[verts],normals[sl])>=.5)
        np.minimum.at(first,verts[ok],ticks[sl][ok])
    earliest=np.full((2,len(areas)),end+1,dtype=np.int32)
    for v in np.flatnonzero(first<=end):
        distances=dijkstra(graph.adjacency,indices=v,directed=False,limit=.025)
        for i,radius in enumerate((.010,.025)):
            near=distances<=radius
            earliest[i,near]=np.minimum(earliest[i,near],first[v])
    c10,c25=[np.minimum(1.,np.cumsum(np.bincount(k,weights=areas/areas.sum(),minlength=end+2))[:end+1])
             for k in earliest]
    np.savez_compressed(folder/'coverage.npz',time_s=np.arange(end+1)*.01,c10=c10,c25=c25)
    _,first_cell=np.unique(np.floor(points/VOXEL_M).astype(int),axis=0,return_index=True)
    first_cell.sort()
    cells=points[first_cell]
    initial=int((ticks[first_cell]==0).sum())
    covariance=np.empty((len(cells),len(cells)),order='F')
    for start in range(0,len(cells),256):
        sl=slice(start,start+256)
        u=cdist(cells[sl],cells)*(np.sqrt(3.)/.025)
        covariance[:,sl]=((1.+u)*np.exp(-u)).T
    covariance.flat[::len(cells)+1]+=NOISE_VARIANCE
    factor=cholesky(covariance,lower=True,overwrite_a=True,check_finite=False)
    information=float(np.sum(np.log(factor.diagonal()[initial:])-.5*np.log(NOISE_VARIANCE)))
    t0=perf_counter()
    ids=ridge.voxel_representatives(points)
    if len(ids)>ridge.CONTACT_BUDGET:
        ids=ids[ridge.farthest_indices(points[ids],ridge.CONTACT_BUDGET)]
    surface,_,device=signed_surface_for_output(points[ids],normals[ids])
    surface.export(folder/f'signed-{end*.01:g}s.ply')
    truth=trimesh.sample.sample_surface(mesh,MESH_SAMPLES,seed=METRIC_SEED+1)[0]
    accuracy=mesh_accuracy(surface,truth)
    signed=accuracy['at_mm']['5']
    result=dict(coverage=float(c10[-1]),coverage25=float(c25[-1]),f5=signed['fscore'],
                precision5=signed['precision'],recall5=signed['recall'],cells=len(cells),total_nat=information,
                light2_s=float(np.sum(qualified[1:].sum(1)>=2)*.01),light3_s=float(np.sum(qualified[1:].sum(1)>=3)*.01),
                zero_tip_s=float(np.sum(live[1:].sum(1)==0)*.01),over8_s=float(np.sum(loads[1:].max(1)>=8)*.01),
                final_surface_error_mm=accuracy['symmetric_mean_mm'],device=device,final_gpis_and_score_s=perf_counter()-t0)
    write_json(folder/'portable-scores.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path)
    p.add_argument('--surface',required=True,type=Path)
    args=p.parse_args()
    print(score(args.run,args.surface))
