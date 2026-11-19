"""Prepare scaled, closed YCB candidates; never grants physics/training admission."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--objects', nargs='+', help='Subset of the frozen candidate/split list')
 a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 a.output.mkdir(parents=True)
 import vhacdx
 # Freeze parent-level splits before physical runs. Existing Mustard is development.
 candidates=[('002_master_chef_can','train'),('003_cracker_box','train'),('021_bleach_cleanser','train'),('024_bowl','train'),('025_mug','validation'),('019_pitcher_base','test')]
 # Additional parents are opt-in; existing held-out identities never change.
 extra=[('001_chips_can','train'),('004_sugar_box','train'),('005_tomato_soup_can','train'),('010_potted_meat_can','train'),('011_banana','train'),('013_apple','train'),('016_pear','train'),('022_windex_bottle','train')]
 if a.objects:candidates += [x for x in extra if x[0] in a.objects]
 if a.objects:
  if set(a.objects)-{n for n,_ in candidates}:raise ValueError('Unknown candidate')
  candidates=[(n,s) for n,s in candidates if n in a.objects]
 rows=[];reject=[]
 def save():
  (a.output/'object_manifest.json').write_text(json.dumps(dict(schema='contact_wm_ycb_pool_v1',objects=rows,rejected=reject,training_admissible=False,split_rule='All scales, orientations, motions and branches of a YCB parent stay together; test not used for DP selection.'),indent=2))
 for name,split in candidates:
  source=a.source/name/'google_16k/textured.obj';start=time.monotonic()
  try:
   m=trimesh.load(source,force='mesh',process=True)
   source_vertex_count=len(m.vertices)
   # OBJ texture/normal seams duplicate geometric vertices. Weld seams only.
   before=m.triangles.copy()
   m.merge_vertices(merge_tex=True,merge_norm=True)
   seam_error=float(np.max(abs(before-m.triangles)))
   if seam_error>1e-8:raise ValueError('Seam welding changed triangle geometry')
   removed_components=[]
   if not m.is_watertight:
    parts=list(m.split(only_watertight=False,repair=False))
    main=max(parts,key=lambda x:len(x.faces))
    debris=[x for x in parts if x is not main]
    # Discard only zero-volume scan debris; never fill holes or vessel mouths.
    if (main.is_watertight and debris and
        all(len(x.faces)<=2 and x.area<1e-7 and abs(x.volume)<1e-12 for x in debris)):
     removed_components=[dict(faces=len(x.faces),area_m2=float(x.area),volume_m3=float(x.volume)) for x in debris]
     m=main
   if not np.isfinite(m.vertices).all() or not m.is_watertight or not m.is_winding_consistent or m.volume<=0:
    raise ValueError('Source not a verified closed consistently oriented solid; no automatic hole filling')
   original=m.extents.copy();order=np.argsort(original)[::-1]
   rotation=np.eye(3)[order]
   if np.linalg.det(rotation)<0:rotation[2]*=-1
   scale=max(1.,.36/original[order[0]],.22/original[order[1]])
   if scale>10:raise ValueError('Scale above10 requires manual geometry review')
   center=m.bounds.mean(0);m.vertices=(m.vertices-center)@rotation.T*scale
   oid='ycb_'+name;folder=a.output/'objects'/oid;folder.mkdir(parents=True);m.export(folder/'surface.obj')
   print('DECOMPOSE',name,'scale',scale,'extents',m.extents.tolist(),flush=True)
   pieces=vhacdx.compute_vhacd(np.asarray(m.vertices,np.float64),np.c_[np.full(len(m.faces),3),m.faces].astype(np.uint32).ravel(),maxConvexHulls=128,resolution=1000000,minimumVolumePercentErrorAllowed=.1,maxRecursionDepth=12,shrinkWrap=True,minEdgeLength=1,maxNumVerticesPerCH=96,asyncACD=False)
   files=[];points=m.vertices[np.random.default_rng(20261111).choice(len(m.vertices),min(2000,len(m.vertices)),replace=False)];gap=np.full(len(points),np.inf)
   for k,(v,f) in enumerate(pieces):
    piece=trimesh.Trimesh(v,f,process=True)
    if not piece.is_watertight or piece.volume<=0:raise ValueError('Invalid decomposition solid')
    path=folder/f'collision_{k:03d}.obj';piece.export(path);files.append(path.name)
    plane=ConvexHull(v).equations;gap=np.minimum(gap,(points@plane[:,:3].T+plane[:,3]).max(1))
   row=dict(object_id=oid,parent_object_id='ycb/'+name,family='large_household',category='household_mesh_and_CAD',source_type='scanned_mesh',split=split,backend='convex_parts',initialization='under_palm',units='metres',source=str(source.resolve()),source_sha256=digest(source),source_extents_m=original.tolist(),source_vertex_count=source_vertex_count,welded_vertex_count=len(m.vertices),seam_weld_max_error_m=seam_error,applied_scale=scale,source_center=center.tolist(),source_to_collection_rotation=rotation.tolist(),extents_m=m.extents.tolist(),anchor=[0.,0.,float(m.bounds[1,2])],collision_parts=files,mesh_sha256=digest(folder/'surface.obj'),collision_sha256={f:digest(folder/f) for f in files},source_sample_union_plane_gap_p95_m=float(np.quantile(abs(gap),.95)),source_sample_union_plane_gap_max_m=float(abs(gap).max()),collision_qa='Sampled directional plane gap only; cavities and actual contact require review. No whole-object hull substitution.',wm_contact_rate_cutoff=None,admission='candidate_geometry_only',elapsed_s=time.monotonic()-start)
   row['removed_zero_volume_components']=removed_components
   (folder/'geometry.json').write_text(json.dumps(row,indent=2));rows.append(row);print('READY',name,len(files),flush=True)
  except Exception as e:
   reject.append(dict(parent=name,reason=str(e)));print('REJECT',name,str(e),flush=True)
  save()
if __name__=='__main__':main()
