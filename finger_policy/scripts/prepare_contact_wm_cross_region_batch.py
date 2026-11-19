"""Freeze real cross-region collections, not resampled versions of one trace."""
import argparse,copy,hashlib,json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    parent=out.parent;scripts=Path(__file__).resolve().parent
    template=json.loads((parent/'box_band_review_20261114/band40.protocol.json').read_text())
    jobs=[]
    for oid in ['ycb_004_sugar_box','ycb_005_tomato_soup_can','ycb_010_potted_meat_can']:
        for region in [2,1,0]:
            old=json.loads((parent/'balanced_ycb_regions_20261113'/f'{oid}_r{region}.protocol.json').read_text())
            for side in [1,-1]:
                p=copy.deepcopy(template);tag=f'cross_region_20261114_r{region}_{"pos" if side==1 else "neg"}'
                name=f'{oid}_r{region}_{"pos" if side==1 else "neg"}'
                p.update(id=name,object=oid,objects=[oid],interior_reference=old['interior_reference'],interior_reference_sha256=old['interior_reference_sha256'])
                p['collector'].update(objects=[oid],fingers=[0,1,2,3],run_tag=tag,feature_edge_sign=side,feature_half_band_m=.040,frames=4500,force_targets_n=[5.],episode_seeds=[202613130])
                p['source_hashes']={n:hashlib.sha256((scripts/n).read_bytes()).hexdigest() for n in p['source_hashes']}
                p['source_hash_migration']={}
                p['scope']='Authorized cross-region batch: four independent single fingers; keep failures; no WM training admission.'
                p['region']=old['region'];p['balance']=dict(lanes_per_finger=1,independent_physical_lanes=4)
                path=out/f'{name}.protocol.json';path.write_text(json.dumps(p,indent=2))
                jobs.append(dict(id=name,protocol=str(path),collector='collect_contact_wm_feature_motion.py',folder=str(Path(p['collector']['root'])/'episodes/single_finger'/tag/(oid+'_mu1_k1')),expected_lanes=4,expected_frames=4500))
    (out/'queue.json').write_text(json.dumps(dict(jobs=jobs,planned_lanes=72,training_admissible=False,apple_excluded=True),indent=2))
    snap=out/'source_snapshot';snap.mkdir()
    hashes={}
    for name in list(template['source_hashes'])+['prepare_contact_wm_cross_region_batch.py','run_contact_wm_collection_queue.py']:
        content=(scripts/name).read_bytes();(snap/name).write_bytes(content);hashes[name]=hashlib.sha256(content).hexdigest()
    (out/'source_snapshot.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(dict(jobs=len(jobs),planned_lanes=72,output=str(out))))

if __name__=='__main__':main()
