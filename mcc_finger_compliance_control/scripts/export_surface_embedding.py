"""Attach a frozen PointNet manifold embedding to the local-planner DP H5."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from train_surface_pointnet import SurfacePointNet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp-file", type=Path, required=True)
    parser.add_argument("--manifold-file", type=Path, required=True)
    parser.add_argument("--pointnet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    checkpoint = torch.load(args.pointnet, map_location="cpu", weights_only=False)
    latent_dim = int(checkpoint["config"]["latent_dim"])
    model = SurfacePointNet(latent_dim=latent_dim)
    model.load_state_dict(checkpoint["model"])
    device = torch.device(args.device)
    model = model.to(device).eval()
    mean = np.asarray(checkpoint["point_mean"], dtype=np.float32)
    std = np.asarray(checkpoint["point_std"], dtype=np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.dp_file, "r") as source, h5py.File(
        args.manifold_file, "r"
    ) as manifold, h5py.File(args.output, "w") as target:
        for name, dataset in source.items():
            source.copy(dataset, target, name=name)
        for key, value in source.attrs.items():
            target.attrs[key] = value
        raw_index = np.asarray(manifold["raw_index"], dtype=np.int64)
        if raw_index.max(initial=-1) >= source["q_hand"].shape[0]:
            raise ValueError("Manifold raw_index exceeds DP source length")
        embedding = target.create_dataset(
            "surface_manifold_embedding",
            shape=(*source["q_hand"].shape[:2], latent_dim),
            dtype="f4",
            chunks=(min(4096, source["q_hand"].shape[0]), 1, latent_dim),
            compression="gzip",
            compression_opts=2,
            fillvalue=0.0,
        )
        with torch.no_grad():
            for start in tqdm(
                range(0, len(raw_index), args.batch_size), desc="PointNet embeddings"
            ):
                stop = min(start + args.batch_size, len(raw_index))
                points = np.asarray(manifold["gp_points"][start:stop], dtype=np.float32)
                points = (points - mean) / std
                latent = model.encode(torch.from_numpy(points).to(device)).cpu().numpy()
                batch_indices = raw_index[start:stop]
                order = np.argsort(batch_indices)
                embedding[batch_indices[order], 0] = latent[order].astype(np.float32)
        target.attrs["dp_state_schema"] = "contact_geometry_planner_manifold"
        target.attrs["surface_manifold_embedding_dim"] = latent_dim
        target.attrs["surface_manifold_pointnet"] = str(args.pointnet)
        target.attrs["surface_manifold_source"] = str(args.manifold_file)
        target.attrs["surface_manifold_causal"] = True
        target.attrs["state_fields"] = (
            "q_hand,fingertip_contact_pos_palm,fingertip_contact_normal_palm,"
            "fingertip_contact_mask,palm_relative_twist_palm,"
            "planner_palm_delta_pose_palm,surface_manifold_embedding"
        )
    print(f"[SUCCESS] manifold-conditioned DP data saved to {args.output}")


if __name__ == "__main__":
    main()
