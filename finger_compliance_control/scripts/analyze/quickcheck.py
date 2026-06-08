#!/usr/bin/env python3
"""
Simple HDF5 inspection script for trajectory data.
Usage: uv run python quickcheck.py
"""
import sys
from pathlib import Path

try:
    import h5py
    import numpy as np
except ImportError as e:
    print(f"Error: {e}")
    print("Run: uv run python quickcheck.py")
    sys.exit(1)

file_path = Path("./finger_compliance_control/data/collect_20260409_165619.h5")

if not file_path.exists():
    print(f"File not found: {file_path}")
    sys.exit(1)

print(f"\n{'='*60}")
print(f"Inspecting: {file_path}")
print(f"{'='*60}\n")

try:
    with h5py.File(file_path, "r") as f:
        print("📊 Datasets in file:")
        print("-" * 60)
        
        for key in sorted(f.keys()):
            dset = f[key] 
            shape = dset.shape # type: ignore
            dtype = dset.dtype # type: ignore
            
            print(f"\n{key}")
            print(f"  Shape: {shape}")
            print(f"  Dtype: {dtype}")
            print(f"  Size: {np.prod(shape)} elements")
            
            # Load data to check for issues
            try:
                data = dset[:] # type: ignore
                
                # Statistics
                if np.issubdtype(dtype, np.floating):
                    print(f"  Min: {np.nanmin(data):.6f}, Max: {np.nanmax(data):.6f}") # type: ignore
                    print(f"  Mean: {np.nanmean(data):.6f}, Std: {np.nanstd(data):.6f}") # type: ignore
                    
                    # Warnings
                    nan_count = np.isnan(data).sum() # type: ignore
                    if nan_count > 0:
                        print(f"  ⚠️  Contains {nan_count} NaNs ({100*nan_count/np.prod(shape):.2f}%)")
                    
                    if np.all(data == 0):
                        print(f"  ⚠️  All zeros!")
                
                # Sample random data points
                if len(shape) >= 2:
                    # Random steps and environments
                    num_steps = min(5, shape[0])
                    random_steps = np.random.choice(shape[0], size=num_steps, replace=False)
                    random_steps = np.sort(random_steps)
                    
                    num_envs = min(3, shape[1])
                    random_envs = np.random.choice(shape[1], size=num_envs, replace=False)
                    
                    # Simple indexing: first select steps, then select envs
                    sample = dset[random_steps][:, random_envs] # type: ignore
                    print(f"  Random sample (steps {random_steps}, envs {random_envs}):\n    {sample}")
                else:
                    # 1D data: random indices
                    num_samples = min(5, shape[0])
                    random_indices = np.random.choice(shape[0], size=num_samples, replace=False)
                    sample = dset[random_indices] # type: ignore
                    print(f"  Random sample (indices {random_indices}):\n    {sample}")
                    
            except Exception as e:
                print(f"  ❌ Error reading data: {e}")
        
        # Metadata
        if f.attrs:
            print(f"\n{'='*60}")
            print("📝 Attributes:")
            print("-" * 60)
            for attr_name, attr_value in f.attrs.items():
                print(f"  {attr_name}: {attr_value}")
        
        print(f"\n{'='*60}")
        print(f"✅ File check complete\n")
        
except Exception as e:
    print(f"❌ Error opening file: {e}")
    sys.exit(1)