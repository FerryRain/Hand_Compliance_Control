"""LEAP Hand tactile simulation using ETac."""

import os
from pathlib import Path


def blind_grasping_assets() -> Path:
    """Locate the upstream checkout containing the released meshes and weights."""
    source = os.environ.get("ETAC_SOURCE_ROOT")
    if source:
        root = Path(source).expanduser().resolve()
    else:
        import ETac

        root = Path(ETac.__file__).resolve().parents[1]
    assets = root / "BlindGrasping"
    if not (assets / "propagation-ckpt" / "linear.pth").is_file():
        raise FileNotFoundError(
            f"Missing ETac propagation weights under {assets}. "
            "Set ETAC_SOURCE_ROOT to the upstream ETac source checkout."
        )
    return assets
