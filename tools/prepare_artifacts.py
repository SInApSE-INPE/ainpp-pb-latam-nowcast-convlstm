"""Extract inference-only ConvLSTM weights and a checksummed bundle."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from convlstm_nowcast._runtime.artifacts import sha256


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--bundle", type=Path, required=True)
    args = cli.parse_args()
    args.bundle.mkdir(parents=True, exist_ok=True)
    target = args.bundle / "convlstm.pth"
    manifest_path = args.bundle / "manifest.json"
    if target.exists() or manifest_path.exists():
        raise FileExistsError("Use a new bundle directory; existing artifacts are preserved")
    (mean,) = args.bundle.glob("*_log_mean_*.npy")
    (std,) = args.bundle.glob("*_log_std_*.npy")
    allowed = [
        (np._core.multiarray.scalar, "numpy._core.multiarray.scalar"),
        (np._core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        np.dtypes.Float64DType,
        np.dtypes.Float32DType,
    ]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    training_args = checkpoint["args"]
    input_steps = training_args["input_timesteps"]
    output_steps = training_args["output_timesteps"]
    if type(input_steps) is not int or type(output_steps) is not int:
        raise ValueError("Training sequence lengths must be integers")
    if input_steps < 1 or output_steps < 1:
        raise ValueError("Training sequence lengths must be positive")
    state = checkpoint["model_state_dict"]
    cleaned = {}
    for key, value in state.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        if key in cleaned or not torch.is_tensor(value):
            raise ValueError("Invalid or ambiguous state dictionary")
        cleaned[key] = value
    torch.save(cleaned, target)
    manifest = {
        "schema_version": 1,
        "version": args.bundle.name,
        "model": "ConvLSTM",
        "architecture": {
            "encoder": "convlstm",
            "input_steps": input_steps,
            "output_steps": output_steps,
            "output_activation": "linear_normalized",
        },
        "reference_input_product": "gsmap_nrt",
        "training_latitude_order": None,
        "training_domain_verified": False,
        "source_checkpoint": {"name": args.checkpoint.name, "sha256": sha256(args.checkpoint)},
        "training_args": training_args,
        "files": {
            role: {"name": path.name, "sha256": sha256(path)}
            for role, path in (("checkpoint", target), ("mean", mean), ("std", std))
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"Created {target}: {input_steps} input hours, {output_steps} recurrent leads")


if __name__ == "__main__":
    main()
