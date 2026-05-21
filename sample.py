"""
sample.py — Sampling for Binary Diffusion synthetic tabular data.

CLI usage:
    python sample.py --ckpt model.ckpt --ckpt_transformation transform.ckpt \\
                     --n_timesteps 100 --out ./outputs/diabetes \\
                     --n_samples 1000 --batch_size 256

Notebook usage:
    from sample import sample

    df = sample(
        ckpt="model.ckpt",
        ckpt_transformation="transform.ckpt",
        n_timesteps=100,
        out="./outputs/diabetes",
        n_samples=1000,
        batch_size=256,
        # all other arguments are optional — see sample() docstring
    )
"""

import argparse
import os
from pathlib import Path
from functools import partial
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from binary_diffusion_tabular import (
    BinaryDiffusion1D,
    SimpleTableGenerator,
    FixedSizeBinaryTableTransformation,
    select_equally_distributed_numbers,
    TASK,
    get_random_labels,
    seed_everything,
)


# ---------------------------------------------------------------------------
# Classifier-free guidance helper
# ---------------------------------------------------------------------------

def cfg_model_fn(
    x_t: torch.Tensor,
    ts: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    guidance_scale: float,
    task: TASK,
    *args,
    **kwargs,
) -> torch.Tensor:
    """Classifier-free guidance denoising step.

    Args:
        x_t:            Noisy sample.
        ts:             Timesteps.
        y:              Conditioning signal.
        model:          Denoising model.
        guidance_scale: CFG scale factor.
        task:           Dataset task ("classification" or "regression").

    Returns:
        Guided noise prediction.
    """
    combine    = torch.cat([x_t, x_t], dim=0)
    combine_ts = torch.cat([ts, ts],   dim=0)

    if task == "classification":
        y_other = torch.zeros_like(y)
    elif task == "regression":
        # Zero-token is -1 because values are min-max normalised to [0, 1].
        y_other = torch.ones_like(y) * -1

    combine_y  = torch.cat([y, y_other], dim=0)
    model_out  = model(combine, combine_ts, y=combine_y)
    cond_eps, uncond_eps = torch.split(model_out, [y.shape[0], y.shape[0]], dim=0)
    return uncond_eps + guidance_scale * (cond_eps - uncond_eps)


# ---------------------------------------------------------------------------
# Core sampling  ← importable from a notebook
# ---------------------------------------------------------------------------

def sample(
    ckpt: str,
    ckpt_transformation: str,
    n_timesteps: int,
    out: str,
    n_samples: int,
    batch_size: int,
    threshold: float = 0.5,
    strategy: str = "target",
    seed: Optional[int] = None,
    guidance_scale: float = 0.0,
    target_column_name: Optional[str] = None,
    device: str = "cuda",
    use_ema: bool = False,
    dropna: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate synthetic tabular samples from a trained Binary Diffusion model
    and save them to disk as a CSV file.

    Parameters
    ----------
    ckpt : str
        Path to the model checkpoint (.ckpt).
    ckpt_transformation : str
        Path to the transformation checkpoint file.
    n_timesteps : int
        Number of diffusion sampling steps.
    out : str
        Output directory where the CSV will be saved.
    n_samples : int
        Total number of rows to generate.
    batch_size : int
        Number of samples per forward pass.
    threshold : float
        Binarisation threshold (default 0.5).
    strategy : str
        Sampling strategy — "target" or "mask" (default "target").
    seed : int, optional
        Random seed for reproducibility.
    guidance_scale : float
        Classifier-free guidance scale (default 0.0 = no guidance).
    target_column_name : str, optional
        Name of the target/label column in the output DataFrame.
        Required when the model is conditional.
    device : str
        Torch device string (default "cuda").
    use_ema : bool
        Load the EMA weights instead of the raw model weights (default False).
    dropna : bool
        Drop rows that contain NaN after inverse transformation (default False).
    verbose : bool
        Show a tqdm progress bar (default True).

    Returns
    -------
    pd.DataFrame
        The generated samples (all rows concatenated, trimmed to n_samples).
        The same DataFrame is also written to `out/samples_<i>.csv`.
    """
    if seed is not None:
        seed_everything(seed)

    path_out = Path(out)
    path_out.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────
    ckpt_data = torch.load(ckpt)

    denoising_model = SimpleTableGenerator.from_config(ckpt_data["config_model"]).to(device)
    denoising_model.eval()

    diffusion = BinaryDiffusion1D.from_config(
        denoise_model=denoising_model,
        config=ckpt_data["config_diffusion"],
    ).to(device)
    diffusion.eval()

    if use_ema:
        ema_state = ckpt_data["diffusion_ema"]
        ema_model_state = {
            k.replace("ema_model.", ""): v
            for k, v in ema_state.items()
            if k.startswith("ema_model.")
        }
        diffusion.load_state_dict(ema_model_state)
    else:
        diffusion.load_state_dict(ckpt_data["diffusion"])

    transformation = FixedSizeBinaryTableTransformation.from_checkpoint(ckpt_transformation)

    # ── Model properties ──────────────────────────────────────────────
    task                      = denoising_model.task
    conditional               = denoising_model.conditional
    n_classes                 = denoising_model.n_classes
    classifier_free_guidance  = denoising_model.classifier_free_guidance

    timesteps_sampling = select_equally_distributed_numbers(
        diffusion.n_timesteps,
        n_timesteps,
    )

    # ── Generation loop ───────────────────────────────────────────────
    n_generated = 0
    dfs = []
    pbar = tqdm(total=n_samples, disable=not verbose)

    while n_generated < n_samples:
        labels = get_random_labels(
            conditional=conditional,
            task=task,
            n_classes=n_classes,
            classifier_free_guidance=classifier_free_guidance,
            n_labels=batch_size,
            device=device,
        )

        x = diffusion.sample(
            model_fn=(
                partial(cfg_model_fn, guidance_scale=guidance_scale, task=task)
                if classifier_free_guidance and guidance_scale > 0
                else None
            ),
            n=batch_size,
            y=labels,
            timesteps=timesteps_sampling,
            threshold=threshold,
            strategy=strategy,
        )

        if conditional:
            if classifier_free_guidance:
                labels = torch.argmax(labels, dim=1)
            x_df, labels_df = transformation.inverse_transform(x, labels)
            x_df[target_column_name] = labels_df
        else:
            x_df = transformation.inverse_transform(x)

        if dropna:
            x_df = x_df.dropna()

        n_generated += len(x_df)
        pbar.update(len(x_df))
        dfs.append(x_df)

    pbar.close()

    # ── Save to disk ──────────────────────────────────────────────────
    df = pd.concat(dfs, ignore_index=True).iloc[:n_samples]

    i = 1
    while os.path.exists(path_out / f"samples_{i}.csv"):
        i += 1
    csv_path = path_out / f"samples_{i}.csv"
    df.to_csv(csv_path, index=False)

    if verbose:
        print(f"Saved {len(df)} samples to {csv_path}")

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample from a trained Binary Diffusion tabular model."
    )
    parser.add_argument("--ckpt",                type=str,   required=True,               help="Path to checkpoint file")
    parser.add_argument("--ckpt_transformation", type=str,   required=True,               help="Path to transformation checkpoint file")
    parser.add_argument("--n_timesteps", "-t",   type=int,   required=True,               help="Number of sampling steps")
    parser.add_argument("--out",         "-o",   type=str,   required=True,               help="Output folder for saved samples")
    parser.add_argument("--n_samples",   "-n",   type=int,   required=True,               help="Number of samples to generate")
    parser.add_argument("--batch_size",  "-b",   type=int,   required=True,               help="Batch size for sampling")
    parser.add_argument("--threshold",           type=float, default=0.5,                 help="Threshold for binarisation")
    parser.add_argument("--strategy",            type=str,   default="target",            help="Sampling strategy",        choices=["target", "mask"])
    parser.add_argument("--seed",        "-s",   type=int,   default=None,                help="Random seed")
    parser.add_argument("--guidance_scale", "-g",type=float, default=0.0,                 help="Guidance scale")
    parser.add_argument("--target_column_name",  type=str,   default=None,                help="Target column name")
    parser.add_argument("--device",      "-d",   type=str,   default="cuda",              help="Torch device")
    parser.add_argument("--use_ema",     "-e",   action="store_true",                     help="Use EMA weights")
    parser.add_argument("--dropna",              action="store_true",                     help="Drop NaN rows after sampling")
    parser.add_argument("--quiet",               action="store_true",                     help="Suppress progress bar and output")
    return parser.parse_args()


def main():
    args = _parse_args()
    sample(
        ckpt=args.ckpt,
        ckpt_transformation=args.ckpt_transformation,
        n_timesteps=args.n_timesteps,
        out=args.out,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        threshold=args.threshold,
        strategy=args.strategy,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        target_column_name=args.target_column_name,
        device=args.device,
        use_ema=args.use_ema,
        dropna=args.dropna,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()