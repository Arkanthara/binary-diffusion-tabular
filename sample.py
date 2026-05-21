import argparse
import os
from pathlib import Path
from functools import partial

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
    seed_everything
)


def get_sampling_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to checkpoint file"
    )
    parser.add_argument("--ckpt_transformation", type=str, help="Path to transformation checkpoint file")
    parser.add_argument(
        "--n_timesteps", "-t", type=int, required=True, help="Number of sampling steps"
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        required=True,
        help="Path to output folder, where to save samples",
    )
    parser.add_argument(
        "--n_samples",
        "-n",
        type=int,
        required=True,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, required=True, help="Batch size for sampling"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for binarization"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="target",
        choices=["target", "mask"],
        help="Sampling strategy to use",
    )
    parser.add_argument("--seed", "-s", type=int, help="Random seed", required=False)
    parser.add_argument(
        "--guidance_scale", "-g", type=float, default=0.0, help="Guidance scale"
    )
    parser.add_argument("--target_column_name", type=str, help="Target column name", required=False)
    parser.add_argument("--device", "-d", type=str, default="cuda", help="Device")
    parser.add_argument("--use_ema", "-e", action="store_true", help="Use EMA")
    parser.add_argument("--dropna", action="store_true", help="Whether to drop rows with nan during sampling")

    return parser


def cfg_model_fn(
    x_t: torch.Tensor,
    ts: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    guidance_scale: float,
    task: TASK,
    *args,
    **kwargs
) -> torch.Tensor:
    """Classifier free guidance sampling function

    Args:
        x_t: noisy sample
        ts: timesteps
        y: conditioning
        model: denoising model
        guidance_scale: guidance scale in classifier free guidance
        task: dataset task

    Returns:
        torch.Tensor: denoiser output
    """

    combine = torch.cat([x_t, x_t], dim=0)
    combine_ts = torch.cat([ts, ts], dim=0)

    if task == "classification":
        y_other = torch.zeros_like(y)
    elif task == "regression":
        # for regression, zero-token is -1, since values are minmax normalized to [0, 1] range
        y_other = torch.ones_like(y) * -1

    combine_y = torch.cat([y, y_other], dim=0)
    model_out = model(combine, combine_ts, y=combine_y)
    cond_eps, uncod_eps = torch.split(model_out, [y.shape[0], y.shape[0]], dim=0)
    eps = uncod_eps + guidance_scale * (cond_eps - uncod_eps)
    return eps


if __name__ == "__main__":
    parser = get_sampling_args_parser()
    cli_args = parser.parse_args()

    if cli_args.seed:
        seed_everything(cli_args.seed)

    path_out = Path(cli_args.out)
    path_out.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(cli_args.ckpt)
    device = cli_args.device
    batch_size = int(cli_args.batch_size)
    guidance_scale = cli_args.guidance_scale
    threshold = cli_args.threshold
    strategy = cli_args.strategy
    target_column_name = cli_args.target_column_name

    denoising_model = SimpleTableGenerator.from_config(ckpt["config_model"]).to(device)
    denoising_model.eval()

    diffusion = BinaryDiffusion1D.from_config(
        denoise_model=denoising_model,
        config=ckpt["config_diffusion"],
    ).to(device)
    diffusion.eval()

    transformation = FixedSizeBinaryTableTransformation.from_checkpoint(cli_args.ckpt_transformation)

    print(ckpt.keys())

    if cli_args.use_ema:
        ema_state = ckpt["diffusion_ema"]

        ema_model_state = {
            k.replace("ema_model.", ""): v
            for k, v in ema_state.items()
            if k.startswith("ema_model.")
        }

        diffusion.load_state_dict(ema_model_state)
    else:
        diffusion.load_state_dict(ckpt["diffusion"])

    n_total_timesteps = diffusion.n_timesteps
    timesteps_sampling = select_equally_distributed_numbers(
        n_total_timesteps,
        cli_args.n_timesteps,
    )
    task = denoising_model.task
    conditional = denoising_model.conditional
    n_classes = denoising_model.n_classes
    classifier_free_guidance = denoising_model.classifier_free_guidance

    n_generated = 0
    n_samples = cli_args.n_samples
    pbar = tqdm(total=n_samples)
    dfs = []

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

        if cli_args.dropna:
            x_df = x_df.dropna()

        n_generated += len(x_df)
        pbar.update(len(x_df))
        dfs.append(x_df)

    df = pd.concat(dfs)
    # Check if path is a dir or a .csv file
    i = 1
    while True:
        if os.path.exists(os.path.join(path_out, f"samples_{i}.csv")):
            i += 1
        else:
            df.to_csv(os.path.join(path_out, f"samples_{i}.csv"), index=False)
            break
    pbar.close()