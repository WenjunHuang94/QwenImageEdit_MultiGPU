import argparse
import logging
import os
import wandb
import json

import torch
from tqdm.auto import tqdm

from accelerate import Accelerator,DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
import datasets
import diffusers
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import (
    QwenImagePipeline,
    QwenImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from preprocess_dataset import loader, path_done_well
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
import transformers
from optimum.quanto import quantize, qfloat8, freeze
import bitsandbytes as bnb

# from diffusers.loaders import AttnProcsLayers
from diffusers import QwenImageEditPipeline
from diffusers.loaders import AttnProcsLayers
import gc

logger = get_logger(__name__, log_level="INFO")

# > tools -----------------------------------------------------------------------------

# fix env for deepspeed
def fix_env_for_deepspeed():
    for src, dst in [
        ("OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK"),
        ("OMPI_COMM_WORLD_RANK", "RANK"),
        ("OMPI_COMM_WORLD_SIZE", "WORLD_SIZE"),
    ]:
        if src in os.environ and dst not in os.environ:
            os.environ[dst] = os.environ[src]

    for k in [
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "OMPI_COMM_WORLD_RANK",
        "OMPI_COMM_WORLD_SIZE",
        "OMPI_COMM_WORLD_NODE_RANK",
    ]:
        os.environ.pop(k, None)
        
# args parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for Qwen Image Edit (Accelerate+DeepSpeed)")

    # Paths / Basics
    parser.add_argument("--output_dir", type=str, default="/storage/v-jinpewang/az_workspace/rico_model/test_lora_saves_edit")
    parser.add_argument("--logging_dir", type=str, default="./logger")
    parser.add_argument("--pretrained_model", type=str, default="/storage/v-jinpewang/az_workspace/rico_model/Qwen-Image-Edit-2509")

    # LoRA / Quant
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--quantize", action="store_true", help="Enable 8-bit quantization for blocks")
    parser.add_argument("--adam8bit", action="store_true", help="Use bitsandbytes Adam8bit optimizer")

    # Optimizer
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)

    # Training loop controls
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=5000)

    # LR schedule
    parser.add_argument("--lr_warmup_steps", type=int, default=10)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--learning_rate", type=float, default=2e-4)

    # System / misc
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Checkpointing
    parser.add_argument("--checkpointing_steps", type=int, default=250)

    # Caches
    parser.add_argument("--txt_cache_dir", type=str, default="/storage/v-jinpewang/az_workspace/rico_model/text_embs/")
    parser.add_argument("--img_cache_dir", type=str, default="/storage/v-jinpewang/az_workspace/rico_model/img_embs/")
    parser.add_argument("--control_img_cache_dir", type=str, default="/storage/v-jinpewang/az_workspace/rico_model/img_embs_control/")

    return parser.parse_args()

# lora_processor: to manage all the lora modules, return dict
def lora_processors(model):
    processors = {}

    def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
        if 'lora' in name:
            processors[name] = module
            print(name)
        for sub_name, child in module.named_children():
            fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)
        return processors
    
    # traversal named_children
    for name, module in model.named_children():
        fn_recursive_add_processors(name, module, processors)

    return processors


# > main -----------------------------------------------------------------------------

def main():
    # > fix env for deepspeed
    fix_env_for_deepspeed()

    # > config
    args = parse_args()
    args.weight_dtype = torch.bfloat16
    args.output_dir, args.logging_dir, args.pretrained_model, args.txt_cache_dir, args.img_cache_dir, args.control_img_cache_dir = path_done_well(
        args.output_dir, args.logging_dir, args.pretrained_model, args.txt_cache_dir, args.img_cache_dir, args.control_img_cache_dir)

    # > deepSpeed & Accelerator
    project_config=ProjectConfiguration(
            project_dir=args.output_dir,
            logging_dir=args.output_dir / args.logging_dir,
        )
    deepspeed = DeepSpeedPlugin(zero_stage=3, gradient_clipping=1.0)
    accelerator = Accelerator(deepspeed_plugin=deepspeed, log_with="wandb", project_config=project_config)
    logger.info(accelerator.state, main_process_only=False)
    

    # > unwape func for safe accelerate (optional)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model
    
    # > log
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

# > load model in -----------------------------------------------------------------------------
    
    # > define image_encoding_pipeline VAE
    vae_cfg_path = args.pretrained_model / "vae/config.json"
    vae = None
    with open(vae_cfg_path, "r") as f:
        vae = json.load(f)

    # > load flux
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        args.pretrained_model,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,)
    
    # optional quant
    if args.quantize:
        quantize(flux_transformer, weights=qfloat8)
        freeze(flux_transformer)
    gc.collect()
    torch.cuda.empty_cache()

    if args.quantize:
        flux_transformer.to(accelerator.device)
    else:
        flux_transformer.to(accelerator.device, dtype=args.weight_dtype)
    
    # > noise schedular
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model,
        subfolder="scheduler",
    )

    def get_sigmas(timesteps, n_dim=4, dtype=args.weight_dtype):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    # > LoRA config
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    
    flux_transformer.add_adapter(lora_config)

    # > training preparing -----------------------------------------------------------------------------
    
    flux_transformer.requires_grad_(False)
    flux_transformer.train()
    # Freeze all parameters by default; enable grad only for LoRA layers.
    # TIP: The printout below helps confirm which layers are trainable.
    for n, param in flux_transformer.named_parameters():
        if 'lora' not in n:
            param.requires_grad = False
        else:
            param.requires_grad = True
    print(sum([p.numel() for p in flux_transformer.parameters() if p.requires_grad]) / 1e6, 'parameters')

    lora_layers = filter(lambda p: p.requires_grad, flux_transformer.parameters())
    lora_layers_model = AttnProcsLayers(lora_processors(flux_transformer))

    flux_transformer.enable_gradient_checkpointing()

    if args.adam8bit:
        optimizer = bnb.optim.Adam8bit(lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),)
    else:
        optimizer = torch.optim.AdamW(
            lora_layers,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    train_dataloader = loader(args.train_batch_size, args.num_workers, txt_cache_dir=args.txt_cache_dir, img_cache_dir=args.img_cache_dir, ctrl_cache_dir=args.control_img_cache_dir,)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    # Accelerator Prepare
    lora_layers_model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers_model, optimizer, train_dataloader, lr_scheduler
    )


    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    vae_scale_factor = 2 ** len(vae.get("temperal_downsample"))

    # > log
    if accelerator.is_main_process:
        # Initialize a W&B run via Accelerate. The config will include your CLI/YAML args.
        cfgs = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "max_train_steps": args.max_train_steps,
        "lr_scheduler": args.lr_scheduler,
        # 其他字段需要再加，但尽量保持 JSON-serializable
    }
        accelerator.init_trackers(
            project_name="qwen-lora",
            config=cfgs,
        )
        # (Optional) track gradients/parameters every 100 steps. Safe to call only on main process.
        try:
            wandb.watch(accelerator.unwrap_model(flux_transformer), log="gradients", log_freq=100)
        except Exception as e:
            print(f"Failed to watch: {e}")
    logger.info("***** Running training *****")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    global_step = 0
    initial_global_step = 0
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    # > training loop -----------------------------------------------------------------------------

    for epoch in range(args.epochs):
        if accelerator.is_main_process:
            accelerator.log({"epoch": epoch}, step=global_step)
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_transformer):
                img, prompt_embeds, prompt_embeds_mask, control_img = batch
                prompt_embeds, prompt_embeds_mask = prompt_embeds.to(dtype=args.weight_dtype,device=accelerator.device), prompt_embeds_mask.to(dtype=torch.int32,device=accelerator.device)
                txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
                control_img = control_img.to(dtype=args.weight_dtype,device=accelerator.device)

                pixel_latents = img.to(dtype=args.weight_dtype,device=accelerator.device)
                # Encode to latents:
                # - `img`      -> target latents (what the model should predict)
                # - `control_img` -> conditioning latents (auxiliary input)
                # Both are 5D (B, F, C, H, W) after VAE and are then permuted
                # to (B, C, F, H, W) before packing.
                pixel_latents = pixel_latents.permute(0, 2, 1, 3, 4)
                control_img = control_img.permute(0, 2, 1, 3, 4)
                latents_mean = (
                    torch.tensor(vae.get("latents_mean"))
                    .view(1, 1, vae.get("z_dim"), 1, 1)
                    .to(pixel_latents.device, pixel_latents.dtype)
                )
                latents_std = 1.0 / torch.tensor(vae.get("latents_std")).view(1, 1, vae.get("z_dim"), 1, 1).to(
                    pixel_latents.device, pixel_latents.dtype
                )
                # Normalize latents using VAE config stats for stable training.
                pixel_latents = (pixel_latents - latents_mean) * latents_std
                control_img = (control_img - latents_mean) * latents_std

                bsz = pixel_latents.shape[0]
                noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=args.weight_dtype)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme="none",
                    batch_size=bsz,
                    logit_mean=0.0,
                    logit_std=1.0,
                    mode_scale=1.29,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=pixel_latents.device)

                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
                # Pack latents into transformer sequence format and then concat
                # the target branch with the condition branch along channel dim.
                packed_noisy_model_input = QwenImageEditPipeline._pack_latents(
                    noisy_model_input,
                    bsz,
                    noisy_model_input.shape[2],
                    noisy_model_input.shape[3],
                    noisy_model_input.shape[4],
                )
                packed_control_img = QwenImageEditPipeline._pack_latents(
                    control_img,
                    bsz,
                    control_img.shape[2],
                    control_img.shape[3],
                    control_img.shape[4],
                )
                # `img_shapes` conveys spatial token grid shapes per branch to
                # the transformer so that RoPE can be properly applied.
                img_shapes = [[(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2),
                              (1, control_img.shape[3] // 2, control_img.shape[4] // 2)]] * bsz
                packed_noisy_model_input_concated = torch.cat([packed_noisy_model_input, packed_control_img], dim=1)
                
                # > forward 
                '''''''''
                Forward args:
                  hidden_states: packed target+condition latents
                  timestep:      normalized timestep for Flow Matching (t/1000)
                  encoder_hidden_states(_mask): text embeddings/mask
                  img_shapes:    spatial token metadata for each branch
                  txt_seq_lens:  sequence lengths per sample (for variable text)
                '''''''''
                #input()
                model_pred = flux_transformer(
                    hidden_states=packed_noisy_model_input_concated,
                    timestep=timesteps / 1000,
                    guidance=None,
                    encoder_hidden_states_mask=prompt_embeds_mask,
                    encoder_hidden_states=prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=txt_seq_lens,
                    return_dict=False,
                )[0]
                model_pred = model_pred[:, : packed_noisy_model_input.size(1)]

                model_pred = QwenImageEditPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[3] * vae_scale_factor,
                    width=noisy_model_input.shape[4] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
                # Flow-Matching loss:
                #   target = noise - pixel_latents  (after packing/unpacking)
                #   weighting = scheme from SD3 utilities (currently "none")
                target = noise - pixel_latents
                target = target.permute(0, 2, 1, 3, 4)
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(flux_transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    
                    save_path = args.output_dir / f"checkpoint-{global_step}"
                    try:
                        save_path.mkdir(exist_ok=False)
                    except Exception as e:
                        print(f"Failed to create checkpoint directory {save_path}: {e}")

                    unwrapped_flux_transformer = unwrap_model(flux_transformer)
                    flux_transformer_lora_state_dict = convert_state_dict_to_diffusers(
                        get_peft_model_state_dict(unwrapped_flux_transformer)
                    )

                    # Save **only** LoRA adapter weights in diffusers format.
                    QwenImagePipeline.save_lora_weights(
                        save_path,
                        flux_transformer_lora_state_dict,
                        safe_serialization=True,
                    )

                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    main()
