import argparse
import os
import wandb
import json
import logging
from pathlib import Path

import torch
from tqdm.auto import tqdm
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from peft import LoraConfig, get_peft_model, PeftModel
from diffusers import QwenImageEditPlusPipeline,QwenImageTransformer2DModel
from QwenEdit import loader, path_done_well, fix_env_for_deepspeed, MultiGPUTransformer

# 使用标准 Python logging 而不是 accelerate.logging（避免需要初始化 accelerate 状态）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# > tools -----------------------------------------------------------------------------

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
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, 
                        help="Path to checkpoint directory to resume training from")
    parser.add_argument("--save_best_model", action="store_true", 
                        help="Save best model based on training loss")
    parser.add_argument("--no_save_optimizer_state", action="store_true",
                        help="Don't save optimizer/scheduler state in checkpoints (default: save for resuming training)")

    # Caches
    parser.add_argument("--txt_cache_dir", type=str, default="/storage/v-jinpewang/az_workspace/rico_model/text_embs/")
    parser.add_argument("--img_cache_dir", type=str, default="/storage/v-jinpewang/az_workspace/rico_model/img_embs/")
    parser.add_argument("--control_img_cache_dir", type=str, default="/storage/v-jinpewang/az_workspace/rico_model/img_embs_control/")

    return parser.parse_args()


# > main -----------------------------------------------------------------------------

def main():
    # > fix env for deepspeed
    fix_env_for_deepspeed()

    # > config
    args = parse_args()
    # dtype 自适配：支持 bf16 则优先，否则用 fp16（V100 无 bf16）
    args.weight_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    args.output_dir, args.logging_dir, args.pretrained_model, args.txt_cache_dir, args.img_cache_dir, args.control_img_cache_dir = path_done_well(
        args.output_dir, args.logging_dir, args.pretrained_model, args.txt_cache_dir, args.img_cache_dir, args.control_img_cache_dir
    )
    # 处理 resume_from_checkpoint 路径
    if args.resume_from_checkpoint:
        args.resume_from_checkpoint = Path(args.resume_from_checkpoint) if not isinstance(args.resume_from_checkpoint, Path) else args.resume_from_checkpoint

    use_fp16_amp = (args.weight_dtype == torch.float16)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16_amp)


# > load model in -----------------------------------------------------------------------------

    # > define image_encoding_pipeline VAE（这里只读配置用于规范化；不做编码）
    vae_cfg_path = args.pretrained_model / "vae/config.json"
    with open(vae_cfg_path, "r") as f:
        vae = json.load(f)

    # 将 Path 对象转换为字符串，避免 PEFT 模型卡片序列化错误
    pretrained_model_str = str(args.pretrained_model)
    flux_transformer = QwenImageTransformer2DModel.from_pretrained(
        pretrained_model_str,
        subfolder="transformer",
        torch_dtype=args.weight_dtype,
    )

    # > LoRA config —— 先插 LoRA 再分片
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=2*args.rank,  # TODO: 需要根据实际效果调整
        init_lora_weights='gaussian',  # TODO:原来是 loftq
        target_modules=[
        "to_k", "to_q", "to_v", "to_out.0", 
        "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",],
    )


    # block 级映射为“流水线/模型并行”
    flux_transformer = MultiGPUTransformer(flux_transformer).auto_split()
    first_device = next(flux_transformer.parameters()).device
    
    # 如果指定了恢复检查点，则加载 LoRA 权重
    if args.resume_from_checkpoint and args.resume_from_checkpoint.exists():
        logger.info(f"Loading LoRA weights from checkpoint: {args.resume_from_checkpoint}")
        # 先创建基础 LoRA 结构
        flux_transformer = get_peft_model(flux_transformer, lora_config)
        # 然后加载权重
        def _unwrap(m):
            return m._orig_mod if hasattr(m, "_orig_mod") else m
        unwrapped_flux_transformer = _unwrap(flux_transformer)
        flux_transformer = PeftModel.from_pretrained(
            unwrapped_flux_transformer, 
            str(args.resume_from_checkpoint), 
            low_cpu_mem_usage=False
        )
    else:
        flux_transformer = get_peft_model(flux_transformer, lora_config)

    # Freeze base, train only LoRA
    for n, p in flux_transformer.named_parameters():
        p.requires_grad = ("lora" in n)

    

    # 统计可训练参数
    trainable_params = [p for p in flux_transformer.parameters() if p.requires_grad]
    print(sum(p.numel() for p in trainable_params) / 1e6, 'parameters (trainable)')

    # 启用大块的梯度检查点，降低显存
    flux_transformer.enable_gradient_checkpointing()

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # 已编码 img_latents / control_latents、prompt_embeds、prompt_mask（预编码管线）
    train_dataloader = loader(
        args.train_batch_size,
        args.num_workers,
        txt_cache_dir=args.txt_cache_dir,
        img_cache_dir=args.img_cache_dir,
        ctrl_cache_dir=args.control_img_cache_dir,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps
    )

    vae_scale_factor = 2 ** len(vae.get("temperal_downsample"))

    # > noise scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        pretrained_model_str,
        subfolder="scheduler",
    )

    def get_sigmas(timesteps, n_dim=4, dtype=args.weight_dtype, device=torch.device("cuda")):
        sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma


    global_step = 0
    initial_global_step = 0
    best_loss = float('inf')  # 跟踪最佳损失
    ema_loss = None  # EMA 平滑损失
    ema_decay = 0.99  # EMA 衰减系数（0.99 表示保留 99% 的历史信息）
    
    # 如果从检查点恢复，尝试加载训练状态
    if args.resume_from_checkpoint and args.resume_from_checkpoint.exists():
        checkpoint_path = args.resume_from_checkpoint
        # 尝试加载训练状态文件
        optimizer_state_path = checkpoint_path / "optimizer.pt"
        scheduler_state_path = checkpoint_path / "scheduler.pt"
        scaler_state_path = checkpoint_path / "scaler.pt"
        training_state_path = checkpoint_path / "training_state.json"
        
        if training_state_path.exists():
            with open(training_state_path, "r") as f:
                training_state = json.load(f)
                global_step = training_state.get("global_step", 0)
                initial_global_step = global_step
                best_loss = training_state.get("best_loss", float('inf'))
                ema_loss = training_state.get("ema_loss", None)  # 恢复 EMA 损失
                ema_loss_str = f"{ema_loss:.6f}" if ema_loss is not None else "None"
                logger.info(f"Resuming from step {global_step}, best_loss: {best_loss:.6f}, ema_loss: {ema_loss_str}")
        
        if optimizer_state_path.exists():
            logger.info(f"Loading optimizer state from {optimizer_state_path}")
            optimizer.load_state_dict(torch.load(str(optimizer_state_path), map_location=first_device))
        
        if scheduler_state_path.exists():
            logger.info(f"Loading scheduler state from {scheduler_state_path}")
            lr_scheduler.load_state_dict(torch.load(str(scheduler_state_path), map_location=first_device))
        
        if scaler_state_path.exists() and use_fp16_amp:
            logger.info(f"Loading scaler state from {scaler_state_path}")
            scaler.load_state_dict(torch.load(str(scaler_state_path), map_location=first_device))
    
    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
    )
    # > wandb init -----------------------------------------------------------------
    wandb.init(
        project="qwen_lora",      # 自定义项目名
        name=f"ppRun-rank{args.rank}",  # 可选：给这次实验命名
        config={
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.train_batch_size,
            "grad_accum_steps": args.gradient_accumulation_steps,
            "dtype": str(args.weight_dtype),
            "rank": args.rank,
            "num_gpus": torch.cuda.device_count(),
        }
    )

    # > training loop -----------------------------------------------------------------------------

    for epoch in range(args.epochs):
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            # 约定 loader 返回的就是预编码后的张量
            # img:        目标 latent (B, F, C, H, W)
            # control:    条件 latent (B, F, C, H, W)
            # prompt_embeds:      (B, T, D_txt)
            # prompt_embeds_mask: (B, T) int32
            img, prompt_embeds, prompt_embeds_mask, control_img = batch

            prompt_embeds = prompt_embeds.to(dtype=args.weight_dtype, device=first_device)
            prompt_embeds_mask = prompt_embeds_mask.to(dtype=torch.int32, device=first_device)
            txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

            control_img = control_img.to(dtype=args.weight_dtype, device=first_device)
            pixel_latents = img.to(dtype=args.weight_dtype, device=first_device)

            # (B, C, F, H, W)
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

            # 规范化（与预编码配置一致）
            pixel_latents = (pixel_latents - latents_mean) * latents_std
            control_img = (control_img - latents_mean) * latents_std

            bsz = pixel_latents.shape[0]
            noise = torch.randn_like(pixel_latents, device=first_device, dtype=pixel_latents.dtype)

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

            # pack
            packed_noisy_model_input = QwenImageEditPlusPipeline._pack_latents(
                noisy_model_input,
                bsz,
                noisy_model_input.shape[2],
                noisy_model_input.shape[3],
                noisy_model_input.shape[4],
            )
            packed_control_img = QwenImageEditPlusPipeline._pack_latents(
                control_img,
                bsz,
                control_img.shape[2],
                control_img.shape[3],
                control_img.shape[4],
            )

            img_shapes = [[(1, noisy_model_input.shape[3] // 2, noisy_model_input.shape[4] // 2),
                            (1, control_img.shape[3] // 2, control_img.shape[4] // 2)]] * bsz

            packed_noisy_model_input_concated = torch.cat([packed_noisy_model_input, packed_control_img], dim=1)
            

            # forward
            model_pred= flux_transformer(
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

            # unpack
            model_pred = QwenImageEditPlusPipeline._unpack_latents(
                model_pred,
                height=noisy_model_input.shape[3] * vae_scale_factor,
                width=noisy_model_input.shape[4] * vae_scale_factor,
                vae_scale_factor=vae_scale_factor,
            )

            weighting = compute_loss_weighting_for_sd3(weighting_scheme="none", sigmas=sigmas)
            target = noise - pixel_latents
            target = target.permute(0, 2, 1, 3, 4)

            loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            )
            loss = loss.mean()

            # Backprop & optimization (PP-only)
            train_loss += loss.detach().item()
            if use_fp16_amp:
                scaler.scale(loss / args.gradient_accumulation_steps).backward()
            else:
                (loss / args.gradient_accumulation_steps).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                if use_fp16_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                progress_bar.update(1)
                global_step += 1

                # 更新 EMA 平滑损失
                current_loss = loss.item()
                if ema_loss is None:
                    ema_loss = current_loss
                else:
                    ema_loss = ema_decay * ema_loss + (1 - ema_decay) * current_loss
                
                # 检查并更新最佳损失（每次 step 都检查，确保及时保存最佳模型）
                if args.save_best_model and current_loss < best_loss:
                    best_loss = current_loss
                    best_path = args.output_dir / "best"
                    try:
                        best_path.mkdir(exist_ok=True, parents=True)
                        logger.info(f"New best loss: {best_loss:.6f}, saving to {best_path}")
                        
                        # unwrap helper
                        def _unwrap(m):
                            return m._orig_mod if hasattr(m, "_orig_mod") else m
                        unwrapped_flux_transformer = _unwrap(flux_transformer)
                        
                        # 确保 base_model 是字符串，避免 YAML 序列化 Path 对象错误
                        if hasattr(unwrapped_flux_transformer, 'base_model') and hasattr(unwrapped_flux_transformer.base_model, 'config'):
                            if hasattr(unwrapped_flux_transformer.base_model.config, '_name_or_path'):
                                unwrapped_flux_transformer.base_model.config._name_or_path = str(unwrapped_flux_transformer.base_model.config._name_or_path)
                        if hasattr(unwrapped_flux_transformer, 'peft_config'):
                            for key in unwrapped_flux_transformer.peft_config:
                                if hasattr(unwrapped_flux_transformer.peft_config[key], 'base_model_name_or_path'):
                                    path_val = unwrapped_flux_transformer.peft_config[key].base_model_name_or_path
                                    if isinstance(path_val, Path):
                                        unwrapped_flux_transformer.peft_config[key].base_model_name_or_path = str(path_val)
                        
                        # 删除可能损坏的模型卡片文件，让 PEFT 重新创建
                        model_card_path = best_path / "README.md"
                        if model_card_path.exists():
                            try:
                                model_card_path.unlink()
                            except Exception as e:
                                logger.warning(f"Failed to remove existing model card: {e}")
                        
                        # 保存最佳模型（只保存模型权重，不保存优化器状态，用于推理/部署）
                        # 将 Path 对象转换为字符串，避免 PEFT 模型卡片序列化错误
                        unwrapped_flux_transformer.save_pretrained(str(best_path), safe_serialization=True)
                        
                        # 只保存训练状态元数据（不保存优化器状态，因为最佳模型主要用于推理）
                        training_state = {
                            "global_step": global_step,
                            "epoch": epoch,
                            "best_loss": best_loss,
                            "current_loss": current_loss,
                            "ema_loss": ema_loss,  # 保存 EMA 损失
                        }
                        with open(best_path / "training_state.json", "w") as f:
                            json.dump(training_state, f, indent=2)
                            
                    except Exception as e:
                        logger.warning(f"Failed to save best model: {e}")

                wandb.log({
                "global_step": global_step,
                "train_loss": loss.item(),  # 瞬时损失（可能有噪声）
                "train_loss_ema": ema_loss,  # EMA 平滑损失（推荐观察这个）
                "best_loss": best_loss,
                "lr": lr_scheduler.get_last_lr()[0],
                "epoch": epoch,
                })

                if (global_step % args.checkpointing_steps) == 0:
                    save_path = args.output_dir / f"checkpoint-{global_step}"
                    try:
                        save_path.mkdir(exist_ok=True, parents=True)
                    except Exception as e:
                        print(f"Failed to create checkpoint directory {save_path}: {e}")
                        continue

                    # unwrap is identity in PP-only, but keep a tiny helper for clarity
                    def _unwrap(m):
                        return m._orig_mod if hasattr(m, "_orig_mod") else m
                    unwrapped_flux_transformer = _unwrap(flux_transformer)
                    
                    # 确保 base_model 是字符串，避免 YAML 序列化 Path 对象错误
                    if hasattr(unwrapped_flux_transformer, 'base_model') and hasattr(unwrapped_flux_transformer.base_model, 'config'):
                        if hasattr(unwrapped_flux_transformer.base_model.config, '_name_or_path'):
                            unwrapped_flux_transformer.base_model.config._name_or_path = str(unwrapped_flux_transformer.base_model.config._name_or_path)
                    if hasattr(unwrapped_flux_transformer, 'peft_config'):
                        for key in unwrapped_flux_transformer.peft_config:
                            if hasattr(unwrapped_flux_transformer.peft_config[key], 'base_model_name_or_path'):
                                path_val = unwrapped_flux_transformer.peft_config[key].base_model_name_or_path
                                if isinstance(path_val, Path):
                                    unwrapped_flux_transformer.peft_config[key].base_model_name_or_path = str(path_val)
                    
                    # 删除可能损坏的模型卡片文件，让 PEFT 重新创建
                    model_card_path = save_path / "README.md"
                    if model_card_path.exists():
                        try:
                            model_card_path.unlink()
                        except Exception as e:
                            logger.warning(f"Failed to remove existing model card: {e}")
                    
                    # 将 Path 对象转换为字符串，避免 PEFT 模型卡片序列化错误
                    unwrapped_flux_transformer.save_pretrained(str(save_path), safe_serialization=True)
                    
                    # 保存训练状态（默认保存，用于恢复训练）
                    if not args.no_save_optimizer_state:
                        torch.save(optimizer.state_dict(), save_path / "optimizer.pt")
                        torch.save(lr_scheduler.state_dict(), save_path / "scheduler.pt")
                        if use_fp16_amp:
                            torch.save(scaler.state_dict(), save_path / "scaler.pt")
                    
                    training_state = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "best_loss": best_loss,
                        "current_loss": loss.item(),
                        "ema_loss": ema_loss,  # 保存 EMA 损失
                    }
                    with open(save_path / "training_state.json", "w") as f:
                        json.dump(training_state, f, indent=2)

                logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)

                if global_step >= args.max_train_steps:
                    break

if __name__ == "__main__":
    main()
