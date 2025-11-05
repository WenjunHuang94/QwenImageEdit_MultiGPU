import torch
from diffusers import (
    QwenImageEditPlusPipeline,
)
from accelerate import dispatch_model
from PIL import Image
# from optimum.quanto import quantize, freeze
import math
import argparse

# > tools -----------------------------------------------------------------------------

# args parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train LoRA for Qwen Image Edit (Accelerate+DeepSpeed)")

    # Paths / Basics
    parser.add_argument("--output_img", type=str, default="output.png")
    parser.add_argument("--pretrained_model", type=str, default="/storage/v-jinpewang/az_workspace/rico_model/Qwen-Image-Edit-2509")

    # LoRA / Quant
    parser.add_argument("--lora_weight", type=str, default="/storage/v-jinpewang/az_workspace/rico_model/test_lora_saves_edit/checkpoint-3000/pytorch_lora_weights.safetensors")
    parser.add_argument("--quant", type=str, default="qfloat8")


    # inputs
    parser.add_argument("--ctrl_img", type=str, default="/storage/v-jinpewang/lab_folder/junchao/data/image_eidt_dataset/processed_data_Accgen/with_textbox/input/000001_00000303_textbox.png")
    parser.add_argument("--prompt", type=str, default="Please return the correctly edited image.")
    parser.add_argument("--neg_prompt", type=str, default="")

    # infer arguments
    parser.add_argument("--target_area", type=int, default=1024*1024, help="Approximate target area (H*W) for 32-aligned resize")
    parser.add_argument("--width", type=int, default=1024, help="Approximate target area (H*W) for 32-aligned resize")
    parser.add_argument("--height", type=int, default=1024, help="Approximate target area (H*W) for 32-aligned resize")
    parser.add_argument("--infer_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-Free Guidance scale.")

    # seed
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")

    return parser.parse_args()

# load image
def get_image(path):
    img = Image.open(path).convert("RGB")
    pass

    return img

# calculate dimension for easy divised by 32
def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio

    width = round(width / 32) * 32
    height = round(height / 32) * 32

    return width, height

# wrapped flux transformer
class MultiGPUTransformer():
    """multi GPU包装器,通过device_map自动分配"""  
    def __init__(self, transformer):
        self.transformer = transformer
        self.num_gpus = torch.cuda.device_count()
        self.total_blocks = len(transformer.transformer_blocks)
        self.split_points = [i*(self.total_blocks // self.num_gpus) for i in range(1, self.num_gpus)]
    
    @property
    def device_map(self):
        device_map = {}
        res = 0
        # main device cuda:0
        for name, _ in self.transformer.named_children():
            if name != "transformer_blocks":
                device_map[name] = "cuda:1"
        # dispatch transformer blocks
        for item, splt in enumerate(self.split_points):
            temp = {f"transformer_blocks.{i}": f"cuda:{item+1}" for i in range(res, splt)}
            res = splt
            device_map.update(temp)

        temp = {f"transformer_blocks.{i}": f"cuda:{self.num_gpus-1}" for i in range(res, self.total_blocks)}
        device_map.update(temp)

        return device_map
        
    def auto_split(self):
        # accelerate dispatch
        try:
            model = dispatch_model(self.transformer, device_map=self.device_map)
            print("Successfully applied device_map using accelerate")
        except Exception as e:
            print(f"Error with accelerate dispatch: {e}")
            model = self.transformer
            pass
            # could add manual split logic here if needed
        return model
    
# > main -----------------------------------------------------------------------------

def main():
    args = parse_args()
    dtype = torch.bfloat16

    pipe = QwenImageEditPlusPipeline.from_pretrained(args.pretrained_model,
                                                    torch_dtype=dtype)

    pipe.vae.to("cuda:0")
    pipe.text_encoder.to("cuda:0")

    flux_transformer = MultiGPUTransformer(pipe.transformer).auto_split()
    pipe.transformer = flux_transformer
    
    if args.lora_weight:
        print(f"Loading LoRA weights from: {args.lora_weight}")
        pipe.load_lora_weights(args.lora_weight, adapter_name="lora", low_cpu_mem_usage=False)

    # NEVER QUANTIZE WHILE DISPATCHING, UNLESS THE DEVICE DISTRIBUTION IS HANDLED MANUALLY (PROPERLY)
    # if not args.quant:
    #    quantize(flux_transformer, weights=args.quant)
    #    freeze(flux_transformer)
    #    print("Transformer quantization complete.")


    # pipe.maybe_free_model_hooks()

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    image = get_image(args.ctrl_img)

    inputs = {
        "image": image,
        "prompt": args.prompt,
        "generator": generator,
        "true_cfg_scale": args.cfg_scale,
        "negative_prompt": args.neg_prompt,
        "num_inference_steps": args.infer_steps,
        "width": args.width,
        "height": args.height,
        "max_sequence_length":1024
    }

    pipe.set_progress_bar_config(disable=None)

    with torch.inference_mode():
        output = pipe(**inputs)
        output_image = output.images[0]
        output_image.save(args.output_img)
    print(f"Image successfully saved to {args.output_img}")

if __name__ == "__main__":
    main()    


