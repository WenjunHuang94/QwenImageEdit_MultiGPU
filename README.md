# QwenImageEdit_MultiGPU
A lightweight implementation of the Qwen-Image-Edit model for inference and LoRA fine-tuning on 8×V100 GPUs
---

## 📦 Training LoRA Installation

**Requirements:**
- Python 3.10

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Install the latest `diffusers` from GitHub:
   ```bash
   pip install git+https://github.com/huggingface/diffusers
   ```

---

## 📦 Inference with LoRA Installation

**Requirements:**
- Python 3.10

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Install the latest `diffusers` from GitHub:
   ```bash
   pip install git+https://github.com/huggingface/diffusers
   ```

3. In case you encounter an error like the following:
   <span style="color: red;">AttributeError: 'dict' object has no attribute 'to_dict'</span>
   **how to fix it**
   ```bash
   pip install --upgrade diffusers transformers accelerate
   ```

--

## 🌟 QuickStart
**Confirm you are all ready for processing the arguments properly**

1. run produce.sh to precompute embedds.

2. run consume.sh to train lora on your Qwen-Image-Edit model.

3. run infer.py to infer with official pipeline with Multi-GPU support.
<span style="color:#999999;">which take 1 hour and 20 minutes to generate i image.</span>

4. run vanillaInfer.py for a rather faster inference, while without CFG.
<span style="color:#999999;">which brings you a more enjoyable experience costing only 20 minutes.</span> 
