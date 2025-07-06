# File: model.py

# Install necessary libraries (remove unsloth and add peft)
# !pip install bitsandbytes accelerate xformers peft trl triton
# !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
# !pip install gTTS
# !apt-get install -y portaudio19-dev # For Colab/Linux audio if needed
# !pip install sounddevice scipy speechrecognition
# !apt-get install -y ffmpeg # For Colab/Linux audio if needed
# !pip install openai-whisper
# !pip install pyngrok

# Import necessary libraries
import torch
from PIL import Image
import requests
from io import BytesIO
import io
import base64
from flask import Flask, request, jsonify
from collections import deque
import gc

# NEW IMPORTS for standard Hugging Face model loading and PEFT
from transformers import AutoModelForCausalLM, AutoProcessor, TextStreamer
from peft import LoraConfig, get_peft_model

# --- Device Configuration ---
# Check for MPS (Apple Silicon), then CUDA (NVIDIA), then CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA (NVIDIA GPU)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# --- Model Loading without Unsloth ---
# Specify the model name
model_id = "unsloth/Llama-3.2-11B-Vision-Instruct" # Note: This is an Unsloth-optimized model.
                                                 # You might consider its base model if this causes issues
                                                 # without Unsloth's specific handling.
                                                 # A generic vision model like "llava-hf/llava-1.5-7b-hf" might be safer
                                                 # if the specific "unsloth/Llama-3.2-11B-Vision-Instruct"
                                                 # has deep Unsloth-specific architecture changes.

print(f"Loading VLM model: {model_id}...")

# Load processor and model
processor = AutoProcessor.from_pretrained(model_id)

# For MPS, load in float16 or bfloat16 for better performance and memory
# (MPS doesn't natively support bitsandbytes 4-bit)
torch_dtype = torch.float16 # or torch.bfloat16 if your model and device support it

# Load base model
# `low_cpu_mem_usage=True` is good practice, especially for large models
vlm_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
)

# --- Apply LoRA manually using PEFT ---
# Define LoRA configuration
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM", # For language models
    # target_modules can be specified if you know the exact linear layers to apply LoRA to.
    # For a general approach, it might be better to let `get_peft_model` discover them
    # or define them based on model architecture. Unsloth's "all-linear" is often a good default.
    # You might need to inspect the model's layers to determine specific target_modules
    # if `get_peft_model` doesn't apply to enough layers by default.
)

# Wrap the base model with PEFT
vlm_model = get_peft_model(vlm_model, lora_config)

# Move the model to the target device
vlm_model.to(device)

print("VLM model loaded and moved to device:", device)

# --- VLMHandler Class ---
class VLMHandler():
    def initialize(self):
        self.model = vlm_model # Use the globally loaded and PEFT-wrapped model
        self.tokenizer = processor.tokenizer # Use the tokenizer from the loaded processor
        self.processor = processor # Store the processor for image handling
        self.model.eval()

    def preprocess(self, image_url):
        response = requests.get(image_url)
        # Ensure image is in RGB format for consistency
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image

    def inference(self, images, instruction):
        # Prepare messages in chat template
        messages = [
            {"role": "user", "content": (
                [{"type": "image"} for _ in images] + [{"type": "text", "text": instruction}]
            )}
        ]
        
        # Apply chat template
        # `processor.apply_chat_template` is often used for VLM to get both text and image inputs
        # For pure text, `tokenizer.apply_chat_template` is common.
        # Given this is a VLM, we should use the processor's method for multi-modal inputs.
        
        # The `process_images=True` argument in processor's call will handle image tokenization
        # and integrate them into the input_ids.
        
        # NOTE: The images should be PIL Images or similar ready for the processor.
        # The `images` argument to `tokenizer()` (now `processor()`) expects PIL images directly.
        
        # First, format the chat template.
        # Ensure 'apply_chat_template' exists on the processor or handle messages differently.
        # Transformers VLMs often have a specific way to handle multimodal input.
        # A common pattern is: `inputs = processor(text=input_text, images=images, return_tensors="pt")`

        # Let's adjust based on typical LLaVA/Vision-Instruct model usage with AutoProcessor
        # The `messages` structure with "type": "image" and "type": "text" is common for LLaVA.
        # AutoProcessor's call handles this.
        
        inputs = self.processor(
            messages,
            images=images, # List of PIL Images
            return_tensors="pt",
            # add_special_tokens=False, # Often handled by apply_chat_template
        ).to(device) # Move inputs to the correct device

        text_streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                streamer=text_streamer,
                max_new_tokens=50,
                use_cache=True,
                temperature=1.5,
                min_p=0.1,
                # generation_config=... # Can set custom generation config
            )
        
        # Decode the output, handling cases where streamer already prints or you need the full text
        # If using streamer, text_output might contain repeated prompt or be empty if streamer prints everything.
        # A safer way to get the final text if streamer is just for console:
        text_output = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


        # Cleanup
        del inputs, outputs
        # Only clear cache if device is MPS or CUDA
        if device.type != 'cpu':
            torch.mps.empty_cache() # Use torch.mps.empty_cache() for MPS
            torch.cuda.empty_cache() # For CUDA
        gc.collect()

        return text_output

    def postprocess(self, inference_output):
        return inference_output.split('\n')[-1].strip() # Added .strip() for clean output

IMAGE_TO_TEXT_MODEL = VLMHandler()
IMAGE_TO_TEXT_MODEL.initialize()
