# %%capture
# import os
# if "COLAB_" not in "".join(os.environ.keys()):
#     !pip install unsloth
# else:
#     # Do this only in Colab notebooks! Otherwise use pip install unsloth
#     !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
#     !pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
#     !pip install --no-deps unsloth

# !pip install gTTS
# !apt-get install -y portaudio19-dev
# !pip install sounddevice scipy speechrecognition
# !apt-get install -y ffmpeg
# !pip install openai-whisper
# !pip install pyngrok

from unsloth import FastVisionModel
import torch
from PIL import Image
import requests
from io import BytesIO
import io
# from IPython.display import display, Javascript
# from google.colab import output
import base64
# from pydub import AudioSegment
# import whisper
# from gtts import gTTS
# from IPython.display import Audio
from flask import Flask, request, jsonify
from collections import deque
from transformers import TextStreamer
import gc

vlm_model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Llama-3.2-11B-Vision-Instruct",
    load_in_4bit = True,
    use_gradient_checkpointing = "unsloth",
)

vlm_model = FastVisionModel.get_peft_model(
    vlm_model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 16,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 16,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)


# # Image URL
# image_url = 'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG'

# # Load and display the image
# response = requests.get(image_url)
# image = Image.open(BytesIO(response.content))
# image


class VLMHandler():
    def initialize(self):
        self.model, self.tokenizer = vlm_model, tokenizer
        self.model.eval()

    def preprocess(self, image_url):
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        return image

    def inference(self, images, instruction):
        messages = [
            {"role": "user", "content": (
                [{"type": "image"} for _ in images] + [{"type": "text", "text": instruction}]
            )}
        ]
        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.tokenizer(
            images,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to("cuda")

        text_streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                streamer=text_streamer,
                max_new_tokens=50,
                use_cache=True,
                temperature=1.5,
                min_p=0.1
            )
        text_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Cleanup
        del inputs, outputs, text_streamer
        torch.cuda.empty_cache()
        gc.collect()

        return text_output

    def postprocess(self, inference_output):
        return inference_output.split('\n')[-1]

IMAGE_TO_TEXT_MODEL = VLMHandler()
IMAGE_TO_TEXT_MODEL.initialize()
