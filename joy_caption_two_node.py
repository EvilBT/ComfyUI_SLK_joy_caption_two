import shutil
import time

import numpy as np
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, \
    AutoModelForCausalLM
from pathlib import Path
import torch
import torch.amp.autocast_mode
from PIL import Image
import os

import comfy.model_management
import comfy.utils
import folder_paths
import torchvision.transforms.functional as TVF

from comfy.model_management import get_torch_device, get_free_memory

from comfy.model_management import load_models_gpu, text_encoder_dtype
from comfy.model_patcher import ModelPatcher
from .uitls import download_hg_model, modify_json_value, clear_cache
from .joy_config import joy_config

DEVICE = get_torch_device()

BASE_MODEL_PATH = Path(folder_paths.models_dir, "Joy_caption_two")

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

class JoyClipVisionModel:
    def __init__(self, load_device, offload_device):
        self.load_device = load_device
        self.offload_device = offload_device
        self.type = text_encoder_dtype()

        # clip
        model_id = "google/siglip-so400m-patch14-384"
        CLIP_PATH = download_hg_model(model_id, "clip")

        clip_model = AutoModel.from_pretrained(
            CLIP_PATH,
            trust_remote_code=True,
            torch_dtype=self.type
        )

        clip_model = clip_model.vision_model

        assert (BASE_MODEL_PATH / "clip_model.pt").exists()
        print("Loading VLM's custom vision model")
        checkpoint = torch.load(BASE_MODEL_PATH / "clip_model.pt", map_location='cpu', weights_only=True)
        checkpoint = {k.replace("_orig_mod.module.", ""): v for k, v in checkpoint.items()}
        clip_model.load_state_dict(checkpoint)
        del checkpoint

        clip_model.eval()
        clip_model.requires_grad_(False)
        self.model = clip_model

        self.patcher = ModelPatcher(self.model, load_device=self.load_device, offload_device=self.offload_device)

    def encode_image(self, pixel_values):
        #print(f"{id(self)}之前 in JoyClipVisionModel: {next(self.model.parameters()).device}")  # 打印模型参数的设备
        load_models_gpu([self.patcher], force_full_load=True, force_patch_weights=True)
        #print(f"之后 in JoyClipVisionModel: {next(self.model.parameters()).device}")  # 打印模型参数的设备
        vision_outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
        return vision_outputs




class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int, ln1: bool, pos_emb: bool, num_image_tokens: int, deep_extract: bool):
        super().__init__()
        self.deep_extract = deep_extract
        if self.deep_extract:
            input_features = input_features * 5

        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
        self.ln1 = nn.Identity() if not ln1 else nn.LayerNorm(input_features)
        self.pos_emb = None if not pos_emb else nn.Parameter(torch.zeros(num_image_tokens, input_features))
        # Other tokens (<|image_start|>, <|image_end|>, <|eot_id|>)
        self.other_tokens = nn.Embedding(3, output_features)
        self.other_tokens.weight.data.normal_(mean=0.0, std=0.02)   # Matches HF's implementation of llama3
    def forward(self, vision_outputs: torch.Tensor):
        if self.deep_extract:
            x = torch.concat((
				vision_outputs[-2],
				vision_outputs[3],
				vision_outputs[7],
				vision_outputs[13],
				vision_outputs[20],
			), dim=-1)
            assert len(x.shape) == 3, f"Expected 3, got {len(x.shape)}"  # batch, tokens, features
            assert x.shape[-1] == vision_outputs[-2].shape[-1] * 5, f"Expected {vision_outputs[-2].shape[-1] * 5}, got {x.shape[-1]}"
        else:
            x = vision_outputs[-2]

        x = self.ln1(x)

        if self.pos_emb is not None:
            assert x.shape[-2:] == self.pos_emb.shape, f"Expected {self.pos_emb.shape}, got {x.shape[-2:]}"
            x = x + self.pos_emb

        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)

        # <|image_start|>, IMAGE, <|image_end|>
        other_tokens = self.other_tokens(torch.tensor([0, 1], device=self.other_tokens.weight.device).expand(x.shape[0], -1))
        assert other_tokens.shape == (x.shape[0], 2, x.shape[2]), f"Expected {(x.shape[0], 2, x.shape[2])}, got {other_tokens.shape}"
        x = torch.cat((other_tokens[:, 0:1], x, other_tokens[:, 1:2]), dim=1)
        return x

    def get_eot_embedding(self):
        return self.other_tokens(torch.tensor([2], device=self.other_tokens.weight.device)).squeeze(0)


class JoyImageAdapter:
    def __init__(self, load_device, offload_device):
        self.load_device = load_device
        self.offload_device = offload_device

        # Image Adapter
        adapter_path = os.path.join(BASE_MODEL_PATH, "image_adapter.pt")

        image_adapter = ImageAdapter(1152, 4096, False, False, 38,
                                     False)  # ImageAdapter(clip_model.config.hidden_size, 4096)
        image_adapter.load_state_dict(torch.load(adapter_path, map_location=self.offload_device, weights_only=True))
        img_dtype = text_encoder_dtype()
        image_adapter = image_adapter.to(img_dtype)
        image_adapter.eval()
        self.image_adapter = image_adapter

        self.patcher = ModelPatcher(self.image_adapter, load_device=self.load_device,
                                    offload_device=self.offload_device)

    def embedded_image(self, hidden_states):
        load_models_gpu([self.patcher], force_full_load=True, force_patch_weights=True)
        embedded_images = self.image_adapter(hidden_states)
        embedded_images.to(self.load_device)
        return embedded_images

class JoyLLM:
    def __init__(self, load_device, offload_device, model_id):
        self.load_device = load_device
        self.offload_device = offload_device
        self.type = text_encoder_dtype()
        self.model_id = model_id
        self.current_model_id = None

        print("Loading tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(BASE_MODEL_PATH, "text_model"), use_fast=True)
        assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer,
                                                                        PreTrainedTokenizerFast), f"Tokenizer is of type {type(tokenizer)}"

        self.tokenizer = tokenizer
        self.text_model = None

    def load_llm_model(self):
        if self.text_model is None or self.current_model_id != self.model_id:
            print(f"Loading LLM: {self.model_id}")
            LLM_PATH = download_hg_model(self.model_id, "LLM")
            text_model_path = os.path.join(BASE_MODEL_PATH, "text_model")
            modify_json_value(os.path.join(text_model_path, "adapter_config.json"), "base_model_name_or_path",
                              LLM_PATH)
            max_retries = 5  # 设置最大重试次数
            retries = 0
            while True:
                free_vram = get_free_memory()/1024/1024
                # print(f"现在的显存{retries}:{free_vram}")
                if free_vram > 6400:
                    text_model = AutoModelForCausalLM.from_pretrained(text_model_path,
                                                              device_map=self.load_device,
                                                              local_files_only=True,
                                                              trust_remote_code=True, torch_dtype=self.type)
                    text_model.eval()
                    self.text_model = text_model
                    break
                else:
                    clear_cache()
                    retries += 1
                    if retries > max_retries:
                        text_model = AutoModelForCausalLM.from_pretrained(text_model_path,
                                                                          device_map=self.load_device,
                                                                          local_files_only=True,
                                                                          trust_remote_code=True,
                                                                          torch_dtype=self.type)
                        text_model.eval()
                        self.text_model = text_model
                        break
                    time.sleep(1 + retries / 2)
            # print(f"现在呢:{get_free_memory()/1024/1024}")
            self.current_model_id = self.model_id
        return self.text_model

    def clear_gpu(self, low_vram):
        del self.text_model
        self.text_model = None
        self.current_model_id = None
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        if low_vram:
            clear_cache()


class JoyTwoPipeline:
    def __init__(self, load_device, offload_device):
        self.load_device = load_device
        self.offload_device = offload_device
        self.clip_model: JoyClipVisionModel | None = None
        self.image_adapter: JoyImageAdapter | None = None
        self.llm: JoyLLM | None = None
        self.parent = None
        self.model = None

    def clearCache(self):
        self.clip_model = None
        self.image_adapter = None
        self.model = None

    def loadModels(self):
        # clip
        self.clip_model = JoyClipVisionModel(self.load_device, self.offload_device)

        self.image_adapter = JoyImageAdapter(self.load_device, self.offload_device)

    def loadLLM(self, model_id):
        if self.llm is None:
            self.llm = JoyLLM(self.load_device, self.offload_device, model_id)
        else:
            self.llm.model_id = model_id


class Joy_caption_two_load:

    def __init__(self):
        self.model = None
        self.load_device = comfy.model_management.text_encoder_device()
        self.offload_device = comfy.model_management.text_encoder_offload_device()
        self.pipeline = JoyTwoPipeline(self.load_device, self.offload_device)
        self.pipeline.parent = self

    @classmethod
    def INPUT_TYPES(s):
        models = joy_config["model"]
        return {
            "required": {
                "model": (models, ),
            }
        }

    CATEGORY = "SLK/LLM"
    RETURN_TYPES = ("JoyTwoPipeline",)
    FUNCTION = "generate"

    def loadModels(self):
        self.pipeline.loadModels()

    def generate(self, model):
        if self.model is None or self.model != model or self.pipeline is None:
            self.model = model
            self.loadModels()
        self.pipeline.model = model
        return (self.pipeline,)


class Joy_caption_two:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        caption_lengths = list(joy_config["CAPTION_LENGTH"])
        caption_types = list(joy_config["CAPTION_TYPE_MAP"].keys())
        return {
            "required": {
                "joy_two_pipeline": ("JoyTwoPipeline",),
                "image": ("IMAGE",),
                "caption_type": (caption_types, {}),
                "caption_length": (caption_lengths, {"default": "long"}),
                "low_vram": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "SLK/LLM"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"

    def generate(self, joy_two_pipeline: JoyTwoPipeline, image, caption_type, caption_length, low_vram):
        torch.cuda.empty_cache()

        if joy_two_pipeline.clip_model is None:
            joy_two_pipeline.parent.loadModels()

        # 'any' means no length specified
        length = None if caption_length == "any" else caption_length

        if isinstance(length, str):
            try:
                length = int(length)
            except ValueError:
                pass

        # Build prompt
        if length is None:
            map_idx = 0
        elif isinstance(length, int):
            map_idx = 1
        elif isinstance(length, str):
            map_idx = 2
        else:
            raise ValueError(f"Invalid caption length: {length}")

        caption_type_map = joy_config["CAPTION_TYPE_MAP"]
        prompt_str = list(caption_type_map[caption_type])[map_idx]

        prompt_str = prompt_str.format(length=caption_length, word_count=caption_length)

        # For debugging
        # print(f"Prompt: {prompt_str}")

        # Preprocess image
        # NOTE: I found the default processor for so400M to have worse results than just using PIL directly
        # image = clip_processor(images=input_image, return_tensors='pt').pixel_values
        image = tensor2pil(image)
        image = image.resize((384, 384), Image.LANCZOS)
        pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to(joy_two_pipeline.load_device)

        # Embed image
        # This results in Batch x Image Tokens x Features
        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = joy_two_pipeline.clip_model.encode_image(pixel_values)
            embedded_images = joy_two_pipeline.image_adapter.embedded_image(vision_outputs.hidden_states)

        if low_vram:
            pixel_values.to(joy_two_pipeline.offload_device)
            clear_cache()

        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt_str,
            },
        ]

        joy_two_pipeline.loadLLM(joy_two_pipeline.model)

        tokenizer = joy_two_pipeline.llm.tokenizer
        # Format the conversation
        convo_string = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        assert isinstance(convo_string, str)

        # Tokenize the conversation
        # prompt_str is tokenized separately so we can do the calculations below
        convo_tokens = tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False)
        prompt_tokens = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False, truncation=False)
        assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
        convo_tokens = convo_tokens.squeeze(0)  # Squeeze just to make the following easier
        prompt_tokens = prompt_tokens.squeeze(0)

        # Calculate where to inject the image
        eot_id_indices = (convo_tokens == tokenizer.convert_tokens_to_ids("<|eot_id|>")).nonzero(as_tuple=True)[
            0].tolist()
        assert len(eot_id_indices) == 2, f"Expected 2 <|eot_id|> tokens, got {len(eot_id_indices)}"

        preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]  # Number of tokens before the prompt


        text_model = joy_two_pipeline.llm.load_llm_model()
        # Embed the tokens
        convo_embeds = text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to(joy_two_pipeline.load_device))

        # Construct the input
        input_embeds = torch.cat([
            convo_embeds[:, :preamble_len],  # Part before the prompt
            embedded_images.to(dtype=convo_embeds.dtype),  # Image
            convo_embeds[:, preamble_len:],  # The prompt and anything after it
        ], dim=1).to(joy_two_pipeline.load_device)

        input_ids = torch.cat([
            convo_tokens[:preamble_len].unsqueeze(0),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
            # Dummy tokens for the image (TODO: Should probably use a special token here so as not to confuse any generation algorithms that might be inspecting the input)
            convo_tokens[preamble_len:].unsqueeze(0),
        ], dim=1).to(joy_two_pipeline.load_device)
        attention_mask = torch.ones_like(input_ids)

        # Debugging
        # print(f"Input to model: {repr(tokenizer.decode(input_ids[0]))}")

        # generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=False, suppress_tokens=None)
        # generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=True, top_k=10, temperature=0.5, suppress_tokens=None)
        generate_ids = text_model.generate(input_ids, inputs_embeds=input_embeds, attention_mask=attention_mask,
                                           max_new_tokens=300, do_sample=True,
                                           suppress_tokens=None)  # Uses the default which is temp=0.6, top_p=0.9

        # Trim off the prompt
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        if generate_ids[0][-1] == tokenizer.eos_token_id or generate_ids[0][-1] == tokenizer.convert_tokens_to_ids(
                "<|eot_id|>"):
            generate_ids = generate_ids[:, :-1]

        caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

        joy_two_pipeline.llm.clear_gpu(low_vram)

        return (caption.strip(), )

class Joy_caption_two_advanced:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        caption_lengths = list(joy_config["CAPTION_LENGTH"])
        caption_types = list(joy_config["CAPTION_TYPE_MAP"].keys())
        return {
            "required": {
                "joy_two_pipeline": ("JoyTwoPipeline",),
                "image": ("IMAGE",),
                "extra_options": ("Extra_Options", ),
                "caption_type": (caption_types, {}),
                "caption_length": (caption_lengths, {"default": "long"}),
                "name": ("STRING", {"default": ""}),
                "custom_prompt": ("STRING", {"default": ""}),
                "low_vram": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    CATEGORY = "SLK/LLM"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"

    def generate(self, joy_two_pipeline: JoyTwoPipeline, image, extra_options, caption_type, caption_length, name, custom_prompt, low_vram, top_p, temperature):
        torch.cuda.empty_cache()

        if joy_two_pipeline.clip_model == None:
            joy_two_pipeline.parent.loadModels()

        # 'any' means no length specified
        length = None if caption_length == "any" else caption_length

        if isinstance(length, str):
            try:
                length = int(length)
            except ValueError:
                pass

        # Build prompt
        if length is None:
            map_idx = 0
        elif isinstance(length, int):
            map_idx = 1
        elif isinstance(length, str):
            map_idx = 2
        else:
            raise ValueError(f"Invalid caption length: {length}")

        caption_type_map = joy_config["CAPTION_TYPE_MAP"]
        prompt_str = list(caption_type_map[caption_type])[map_idx]

        # Add extra options
        if len(extra_options) > 0:
            prompt_str += " " + " ".join(extra_options)

        # Add name, length, word_count
        prompt_str = prompt_str.format(name=name, length=caption_length, word_count=caption_length)

        if custom_prompt.strip() != "":
            prompt_str = custom_prompt.strip()

        # For debugging
        print(f"Prompt: {prompt_str}")

        # Preprocess image
        # NOTE: I found the default processor for so400M to have worse results than just using PIL directly
        # image = clip_processor(images=input_image, return_tensors='pt').pixel_values
        image = tensor2pil(image)
        image = image.resize((384, 384), Image.LANCZOS)
        pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to(joy_two_pipeline.load_device)

        # Embed image
        # This results in Batch x Image Tokens x Features
        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = joy_two_pipeline.clip_model.encode_image(pixel_values)
            embedded_images = joy_two_pipeline.image_adapter.embedded_image(vision_outputs.hidden_states)

        if low_vram:
            pixel_values.to(joy_two_pipeline.offload_device)
            clear_cache()

        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt_str,
            },
        ]

        joy_two_pipeline.loadLLM(joy_two_pipeline.model)

        tokenizer = joy_two_pipeline.llm.tokenizer
        # Format the conversation
        convo_string = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        assert isinstance(convo_string, str)

        # Tokenize the conversation
        # prompt_str is tokenized separately so we can do the calculations below
        convo_tokens = tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False)
        prompt_tokens = tokenizer.encode(prompt_str, return_tensors="pt", add_special_tokens=False, truncation=False)
        assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
        convo_tokens = convo_tokens.squeeze(0)  # Squeeze just to make the following easier
        prompt_tokens = prompt_tokens.squeeze(0)

        # Calculate where to inject the image
        eot_id_indices = (convo_tokens == tokenizer.convert_tokens_to_ids("<|eot_id|>")).nonzero(as_tuple=True)[
            0].tolist()
        assert len(eot_id_indices) == 2, f"Expected 2 <|eot_id|> tokens, got {len(eot_id_indices)}"

        preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]  # Number of tokens before the prompt


        text_model = joy_two_pipeline.llm.load_llm_model()
        # Embed the tokens
        convo_embeds = text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to(joy_two_pipeline.load_device))
        # Construct the input
        input_embeds = torch.cat([
            convo_embeds[:, :preamble_len],  # Part before the prompt
            embedded_images.to(dtype=convo_embeds.dtype),  # Image
            convo_embeds[:, preamble_len:],  # The prompt and anything after it
        ], dim=1).to(joy_two_pipeline.load_device)

        input_ids = torch.cat([
            convo_tokens[:preamble_len].unsqueeze(0),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
            # Dummy tokens for the image (TODO: Should probably use a special token here so as not to confuse any generation algorithms that might be inspecting the input)
            convo_tokens[preamble_len:].unsqueeze(0),
        ], dim=1).to(joy_two_pipeline.load_device)
        attention_mask = torch.ones_like(input_ids)

        # Debugging
        # print(f"Input to model: {repr(tokenizer.decode(input_ids[0]))}")

        # generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=False, suppress_tokens=None)
        # generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=True, top_k=10, temperature=0.5, suppress_tokens=None)
        generate_ids = text_model.generate(input_ids, inputs_embeds=input_embeds, attention_mask=attention_mask,
                                           max_new_tokens=300, do_sample=True,
                                           top_p=top_p, temperature=temperature,
                                           suppress_tokens=None)  # Uses the default which is temp=0.6, top_p=0.9

        # Trim off the prompt
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        if generate_ids[0][-1] == tokenizer.eos_token_id or generate_ids[0][-1] == tokenizer.convert_tokens_to_ids(
                "<|eot_id|>"):
            generate_ids = generate_ids[:, :-1]

        caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

        joy_two_pipeline.llm.clear_gpu(low_vram)

        return (caption.strip(), )

class Batch_joy_caption_two:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        caption_lengths = list(joy_config["CAPTION_LENGTH"])
        caption_types = list(joy_config["CAPTION_TYPE_MAP"].keys())
        return {
            "required": {
                "joy_two_pipeline": ("JoyTwoPipeline",),
                "input_dir": ("STRING", {"default": ""}),
                "output_dir": ("STRING", {"default": ""}),
                "caption_type": (caption_types, {}),
                "caption_length": (caption_lengths, {"default": "long"}),
                "low_vram": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "SLK/LLM"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"

    def generate_caption(self, joy_two_pipeline: JoyTwoPipeline, image, prompt, low_vram=True):
        torch.cuda.empty_cache()
        pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to(joy_two_pipeline.load_device)

        # Embed image
        # This results in Batch x Image Tokens x Features
        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = joy_two_pipeline.clip_model.encode_image(pixel_values)
            embedded_images = joy_two_pipeline.image_adapter.embedded_image(vision_outputs.hidden_states)

        if low_vram:
            pixel_values.to(joy_two_pipeline.offload_device)
            clear_cache()

        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        joy_two_pipeline.loadLLM(joy_two_pipeline.model)

        tokenizer = joy_two_pipeline.llm.tokenizer
        # Format the conversation
        convo_string = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        assert isinstance(convo_string, str)

        # Tokenize the conversation
        # prompt_str is tokenized separately so we can do the calculations below
        convo_tokens = tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False)
        prompt_tokens = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
        assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
        convo_tokens = convo_tokens.squeeze(0)  # Squeeze just to make the following easier
        prompt_tokens = prompt_tokens.squeeze(0)

        # Calculate where to inject the image
        eot_id_indices = (convo_tokens == tokenizer.convert_tokens_to_ids("<|eot_id|>")).nonzero(as_tuple=True)[
            0].tolist()
        assert len(eot_id_indices) == 2, f"Expected 2 <|eot_id|> tokens, got {len(eot_id_indices)}"

        preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]  # Number of tokens before the prompt

        text_model = joy_two_pipeline.llm.load_llm_model()
        # Embed the tokens
        convo_embeds = text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to(joy_two_pipeline.load_device))
        # Construct the input
        input_embeds = torch.cat([
            convo_embeds[:, :preamble_len],  # Part before the prompt
            embedded_images.to(dtype=convo_embeds.dtype),  # Image
            convo_embeds[:, preamble_len:],  # The prompt and anything after it
        ], dim=1).to(joy_two_pipeline.load_device)

        input_ids = torch.cat([
            convo_tokens[:preamble_len].unsqueeze(0),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
            # Dummy tokens for the image (TODO: Should probably use a special token here so as not to confuse any generation algorithms that might be inspecting the input)
            convo_tokens[preamble_len:].unsqueeze(0),
        ], dim=1).to(joy_two_pipeline.load_device)
        attention_mask = torch.ones_like(input_ids)

        # Debugging
        # print(f"Input to model: {repr(tokenizer.decode(input_ids[0]))}")

        # generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=False, suppress_tokens=None)
        # generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=True, top_k=10, temperature=0.5, suppress_tokens=None)
        generate_ids = text_model.generate(input_ids, inputs_embeds=input_embeds, attention_mask=attention_mask,
                                           max_new_tokens=300, do_sample=True,
                                           suppress_tokens=None)  # Uses the default which is temp=0.6, top_p=0.9

        # Trim off the prompt
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        if generate_ids[0][-1] == tokenizer.eos_token_id or generate_ids[0][-1] == tokenizer.convert_tokens_to_ids(
                "<|eot_id|>"):
            generate_ids = generate_ids[:, :-1]

        caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

        return caption.strip()

    def generate(self, joy_two_pipeline: JoyTwoPipeline, input_dir, output_dir, caption_type, caption_length, low_vram):
        torch.cuda.empty_cache()

        if joy_two_pipeline.clip_model == None:
            joy_two_pipeline.parent.loadModels()

        # 'any' means no length specified
        length = None if caption_length == "any" else caption_length

        if isinstance(length, str):
            try:
                length = int(length)
            except ValueError:
                pass

        # Build prompt
        if length is None:
            map_idx = 0
        elif isinstance(length, int):
            map_idx = 1
        elif isinstance(length, str):
            map_idx = 2
        else:
            raise ValueError(f"Invalid caption length: {length}")

        caption_type_map = joy_config["CAPTION_TYPE_MAP"]
        prompt_str = list(caption_type_map[caption_type])[map_idx]

        # Add name, length, word_count
        prompt_str = prompt_str.format(length=caption_length, word_count=caption_length)

        # For debugging
        print(f"Prompt: {prompt_str}")

        if output_dir is None or output_dir.strip() == "":
            output_dir = input_dir

        finished_image_count = 0
        error_image_count = 0
        image_count = 0

        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".webp")):
                image_count += 1

        pbar = comfy.utils.ProgressBar(image_count)
        step = 0
        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".webp")):
                image_path = os.path.join(input_dir, filename)
                text_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')

                try:
                    print(f"打开{image_path}")
                    with Image.open(image_path) as img:
                        if img.mode == 'RGBA':
                            img = img.convert('RGB')
                        pbar.update_absolute(step, image_count)
                        image = img.resize((384, 384), Image.LANCZOS)
                        caption = self.generate_caption(joy_two_pipeline, image, prompt_str)
                        with open(text_path, 'w', encoding='utf-8') as f:
                            f.write(caption)
                    finished_image_count += 1
                except Exception as e:
                    print(f"Error processing {filename} :{e}")
                    error_image_count += 1
                step += 1


        # Preprocess image
        # NOTE: I found the default processor for so400M to have worse results than just using PIL directly
        # image = clip_processor(images=input_image, return_tensors='pt').pixel_values

        joy_two_pipeline.llm.clear_gpu(low_vram)

        return (f"result: finished count: {finished_image_count}, error count: {error_image_count}", )

class Batch_joy_caption_two_advanced:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        caption_lengths = list(joy_config["CAPTION_LENGTH"])
        caption_types = list(joy_config["CAPTION_TYPE_MAP"].keys())
        return {
            "required": {
                "joy_two_pipeline": ("JoyTwoPipeline",),
                "input_dir": ("STRING", {"default": ""}),
                "output_dir": ("STRING", {"default": ""}),
                "rename": ("BOOLEAN", {"default": False}),
                "prefix_name": ("STRING", {"default": ""}),
                "start_index": ("INT", {"default": 1, "min": 0, "max": 9999999, "step": 1}),
                "extra_options": ("Extra_Options", ),
                "caption_type": (caption_types, {}),
                "caption_length": (caption_lengths, {"default": "long"}),
                "name": ("STRING", {"default": ""}),
                "custom_prompt": ("STRING", {"default": ""}),
                "low_vram": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                "prefix_caption": ("STRING", {"default": ""}),
                "suffix_caption": ("STRING", {"default": ""}),
            }
        }

    CATEGORY = "SLK/LLM"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"

    def generate_caption(self, joy_two_pipeline: JoyTwoPipeline, image, prompt, top_p=0.9, temperature=0.6, low_vram=True):
        torch.cuda.empty_cache()
        pixel_values = TVF.pil_to_tensor(image).unsqueeze(0) / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5], [0.5])
        pixel_values = pixel_values.to(joy_two_pipeline.load_device)

        # Embed image
        # This results in Batch x Image Tokens x Features
        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = joy_two_pipeline.clip_model.encode_image(pixel_values)
            embedded_images = joy_two_pipeline.image_adapter.embedded_image(vision_outputs.hidden_states)

        if low_vram:
            pixel_values.to(joy_two_pipeline.offload_device)
            clear_cache()

        # Build the conversation
        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        joy_two_pipeline.loadLLM(joy_two_pipeline.model)

        tokenizer = joy_two_pipeline.llm.tokenizer
        # Format the conversation
        convo_string = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        assert isinstance(convo_string, str)

        # Tokenize the conversation
        # prompt_str is tokenized separately so we can do the calculations below
        convo_tokens = tokenizer.encode(convo_string, return_tensors="pt", add_special_tokens=False, truncation=False)
        prompt_tokens = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
        assert isinstance(convo_tokens, torch.Tensor) and isinstance(prompt_tokens, torch.Tensor)
        convo_tokens = convo_tokens.squeeze(0)  # Squeeze just to make the following easier
        prompt_tokens = prompt_tokens.squeeze(0)

        # Calculate where to inject the image
        eot_id_indices = (convo_tokens == tokenizer.convert_tokens_to_ids("<|eot_id|>")).nonzero(as_tuple=True)[
            0].tolist()
        assert len(eot_id_indices) == 2, f"Expected 2 <|eot_id|> tokens, got {len(eot_id_indices)}"

        preamble_len = eot_id_indices[1] - prompt_tokens.shape[0]  # Number of tokens before the prompt

        text_model = joy_two_pipeline.llm.load_llm_model()
        # Embed the tokens
        convo_embeds = text_model.model.embed_tokens(convo_tokens.unsqueeze(0).to(joy_two_pipeline.load_device))
        # Construct the input
        input_embeds = torch.cat([
            convo_embeds[:, :preamble_len],  # Part before the prompt
            embedded_images.to(dtype=convo_embeds.dtype),  # Image
            convo_embeds[:, preamble_len:],  # The prompt and anything after it
        ], dim=1).to(joy_two_pipeline.load_device)

        input_ids = torch.cat([
            convo_tokens[:preamble_len].unsqueeze(0),
            torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
            # Dummy tokens for the image (TODO: Should probably use a special token here so as not to confuse any generation algorithms that might be inspecting the input)
            convo_tokens[preamble_len:].unsqueeze(0),
        ], dim=1).to(joy_two_pipeline.load_device)
        attention_mask = torch.ones_like(input_ids)

        # Debugging
        # print(f"Input to model: {repr(tokenizer.decode(input_ids[0]))}")

        # generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=False, suppress_tokens=None)
        # generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=True, top_k=10, temperature=0.5, suppress_tokens=None)
        generate_ids = text_model.generate(input_ids, inputs_embeds=input_embeds, attention_mask=attention_mask,
                                           max_new_tokens=300, do_sample=True,
                                           top_p=top_p, temperature=temperature,
                                           suppress_tokens=None)  # Uses the default which is temp=0.6, top_p=0.9

        # Trim off the prompt
        generate_ids = generate_ids[:, input_ids.shape[1]:]
        if generate_ids[0][-1] == tokenizer.eos_token_id or generate_ids[0][-1] == tokenizer.convert_tokens_to_ids(
                "<|eot_id|>"):
            generate_ids = generate_ids[:, :-1]

        caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

        return caption.strip()

    def generate(self, joy_two_pipeline: JoyTwoPipeline, input_dir, output_dir, rename, prefix_name, start_index,
                 extra_options, caption_type, caption_length, name, custom_prompt, low_vram, top_p, temperature,
                 prefix_caption, suffix_caption):
        torch.cuda.empty_cache()

        if joy_two_pipeline.clip_model == None:
            joy_two_pipeline.parent.loadModels()

        # 'any' means no length specified
        length = None if caption_length == "any" else caption_length

        if isinstance(length, str):
            try:
                length = int(length)
            except ValueError:
                pass

        # Build prompt
        if length is None:
            map_idx = 0
        elif isinstance(length, int):
            map_idx = 1
        elif isinstance(length, str):
            map_idx = 2
        else:
            raise ValueError(f"Invalid caption length: {length}")

        caption_type_map = joy_config["CAPTION_TYPE_MAP"]
        prompt_str = list(caption_type_map[caption_type])[map_idx]

        # Add extra options
        if len(extra_options) > 0:
            prompt_str += " " + " ".join(extra_options)

        # Add name, length, word_count
        prompt_str = prompt_str.format(name=name, length=caption_length, word_count=caption_length)

        if custom_prompt.strip() != "":
            prompt_str = custom_prompt.strip()

        # For debugging
        print(f"Prompt: {prompt_str}")

        if output_dir is None or output_dir.strip() == "":
            output_dir = input_dir

        finished_image_count = 0
        error_image_count = 0
        image_count = 0

        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".webp")):
                image_count += 1

        pbar = comfy.utils.ProgressBar(image_count)
        step = 0
        for filename in os.listdir(input_dir):
            if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp", ".webp")):
                image_path = os.path.join(input_dir, filename)
                text_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')

                try:
                    print(f"打开{image_path}")
                    with Image.open(image_path) as img:
                        if img.mode == 'RGBA':
                            image = img.convert('RGB').resize((384, 384), Image.LANCZOS)
                        else:
                            image = img.resize((384, 384), Image.LANCZOS)
                        pbar.update_absolute(step, image_count)
                        caption = self.generate_caption(joy_two_pipeline, image, prompt_str, top_p, temperature)
                        if prefix_caption:
                            caption = f"{prefix_caption} {caption}"
                        if suffix_caption:
                            caption = f"{caption} {suffix_caption}"
                        if rename:
                            new_filename = f"{prefix_name}_{start_index + finished_image_count}{os.path.splitext(filename)[1]}"
                            new_image_path = os.path.join(output_dir, new_filename)
                            shutil.copyfile(image_path, new_image_path)
                            text_path = os.path.join(output_dir, os.path.splitext(new_filename)[0] + '.txt')

                        with open(text_path, 'w', encoding='utf-8') as f:
                            f.write(caption)
                    finished_image_count += 1
                except Exception as e:
                    print(f"Error processing {filename} :{e}")
                    error_image_count += 1
                step += 1


        # Preprocess image
        # NOTE: I found the default processor for so400M to have worse results than just using PIL directly
        # image = clip_processor(images=input_image, return_tensors='pt').pixel_values

        joy_two_pipeline.llm.clear_gpu(low_vram)

        return (f"result: finished count: {finished_image_count}, error count: {error_image_count}", )

class Joy_extra_options:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        options = list(joy_config["EXTRA_OPTIONS"])
        required = {}
        for option in options:
            required[option] = ("BOOLEAN", {"default": False})
        return {
            "required": required
        }

    CATEGORY = "SLK/LLM"
    RETURN_TYPES = ("Extra_Options",)
    FUNCTION = "run"

    def run(self, **kwargs):
        # 转为列表
        options_selected = list(kwargs.values())
        options = list(joy_config["EXTRA_OPTIONS"])
        values = []
        for selected, option in zip(options_selected, options):
            if selected:
                values.append(option)
        return (values, )