import folder_paths
import torch
import torch.nn.functional as F
import os
import json
import torchaudio

from comfy.utils import load_torch_file, common_upscale
import comfy.model_management as mm

# 从标准库导入，需要确保您的环境中安装了 accelerate 库
from accelerate import init_empty_weights 
# 修复相对导入：用标准库的日志和简单的占位函数替换
import logging
log = logging.getLogger(__name__)

script_directory = os.path.dirname(os.path.abspath(__file__))
device = mm.get_torch_device()
offload_device = mm.unet_offload_device()

# -----------------------------------------------------------------
# 辅助函数：linear_interpolation_fps (来自原文件)
# -----------------------------------------------------------------
def linear_interpolation_fps(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)  # [1, C, T]
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)

# -----------------------------------------------------------------
# 辅助函数：get_audio_emb_window (来自原文件)
# -----------------------------------------------------------------
def get_audio_emb_window(audio_emb, frame_num, frame0_idx, audio_shift=2):
    zero_audio_embed = torch.zeros((audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)
    zero_audio_embed_3 = torch.zeros((3, audio_emb.shape[1], audio_emb.shape[2]), dtype=audio_emb.dtype, device=audio_emb.device)
    iter_ = 1 + (frame_num - 1) // 4
    audio_emb_wind = []
    for lt_i in range(iter_):
        if lt_i == 0:
            st = frame0_idx + lt_i - 2
            ed = frame0_idx + lt_i + 3
            wind_feat = torch.stack([
                audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                for i in range(st, ed)
            ], dim=0)
            wind_feat = torch.cat((zero_audio_embed_3, wind_feat), dim=0)
        else:
            st = frame0_idx + 1 + 4 * (lt_i - 1) - audio_shift
            ed = frame0_idx + 1 + 4 * lt_i + audio_shift
            wind_feat = torch.stack([
                audio_emb[i] if (0 <= i < audio_emb.shape[0]) else zero_audio_embed
                for i in range(st, ed)
            ], dim=0)
        audio_emb_wind.append(wind_feat)
    audio_emb_wind = torch.stack(audio_emb_wind, dim=0)

    return audio_emb_wind, ed - audio_shift

# -----------------------------------------------------------------
# 修复依赖：set_module_tensor_to_device (使用一个简单占位函数，但建议用户检查是否需要完整功能)
# -----------------------------------------------------------------
def set_module_tensor_to_device(module, tensor_name, device, dtype=None, value=None):
    if value is not None and hasattr(module, tensor_name):
        # 实际操作：将tensor设置到指定设备
        getattr(module, tensor_name).data = value.to(device, dtype=dtype)
    elif value is not None:
         # 占位：如果module没有这个属性（通常不会），则跳过
         pass

# -----------------------------------------------------------------
# 关键依赖：WanVideoEncodeLatentBatch (需要从原节点包中复制)
# -----------------------------------------------------------------
# ⚠️ 警告：WanVideoEncodeLatentBatch 的定义不在您提供的文件中，它来自原节点包的另一个文件。
# 为了让代码能够运行，我保留了其名称，但您必须确保该类的完整定义已存在于您的环境中或复制到此文件中。
# 如果运行时报错 'WanVideoEncodeLatentBatch' not defined，请将原节点包中该类的完整代码粘贴到此处。
class WanVideoEncodeLatentBatch:
    # 这是一个占位符，您需要将原节点包中该类的完整实现代码复制到这里
    # 否则，代码会因为调用 encode 方法而失败。
    @classmethod
    def encode(self, vae, pixels, tiled_vae, tile_size, fast_tile_decode, tile_padding, force_zero_area, batch_size):
        raise NotImplementedError("WanVideoEncodeLatentBatch 尚未定义。请将原节点包中该类的完整代码粘贴到此处。")

# -----------------------------------------------------------------
# 节点类定义：WhisperModelLoader (更名为 PainterWhisperModelLoader)
# -----------------------------------------------------------------
class PainterWhisperModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("audio_encoders"), {"tooltip": "These models are loaded from the 'ComfyUI/models/audio_encoders' folder",}),
                "base_precision": (["fp32", "bf16", "fp16"], {"default": "fp16"}),
                "load_device": (["main_device", "offload_device"], {"default": "main_device", "tooltip": "Initial device to load the model to, NOT recommended with the larger models unless you have 48GB+ VRAM"}),
            },
        }

    RETURN_TYPES = ("WHISPERMODEL",)
    RETURN_NAMES = ("whisper_model", )
    FUNCTION = "loadmodel"
    CATEGORY = "painter-humo-i2v"

    def loadmodel(self, model, base_precision, load_device):
        from transformers import WhisperConfig, WhisperModel, WhisperFeatureExtractor

        # 注意：此处使用的 torch.float8_e4m3fn 等类型，在旧版 PyTorch 中可能不存在
        base_dtype = {"fp8_e4m3fn": torch.float8_e4m3fn, "fp8_e4m3fn_fast": torch.float8_e4m3fn, "bf16": torch.bfloat16, "fp16": torch.float16, "fp16_fast": torch.float16, "fp32": torch.float32}[base_precision]
        
        if load_device == "offload_device":
            transformer_load_device = offload_device
        else:
            transformer_load_device = device

        config_path = os.path.join(script_directory, "whisper_config.json")
        
        # ❗ 注意：如果您没有将 'whisper_config.json' 复制到新文件夹中，这里会报错。
        whisper_config = WhisperConfig(**json.load(open(config_path)))

        with init_empty_weights():
            whisper = WhisperModel(whisper_config).eval()
            whisper.decoder = None  # we only need the encoder

        feature_extractor_config = {
            "chunk_length": 30,
            "feature_extractor_type": "WhisperFeatureExtractor",
            "feature_size": 128,
            "hop_length": 160,
            "n_fft": 400,
            "n_samples": 480000,
            "nb_max_frames": 3000,
            "padding_side": "right",
            "padding_value": 0.0,
            "processor_class": "WhisperProcessor",
            "return_attention_mask": False,
            "sampling_rate": 16000
            }

        feature_extractor = WhisperFeatureExtractor(**feature_extractor_config)

        model_path = folder_paths.get_full_path_or_raise("audio_encoders", model)
        sd = load_torch_file(model_path, device=transformer_load_device, safe_load=True)

        for name, param in whisper.named_parameters():
            key = "model." + name
            value=sd[key]
            set_module_tensor_to_device(whisper, name, device=offload_device, dtype=base_dtype, value=value)

        whisper_model = {
            "feature_extractor": feature_extractor,
            "model": whisper,
            "dtype": base_dtype,
        }

        return (whisper_model,)

# -----------------------------------------------------------------
# 节点类定义：HuMoEmbeds (更名为 PainterHuMoEmbeds) - 您的修改部分
# -----------------------------------------------------------------
class PainterHuMoEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_frames": ("INT", {"default": 81, "min": -1, "max": 10000, "step": 1, "tooltip": "The total frame count to generate."}),
                "width": ("INT", {"default": 832, "min": 64, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 480, "min": 64, "max": 4096, "step": 16}),
                "audio_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Strength of the audio conditioning"}),
                "audio_cfg_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "When not 1.0, an extra model pass without audio conditioning is done: slower inference but more motion is allowed"}),
                "audio_start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The percent of the video to start applying audio conditioning"}),
                "audio_end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The percent of the video to stop applying audio conditioning"})
            },
            "optional": {
                "whisper_model": ("WHISPERMODEL",),
                "vae": ("WANVAE", ),
                "reference_images": ("IMAGE", {"tooltip": "reference images for the humo model"}),
                "audio": ("AUDIO",),
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Use tiled VAE encoding for reduced memory use"}),
                "start_image": ("IMAGE", {"tooltip": "Optional starting image for the video (will be used as first frame)"}),
            }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS", )
    RETURN_NAMES = ("image_embeds", )
    FUNCTION = "process"
    CATEGORY = "painter-humo-i2v"

    def process(self, num_frames, width, height, audio_scale, audio_cfg_scale, audio_start_percent, audio_end_percent,
                whisper_model=None, vae=None, reference_images=None, audio=None, tiled_vae=False, start_image=None):

        # === 自动用 start_image 生成 reference_images（您修改的代码）===
        if start_image is not None and len(start_image) > 0:
            reference_images = start_image[:1].repeat(8, 1, 1, 1)

        if reference_images is not None and vae is None:
            raise ValueError("VAE is required when reference images are provided")
        if whisper_model is None and audio is not None:
            raise ValueError("Whisper model is required when audio is provided")

        # === 音频处理 ===
        model = whisper_model["model"] if whisper_model else None
        feature_extractor = whisper_model["feature_extractor"] if whisper_model else None
        dtype = whisper_model["dtype"] if whisper_model else torch.float32
        sampling_rate = 16000
        if audio is not None:
            audio_input = audio["waveform"][0]
            sample_rate = audio["sample_rate"]
            if sample_rate != sampling_rate:
                audio_input = torchaudio.functional.resample(audio_input, sample_rate, sampling_rate)
            if audio_input.shape[1] == 2:
                audio_input = audio_input.mean(dim=0, keepdim=False)
            else:
                audio_input = audio_input[0]
            model.to(device)
            audio_len = len(audio_input) // 640
            audio_features = []
            window = 750 * 640
            for i in range(0, len(audio_input), window):
                audio_feature = feature_extractor(audio_input[i:i+window], sampling_rate=sampling_rate, return_tensors="pt").input_features
                audio_features.append(audio_feature)
            audio_features = torch.cat(audio_features, dim=-1).to(device, dtype)
            window = 3000
            audio_prompts = []
            for i in range(0, audio_features.shape[-1], window):
                audio_prompt = model.encoder(audio_features[:, :, i:i+window], output_hidden_states=True).hidden_states
                audio_prompt = torch.stack(audio_prompt, dim=2)
                audio_prompts.append(audio_prompt)
            model.to(offload_device)
            audio_prompts = torch.cat(audio_prompts, dim=1)
            audio_prompts = audio_prompts[:, :audio_len * 2]
            feat0 = linear_interpolation_fps(audio_prompts[:, :, 0:8].mean(dim=2), 50, 25)
            feat1 = linear_interpolation_fps(audio_prompts[:, :, 8:16].mean(dim=2), 50, 25)
            feat2 = linear_interpolation_fps(audio_prompts[:, :, 16:24].mean(dim=2), 50, 25)
            feat3 = linear_interpolation_fps(audio_prompts[:, :, 24:32].mean(dim=2), 50, 25)
            feat4 = linear_interpolation_fps(audio_prompts[:, :, 32], 50, 25)
            audio_emb = torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]
        else:
            audio_emb = torch.zeros(num_frames, 5, 1280, device=device)
            audio_len = num_frames

        pixel_frame_num = num_frames if num_frames != -1 else audio_len
        pixel_frame_num = 4 * ((pixel_frame_num - 1) // 4) + 1
        latent_frame_num = (pixel_frame_num - 1) // 4 + 1
        log.info(f"HuMo set to generate {pixel_frame_num} frames")

        num_refs = 0
        if reference_images is not None:
            if reference_images.shape[1] != height or reference_images.shape[2] != width:
                reference_images_in = common_upscale(reference_images.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)
            else:
                reference_images_in = reference_images
            
            # 调用 WanVideoEncodeLatentBatch 的 encode 方法
            samples, = WanVideoEncodeLatentBatch.encode(self, vae, reference_images_in, tiled_vae, None, None, None, None)
            samples = samples["samples"].transpose(0, 2).squeeze(0)
            num_refs = samples.shape[1]

        vae.to(device)

        # === 创建全零视频帧（像素空间）===
        total_frames = pixel_frame_num + 4 * num_refs
        zero_frames = torch.zeros(1, 3, total_frames, height, width, device=device, dtype=vae.dtype)

        # 编码为 latent
        zero_latents = vae.encode(zero_frames, device=device, tiled=tiled_vae)[0].to(offload_device)
        vae.to(offload_device)
        mm.soft_empty_cache()

        # === 注入 start_image 的 latent（您修改的代码）===
        if start_image is not None and len(start_image) > 0:
            img = start_image[0:1]  # [1, H, W, C]
            img = img.permute(0, 3, 1, 2).unsqueeze(2)  # [1, C, 1, H, W] ← 关键：加时间维度
            if img.shape[3] != height or img.shape[4] != width:
                img = common_upscale(img, width, height, "lanczos", "disabled")
            vae.to(device)
            with torch.no_grad():
                start_latent = vae.encode(img.to(device, dtype=vae.dtype), device=device, tiled=tiled_vae)[0]  # [16, 1, h, w]
            zero_latents = zero_latents.to(device)
            zero_latents[:, 0, :, :] = start_latent.squeeze(1) 
            zero_latents = zero_latents.to(offload_device)
            vae.to(offload_device)
            mm.soft_empty_cache()

        target_shape = (16, latent_frame_num + num_refs, height // 8, width // 8)
        mask = torch.ones(4, target_shape[1], target_shape[2], target_shape[3], device=offload_device, dtype=vae.dtype)
        if reference_images is not None:
            mask[:, :-num_refs] = 0
            image_cond = torch.cat([zero_latents[:, :(target_shape[1] - num_refs)], samples], dim=1)
        else:
            image_cond = zero_latents
            mask = torch.zeros_like(mask)

        image_cond_neg = torch.cat([mask, zero_latents], dim=0)
        image_cond = torch.cat([mask, image_cond], dim=0)

        embeds = {
            "humo_audio_emb": audio_emb,
            "humo_audio_emb_neg": torch.zeros_like(audio_emb, dtype=audio_emb.dtype, device=audio_emb.device),
            "humo_image_cond": image_cond,
            "humo_image_cond_neg": image_cond_neg,
            "humo_reference_count": num_refs,
            "target_shape": target_shape,
            "num_frames": pixel_frame_num,
            "humo_audio_scale": audio_scale,
            "humo_audio_cfg_scale": audio_cfg_scale,
            "humo_start_percent": audio_start_percent,
            "humo_end_percent": audio_end_percent,
        }
        return (embeds, )


    
class WanVideoEncodeLatentBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "vae": ("WANVAE",),
                    "images": ("IMAGE",),
                    "enable_vae_tiling": ("BOOLEAN", {"default": False, "tooltip": "Drastically reduces memory use but may introduce seams"}),
                    "tile_x": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_y": ("INT", {"default": 272, "min": 64, "max": 2048, "step": 1, "tooltip": "Tile size in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_x": ("INT", {"default": 144, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    "tile_stride_y": ("INT", {"default": 128, "min": 32, "max": 2048, "step": 32, "tooltip": "Tile stride in pixels, smaller values use less VRAM, may introduce more seams"}),
                    },
                }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "encode"
    CATEGORY = "WanVideoWrapper"
    DESCRIPTION = "Encodes a batch of images individually to create a latent video batch where each video is a single frame, useful for I2V init purposes, for example as multiple context window inits"

    def encode(self, vae, images, enable_vae_tiling, tile_x, tile_y, tile_stride_x, tile_stride_y, latent_strength=1.0):
        vae.to(device)

        images = images.clone()

        B, H, W, C = images.shape
        if W % 16 != 0 or H % 16 != 0:
            new_height = (H // 16) * 16
            new_width = (W // 16) * 16
            log.warning(f"Image size {W}x{H} is not divisible by 16, resizing to {new_width}x{new_height}")
            images = common_upscale(images.movedim(-1, 1), new_width, new_height, "lanczos", "disabled").movedim(1, -1)

        if images.shape[-1] == 4:
            images = images[..., :3]
        images = images.to(vae.dtype).to(device) * 2.0 - 1.0

        latent_list = []
        for img in images:
            if enable_vae_tiling and tile_x is not None:
                latent = vae.encode(img.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3), device=device, tiled=enable_vae_tiling, tile_size=(tile_x//vae.upsampling_factor, tile_y//vae.upsampling_factor), tile_stride=(tile_stride_x//vae.upsampling_factor, tile_stride_y//vae.upsampling_factor))
            else:
                latent = vae.encode(img.unsqueeze(0).unsqueeze(0).permute(0, 4, 1, 2, 3), device=device, tiled=enable_vae_tiling)
            
            if latent_strength != 1.0:
                latent *= latent_strength
            latent_list.append(latent.squeeze(0).cpu())
        latents_out = torch.stack(latent_list, dim=0)

        log.info(f"WanVideoEncode: Encoded latents shape {latents_out.shape}")
        vae.to(offload_device)
        mm.soft_empty_cache()

        return ({"samples": latents_out},)

    
# -----------------------------------------------------------------
# 节点类定义：WanVideoCombineEmbeds (更名为 PainterCombineEmbeds)
# -----------------------------------------------------------------
class PainterCombineEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "embeds_1": ("WANVIDIMAGE_EMBEDS",),
                    "embeds_2": ("WANVIDIMAGE_EMBEDS",),
                }
        }

    RETURN_TYPES = ("WANVIDIMAGE_EMBEDS",)
    RETURN_NAMES = ("image_embeds",)
    FUNCTION = "add"
    CATEGORY = "painter-humo-i2v"
    EXPERIMENTAL = True

    def add(self, embeds_1, embeds_2):
        # Combine the two sets of embeds
        combined = {**embeds_1, **embeds_2}
        return (combined,)


# -----------------------------------------------------------------
# 节点映射更新
# -----------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "PainterWhisperModelLoader": PainterWhisperModelLoader,
    "PainterHuMoEmbeds": PainterHuMoEmbeds,
    "PainterCombineEmbeds": PainterCombineEmbeds,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PainterWhisperModelLoader": "I2V Whisper Model Loader (P)",
    "PainterHuMoEmbeds": "Painter-HuMo-i2v",
    "PainterCombineEmbeds": "WanVideo Combine Embeds (P)",
}
