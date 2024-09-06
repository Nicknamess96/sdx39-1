import torch
from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline, AutoencoderTiny
from pipelines.models import TextToImageRequest
from torch import Generator
import os
from huggingface_hub import hf_hub_download
import xformers
import triton
from sfast.compilers.diffusion_pipeline_compiler import (compile, CompilationConfig)

def load_pipeline() -> StableDiffusionXLPipeline:
    model_id = "AIArchitect23/edge_1"
    revision = "361cef68a38d1dc54e5b32d85898afce7f1cb956"
    cache_dir = "./models"

    # List of essential files for the model
    essential_files = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/pytorch_model.bin",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.bin",
        "vae/config.json",
        "vae/diffusion_pytorch_model.bin",
    ]

    model_path = os.path.join(cache_dir, "models--stablediffusionapi--newdream-sdxl-20", "snapshots", revision)

    # Check if all essential files exist
    missing_files = [file for file in essential_files if not os.path.exists(os.path.join(model_path, file))]

    if missing_files:
        print(f"Some model files are missing. Attempting to download them...")
        for file in missing_files:
            try:
                hf_hub_download(repo_id=model_id, filename=file, revision=revision, cache_dir=cache_dir)
                print(f"Downloaded: {file}")
            except Exception as e:
                print(f"Error downloading {file}: {str(e)}")
                raise

    print(f"Loading model from: {model_path}")

    vae = AutoencoderTiny.from_pretrained(
      'madebyollin/taesdxl',
      use_safetensors=True,
      torch_dtype=torch.float16,
    ).to('cuda')

    pipeline = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        local_files_only=True,
        use_safetensors=True,
        torch_dtype=torch.float16,
        variant='fp16',
        vae=vae
    )

    config = CompilationConfig.Default()

    config.enable_xformers = True
    config.enable_triton = True
    config.enable_cuda_graph = True

    pipeline = compile(pipeline, config)

    pipeline = pipeline.to("cuda")
    pipeline(prompt="")  # Warm-up run

    return pipeline

def infer(request: TextToImageRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    generator = Generator(pipeline.device).manual_seed(request.seed) if request.seed else None

    return pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        generator=generator,
    ).images[0]