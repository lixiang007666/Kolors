COMPILER = "oneflow"
COMPILER_CONFIG = None
QUANTIZE_CONFIG = None

import os, torch
import argparse
# from PIL import Image
from kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from kolors.models.modeling_chatglm import ChatGLMModel
from kolors.models.tokenization_chatglm import ChatGLMTokenizer
from diffusers import UNet2DConditionModel, AutoencoderKL
from diffusers import EulerDiscreteScheduler

from onediffx import compile_pipe, quantize_pipe # quantize_pipe currently only supports the nexfort backend.
from onediff.infer_compiler import oneflow_compile

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str, help="Text prompt for image generation")
    parser.add_argument(
        "--compiler",
        type=str,
        default=COMPILER,
        choices=["none", "oneflow", "nexfort", "compile", "compile-max-autotune"],
    )
    parser.add_argument(
        "--compiler-config",
        type=str,
        default=COMPILER_CONFIG,
    )
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument(
        "--quantize-config",
        type=str,
        default=QUANTIZE_CONFIG,
    )
    parser.add_argument("--quant-submodules-config-path", type=str, default=None)
    return parser.parse_args()

def infer(args):
    ckpt_dir = f'{root_dir}/weights/Kolors'
    text_encoder = ChatGLMModel.from_pretrained(
        f'{ckpt_dir}/text_encoder',
        torch_dtype=torch.float16).half()
    tokenizer = ChatGLMTokenizer.from_pretrained(f'{ckpt_dir}/text_encoder')
    vae = AutoencoderKL.from_pretrained(f"{ckpt_dir}/vae", revision=None).half()
    scheduler = EulerDiscreteScheduler.from_pretrained(f"{ckpt_dir}/scheduler")
    unet = UNet2DConditionModel.from_pretrained(f"{ckpt_dir}/unet", revision=None).half()
    pipe = StableDiffusionXLPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            force_zeros_for_empty_prompt=False)
    pipe = pipe.to("cuda")

    if args.compiler == "none":
        pass
    elif args.compiler == "oneflow":
        print("Oneflow backend is now active...")
        pipe.unet = oneflow_compile(pipe.unet)
        # pipe.vae.decoder = oneflow_compile(pipe.vae.decoder)
    elif args.compiler == "nexfort":
        print("Nexfort backend is now active...")
        if args.quantize:
            if args.quantize_config is not None:
                quantize_config = json.loads(args.quantize_config)
            else:
                quantize_config = '{"quant_type": "fp8_e4m3_e4m3_dynamic"}'
            if args.quant_submodules_config_path:
                pipe = quantize_pipe(
                    pipe,
                    quant_submodules_config_path=args.quant_submodules_config_path,
                    ignores=[],
                    **quantize_config,
                )
            else:
                pipe = quantize_pipe(pipe, ignores=[], **quantize_config)
        if args.compiler_config is not None:
            # config with dict
            options = json.loads(args.compiler_config)
        else:
            # config with string
            options = '{"mode": "max-optimize:max-autotune:low-precision", "memory_format": "channels_last"}'
        pipe = compile_pipe(
            pipe, backend="nexfort", options=options, fuse_qkv_projections=True
        )

    # pipe.enable_model_cpu_offload()
    image = pipe(
        prompt=args.prompt,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=5.0,
        num_images_per_prompt=1,
        generator= torch.Generator(pipe.device).manual_seed(66)).images[0]
    image.save(f'{root_dir}/scripts/outputs/sample_test.jpg')


if __name__ == '__main__':
    args = parse_args()
    infer(args)
