import os, json, requests, runpod

import torch
import base64
from io import BytesIO
from PIL import Image
from fastapi import FastAPI

Image.MAX_IMAGE_PIXELS = None

from modules import initialize_util
from modules import initialize
initialize.imports()
initialize.check_versions()
initialize.initialize()
app = FastAPI()
initialize_util.setup_middleware(app)

from modules.api.api import Api
from modules.call_queue import queue_lock
from modules.api.models import StableDiffusionImg2ImgProcessingAPI
api = Api(app, queue_lock)

def download_file(url, save_dir='/content'):
    os.makedirs(save_dir, exist_ok=True)
    file_name = url.split('/')[-1]
    file_path = os.path.join(save_dir, file_name)
    response = requests.get(url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)
    return file_path

@torch.inference_mode()
def generate(input):
    values = input["input"]

    img2img_input_image_url = values['input_image_check']
    img2img_input_image_path = download_file(img2img_input_image_url)
    img2img_prompt = values['img2img_prompt'] 
    img2img_negative_prompt = values['img2img_negative_prompt']
    img2img_steps = values['img2img_steps']
    img2img_cfg_scale = values['img2img_cfg_scale']
    img2img_seed = values['img2img_seed']
    img2img_sampler_name = values['img2img_sampler_name']
    img2img_denoising_strength = values['img2img_denoising_strength']
    img2img_scale_by = values['img2img_scale_by']
    tiled_diffusion_tile_width = values['tiled_diffusion_tile_width']
    tiled_diffusion_tile_height = values['tiled_diffusion_tile_height']
    tiled_diffusion_overlap = values['tiled_diffusion_overlap']
    tiled_diffusion_tile_batch_size = values['tiled_diffusion_tile_batch_size']
    tiled_diffusion_scale_factor = values['tiled_diffusion_scale_factor']
    tiled_vae_encoder_tile_size = values['tiled_vae_encoder_tile_size']
    tiled_vae_decoder_tile_size = values['tiled_vae_decoder_tile_size']
    controlnet_tile_weight = values['controlnet_tile_weight']
    with open(img2img_input_image_path, "rb") as img_file:
        img_data = img_file.read()
        base64_image = base64.b64encode(img_data).decode("utf-8")
    payload = {
        "override_settings": {
            "sd_model_checkpoint": "juggernaut_reborn.safetensors",
            "sd_vae": "vae-ft-mse-840000-ema-pruned.safetensors",
            "CLIP_stop_at_last_layers": 1,
        },
        "override_settings_restore_afterwards": False,
        "init_images": [base64_image],
        "prompt": img2img_prompt,
        "negative_prompt": img2img_negative_prompt,
        "steps": img2img_steps,
        "cfg_scale": img2img_cfg_scale,
        "seed": img2img_seed,
        "do_not_save_samples": True,
        "sampler_name": img2img_sampler_name,
        "denoising_strength": img2img_denoising_strength,
        "scale_by": img2img_scale_by,
        "alwayson_scripts": {
            "Tiled Diffusion": {
                "args": [
                    True,
                    "MultiDiffusion",
                    True,
                    True,
                    1,
                    1,
                    tiled_diffusion_tile_width,
                    tiled_diffusion_tile_height,
                    tiled_diffusion_overlap,
                    tiled_diffusion_tile_batch_size,
                    "4x-UltraSharp",
                    tiled_diffusion_scale_factor, 
                    False,
                ]
            },
            "Tiled VAE": {
                "args": [
                    True,
                    tiled_vae_encoder_tile_size,
                    tiled_vae_decoder_tile_size,
                    True,
                    True,
                    True,
                    True,
                ]

            },
            "controlnet": {
                "args": [
                    {
                        "enabled": True,
                        "module": "tile_resample",
                        "model": "control_v11f1e_sd15_tile",
                        "weight": controlnet_tile_weight,
                        "image": base64_image,
                        "lowvram": False,
                        "downsample": 1.0,
                        "guidance_start": 0.0,
                        "guidance_end": 1.0,
                        "pixel_perfect": True,
                        "threshold_a": 1,
                        "threshold_b": 1,
                        "save_detected_map": False,
                        "processor_res": 512,
                    }
                ]
            }
        }
    }
    req = StableDiffusionImg2ImgProcessingAPI(**payload)
    resp = api.img2imgapi(req)
    # info = json.loads(resp.info)
    base64_image = resp.images[0]
    gen_bytes = BytesIO(base64.b64decode(base64_image))
    imageObject = Image.open(gen_bytes)
    imageObject.save("/content/output_image.png")

    result = "/content/output_image.png"
    try:
        notify_uri = values['notify_uri']
        del values['notify_uri']
        notify_token = values['notify_token']
        del values['notify_token']
        discord_id = values['discord_id']
        del values['discord_id']
        if(discord_id == "discord_id"):
            discord_id = os.getenv('com_camenduru_discord_id')
        discord_channel = values['discord_channel']
        del values['discord_channel']
        if(discord_channel == "discord_channel"):
            discord_channel = os.getenv('com_camenduru_discord_channel')
        discord_token = values['discord_token']
        del values['discord_token']
        if(discord_token == "discord_token"):
            discord_token = os.getenv('com_camenduru_discord_token')
        job_id = values['job_id']
        del values['job_id']
        default_filename = os.path.basename(result)
        with open(result, "rb") as file:
            files = {default_filename: file.read()}
        payload = {"content": f"{json.dumps(values)} <@{discord_id}>"}
        response = requests.post(
            f"https://discord.com/api/v9/channels/{discord_channel}/messages",
            data=payload,
            headers={"Authorization": f"Bot {discord_token}"},
            files=files
        )
        response.raise_for_status()
        result_url = response.json()['attachments'][0]['url']
        notify_payload = {"jobId": job_id, "result": result_url, "status": "DONE"}
        web_notify_uri = os.getenv('com_camenduru_web_notify_uri')
        web_notify_token = os.getenv('com_camenduru_web_notify_token')
        if(notify_uri == "notify_uri"):
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
        else:
            requests.post(web_notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            requests.post(notify_uri, data=json.dumps(notify_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        return {"jobId": job_id, "result": result_url, "status": "DONE"}
    except Exception as e:
        error_payload = {"jobId": job_id, "status": "FAILED"}
        try:
            if(notify_uri == "notify_uri"):
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
            else:
                requests.post(web_notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": web_notify_token})
                requests.post(notify_uri, data=json.dumps(error_payload), headers={'Content-Type': 'application/json', "Authorization": notify_token})
        except:
            pass
        return {"jobId": job_id, "result": f"FAILED: {str(e)}", "status": "FAILED"}
    finally:
        if os.path.exists(result):
            os.remove(result)

runpod.serverless.start({"handler": generate})