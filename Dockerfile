FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod xformers==0.0.25 && \
    git clone -b v3.0 https://github.com/camenduru/stable-diffusion-webui /content/stable-diffusion-webui && \
    git clone -b v3.0 https://github.com/camenduru/sd-webui-controlnet /content/stable-diffusion-webui/extensions/sd-webui-controlnet && \
    git clone -b v3.0 https://github.com/camenduru/multidiffusion-upscaler-for-automatic1111 /content/stable-diffusion-webui/extensions/multidiffusion-upscaler-for-automatic1111 && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/clarity-upscaler/resolve/main/embeddings/JuggernautNegative-neg.pt -d /content/stable-diffusion-webui/embeddings -o JuggernautNegative-neg.pt && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/clarity-upscaler/resolve/main/embeddings/verybadimagenegative_v1.3.pt -d /content/stable-diffusion-webui/embeddings -o verybadimagenegative_v1.3.pt && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/clarity-upscaler/resolve/main/models/ControlNet/control_v11f1e_sd15_tile.pth -d /content/stable-diffusion-webui/models/ControlNet -o control_v11f1e_sd15_tile.pth && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/clarity-upscaler/resolve/main/models/ESRGAN/4x-UltraSharp.pth -d /content/stable-diffusion-webui/models/ESRGAN -o 4x-UltraSharp.pth && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/clarity-upscaler/resolve/main/models/Lora/SDXLrender_v2.0.safetensors -d /content/stable-diffusion-webui/models/Lora -o SDXLrender_v2.0.safetensors && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/clarity-upscaler/resolve/main/models/Lora/more_details.safetensors -d /content/stable-diffusion-webui/models/Lora -o more_details.safetensors && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/clarity-upscaler/resolve/main/models/Stable-diffusion/epicrealism_naturalSinRC1VAE.safetensors -d /content/stable-diffusion-webui/models/Stable-diffusion -o epicrealism_naturalSinRC1VAE.safetensors && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/clarity-upscaler/resolve/main/models/Stable-diffusion/flat2DAnimerge_v45Sharp.safetensors -d /content/stable-diffusion-webui/models/Stable-diffusion -o flat2DAnimerge_v45Sharp.safetensors && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/clarity-upscaler/resolve/main/models/Stable-diffusion/juggernaut_reborn.safetensors -d /content/stable-diffusion-webui/models/Stable-diffusion -o juggernaut_reborn.safetensors && \
	aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/ckpt/clarity-upscaler/resolve/main/models/VAE/vae-ft-mse-840000-ema-pruned.safetensors -d /content/stable-diffusion-webui/models/VAE -o vae-ft-mse-840000-ema-pruned.safetensors

COPY ./worker_runpod.py /content/stable-diffusion-webui/worker_runpod.py
WORKDIR /content/stable-diffusion-webui
RUN python -c "import sys; sys.argv.append('--skip-torch-cuda-test'); sys.argv.append('--xformers'); from modules import launch_utils; launch_utils.prepare_environment();"
CMD python worker_runpod.py