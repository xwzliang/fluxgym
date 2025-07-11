#!/bin/sh
# git clone https://github.com/cocktailpeanut/fluxgym.git /app/fluxgym; 
# git clone -b sd3 https://github.com/kohya-ss/sd-scripts /app/fluxgym/sd-scripts; 

# cd /app/fluxgym/sd-scripts && 
# pip install -r requirements.txt && 
# echo "fluxgym requirements installed successfully."; 

# cd /app/fluxgym && 
# pip install -r requirements.txt && 
# echo "fluxgym requirements installed successfully."; 

# pip install --pre torch==2.4 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && 
# echo "PyTorch installed successfully.";

# echo "Downloading models is ${DOWNLOAD_MODELS}";

if [ "$DOWNLOAD_MODELS" = "true" ]; then
    echo "Downloading models..." && 
    wget -O /app/fluxgym/models/unet/flux1-dev.sft https://huggingface.co/cocktailpeanut/xulf-dev/resolve/main/flux1-dev.sft?download=true && 
    wget -O /app/fluxgym/models/unet/flux1-schnell.safetensors https://huggingface.co/cocktailpeanut/xulf-schnell/resolve/main/flux1-schnell.safetensors?download=true && 
    wget -O /app/fluxgym/models/clip/clip_l.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true && 
    wget -O /app/fluxgym/models/clip/t5xxl_fp16.safetensors https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors?download=true && 
    wget -O /app/fluxgym/models/vae/ae.sft https://huggingface.co/cocktailpeanut/xulf-dev/resolve/main/ae.sft?download=true && 
    echo "Models downloaded successfully."; 
else
    echo "Skipping model download.";
fi 

export GRADIO_SERVER_NAME="0.0.0.0"

cd /app/fluxgym

python /app/fluxgym/app.py
