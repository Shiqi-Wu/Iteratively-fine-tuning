conda deactivate
conda remove --name fine-tuning --all

# Create a new conda environment named "myenv"
conda create --name fine-tuning python=3.8 -y

# Activate the environment
conda activate fine-tuning

# Install PyTorch 1.6.0 with CUDA 10.1 support
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install timm 0.4.8
pip install timm==0.4.9

# Install einops, easydict, easydict, tensorboard, six, packaging
pip install einops easydict tensorboard six packaging