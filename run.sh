
sudo apt update

wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh

conda create -n LLM_OMP 
conda activate LLM_OMP
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
export PATH="$PATH:/home/cc/.local/bin"
pip install transformers


huggingface-cli login
pip install 'accelerate>=0.26.0'

git config --global user.email "hhh"
git config --global user.name "hhh2"