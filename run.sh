# init
sudo apt update

# conda
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
conda create -n LLM_OMP 
conda activate LLM_OMP

# Pytorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
export PATH="$PATH:/home/cc/.local/bin"
pip install transformers

# huggingface
huggingface-cli login
pip install 'accelerate>=0.26.0'

# github
git config --global user.email "hhh"
git config --global user.name "hhh2"

# intel oneapi
# continuous with anoymous
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?packages=oneapi-toolkit&oneapi-toolkit-os=linux&oneapi-lin=online

# source /home/cc/intel/oneapi/setvars.sh
vi ~/.bashrc 

# Vtune
# vtune -collect threading -r my_result -- ./my_app
sudo sysctl -w kernel.yama.ptrace_scope=0

# ssh -i C:\Users\11062\.ssh\Par2.pem -L 6005:localhost:6006 cc@129.114.108.223

https://github.com/settings/keys
