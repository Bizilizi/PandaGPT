# create necessary directories
mkdir pretrained_ckpt/imagebind_ckpt -p
mkdir pretrained_ckpt/llama_ckpt/7b -p
mkdir pretrained_ckpt/llama_ckpt/13b -p
mkdir pretrained_ckpt/pandagpt_ckpt/7b -p
mkdir pretrained_ckpt/pandagpt_ckpt/13b -p
mkdir pretrained_ckpt/vicuna_ckpt/7b_v0 -p
mkdir pretrained_ckpt/vicuna_ckpt/13b_v0 -p
mkdir pretrained_ckpt/vicuna_delta_ckpt/7b -p
mkdir pretrained_ckpt/vicuna_delta_ckpt/13b -p

# download pretrained models
# llama
huggingface-cli download huggyllama/llama-7b --local-dir pretrained_ckpt/llama_ckpt/7b --local-dir-use-symlinks False
huggingface-cli download huggyllama/llama-13b --local-dir pretrained_ckpt/llama_ckpt/13b --local-dir-use-symlinks False
# vicuna
huggingface-cli download lmsys/vicuna-7b-delta-v0 --local-dir pretrained_ckpt/vicuna_delta_ckpt/7b --local-dir-use-symlinks False
huggingface-cli download lmsys/vicuna-13b-delta-v0 --local-dir pretrained_ckpt/vicuna_delta_ckpt/13b --local-dir-use-symlinks False
# pandagpt
huggingface-cli download openllmplayground/pandagpt_7b_max_len_1024 --local-dir pretrained_ckpt/pandagpt_ckpt/7b --local-dir-use-symlinks False
huggingface-cli download openllmplayground/pandagpt_13b_max_len_256 --local-dir pretrained_ckpt/pandagpt_ckpt/13b --local-dir-use-symlinks False

# Check if mamba command exists and install squashfs
if command -v mamba &> /dev/null; then
    echo "Installing squashfs using mamba..."
    mamba install conda-forge::squashfuse
else
    echo "Mamba not found, installing squashfs using conda..."
    conda install conda-forge::squashfuse
fi
