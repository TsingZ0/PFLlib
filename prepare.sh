# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

# Configure ~/.bashrc
echo alias p=\"ps -aux|grep zhangjq|grep 'python -u'\" >> ~/.bashrc
echo alias n=\'nvidia-smi\' >> ~/.bashrc
echo alias d=\'du -hs * | sort -h\' >> ~/.bashrc
echo alias del_pycache=\'find . -type d -name __pycache__ -prune -exec rm -rf {} \;\' >> ~/.bashrc

echo export PIP_CACHE_DIR='$PWD'/tmp >> ~/.bashrc
echo # export TMPDIR='$PWD'/tmp >> ~/.bashrc

# Install python packages
source ~/.bashrc
conda env create -f env_cuda_latest.yaml
conda activate pfllib