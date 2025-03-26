wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 755 Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
git clone https://github.com/maxlautenbach/CIExMAS.git
cd CIExMAS/
conda create --name CIExMAS python=3.11
conda activate CIExMAS
pip install vllm
pip install -r requirements.txt
mkdir -p /ceph/$(whoami)/CIExMAS/models
mkdir -p /ceph/$(whoami)/CIExMAS/datasets