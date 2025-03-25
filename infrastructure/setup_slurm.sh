wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod 755 Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
curl -LO https://github.com/abiosoft/colima/releases/latest/download/colima-$(uname)-$(uname -m)
mkdir bin/
mv colima-$(uname)-$(uname -m) bin/colima
chmod 755 bin/colima
git clone https://github.com/maxlautenbach/CIExMAS.git
cd CIExMAS/
conda create --name CIExMAS python=3.11
conda activate CIExMAS
pip install vllm
pip install -r requirements.txt