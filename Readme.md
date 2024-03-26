# Quantization Workbench


## Conda Installation Instructions

```bash
conda create --prefix "C:\\Users\\rs659\\Desktop\\quantization-workbench\\wincondaprojenv" python=3.9
conda activate "C:\\Users\\rs659\\Desktop\\quantization-workbench\\wincondaprojenv"
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.37.0 bitsandbytes==0.43.0 accelerate==0.27.2 ipywidgets
onda install cudatoolkit
conda install cudnn

!git clone https://github.com/casper-hansen/AutoAWQ_kernels 
cd AutoAWQ_kernels 
!pip install -e . 
```

If you are unable to clone the repo on Windows due to Filename too long error, run the following:
```git config --global core.longpaths true```




## MacOS Installation Instructions
```bash
conda create --prefix "/Users/rishabhshah/Desktop/quantization-workbench/wincondaprojenv" python=3.9
conda activate "/Users/rishabhshah/Desktop/quantization-workbench/wincondaprojenv"
pip3 install torch torchvision torchaudio   
pip install transformers==4.37.0 bitsandbytes==0.43.0 accelerate==0.27.2 ipywidgets
```


## WSL Installation Instructions
```bash
cd /mnt/c/Users/rs659/Desktop/quantization-workbench
export PATH=~/anaconda3/bin:$PATH
conda create --prefix=/mnt/c/Users/rs659/Desktop/quantization-workbench/wslenv python=3.9
conda activate /mnt/c/Users/rs659/Desktop/quantization-workbench/wslenv
conda install -p /mnt/c/Users/rs659/Desktop/quantization-workbench/wslenv ipykernel --update-deps --force-reinstall
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/casper-hansen/AutoAWQ.git
pip install vllm
pip install -U ipywidgets
```




