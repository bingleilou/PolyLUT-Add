# PolyLUT-Add
PolyLUT-Add is a technique that enhances neuron connectivity by combining E base PolyLUT models to improve accuracy. Moreover, we describe a novel architecture to improve its scalability.
This project is a derivative work based on PolyLUT (https://github.com/MartaAndronic/PolyLUT) which is licensed under the Apache License 2.0.

We provide this code for reproducibility purposes in the FPL24 submission, the toolflow is inherited form PolyLUT work. (Examples of Table.III are provided in this project. We'll open source whole project later.)
## Setup
**Install Vivado Design Suite**
Vivado 2020.1

**Create a Conda environment**
```
conda create --name plutadd python=3.8
conda activate plutadd
pip install torch==1.4.0
pip install torchvision==0.5.0
pip install numpy==1.19.0
pip install scikit-learn==1.3.2
pip install tqdm
pip install h5py
pip install pandas
```

## Install Brevitas
```
conda install -y packaging pyparsing
conda install -y docrep -c conda-forge
pip install --no-cache-dir git+https://github.com/Xilinx/brevitas.git@67be9b58c1c63d3923cac430ade2552d0db67ba5
```

## Install PolyLUT package
```
cd PolyLUT-Add
pip install .
python setup.py install
```
## Install wandb + login
```
pip install pyverilator
pip install wandb
wandb.login()
```

## Citation
This work is submitted to the International Conference on Field-Programmable Logic and Applications (FPL) 2024 on 29 March 2024.
