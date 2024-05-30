# PolyLUT-Add
PolyLUT-Add is a technique that enhances neuron connectivity by combining multiple base PolyLUT models to improve accuracy. Moreover, we describe a novel architecture to improve its scalability.
This project is a derivative work based on PolyLUT (https://github.com/MartaAndronic/PolyLUT) which is licensed under the Apache License 2.0.

We provide this code for reproducibility purposes; the toolflow is inherited from PolyLUT work.

## LUT implementation example
```verilog
module layer0_N0_E0 ( input [2:0] M0, output [1:0] M1 );
    (*rom_style = "distributed" *) reg [1:0] M1r;
        assign M1 = M1r;
        always @ (M0) begin
            case (M0)
                3'b000: M1r = 2'b10;
                3'b001: M1r = 2'b11;
                3'b010: M1r = 2'b10;
                3'b011: M1r = 2'b11;
                3'b100: M1r = 2'b00;
                3'b101: M1r = 2'b01;
                3'b110: M1r = 2'b11;
                3'b111: M1r = 2'b11;
            endcase
        end
endmodule
```

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
This work is accepted by the International Conference on Field-Programmable Logic and Applications (FPL) 2024.
If you think this work is useful to your project, please cite the paper 
```bibtex
@article{lou2024polylutadd,
  author       = {Binglei Lou, Richard Rademacher, David Boland and Philip HW Leong},
  title        = {PolyLUT-Add: FPGA-based LUT Inference with Wide Inputs},
  conference   = {International Conference on Field-Programmable Logic and Applications (FPL)},
  year         = {2024}
}
```
