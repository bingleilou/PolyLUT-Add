# PolyLUT-Add
PolyLUT-Add is a technique that enhances neuron connectivity by combining multiple base PolyLUT models to improve accuracy.
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
If you think this work is useful to your project, please cite the paper: https://arxiv.org/abs/2406.04910
```bibtex
@misc{lou2024polylutadd,
      title={PolyLUT-Add: FPGA-based LUT Inference with Wide Inputs}, 
      author={Binglei Lou and Richard Rademacher and David Boland and Philip H. W. Leong},
      year={2024},
      eprint={2406.04910},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

The updated post-Place&Route results for Table III in this paper are as follows.

When citing this data, we recommend using these post-Place&Route results for a more consistent comparison with other methods (e.g., PolyLUT and LogicNets).

| Dataset                                 | Model                        | Acc.  | LUT | FF     | DSP | BRAM | F_max(MHz) | Latency(ns) | 
|-----------------------------------------|------------------------------|--|--|--|--|--|---|--|
| **MNIST** (post systhesis)              | PolyLUT-Add (HDR-Add2, $D$=3)        | 96%  | 15272 | 2880 | 0 | 0  | 833 | 7  |
| **MNIST** (post Place&Route)            | PolyLUT-Add (HDR-Add2, $D$=3)        | 96%  | 14810 | 2609 | 0 | 0  | 625 | 10 |
| **JSC-HC** (post systhesis)             | PolyLUT-Add (JSC-XL-Add2, $D$=3)     | 75%  | 47639 | 1712 | 0 | 0  | 400 | 13 |
| **JSC-HC** (post Place&Route)           | PolyLUT-Add (JSC-XL-Add2, $D$=3)     | 75%  | 36484 | 1209 | 0 | 0  | 315 | 16 |
| **JSC-LC** (post systhesis)             | PolyLUT-Add (JSC-M Lite-Add2, $D$=3) | 72%  | 1618  | 336  | 0 | 0  | 800 | 4  |
| **JSC-LC** (post Place&Route)           | PolyLUT-Add (JSC-M Lite-Add2, $D$=3) | 72%  | 895   | 189  | 0 | 0  | 750 | 4  |
| **UNSW-NB15** (post systhesis)          | PolyLUT-Add (NID-Add2, $D$=1)        | 92%  | 2591  | 1193 | 0 | 0  | 620 | 8  |
| **UNSW-NB15** (post Place&Route)        | PolyLUT-Add (NID-Add2, $D$=1)        | 92%  | 1649  | 830  | 0 | 0  | 620 | 8  |



