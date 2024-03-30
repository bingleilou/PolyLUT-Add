## PolyLUT-Add on the jet substructure tagging dataset

To reproduce the results in our paper follow the steps below. Subsequently, compile the Verilog files using the following settings (utilize Vivado 2020.1, target the xcvu9p-flgb2104-2-i FPGA part, use the Vivado Flow_PerfOptimized_high settings, and perform synthesis in the Out-of-Context (OOC) mode).

## Download dataset
Navigate to the jet_substructure directory.
```
mkdir -p data
wget https://cernbox.cern.ch/index.php/s/jvFd5MoWhGs1l5v/download -O data/processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_truth.z
```

```
python train.py --arch jsc-xl-add2 --log_dir _xl_add2
python neq2lut.py --arch jsc-xl-add2 --checkpoint ./test_fan_2_xl_add2/best_accuracy.pth --log-dir ./test_fan_2_xl_add2/verilog/ --add-registers --seed 1234 --device 0

python train.py --arch jsc-m-lite-add2 --log_dir _mlite_add2
python neq2lut.py --arch jsc-m-lite-add2 --checkpoint ./test_fan_2_mlite_add2/best_accuracy.pth --log-dir ./test_fan_2_mlite_add2/verilog/ --add-registers --seed 1697 --device 0
```


During the Vivado synthesis step, we offer two pipeline methods as examples located in ./pipeline-example for reproducibility purposes in the FPL24 submission. One can manually adjust them according to the provided examples temporarily.

Note: The "Summary Metrics" results in wandb are sometimes incorrect, one could select a specfic chart (e.g. test chart) and then "add to report" -> "add panel" -> "Run Comparer" for correct values.
