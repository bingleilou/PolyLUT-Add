## PolyLUT-Add on the network intrusion dataset (UNSW-NB15)

To reproduce the results in our paper follow the steps below. Subsequently, compile the Verilog files using the following settings (utilize Vivado 2020.1, target the xcvu9p-flgb2104-2-i FPGA part, use the Vivado Flow_PerfOptimized_high settings, and perform synthesis in the Out-of-Context (OOC) mode).


```
python train.py --arch nid-add2  --log_dir _add2
python neq2lut.py --arch nid-add2 --checkpoint ./test_fan_7_add2/best_accuracy.pth --log-dir ./test_fan_7_add2/verilog/ --add-registers --seed 1699 --device 0
```


During the Vivado synthesis step, we offer two pipeline methods as examples located in ./pipeline-example for reproducibility purposes in the FPL24 submission. One can manually adjust them according to the provided examples temporarily.

Note: The "Summary Metrics" results in wandb are sometimes incorrect, one could refresh the webpage or select a specfic chart (e.g. test chart) and then "add to report" -> "add panel" -> "Run Comparer" for correct values.


