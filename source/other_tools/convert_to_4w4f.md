# QAT 4W4F support

QAT (Quantization-Aware Training) refers to simulating the quantization process during model training to adapt the model to low-precision calculations and reduce accuracy loss after quantization. QAT usually inserts fake quantization operations in forward propagation to simulate low-bit quantization, but still uses FP32 to calculate gradients during back propagation.

For 4-bit quantization, please refer to the [resnet50/config_4w4f](https://github.com/AXERA-TECH/QAT.axera/blob/cc4c50293317e21dc1b7f52854d992df48d4ffd8/resnet50/config_4w4f.json) configuration, and use [simplify_and_fix_4bit_dtype](https://github.com/AXERA-TECH/QAT.axera/blob/cc4c50293317e21dc1b7f52854d992df48d4ffd8/utils/quant_utils.py#L12) to replace onnxsim/onnxslim.
