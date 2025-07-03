================
QAT 4W8F support
================

QAT (Quantization-Aware Training) refers to simulating the quantization process during model training to adapt the model to low-precision calculations to reduce the accuracy loss after quantization. QAT usually inserts fake quantization operations in forward propagation to simulate INT8 quantization, but still uses FP32 to calculate gradients during back propagation.

In quantization-aware training, we can use a mixed precision quantization strategy of 4-bit weight quantization (Weights) + 8-bit activation quantization (Feature maps/Activations), which aims to balance model compression rate and inference accuracy. This configuration means that in the quantization-aware training process:

- Weights will be quantized to 4 bits (INT4) to reduce model size and computation.
- Activations remain at 8 bits (INT8) to balance computational efficiency and accuracy

Considering that PyTorch does not support the official INT4 weight format, users still use int8 to represent ONNX models trained and exported from QAT, but the bit width is limited to the INT4 range. First, you need to execute the following command to convert the ONNX model into a real INT4 ONNX model:

.. code:: bash

   onnxslim model_qat_4w8f.onnx model_qat_4w8f_slim.onnx
   convert_to_4w8f_cli --input model_qat_4w8f_slim.onnx --output model_qat_4w8f_opt.onnx

Here, we assume that the QDQ ONNX model name obtained by QAT is "model_qat_4w8f.onnx", and the converted model "model_qat_4w8f_opt.onnx" is a model with Weights marked as INT4 that meets ONNX semantics.zhang'chen'xuan