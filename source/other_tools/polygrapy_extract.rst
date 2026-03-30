===========================
Polygraphy tool overview
===========================

Polygraphy is a Python toolkit for deep learning model inference and validation, maintained by NVIDIA developers. It is mainly used for model conversion, performance analysis, and correctness verification to help developers optimize and debug models. The ``surgeon extract`` feature can be used to conveniently split a QuantAxmodel to locate the minimal subgraph that causes issues.

This tool can be flexibly applied to models expressed in the ONNX container format, such as ONNX, QuantAxModel, and OptimizedAxModel.

Common parameters
-----------------

.. code-block:: text

   --inputs INPUT_META [INPUT_META ...]
       Input metadata of the subgraph (name, shape, dtype).
       Use auto to let extract infer these values automatically.
       Format:
       --inputs <name>:<shape>:<dtype>
       example:
       --inputs input0:[1,3,224,224]:float32 input1:auto:auto
       If omitted, the current model input configuration is used.

   --outputs OUTPUT_META [OUTPUT_META ...]
       Output metadata of the subgraph (name and dtype).
       Use auto to let extract infer these values automatically.
       Format:
       --outputs <name>:<dtype>
       example:
       --outputs output0:float32 output1:auto
       If omitted, the current model output configuration is used.

   -o SAVE_ONNX, --output SAVE_ONNX
       The output ONNX path of the extracted subgraph.

Quick start
-----------

.. code-block:: bash

   polygraphy surgeon extract output/quant/quant_axmodel.onnx        --inputs your_inputs1:auto:auto your_inputs2:auto:auto        --outputs your_outputs1:auto your_outputs2:auto        -o output_quant_axmodel.onnx
