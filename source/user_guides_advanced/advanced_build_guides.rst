===================================
Advanced Guide to Model Conversion
===================================

-----------------------
Overview
-----------------------

``pulsar2 build`` is used for model **graph optimization**, **quantification**, **compilation** and other operations. Its operation diagram is as follows:

.. figure:: ../media/pulsar2-build-pipeline.png
     :alt: pipeline
     :align: center

* ``pulsar2 build`` uses the input model (``model.onnx``), PTQ calibration data (``calibration.tar``) and configuration file (``config.json``) to generate the output model (``axmodel``).
* The command line parameters of ``pulsar2 build`` will overwrite certain corresponding parts in the configuration file, and cause ``pulsar2 build`` to output the overwritten configuration file. For a detailed introduction to the configuration file, see :ref:`《Configuration file details》 <config_details>`.

------------------------------------------
Detailed explanation of model compilation
------------------------------------------

This section introduces the complete use of the ``pulsar2 build`` command.

``pulsar2 build -h`` can display detailed command line parameters: 

.. code-block:: shell
  :name: pulsar_build_help
  :linenos:

    usage: main.py build [-h] [--config] [--input] [--output_dir] [--output_name]
                         [--work_dir] [--model_type] [--target_hardware]
                         [--npu_mode] [--input_shapes]
                         [--onnx_opt.disable_onnx_optimization]
                         [--onnx_opt.enable_onnxsim] [--onnx_opt.model_check]
                         [--onnx_opt.disable_transformation_check]
                         [--quant.calibration_method] [--quant.precision_analysis]
                         [--quant.precision_analysis_method]
                         [--quant.precision_analysis_mode]
                         [--quant.highest_mix_precision]
                         [--quant.conv_bias_data_type]
                         [--quant.refine_weight_threshold]
                         [--quant.enable_smooth_quant]
                         [--quant.transformer_opt_level]
                         [--quant.input_sample_dir] [--quant.ln_scale_data_type]
                         [--compiler.static_batch_sizes [...]]
                         [--compiler.max_dynamic_batch_size]
                         [--compiler.disable_ir_fix] [--compiler.check]
                         [--compiler.debug] [--compiler.input_sample_dir]
    
    optional arguments:
      -h, --help            show this help message and exit
      --config              config file path, supported formats: json / yaml /
                            toml / prototxt. type: string. required: false.
                            default:.
      --input               input model file path. type: string. required: true.
      --output_dir          axmodel output directory. type: string. required:
                            true.
      --output_name         rename output axmodel. type: string. required: false.
                            default: compiled.axmodel.
      --work_dir            temporary data output directory. type: string.
                            required: false. default: same with ${output_dir}.
      --model_type          input model type. type: enum. required: false.
                            default: ONNX. option: ONNX, QuantAxModel, QuantONNX.
      --target_hardware     target hardware. type: enum. required: false. default:
                            AX650. option: AX650, AX620E, M76H.
      --npu_mode            npu mode. while ${target_hardware} is AX650, npu mode
                            can be NPU1 / NPU2 / NPU3. while ${target_hardware} is
                            AX620E, npu mode can be NPU1 / NPU2. type: enum.
                            required: false. default: NPU1.
      --input_shapes        modify model input shape, this feature will take
                            effect before the `input_processors` configuration.
                            format: input1:1x3x224x224;input2:1x1x112x112. type:
                            string. required: false. default: .
      --onnx_opt.disable_onnx_optimization 
                            disable onnx optimization. type: bool. required:
                            false. default: false.
      --onnx_opt.enable_onnxsim 
                            enable onnx simplify by
                            https://github.com/daquexian/onnx-simplifier. type:
                            bool. required: false. default: false.
      --onnx_opt.model_check 
                            enable model check. type: bool. required: false.
                            default: false.
      --onnx_opt.disable_transformation_check 
                            disable transformation check. type: bool. required:
                            false. default: false.
      --quant.calibration_method 
                            quantize calibration method. type: enum. required:
                            false. default: MinMax. option: MinMax, Percentile,
                            MSE.
      --quant.precision_analysis 
                            enable quantization precision analysis. type: bool.
                            required: false. default: false.
      --quant.precision_analysis_method 
                            precision analysis method. type: enum. required:
                            false. default: PerLayer. option: PerLayer, EndToEnd.
      --quant.precision_analysis_mode 
                            precision analysis mode. type: enum. required: false.
                            default: Reference. option: Reference, NPUBackend.
      --quant.highest_mix_precision 
                            enable highest mix precision quantization. type: bool.
                            required: false. default: false.
      --quant.conv_bias_data_type 
                            conv bias data type. type: enum. required: false.
                            default: S32. option: S32, FP32.
      --quant.refine_weight_threshold 
                            refine weight threshold, should be a legal float
                            number, like 1e-6. -1 means disable this feature.
                            type: float. required: false. default: 1e-6.
                            limitation: 0 or less than 0.0001.
      --quant.enable_smooth_quant 
                            enalbe smooth quant strategy for conv 1x1. type: bool.
                            required: false. default: false.
      --quant.transformer_opt_level 
                            tranformer opt level. type: int. required: false.
                            default: 0. limitation: 0~2.
      --quant.input_sample_dir 
                            input sample data dir for precision analysis. type:
                            string. required: false. default: .
      --quant.ln_scale_data_type 
                            LayerNormalization scale data type. type: enum.
                            required: false. default: FP32. option: FP32, S32,
                            U32.
      --compiler.static_batch_sizes [ ...]
                            static batch sizes. type: int array. required: false.
                            default: [].
      --compiler.max_dynamic_batch_size 
                            max dynamic batch. type: int, required: false.
                            default: 0.
      --compiler.disable_ir_fix 
                            disable ir fix, only work in multi-batch compilation.
                            type: bool. required: false. default: false.
      --compiler.check      compiler check level, 0: no check; 1: simulate compile
                            result; 2: simulate and check compile result (for
                            debug). type: int. required: false. default: 0.
      --compiler.debug      compiler debug level. type: int. required: false.
                            default: 0.
      --compiler.input_sample_dir 
                            input sample data dir for compiler check. type:
                            string. required: false. default: .

.. hint::

  - Users can write configuration files in the format of ``json/yaml/toml/prototxt`` according to parameter specifications, and point to the configuration file through the command line parameter ``--config``
  - Some compilation parameters support command line input and have higher priority than configuration files. Use pulsar2 build -h to view the supported command line compilation parameters. For example, the command line parameter ``--quant.calibration_method`` is equivalent to the ``calibration_method`` field of the ``QuantConfig`` structure is configured

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Detailed explanation of parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. data:: pulsar2 build's parameter explanation

    --config

        - type of data: string
        - required or not:  yes
        - description：configuration file path, supports ``json/yaml/toml/prototxt`` format, see :ref:`《Configuration File Detailed Description》 <config_details>` for the structure

    --work_dir

        - type of data: string
        - required or not:  no
        - default value: same as output_dir
        - description： intermediate result output directory

    --input

        - type of data: string
        - required or not:  yes
        - description： model enter path

    --output_dir

        - type of data: string
        - required or not:  yes
        - description： compilation result output directory, the compiled model is named compiled.axmodel

    --model_type

        - type of data: enum
        - required or not:  no
        - default value: ONNX
        - description： input model type, supports enumeration: ``ONNX``, ``QuantAxModel``, ``QuantONNX``

    --target_hardware

        - type of data: enum
        - required or not:  no
        - default value: AX650
        - description： the target soc platform type for model compilation, supports ``AX650``, ``AX620E``, ``M76H``

    --npu_mode

        - type of data: enum
        - required or not:  no
        - default value: NPU1
        - description： model compilation mode

            * When the soc platform is ``AX650``, enumeration is supported: ``NPU1``, ``NPU2``, ``NPU3``
            * When the SOC platform is ``AX620E``, enumeration is supported: ``NPU1``, ``NPU2``

      .. warning:: npu_mode refers to the number of NPU cores used, not the vNPU number, please don't be confused.

    --input_shapes

        - type of data: string
        - required or not:  no
        - default value: empty
        - description： during the model compilation process, modify the input size of the model in the format: ``input1:1x3x224x224;input2:1x1x112x112``.

    --onnx_opt

        - disable_onnx_optimization

            - type of data: bool
            - required or not:  no
            - default value: false
            - description： whether to enable the floating-point ONNX model graph optimization module.

        - enable_onnxsim

            - type of data: bool
            - required or not:  no
            - default value: false
            - description： whether to use the `onnxsim` tool to simplify floating point ONNX, https://github.com/daquexian/onnx-simplifier.

        - model_check

            - type of data: bool
            - required or not:  no
            - default value: false
            - description： whether to enable the bisection function of the floating-point ONNX model graph after optimization with the original ONNX model.

        - disable_transformation_check

            - type of data: bool
            - required or not:  no
            - default value: false
            - description： whether to disable the subgraph bisection function after each subgraph transformation during floating-point ONNX model graph optimization.

    --quant

        A member variable named quant in BuiildConfig

        - calibration_method

            - type of data: enum
            - required or not:  no
            - default value: MinMax
            - description： Quantization algorithm, supported enumerations ``MinMax``, ``Percentile``, ``MSE``, the structure can be found in :ref:`《Configuration File Detailed Description》 <config_details>`

        - precision_analysis

            - type of data: bool
            - required or not:  no
            - default value: false
            - description： whether to analyze the quantification accuracy of Quant AXModel layer by layer

        - precision_analysis_method

            - type of data: enum
            - required or not:  no
            - default value: PerLayer
            - description： precision analysis method, optional PerLayer / EndToEnd. PerLayer means that each layer uses the layer input corresponding to the floating point model, and calculates the similarity between the output of each layer and the output of the floating point model. EndToEnd means that the first layer adopts floating point model input, then simulates the complete model, and calculates the similarity between the final output result and the floating point model output.

        - precision_analysis_mode

            - type of data: enum
            - required or not:  no
            - default value: Reference
            - description： Implementation of layer-by-layer simulation, optional Reference / NPUBackend. Reference can run all models supported by the compiler (supports models including CPU and NPU subgraphs), but the calculation results will have a small error compared to the final board results (basically the difference is within plus or minus 1, and there is no systematic error ). NPUBackend can run models containing only NPU subgraphs, but the calculation results are bit-aligned with the upper-board results.

        - highest_mix_precision

            - type of data: bool
            - required or not:  no
            - default value: false
            - description： whether to enable the highest precision quantization mode.

        - conv_bias_data_type

            - type of data: enum
            - required or not:  no
            - default value: S32
            - description： the data type saved by the Bias attribute of the Conv operator during quantization, optional S32/FP32.

        - refine_weight_threshold

            - type of data: float
            - required or not:  no
            - default value: 1e-6
            - description： adjusts the weight quantization threshold to the specified value.

        - enable_smooth_quant

            - type of data: bool
            - required or not:  no
            - default value: false
            - description： enable smooth quant quantization strategy to improve quantization accuracy.

        - transformer_opt_level

            - type of data: int
            - required or not:  no
            - default value: 0
            - description： Quantization mode configuration of Transformer network.

        - input_sample_dir

            - type of data: string
            - required or not:  no
            - default value: 空
            - description： Configures the input data directory used for quantification accuracy analysis. If not specified, data from the quantification calibration set is used.

            .. note::

                Note that the format of the input data in ``--quant.input_sample_dir`` should be the same as the original model.

    --compiler

        A member variable named compiler in BuildConfig

        - static_batch_sizes

            - type of data: list of int
            - required or not:  no
            - default value: 0
            - description： The compiler compiles according to the batch combination provided by the user. Based on this set of batch models, it can support efficient inference of any batch_size input at runtime. For details, please refer to: :ref:`《Static multi-batch mode》<multi_batch_static_compile>`.

        - max_dynamic_batch_size

            - type of data: int
            - required or not:  no
            - default value: 0
            - description： The compiler automatically derives a batch model combination that the NPU can run efficiently and is no larger than max_dynamic_batch_size. Based on this set of batch models, efficient inference of any batch_size input can be supported at runtime. For details, please refer to: :ref:`《Dynamic multi-batch mode》<multi_batch_dynamic_compile>`.

        - ddr_bw_limit

            - type of data: float
            - required or not: No
            - default value: 0
            - description：: Set the compile-time emulation ddr bandwidth limit in GB.

        - disable_ir_fix

            - type of data: bool
            - required or not:  no
            - default value: false
            - description： whether to disable the compiler's default Reshape operator attribute modification behavior during multi-batch compilation.

        - npu_perf

            - type of data: bool
            - required or not:  no
            - default value: false
            - description：: export debug files during NPU compilation.

        - check

            - type of data: int
            - required or not:  no
            - default value: 0
            - description： whether to check the correctness of the compilation results through simulation, 0 means no checking; 1 means checking whether the compilation results can run correctly; 2 means checking whether the output data of the model is correct.

        - check_mode

          - type of data: enum
          - required or not:  no
          - default value: 0
          - description：bisection mode, CheckOutput means that only the result is bisected. CheckPerLayer means bisection layer by layer.

        - check_rtol

          - type of data: float
          - required or not:  no
          - default value: 1e-5
          - description：this parameter is effective when the --compiler.check parameter is 1. This parameter is the relative error parameter.

        - check_atol

          - type of data: float
          - required or not:  no
          - default value: 0
          - description：this parameter is effective when the --compiler.check parameter is 1. This parameter is the relative error parameter.

        - check_cosine_simularity

          - type of data: float
          - required or not:  no
          - default value: 0.999
          - description：this parameter is only valid when the --compiler.check parameter is 3. This parameter specifies the tensor cosine similarity check threshold.

        - check_tensor_black_list

          - type of data: list of string
          - required or not:  no
          - default value: []
          - description：a list of tensors that are not included in the check. Regular expression matching is supported.

        - input_sample_dir

            - type of data: string
            - required or not:  no
            - default value: empty
            - description： configures the input data directory used for compiler checks. If not specified, quantization calibration data will be used in preference.

            .. note::

                Note that the input data in ``--compiler.input_sample_dir`` should be in the same format as the compiled model (including preprocessing).

-------------------------------------------------
Detailed explanation of multi-core compilation
-------------------------------------------------

Users can flexibly configure the **NPU compilation mode** by modifying the ``--npu_mode`` option in ``pulsar2 build`` to make full use of computing power.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NPU single core mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default configuration of ``--npu_mode`` is ``NPU1``, which is ``1 NPU core`` mode. The previous :ref:`《Model Compilation》 <model_compile>` chapter used the default configuration of ``NPU1`` for explanation.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NPU dual core mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``--npu_mode`` configuration is modified to ``NPU2``, that is, ``2 NPU core`` mode. Taking the example of converting the ``mobilenetv2`` model, modify the configuration file as follows:

.. code-block::

    {
      "model_type": "ONNX",
      "npu_mode": "NPU2",   # 只需要修改这里，默认配置是 NPU1
      "quant": {
        "input_configs": [
          {
            "tensor_name": "input",
            "calibration_dataset": "./dataset/imagenet-32-images.tar",
            "calibration_size": 32,
            "calibration_mean": [103.939, 116.779, 123.68],
            "calibration_std": [58.0, 58.0, 58.0]
          }
        ],
        "calibration_method": "MinMax",
        "precision_analysis": false
      },
      "input_processors": [
        {
          "tensor_name": "input",
          "tensor_format": "BGR",
          "src_format": "BGR",
          "src_dtype": "U8",
          "src_layout": "NHWC",
          "csc_mode": "NoCSC"
        }
      ],
      "compiler": {
        "check": 0
      }
    }

The compilation command of ``pulsar2 build`` as follows:

.. code-block::

    root@xxx:/data# pulsar2 build --input model/mobilenetv2-sim.onnx --output_dir output --config config/mobilenet_v2_build_config.json

.. _multi_batch_compile:

-------------------------------------------------
Detailed explanation of multi-batch compilation
-------------------------------------------------

``pulsar2 build`` supports users to configure the batch_size of the model, which is divided into two modes: static multi-batch and dynamic multi-batch compilation. These two modes are mutually exclusive. This chapter uses ``AX650`` as an example.

.. _multi_batch_static_compile:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Static multi-batch mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The compiler compiles according to the batch combination provided by the user, and supports two ways of configuring the command line parameter ``--compiler.static_batch_sizes`` and modifying ``compiler.static_batch_sizes`` in the configuration file.

* After setting up static multi-batch compilation, when viewing ``compiled.axmodel`` through ``onnx inspect -m -n -t``, the batch dimension of the input and output shape will become the maximum batch specified by the user.

     * **Weight data will be reused as much as possible between batches**, so the model size is smaller than the sum of the model sizes of each batch compiled separately.

.. hint::

    Taking the mobilenetv2 model as an example, the original model input ``input`` shape is ``[1, 224, 224, 3]``,
    After static multi-batch compilation with ``static_batch_sizes`` equal to [1, 2, 4], the shape will become ``[4, 224, 224, 3]``.

.. _multi_batch_dynamic_compile:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dynamic multi-batch mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The compiler automatically derives a batch model combination that the NPU can run efficiently and is no larger than ``max_dynamic_batch_size``. Based on this set of batch models, efficient inference of any batch_size input can be supported at runtime. Supports two ways of configuring the command line parameter ``--compiler.max_dynamic_batch_size`` and modifying ``compiler.max_dynamic_batch_size`` in the configuration file.

* The compiler will start from batch 1 and compile in 2-fold increments. It will stop when the batch is larger than the set ``max_dynamic_batch_size`` or the theoretical reasoning efficiency of the current batch is lower than that of the previous batch.

     * Batch theoretical reasoning efficiency: theoretical reasoning takes / batch_size.

     * **Weight data will be reused as much as possible between batches**, so the model size is smaller than the sum of the model sizes of each batch compiled separately.

* After setting up dynamic multi-batch compilation, when viewing ``compiled.axmodel`` through ``onnx inspect -m -n -t``, the batch dimension of the input and output shape will become ``max_dynamic_batch_size``.

.. hint::

     Taking the mobilenetv2 model as an example, the original model input ``input`` shape is ``[1, 224, 224, 3]``,
     After doing a dynamic multi-batch compilation with max_dynamic_batch_size equal to 4, the shape will become ``[4, 224, 224, 3]``.

* When running, it will find the appropriate batch combination and perform multiple inferences based on the dynamic batch size set by the caller during inference.

.. hint::

     When the theoretical inference efficiency of a model increases as the number of batches increases and there is only one NPU subgraph after compilation and ``max_dynamic_batch_size`` is set to 4, the compiled ``compiled.axmodel`` will contain [1, 2 , 4] model of three batches.

     When reasoning or simulating:

     * If the dynamic batch value is set to 3, the ``axengine inference framework`` and ``pulsar2 run emulator`` will internally perform batch 2 + batch 1 twice for NPU inference or simulation.
     * If the dynamic batch value is set to 9, the ``axengine inference framework`` and ``pulsar2 run emulator`` will perform batch 4 + batch 4 + batch 1 three times of NPU inference or simulation internally.
  
During dynamic multi-batch compilation, the following logs will show the compiled batch combinations:

.. code-block:: bash

    ...
    2023-07-09 20:06:02.690 | INFO     | yamain.command.build:compile_npu_subgraph:985 - QuantAxModel macs: 280,262,480
    2023-07-09 20:06:06.786 | WARNING  | yamain.command.build:compile_npu_subgraph:1035 - graph [subgraph_npu_0] batchs [1, 2]
    2023-07-09 20:06:06.795 | INFO     | yamain.command.build:compile_ptq_model:924 - fuse 1 subgraph(s)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-batch compilation of models containing the Reshape operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If the model contains the ``Reshape`` operator, ``pulsar2`` may not be able to correctly infer the output shape of the ``Reshape`` operator when performing multi-batch compilation.

At this time, the user can modify the ``shape`` input data of the ``Reshape`` operator through the :ref:`《Constant Data Modification》<Const_patch>` function, so that ``pulsar2`` can be correct during the multi-batch compilation process. Derive the output shape.

For example, there is a ``Reshape`` operator with a shape of ``[2, 1, 64]``. Assuming that the first dimension is batch, the constant tensor corresponding to the shape is modified to ``[ through the constant data modification function. 2, -1, 64]`` or ``[2, 0, 64]`` to support multi-batch compilation.

In addition, if the user does not explicitly configure the operator attribute modification, then ``pulsar2`` will modify the 0th dimension of the ``Reshape`` operator**shape to -1, and try to perform multi-batch compilation**.

.. hint::

     pulsar2 supports configuring ``0`` or ``-1`` in the shape of ``Reshape``. ``0`` represents the same value as the corresponding dimension of the input tensor; ``-1`` represents the unknown dimension size calculated based on the input tensor.

.. _perlayer_precision_debug:

------------------------------------
Detailed explanation layer by layer
------------------------------------

``pulsar2 build`` provides a set of layer-by-layer accuracy comparison tools between floating-point models and quantized models. There are two ways to configure it, just choose one:

* Configure ``--quant.precision_analysis 1`` on the command line to enable the precision comparison function; the following is an example:

.. code-block:: bash

     root@xxx:/data/quick_start_example# pulsar2 build --input model/mobilenetv2-sim.onnx --output_dir output --config config/config_mobilenet_v2_onnx.json --quant.precision_analysis 1

* The compilation command remains unchanged, and ``"precision_analysis": false`` is modified to ``"precision_analysis": true`` in the configuration file:

.. code-block:: bash

    {
      "model_type": "ONNX",
      "npu_mode": "NPU1",
      "quant": {
        "input_configs": [
          {
            "tensor_name": "input",
            "calibration_dataset": "./dataset/imagenet-32-images.tar",
            "calibration_size": 32,
            "calibration_mean": [103.939, 116.779, 123.68],
            "calibration_std": [58.0, 58.0, 58.0]
          }
        ],
        "calibration_method": "MinMax",
        "precision_analysis": true  # 这里修改为 true, 默认是 false
      },
      "input_processors": [
        {
          "tensor_name": "input",
          "tensor_format": "BGR",
          "src_format": "BGR",
          "src_dtype": "U8",
          "src_layout": "NHWC",
          "csc_mode": "NoCSC"
        }
      ],
      "compiler": {
        "check": 0
      }
    }

After re-executing the compilation process, you can get the following output information with ``Quant Precision Table``, including **node name, type, output name, data type, output shape, cosine similarity**, etc.:
At the same time, a quantized similarity graph file in mmd format will be saved. Different similarities can be distinguished by color, which can more intuitively locate precision problems. The file path can be found through the ``save precision analysis graph to`` keyword in the log.

.. code-block:: bash

    root@xxx:/data# pulsar2 build --input model/mobilenetv2-sim.onnx --output_dir output --config config/mobilenet_v2_build_config.json
    ...

    Building native ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
                                                 Quant Precision Table【PerLayer Reference】
    ┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
    ┃ Operator                ┃          Type          ┃ Output Tensor ┃       Shape       ┃ DType ┃ QDType ┃ Cosin   ┃ MSE     ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
    │ Conv_0                  │    AxQuantizedConv     │ 317           │ (1, 32, 112, 112) │ FP32  │   U8   │ 0.99993 │ 0.00003 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_2                  │    AxQuantizedConv     │ 320           │ (1, 32, 112, 112) │ FP32  │   U8   │ 0.99945 │ 0.00070 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_4                  │    AxQuantizedConv     │ 480           │ (1, 16, 112, 112) │ FP32  │   U8   │ 0.99904 │ 0.00046 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_5                  │    AxQuantizedConv     │ 325           │ (1, 96, 112, 112) │ FP32  │   U8   │ 0.99939 │ 0.00008 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_7                  │    AxQuantizedConv     │ 328           │  (1, 96, 56, 56)  │ FP32  │   U8   │ 0.99919 │ 0.00020 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_9                  │    AxQuantizedConv     │ 489           │  (1, 24, 56, 56)  │ FP32  │   U8   │ 0.99912 │ 0.00027 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_10                 │    AxQuantizedConv     │ 333           │ (1, 144, 56, 56)  │ FP32  │   U8   │ 0.99982 │ 0.00002 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_12                 │    AxQuantizedConv     │ 336           │ (1, 144, 56, 56)  │ FP32  │   U8   │ 0.99957 │ 0.00005 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_14                 │    AxQuantizedConv     │ 498           │  (1, 24, 56, 56)  │ FP32  │   U8   │ 0.99933 │ 0.00026 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Add_15                  │     AxQuantizedAdd     │ 339           │  (1, 24, 56, 56)  │ FP32  │   U8   │ 0.99930 │ 0.00050 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_16                 │    AxQuantizedConv     │ 342           │ (1, 144, 56, 56)  │ FP32  │   U8   │ 0.99969 │ 0.00002 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_18                 │    AxQuantizedConv     │ 345           │ (1, 144, 28, 28)  │ FP32  │   U8   │ 0.99979 │ 0.00004 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_20                 │    AxQuantizedConv     │ 507           │  (1, 32, 28, 28)  │ FP32  │   U8   │ 0.99970 │ 0.00013 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_21                 │    AxQuantizedConv     │ 350           │ (1, 192, 28, 28)  │ FP32  │   U8   │ 0.99989 │ 0.00001 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_23                 │    AxQuantizedConv     │ 353           │ (1, 192, 28, 28)  │ FP32  │   U8   │ 0.99936 │ 0.00003 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_25                 │    AxQuantizedConv     │ 516           │  (1, 32, 28, 28)  │ FP32  │   U8   │ 0.99955 │ 0.00008 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Add_26                  │     AxQuantizedAdd     │ 356           │  (1, 32, 28, 28)  │ FP32  │   U8   │ 0.99969 │ 0.00020 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_27                 │    AxQuantizedConv     │ 359           │ (1, 192, 28, 28)  │ FP32  │   U8   │ 0.99989 │ 0.00000 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_29                 │    AxQuantizedConv     │ 362           │ (1, 192, 28, 28)  │ FP32  │   U8   │ 0.99974 │ 0.00001 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_31                 │    AxQuantizedConv     │ 525           │  (1, 32, 28, 28)  │ FP32  │   U8   │ 0.99950 │ 0.00006 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Add_32                  │     AxQuantizedAdd     │ 365           │  (1, 32, 28, 28)  │ FP32  │   U8   │ 0.99966 │ 0.00026 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_33                 │    AxQuantizedConv     │ 368           │ (1, 192, 28, 28)  │ FP32  │   U8   │ 0.99984 │ 0.00001 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_35                 │    AxQuantizedConv     │ 371           │ (1, 192, 14, 14)  │ FP32  │   U8   │ 0.99991 │ 0.00002 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_37                 │    AxQuantizedConv     │ 534           │  (1, 64, 14, 14)  │ FP32  │   U8   │ 0.99968 │ 0.00012 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_38                 │    AxQuantizedConv     │ 376           │ (1, 384, 14, 14)  │ FP32  │   U8   │ 0.99994 │ 0.00000 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_40                 │    AxQuantizedConv     │ 379           │ (1, 384, 14, 14)  │ FP32  │   U8   │ 0.99975 │ 0.00001 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_42                 │    AxQuantizedConv     │ 543           │  (1, 64, 14, 14)  │ FP32  │   U8   │ 0.99979 │ 0.00004 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Add_43                  │     AxQuantizedAdd     │ 382           │  (1, 64, 14, 14)  │ FP32  │   U8   │ 0.99976 │ 0.00011 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_44                 │    AxQuantizedConv     │ 385           │ (1, 384, 14, 14)  │ FP32  │   U8   │ 0.99994 │ 0.00000 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_46                 │    AxQuantizedConv     │ 388           │ (1, 384, 14, 14)  │ FP32  │   U8   │ 0.99985 │ 0.00001 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_48                 │    AxQuantizedConv     │ 552           │  (1, 64, 14, 14)  │ FP32  │   U8   │ 0.99973 │ 0.00003 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Add_49                  │     AxQuantizedAdd     │ 391           │  (1, 64, 14, 14)  │ FP32  │   U8   │ 0.99973 │ 0.00013 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_50                 │    AxQuantizedConv     │ 394           │ (1, 384, 14, 14)  │ FP32  │   U8   │ 0.99989 │ 0.00000 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_52                 │    AxQuantizedConv     │ 397           │ (1, 384, 14, 14)  │ FP32  │   U8   │ 0.99954 │ 0.00002 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_54                 │    AxQuantizedConv     │ 561           │  (1, 64, 14, 14)  │ FP32  │   U8   │ 0.99893 │ 0.00016 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Add_55                  │     AxQuantizedAdd     │ 400           │  (1, 64, 14, 14)  │ FP32  │   U8   │ 0.99954 │ 0.00024 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_56                 │    AxQuantizedConv     │ 403           │ (1, 384, 14, 14)  │ FP32  │   U8   │ 0.99987 │ 0.00000 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_58                 │    AxQuantizedConv     │ 406           │ (1, 384, 14, 14)  │ FP32  │   U8   │ 0.99984 │ 0.00002 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_60                 │    AxQuantizedConv     │ 570           │  (1, 96, 14, 14)  │ FP32  │   U8   │ 0.99963 │ 0.00007 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_61                 │    AxQuantizedConv     │ 411           │ (1, 576, 14, 14)  │ FP32  │   U8   │ 0.99982 │ 0.00000 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_63                 │    AxQuantizedConv     │ 414           │ (1, 576, 14, 14)  │ FP32  │   U8   │ 0.99934 │ 0.00003 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_65                 │    AxQuantizedConv     │ 579           │  (1, 96, 14, 14)  │ FP32  │   U8   │ 0.99818 │ 0.00018 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Add_66                  │     AxQuantizedAdd     │ 417           │  (1, 96, 14, 14)  │ FP32  │   U8   │ 0.99941 │ 0.00016 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_67                 │    AxQuantizedConv     │ 420           │ (1, 576, 14, 14)  │ FP32  │   U8   │ 0.99895 │ 0.00002 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_69                 │    AxQuantizedConv     │ 423           │ (1, 576, 14, 14)  │ FP32  │   U8   │ 0.99857 │ 0.00006 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_71                 │    AxQuantizedConv     │ 588           │  (1, 96, 14, 14)  │ FP32  │   U8   │ 0.99615 │ 0.00052 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Add_72                  │     AxQuantizedAdd     │ 426           │  (1, 96, 14, 14)  │ FP32  │   U8   │ 0.99804 │ 0.00078 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_73                 │    AxQuantizedConv     │ 429           │ (1, 576, 14, 14)  │ FP32  │   U8   │ 0.99914 │ 0.00002 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_75                 │    AxQuantizedConv     │ 432           │  (1, 576, 7, 7)   │ FP32  │   U8   │ 0.99953 │ 0.00005 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_77                 │    AxQuantizedConv     │ 597           │  (1, 160, 7, 7)   │ FP32  │   U8   │ 0.99265 │ 0.00047 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_78                 │    AxQuantizedConv     │ 437           │  (1, 960, 7, 7)   │ FP32  │   U8   │ 0.99659 │ 0.00008 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_80                 │    AxQuantizedConv     │ 440           │  (1, 960, 7, 7)   │ FP32  │   U8   │ 0.99807 │ 0.00007 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_82                 │    AxQuantizedConv     │ 606           │  (1, 160, 7, 7)   │ FP32  │   U8   │ 0.99201 │ 0.00042 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Add_83                  │     AxQuantizedAdd     │ 443           │  (1, 160, 7, 7)   │ FP32  │   U8   │ 0.98304 │ 0.00211 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_84                 │    AxQuantizedConv     │ 446           │  (1, 960, 7, 7)   │ FP32  │   U8   │ 0.99485 │ 0.00011 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_86                 │    AxQuantizedConv     │ 449           │  (1, 960, 7, 7)   │ FP32  │   U8   │ 0.99866 │ 0.00007 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_88                 │    AxQuantizedConv     │ 615           │  (1, 160, 7, 7)   │ FP32  │   U8   │ 0.98717 │ 0.00190 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Add_89                  │     AxQuantizedAdd     │ 452           │  (1, 160, 7, 7)   │ FP32  │   U8   │ 0.97100 │ 0.00809 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_90                 │    AxQuantizedConv     │ 455           │  (1, 960, 7, 7)   │ FP32  │   U8   │ 0.98869 │ 0.00006 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_92                 │    AxQuantizedConv     │ 458           │  (1, 960, 7, 7)   │ FP32  │   U8   │ 0.99952 │ 0.00002 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_94                 │    AxQuantizedConv     │ 624           │  (1, 320, 7, 7)   │ FP32  │   U8   │ 0.99873 │ 0.00012 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Conv_95                 │    AxQuantizedConv     │ 463           │  (1, 1280, 7, 7)  │ FP32  │   U8   │ 0.99990 │ 0.00024 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ GlobalAveragePool_97    │ AxQuantizedGlobAvgPool │ 464           │  (1, 1280, 1, 1)  │ FP32  │   U8   │ 0.99998 │ 0.00002 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ Reshape_103             │       AxReshape        │ 472           │     (1, 1280)     │ FP32  │   U8   │ 0.99998 │ 0.00002 │
    ├─────────────────────────┼────────────────────────┼───────────────┼───────────────────┼───────┼────────┼─────────┼─────────┤
    │ output_DequantizeLinear │   AxDequantizeLinear   │ output        │     (1, 1000)     │ FP32  │  FP32  │ 0.99990 │ 0.00173 │
    └─────────────────────────┴────────────────────────┴───────────────┴───────────────────┴───────┴────────┴─────────┴─────────┘
    2024-09-25 11:47:01.640 | INFO     | yamain.command.precision_analysis:quant_precision_analysis:401 - save precision analysis table to [output/quant/debug/precision_analysis_table.txt]
    2024-09-25 11:47:01.641 | INFO     | yamain.command.precision_analysis:quant_precision_analysis:409 - save precision analysis graph to [output/quant/debug/precision_analysis.mmd]
    ...


Open the ``output/quant/debug/precision_analysis.mmd`` file with an editing tool that supports mermaid flowchart and you can see the following quantitative similarity graph

.. figure:: ../media/precision_analysis.png
        :alt: precision_analysis_group
        :align: center

.. hint::

    For more details, please refer to :ref:`《Quantitative Precision Analysis Parameter Description》 <quant_precision_analysis_config_define>`.

.. note::

    If ``"precision_analysis": false`` is in the configuration file and the compilation command contains ``--quant.precision_analysis 1``, the precision comparison function will still be enabled.


.. _custom_calib_dataset:

-------------------------------------------------
Detailed explanation of loading custom data sets
-------------------------------------------------

``pulsar2 build`` supports loading user-defined data sets for quantification, and supports ``.npy`` and ``.bin`` file formats with suffixes.

~~~~~~~~~~~~~~~~
Prepare dataset
~~~~~~~~~~~~~~~~

It is recommended that when processing images, try to be the same as the preprocessing during inference, and try to avoid using data enhancement during training. Some reference steps are as follows:

     - read pictures
     - Align image ``rbg channel`` sequentially to model input
     - Zoom pictures
     - Normalized

The above steps are for reference only and can be adjusted and deleted according to the actual situation. If some models do not require normalization of images, the normalization step can be omitted for such models.

After processing the images, package the corresponding format files into compressed files.

.. note::

     The ``npy`` suffix file refers to a file saved in the ``Numpy`` array format. When using this file format, you need to ensure that the data type and shape of the array when saving are consistent with the corresponding model input, and the suffix name is ``. npy``.

     The ``bin`` suffix file refers to a file saved in binary format. When using this file format, the data should be saved in binary with the suffix ``.bin``.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configuration and compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modify the ``quant.input_configs.calibration_format`` field to ``Numpy`` or ``Binary``. A complete example is as follows:

.. code-block:: shell

    {
      "model_type": "ONNX",
      "npu_mode": "NPU1",
      "quant": {
        "input_configs": [
          {
            "tensor_name": "input",
            "calibration_dataset": "./dataset/npy_dataset.tar",
            "calibration_size": 10,
            "calibration_mean": [103.939, 116.779, 123.68],
            "calibration_std": [58.0, 58.0, 58.0],
            "calibration_format": "Numpy", # change to Numpy or Binary, the default is Image
          }
        ],
        "calibration_method": "MinMax",
      },
      "input_processors": [
        {
          "tensor_name": "input",
          "tensor_format": "BGR",
          "src_format": "BGR",
          "src_dtype": "U8",
          "src_layout": "NHWC",
          "csc_mode": "NoCSC"
        }
      ],
      "compiler": {
        "check": 0
      }
    }

After compilation is executed, the ``Data Format`` field in ``Quant Config Table`` is changed to ``Numpy``. The result is as follows:

.. code-block:: bash

    root@aa:/data/quick_start_example# pulsar2 build --input model/mobilenetv2-sim.onnx --output_dir npy_output/ --config config/npy_config_mobilenet_v2_onnx.json
    ...
                                                                            Quant Config Table
    ┏━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
    ┃ Input ┃ Shape            ┃ Dataset Directory ┃ Data Format ┃ Tensor Format ┃ Mean                                                         ┃ Std                ┃
    ┡━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
    │ input │ [1, 3, 224, 224] │ input             │ Numpy       │ BGR           │ [103.93900299072266, 116.77899932861328, 123.68000030517578] │ [58.0, 58.0, 58.0] │
    └───────┴──────────────────┴───────────────────┴─────────────┴───────────────┴──────────────────────────────────────────────────────────────┴────────────────────┘
    ...

----------------------------------------------------------
Multi-input model configuration quantitative data set
----------------------------------------------------------

For models with multiple inputs, different inputs require different calibration sets, which can be achieved by modifying the configuration.

The field ``input_configs`` supports configuring multiple inputs. ``tensor_name`` is used to specify the input name of the model. The following is a configuration example:

.. code-block:: shell

    {
      "quant": {
        "input_configs": [
          {
            "tensor_name": "input1", # 输入 1
            "calibration_dataset": "input1_dataset.tar",
            "calibration_size": 10,
            "calibration_mean": [103.939, 116.779, 123.68],
            "calibration_std": [58.0, 58.0, 58.0],
            "calibration_format": "Image", # 
          },
          {
            "tensor_name": "input2", # 输入 2
            "calibration_dataset": "input2_dataset.tar",
            "calibration_size": 10,
            "calibration_mean": [103.939, 116.779, 123.68],
            "calibration_std": [58.0, 58.0, 58.0],
            "calibration_format": "Image", 
          },
        ],
      }
    }

In a multi-input model, different inputs may need to be matched in each ``batch``. In this case, you can change the calibration sets of different inputs to the same ``batch`` to the same name. When quantizing, the inputs with the same name will be selected as one ``batch`` for quantization.

Taking the above configuration file as an example, the following is an example of the directory structure of the corresponding quantization file:

.. code-block:: shell

    .
    ├── input1
    │   ├── 1.bin
    │   └── 2.bin
    └── input2
        ├── 1.bin
        └── 2.bin

When the quantization module calibrates the data, it takes ``1.bin`` of ``input1`` and ``1.bin`` of ``input2`` as the first ``batch``.    

------------------------------------------------------------------------
Multi-input model configuration quantization data set (NumpyObject)
------------------------------------------------------------------------

For models with multiple inputs, different inputs require different calibration sets, which can also be achieved by using `NumpyObject`.

The field ``input_configs`` supports configuring multiple inputs. ``tensor_name`` is used to specify the input name of the model. The following is a configuration example:

.. code-block:: shell

    {
      "quant": {
        "input_configs": [
          {
            "tensor_name": "DEFAULT", 
            "calibration_dataset": "dataset.tar",
            "calibration_size": -1,
            "calibration_format": "NumpyObject", # 数据类型
          },

        ],
      }
    }

~~~~~~~~~~~~~~~~~~~~~~~~~
Prepare the dataset
~~~~~~~~~~~~~~~~~~~~~~~~~    

`NumpyObject` is a dictionary data type provided by `Numpy`. Dictionary data corresponds to `input` in the model, where `key` is the name of the `input` of the model.
`value` is the calibration data, and its type and shape should be the same as the corresponding `input`, that is, the data directly input to the model after preprocessing, and the format is `numpy.ndarray`.
The data processing of `value` is the same as :ref:`《Detailed Explanation of Loading Custom Datasets》 <custom_calib_dataset>`.

Assume that the model has two inputs as shown below:

.. figure:: ../media/multy_inputs.png
        :alt: pipeline
        :align: center

The following is a simple example of how to generate code:

.. code-block:: python

    import numpy as np

    calib_data = {}
    calib_data["rgb"] = np.random.randn(1, 2, 3, 224, 224).astype(np.float32)
    calib_data["inst_emb"] = np.random.randn(1, 384).astype(np.float32)

    np.save("data.npy", calib_data)

In a production environment, it is recommended to call the `dataloader` of the inference code, traverse it, convert the traversed data into the `Numpy.ndarray` type, and save it as a `NumpyObject` according to the dictionary, so that you can directly obtain the preprocessed data.

.. _mix_precision_quantization:

-----------------------------------------------------
Detailed explanation of mixed precision quantization
-----------------------------------------------------

``pulsar2 build`` supports mixed precision quantization, and you can set the quantization precision for a specified ``operator`` or ``a type of operator`` or ``a certain subgraph``.

~~~~~~~~~~~~~~~~
Configuration
~~~~~~~~~~~~~~~~

Modify the ``quant.layer_configs`` field. The currently supported enumerations for quantization precision are: ``U8`` ， ``U16`` ， ``FP32``。
The following is an example configuration:

.. code-block:: shell

    {
      "model_type": "ONNX",
      "npu_mode": "NPU1",
      "quant": {
        "input_configs": [
          {
            "tensor_name": "DEFAULT",
            "calibration_dataset": "./dataset/imagenet-32-images.tar",
            "calibration_size": 32,
            "calibration_mean": [103.939, 116.779, 123.68],
            "calibration_std": [58.0, 58.0, 58.0]
          }
        ],
        "layer_configs": [ 
            {
              "op_type": "Add", # specifies the quantization precision of operators of type Add
              "data_type": "U16"
            },
            {
              "layer_name": "conv6_4", # specify the quantization precision of the conv6_4 operator
              "data_type": "U16"
            },
            {
              # specify the quantization accuracy of the operators contained in the subgraph between conv2_1_linear_bn and relu2_2_dwise
              "start_tensor_names": "conv2_1_linear_bn",
              "end_tensor_names": "relu2_2_dwise",
              "data_type": "U16"
            }
        ],
        "calibration_method": "MinMax",
        "precision_analysis": false
      },
      "input_processors": [
        {
          "tensor_name": "input",
          "tensor_format": "BGR",
          "src_format": "BGR",
          "src_dtype": "U8",
          "src_layout": "NHWC",
          "csc_mode": "NoCSC"
        }
      ],
      "compiler": {
        "check": 0
      }
    }

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sub-image configuration instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When configuring a ``subgraph`` to a specific type, it is important to note that ``start_tensor_names`` and ``end_tensor_names`` specify ``tensor_name``, not ``node_name``.

.. figure:: ../media/nodename_vs_tensorname.png
        :alt: pipeline
        :align: center

If you want to configure the entire model for a certain quantization type, you can set ``start_tensor_names`` and ``end_tensor_names`` to ``[''DEFAULT'']``. Here is an example:

.. code-block:: shell

    {
      "layer_configs": [ 
          {
              "start_tensor_names": ["DEFAULT"], # string of list
              "end_tensor_names": ["DEFAULT"],   # string of list
              "data_type": "U16"
          }
        ]
    }

The ``Conv`` type operator does not support the configuration of ``data_type`` as ``FP32``, but its output can be configured to support ``FP32`` separately, which can be achieved through the following configuration:

.. code-block:: shell

    {
      "layer_configs": [ 
          {
            "op_type": "Conv", 
            "data_type": "U8",
            "output_data_type": "FP32", # 配置输出为FP32, 该配置目前只对Conv算子生效
          }
        ]
    }

The following is the configuration of the entire model except for ``Conv``, and the rest of the operators are quantized to ``FP32``:

.. code-block:: shell

    {
      "layer_configs": [ 
          {
            "op_type": "Conv", 
            "data_type": "U8",
            "output_data_type": "FP32", # 配置输出为FP32, 该配置目前只对Conv算子生效
          },
          {
              "start_tensor_names": ["DEFAULT"], # string of list
              "end_tensor_names": ["DEFAULT"],   # string of list
              "data_type": "FP32"
          }
        ]
    }

.. note::

    For an operator, there may be three quantization precision configurations: ``specified operator`` or ``a class of operators`` or ``a subgraph``. The priority is:
    ``specified operator`` > ``a class of operators`` > ``a subgraph``

.. attention::

    Currently, the ``FP32`` configuration supports limited operators. The verified operators include ``LeayRelu`` ``Sigmoid`` ``Relu`` ``Add`` ``Mul`` ``Div``
    ``Sub`` ``Concat`` ``Softmax``.

~~~~~~~~~~~~~~~~~~~~~~~~
Compilation and results
~~~~~~~~~~~~~~~~~~~~~~~~

There will be a ``Layer Config Table`` when compiling to display the current layer_configs configuration.

.. code-block:: bash

    root@aa:/data/quick_start_example# pulsar2 build --input model/mobilenetv2-sim.onnx --output_dir output --config config/mobilenet_v2_mix_precision_config.json
    ...
                                                                            Quant Config Table
    ┏━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
    ┃ Input ┃ Shape            ┃ Dataset Directory ┃ Data Format ┃ Tensor Format ┃ Mean                                                         ┃ Std                ┃
    ┡━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
    │ input │ [1, 3, 224, 224] │ input             │ Image       │ BGR           │ [103.93900299072266, 116.77899932861328, 123.68000030517578] │ [58.0, 58.0, 58.0] │
    └───────┴──────────────────┴───────────────────┴─────────────┴───────────────┴──────────────────────────────────────────────────────────────┴────────────────────┘
            Layer Config Table
    ┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
    ┃ Op Type / Layer name ┃ Precision ┃
    ┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
    │ Add                  │ U16       │
    ├──────────────────────┼───────────┤
    │ conv6_4              │ U16       │
    └──────────────────────┴───────────┘
    ...

After compilation, a ``quant_axmodel.json`` file will be generated in the ``output/quant`` directory, which records the quantization configuration information of each operator. A part of it is excerpted below for use as an example.

.. code-block:: shell

    "Add_26": {
      "507": {
        "bit_width": 16,
        "policy": {
          "PER_TENSOR": true,
          "PER_CHANNEL": false,
          "LINEAR": true,
          "EXPONENTIAL": false,
          "SYMMETRICAL": false,
          "ASYMMETRICAL": true,
          "POWER_OF_2": false
        },
        "state": "ACTIVATED",
        "quant_min": 0,
        "quant_max": 65535,
        "hash": 762206185,
        "dominator": 762206185
      },
      "516": {
        "bit_width": 16,
        "policy": {
          "PER_TENSOR": true,
          "PER_CHANNEL": false,
          "LINEAR": true,
          "EXPONENTIAL": false,
          "SYMMETRICAL": false,
          "ASYMMETRICAL": true,
          "POWER_OF_2": false
        },
        "state": "OVERLAPPED",
        "quant_min": 0,
        "quant_max": 65535,
        "hash": 3471866632,
        "dominator": 4099361028
      }
    }

.. _change_input_size:

------------------------------------
Enter size modification
------------------------------------

By modifying the configuration file, the dimensions of each input can be modified during the model conversion process.

Next, based on ``mobilenetv2``, modify the model input to ``384*384``

1. Command line mode, add parameters: ``--input_shapes data:1x3x384x384``

2. Configuration file method, add parameters to the root node:

.. code-block:: shell

    {
       ...
       "input_shapes": "data:1x3x384x384",
       ...
    }


During the model conversion process, the following log will appear, indicating that the model input size has been modified successfully:

.. code-block:: shell

    INFO[0006] 2023-08-24 20:04:59.530 | WARNING  | yamain.command.load_model:optimize_onnx_model:640 - change input shape to {'data': (1, 3, 384, 384)}

.. note::

    The model input size modification occurs before quantization, and the size of the quantized dataset needs to be consistent with the modified size.

    Multiple input groups are separated by semicolons. For details, refer to the parameter explanation section.

.. _multi_input_size:

-------------------------------------------
Configure model additional input dimensions
-------------------------------------------

By configuring the model compilation process, in addition to the main dimensions of the original model, additional sets of dimensions can be output. These dimensions will be saved in the same ``compiled.axmodel``.

The same set of weight data will be reused between multiple groups of sizes (the quantization tool will quantize the model based on its original size). Users need to evaluate the accuracy issues that may be caused by the difference between the size during quantization and the size during inference.

Next, we will take ``mobilenetv2`` as an example. Based on the original input size ``224*224``, we will add an additional size ``384*384``, and then select the size for simulation through the ``pulsar2 run`` tool.

- Modify the configuration file. In the ``input_processors`` node, configure a ``src_extra_shapes`` child node for the input:

.. code-block:: shell

    {
      ...
      "input_processors": [
        {
          "tensor_name": "DEFAULT",
          "tensor_format": "BGR",
          "src_format": "BGR",
          "src_dtype": "U8",
          "src_layout": "NHWC",
          "src_extra_shapes": [
            {
              "shape": [1, 3, 384, 384]
            }
          ],
          "csc_mode": "NoCSC",
          "csc_mat": [
            1.164, 2.017, 0, -276.8, 1.164, -0.392, -0.813, 135.616, 1.164, 0,
            1.596, -221.912
          ]
        }
      ],
      ...
    }

- During the model compilation process, the following log appears to confirm that the configuration is effective:  

.. code-block:: shell

    2024-01-01 21:27:02.082 | INFO     | yamain.command.build:compile_ptq_model:973 - extra input shape, index: 1, shape: {'data': (1, 3, 384, 384)}

- After compilation, ``compiled.axmodel`` will contain two subgraphs of independent sizes, which can be used for inference separately.

.. figure:: ../media/multi_shape_compiled_axmodel.png
    :alt: multi_shape
    :align: center

- ``pulsar2 run`` includes ``--group_index`` parameter, which is used to select sub-graphs of different sizes for simulation. The default value of this parameter is 0, which corresponds to the sub-graph of the original resolution (224*224). 1 corresponds to the sub-graph of the additional resolution (384*384).

- ``AXEngine`` For how to select different sizes when inferring models with additional input sizes, please refer to the ``AXEngine documentation``.

.. _op_attr_patch:

------------------------------------
Operator attribute modification
------------------------------------

By modifying the configuration file, the properties of a specific operator can be modified during the model conversion process.

Next, based on ``mobilenetv2``, modify the ``ceil_mode`` of the ``AveragePool`` operator named ``pool6`` to ``1``, and add the following content to the configuration file:

.. code-block:: shell

    "op_processors": [
      {
        "op_name": "pool6",
        "attrs": {
          "ceil_mode": 1
        }
      }
    ],

When using pulsar2 build to convert the model, the following log will appear, indicating that the operator attributes have been modified successfully:

.. code-block:: shell

    2023-05-07 18:47:34.274 | INFO     | yamain.command.load_model:op_attr_patch:488 - set op [pool6] attr [ceil_mode] to 1

.. _const_patch:

------------------------------------
Constant data modification
------------------------------------

By modifying the configuration file, specific constant data can be modified during the model conversion process.

Assume that a model contains a ``Reshape`` operator named ``reshape_0``. The ``shape`` input of this operator is a constant data named ``reshape_0_shape``, and the original data is `` [1, 96, 48]``.

Add the following content to the configuration file to modify the constant data to ``[-1, 96, 48]``.

.. code-block:: shell

    "const_processors": [
      {
        "name": "reshape_0_shape",
        "data": [-1, 96, 48]
      }
    ],

When using ``pulsar2 build`` to convert the model, the following log will appear, indicating that the constant data has been modified successfully:

.. code-block:: shell

    2023-05-07 18:15:41.464 | WARNING  | yamain.command.load_model:const_patch:512 - update data of const tensor [reshape_0_shape], (-1,, 96, 48), S64

.. _transformer_optimize:

----------------------------------------
Transformer model configuration details
----------------------------------------

For the Transformer model, you can set different levels of optimization through ``quant.transformer_opt_level``.

Currently, three levels of settings ``0``, ``1``, and ``2`` are supported.

The following is an example of the ``Swin-T`` model, the configuration is as follows

.. code-block:: shell

    "quant": {
        "input_configs": [
          {
            "tensor_name": "DEFAULT",
            "calibration_dataset": "dataset.tar",
            "calibration_format": "Image",
            "calibration_size": 32,
            "calibration_mean": [123.68, 116.779, 103.939],
            "calibration_std": [58.62, 57.34, 57.6]
          }
        ],
        "calibration_method": "MSE",
        "transformer_opt_level": 2 # set the transformer optimization level to 2
    },

When using ``pulsar2 build`` to convert the model, the following log will appear, indicating that the configuration modification is successful:

.. code-block:: shell

    INFO[0176] Transformer optimize level: 2

The following table shows the accuracy and performance of ``Swin-T`` under different optimization levels. The floating point accuracy (acc1) of this model is ``81.2%``

================== ============== ==============
Optimization level Accuracy(acc1) time consuming
================== ============== ==============
1                  80.488%        7.266ms
2                  80.446%        7.114ms
================== ============== ==============

.. note::

     The current version recommends setting level ``1``. In actual measurements, level ``2`` has a smaller performance improvement than level ``1``, while the accuracy of level ``1`` is slightly better.

.. note::

     The models that have been verified so far include ``Swin`` series, ``SwinV2`` series, ``Deit`` series, and ``Vit`` series.

------------------------------------
Quantized ONNX model import
------------------------------------

In order to support customers' self-quantized models (including 4-bit QAT quantization), ``AX650`` and ``M76H`` support quantized models in Quantized ONNX format as input. The model format uses ONNX QDQ format.

The following takes the resnet50 and yolov5s 4w8f models as examples to demonstrate how to compile models in Quantized ONNX format.

First, please download the model we have converted, :download:`Click to download resnet50 <../examples/resnet50_qdq_4w8f.onnx>`, :download:`Click to download yolov5s <../examples/yolov5s_qdq_4w8f.onnx>`

Then use the following configuration file for resnet50:

.. code-block:: json

    {
      "model_type": "QuantONNX",
      "npu_mode": "NPU1",
      "quant": {
        "input_configs": [
          {
            "tensor_name": "DEFAULT",
            "calibration_dataset": "s3://npu-ci/data/dataset_v04.zip",
            "calibration_size": 64,
            "calibration_mean": [103.939, 116.779, 123.68],
            "calibration_std": [1.0, 1.0, 1.0]
          }
        ],
        "calibration_method": "MinMax"
      },
      "input_processors": [
        {
          "tensor_name": "data",
          "src_format": "BGR",
          "src_dtype": "U8",
          "src_layout": "NHWC"
        }
      ],
      "compiler": {
        "check": 0
      }
    }

Finally, use the pulsar2 build command to compile, and you will get the ``compiled.axmodel`` file.

.. code-block:: shell

    pulsar2 build --target_hardware AX650 --input path/to/model.onnx  --config path/to/config.json --output_dir output

.. hint::
   
    Specify the input model type as Quantized ONNX through ``"model_type": "QuantONNX"`` in the configuration file.

Using a similar method, we can compile the yolov5s Quantized ONNX format model. We only need to replace it with the following configuration file for compilation:

.. code-block:: json

    {
      "model_type": "QuantONNX",
      "npu_mode": "NPU1",
      "quant": {
        "input_configs": [
          {
            "tensor_name": "DEFAULT",
            "calibration_dataset": "s3://npu-ci/data/coco_calib_image.tar",
            "calibration_size": 32,
            "calibration_mean": [0, 0, 0],
            "calibration_std": [255.0, 255.0, 255.0]
          }
        ],
        "layer_configs": [
          {
            "op_type": "Silu",
            "data_type": "U16"
          }
        ],
        "calibration_method": "MSE"
      },
      "input_processors": [
        {
          "tensor_name": "DEFAULT",
          "tensor_format": "RGB",
          "tensor_layout": "NCHW",
          "src_format": "BGR",
          "src_layout": "NHWC",
          "src_dtype": "U8"
        }
      ],
      "compiler": {
        "check": 0
      }
    }

------------------------------------
Color space conversion configuration
------------------------------------

Support customers to add color space conversion function in the model through configuration, and the NPU completes the conversion from YUV to RGB. For detailed configuration, please refer to Pre-processing and Post-processing Parameters <processing_arg_details>

.. code-block:: shell
  
    {
      "input_processors": [
        {
          "tensor_name": "DEFAULT",
          "tensor_format": "BGR",     
          "src_format": "YUV420SP",   # Specify the input color space of the compiled model
          "src_dtype": "U8",
          "src_layout": "NHWC",
          "csc_mode": "LimitedRange"
        }
      ]
    }
