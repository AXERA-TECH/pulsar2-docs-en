.. _config_details:

============================
Configuration file details
============================

This section will introduce the **config** file in ``pulsar2 build`` in detail.

------------------------------------
Profile overview
------------------------------------

- For the definition of all compilation parameters supported by the tool chain, please refer to :ref:`《proto Configuration Definition》<config_define>`, the basic data structure is ``BuildConfig``;

- Users can write configuration files in the format of ``prototxt/relaxed json/yaml/toml`` according to parameter specifications, and point to the configuration file through the command line parameter ``--config``;
  
     - Relaxed ``json`` format: supports ``json`` files containing ``js-style`` or ``python-style`` comments;

- Some compilation parameters support command line input and have higher priority than configuration files. Use pulsar2 build -h to view the supported command line compilation parameters. For example, the command line parameter ``--quant.calibration_method`` is equivalent to The ``calibration_method`` field of the ``QuantConfig`` structure is configured.

--------------------------------------
Complete json configuration reference
--------------------------------------

.. code-block:: json

    {
      // input model file path. type: string. required: true.
      "input": "/path/to/lenet5.onnx",
      // axmodel output directory. type: string. required: true.
      "output_dir": "/path/to/output_dir",
      // rename output axmodel. type: string. required: false. default: compiled.axmodel.
      "output_name": "compiled.axmodel",
      // temporary data output directory. type: string. required: false. default: same with ${output_dir}.
      "work_dir": "",
      // input model type. type: enum. required: false. default: ONNX. option: ONNX, QuantAxModel, QuantONNX.
      "model_type": "ONNX",
      // target hardware. type: enum. required: false. default: AX650. option: AX650, AX620E, M76H, M57.
      "target_hardware": "AX650",
      // npu mode. while ${target_hardware} is AX650, npu mode can be NPU1 / NPU2 / NPU3. while ${target_hardware} is AX620E, npu mode can be NPU1 / NPU2. type: enum. required: false. default: NPU1.
      "npu_mode": "NPU1",
      // modify model input shape of input model, this feature will take effect before the `input_processors` configuration. format: input1:1x3x224x224;input2:1x1x112x112. type: string. required: false. default: .
      "input_shapes": "input:1x1x28x28",
      "onnx_opt": {
        // disable onnx optimization. type: bool. required: false. default: false.
        "disable_onnx_optimization": false,
        // enable onnx simplify by https://github.com/daquexian/onnx-simplifier. type: bool. required: false. default: false.
        "enable_onnxsim": false,
        // enable model check. type: bool. required: false. default: false.
        "model_check": false,
        // disable transformation check. type: bool. required: false. default: false.
        "disable_transformation_check": false,
        // save tensors data to optimize memory footprint. type: bool. required: false. default: false.
        "save_tensors_data": false
      },
      "quant": {
        "input_configs": [
          {
            // input tensor name in origin model. "DEFAULT" means input config for all input tensors. type: string. required: true.
            "tensor_name": "input",
            // quantize calibration dataset archive file path. type: string. required: true. limitation: tar, tar.gz, zip.
            "calibration_dataset": "/path/to/dataset",
            // quantize calibration data format. type: enum. required: false. default: Image. option: Image, Numpy, Binary, NumpyObject.
            "calibration_format": "Image",
            // quantize calibration data size is min(${calibration_size}, size of ${calibration_dataset}), "-1" means load all dataset. type: int. required: false. default: 32.
            "calibration_size": 32,
            // quantize mean parameter of normlization. type: float array. required: false. default: [].
            "calibration_mean": [127],
            // quantize std parameter of normlization. type: float array. required: false. default: [].
            "calibration_std": [1]
          }
        ],
        "layer_configs": [
          {
            // set layer quantize precision. type: string. required: must choose between `layer_name` and `op_type` and `layer_names` and `op_types`. default: .
            "layer_name": "Conv_0",
            // quantize data type. type: enum. required: false. default: U8. option: U8, S8, U16, S16, FP32.
            "data_type": "U8",
            // quantize data type for Conv. type: enum. required: false. default: U8. option: U8, S8, U16, S16, FP32.
            "output_data_type": "U8",
            // quantize weight type for Conv. type: enum. required: false. default: S8. option: S8, FP32.
            "weight_data_type": "S8"
          },
          {
            // set quantize precision by operator type. type: string. required: must choose between `layer_name` and `op_type` and `layer_names` and `op_types`. default: .
            "op_type": "MaxPool",
            // quantize data type. type: enum. required: false. default: U8. option: U8, S8, U16, S16, FP32.
            "data_type": "U8"
          },
          {
            // set layer quantize precision by layers name. type: enum. required: must choose between `layer_name` and `op_type` and `layer_names` and `op_types`. default: [].
            "layer_names": ["Conv_2"],
            // quantize data type. type: enum. required: false. default: U8. option: U8, S8, U16, S16, FP32.
            "data_type": "U8",
            // quantize data type for Conv. type: enum. required: false. default: U8. option: U8, S8, U16, S16, FP32.
            "output_data_type": "U8",
            // quantize weight type for Conv. type: enum. required: false. default: S8. option: S8, FP32.
            "weight_data_type": "S8"
          },
          {
            // set quantize precision by operator types. type: enum. required: must choose between `layer_name` and `op_type` and `layer_names` and `op_types`. default: [].
            "op_types": ["Gemm"],
            // quantize data type. type: enum. required: false. default: U8. option: U8, S8, U16, S16, FP32.
            "data_type": "U8"
          },
          {
            // start tensor names of subgraph quantization config. type: string array. required: false. default: [].
            "start_tensor_names": ["13"],
            // end tensor names of subgraph quantization config. type: string array. required: false. default: [].
            "end_tensor_names": ["15"],
            // quantize data type. type: enum. required: false. default: U8. option: U8, S8, U16, S16, FP32.
            "data_type": "U16"
          }
        ],
        // quantize calibration method. type: enum. required: false. default: MinMax. option: MinMax, Percentile, MSE, KL.
        "calibration_method": "MinMax",
        // enable quantization precision analysis. type: bool. required: false. default: false.
        "precision_analysis": false,
        // precision analysis method. type: enum. required: false. default: PerLayer. option: PerLayer, EndToEnd.
        "precision_analysis_method": "PerLayer",
        // precision analysis mode. type: enum. required: false. default: Reference. option: Reference, NPUBackend.
        "precision_analysis_mode": "Reference",
        // input sample data dir for precision analysis. type: string. required: false. default: .
        "input_sample_dir": "",
        // enable highest mix precision quantization. type: bool. required: false. default: false.
        "highest_mix_precision": false,
        // conv bias data type. type: enum. required: false. default: S32. option: S32, FP32.
        "conv_bias_data_type": "S32",
        // LayerNormalization scale data type. type: enum. required: false. default: FP32. option: FP32, S32, U32.
        "ln_scale_data_type": "FP32",
        // refine weight threshold, should be a legal float number, like 1e-6. -1 means disable this feature. type: float. required: false. default: 1e-6. limitation: 0 or less than 0.0001.
        "refine_weight_threshold": 1e-6,
        // enalbe smooth quant strategy. type: bool. required: false. default: false.
        "enable_smooth_quant": false,
        // smooth quant threshold. The larger the threshold, the more operators will be involved in performing SmoothQuant. limitation: 0~1.
        "smooth_quant_threshold": 2e-1,
        // smooth quant strength, a well-balanced point to evenly split the quantization difficulty.
        "smooth_quant_strength": 6e-1,
        // tranformer opt level. type: int. required: false. default: 0. limitation: 0~2.
        "transformer_opt_level": 0,
        // quant check level, 0: no check; 1: check node dtype. type: int. required: false. default: 0.
        "check": 0,
        // refine weight scale and input scale, type: bool. required: false. default: false.
        "disable_auto_refine_scale": false,
        // enable easyquant; type bool. required: false. default: false.
        "enable_easy_quant": false,
        // disable quant optimization; type bool. required: false. default: false.
        "disable_quant_optimization": false,
        // enable brecq quantize strategy; type bool. required: false. default: false.
        "enable_brecq": false,
        // enable lsq quantize strategy; type bool. required: false. default: false.
        "enable_lsq": false,
        // enable adaround quantize strategy; type bool. required: false. default: false.
        "enable_adaround": false,
        // finetune epochs when enable finetune algorithm; type int32. required: false. default: 500.
        "finetune_epochs": 500,
        // finetune split block size when enable finetune algorithm; type int32. required: false. default: 4.
        "finetune_block_size": 4,
        // finetune batch size when enable finetune algorithm; type int32. required: false. default: 1.
        "finetune_batch_size": 1,
        // learning rate when enable finetune algorithm; type float. required: false. default: 1e-3.
        "finetune_lr": 1e-3,
        // device for quant calibration. type: string. required: false. default: cpu. option: cpu, cuda:0, cuda:1, ..., cuda:7.
        "device": "cpu"
      },
      "input_processors": [
        {
          // input tensor name in origin model. "DEFAULT" means processor for all input tensors. type: string. required: true.
          "tensor_name": "input",
          // input tensor format in origin model. type: enum. required: false. default: AutoColorSpace. option: AutoColorSpace, BGR, RGB, GRAY.
          "tensor_format": "AutoColorSpace",
          // input tensor layout in origin model. type: enum. required: false. default: NCHW. option: NHWC, NCHW.
          "tensor_layout": "NCHW",
          // input format in runtime. type: enum. required: false. default: AutoColorSpace. option: AutoColorSpace, GRAY, BGR, RGB, YUYV422, UYVY422, YUV420SP, YVU420SP, RAW.
          "src_format": "AutoColorSpace",
          // input layout in runtime; if `src_format` is YUV/YVU, `src_layout` will be changed to NHWC. type: enum. required: false. default: NCHW. option: NHWC, NCHW.
          "src_layout": "NHWC",
          // input data type in runtime. type: enum. required: false. default: FP32. option: U8, S8, U16, S16, U32, S32, FP16, FP32.
          "src_dtype": "U8",
          // extra compiler shapes for this input. src_extra_shapes size of every input should be the same. shape at the same index of every input will be treated as a input group which can inference independently at runtime. type: list of Shape. required: false. default [].
          "src_extra_shapes": [],
          // color space mode. type: enum. required: false. default: NoCSC. option: NoCSC, Matrix, FullRange, LimitedRange.
          "csc_mode": "NoCSC",
          // color space conversion matrix, 12 elements array that represents a 3x4 matrix. type: float array. required: false. default: [].
          "csc_mat": [1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4],
          // mean parameter of normlization in runtime. type: float array. required: false. default: same with ${quant.input_configs.calibration_mean}.
          "mean": [],
          // std parameter of normlization in runtime. type: float array. required: false. default: same with ${quant.input_configs.calibration_std}.
          "std": [],
          // list containing the number of start and end pad values for axis when padding. type: int32 array. required: false. default: [].
          "padding": [],
          // padding mode. type: string. required: false. default: constant.
          "padding_mode": "constant",
          // padding constant value. type: int32. required: false. default: 0.
          "padding_constant_value": 0,
          // list containing the number of start and end pad values for axis when slicing. type: int32 array. required: false. default: [].
          "slicing": []
        }
      ],
      "output_processors": [
        {
          // output tensor name in origin model. "DEFAULT" means processor for all output tensors. type: string. required: true.
          "tensor_name": "output",
          // permute the output tensor. type: int32 array. required: false. default: [].
          "dst_perm": [0, 1],
          // output data type. type: enum. required: false. default: FP32. option: FP32, U8.
          "output_dtype": "FP32"
        }
      ],
      "const_processors": [
        {
          // const tensor name in origin model. type: string. required: true.
          "name": "fc2.bias",
          // const tensor data array. type: list of double. required: false.
          "data": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
          // const tensor data file path, support .bin / .npy / .txt. type: string. required: false.
          "data_path": "replaced_data_file_path"
        }
      ],
      "quant_op_processors": [
        {
          // operator name in origin model. type: string. required: true.
          "op_name": "MaxPool_3",
          // operator attributes to be patched. type: dict. default: {}. required: true.
          "attrs": {
            "ceil_mode": 0
          }
        },
        {
          "op_name": "Flatten_4", // AxReshape
          "attrs": {
            "shape": [0, 800]
          }
        }
      ],
      "compiler": {
        // static batch sizes. type: int array. required: false. default: [].
        "static_batch_sizes": [],
        // max dynamic batch. type: int, required: false. default: 0.
        "max_dynamic_batch_size": 0,
        // ddr bandwidth limit in GB, 0 means no limit. type: int. required: false. default: 0.
        "ddr_bw_limit": 0,
        // disable ir fix, only work in multi-batch compilation. type: bool. required: false. default: false.
        "disable_ir_fix": false,
        // compiler check level, 0: no check; 1: assert all close; 2: assert all equal; 3: check cosine simularity. type: int. required: false. default: 0.
        "check": 0,
        // dump npu perf information for profiling. type: bool. required: false. default: false.
        "npu_perf": false,
        // compiler check mode, CheckOutput: only check model output; CheckPerLayer: check model intermediate tensor and output. type: enum. required: false. default: CheckOutput. option: CheckOutput, CheckPerLayer.
        "check_mode": "CheckOutput",
        // relative tolerance when check level is 1. type: float. required: false. default: 1e-5.
        "check_rtol": 1e-5,
        // absolute tolerance when check level is 1. type: float. required: false. default: 0.
        "check_atol": 0,
        // cosine simularity threshold when check level is 3. type: float. required: false. default: 0.999.
        "check_cosine_simularity": 0.999,
        // tensor black list for per layer check, support regex. type: list of string. required: false. default: [].
        "check_tensor_black_list": [],
        // input sample data dir for compiler check. type: string. required: false. default: .
        "input_sample_dir": "",
        // enable slice mode scheduler. type: bool. required: false. default: false.
        "enable_slice_mode": false,
        // enable tile mode scheduler. type: bool. required: false. default: false.
        "enable_tile_mode": false,
        // enable data soft compression. type: bool. required: false. default: false.
        "enable_data_soft_compression": false
      }
    }


.. _config_define:


------------------------------------
Quantitative parameter description
------------------------------------

- ``tensor_name`` in ``input_configs`` needs to be set according to the actual input/output node name of the model.
- ``tensor_name`` in ``input_configs`` can be set to ``DEFAULT`` to indicate that the quantization configuration applies to all inputs.
- The color space of the model input is expressed by the ``tensor_format`` parameter in the preprocessing ``input_processors`` configuration.
- When the tool chain reads the quantization calibration set, it will automatically convert the color space of the calibration set data according to the ``tensor_format`` parameter in ``input_processors``.
- The ``layer_name`` and ``op_type`` options in ``layer_configs`` cannot be configured at the same time.
- ``transformer_opt_level`` sets optimization options for the ``Transformer`` model.

.. _quant_precision_analysis_config_define:

------------------------------------------------------------
Quantitative precision analysis parameter description
------------------------------------------------------------

- Precision analysis calculation method, ``precision_analysis_mode`` field.

    - ``Reference`` can run all models supported by the compiler (supports models including CPU and NPU subgraphs), but the calculation results will have a small error compared to the final board results (basically the difference is within plus or minus 1, and no systematic errors).
    - ``NPUBackend`` can run models containing only NPU subgraphs, but the calculation results are bit aligned with the upper board results.

- Precision analysis method, ``precision_analysis_method`` field.

    - ``PerLayer`` means that each layer uses the layer input corresponding to the floating point model, and calculates the similarity between the output of each layer and the output of the floating point model.
    - ``EndToEnd`` means that the first layer adopts floating point model input, then simulates the complete model, and calculates the similarity between the final output result and the floating point model output.

.. _processing_arg_details:

------------------------------------------------------------
Preprocessing and postprocessing parameter description
------------------------------------------------------------

- ``input_processors`` / ``output_processors`` configuration instructions

     - ``tensor_name`` needs to be set according to the actual input/output node name of the model.
     - ``tensor_name`` can be set to ``DEFAULT`` to indicate that the configuration applies to all inputs or outputs.
     - Parameters prefixed with ``tensor_`` represent the input and output attributes in the original model.
     - Parameters prefixed with ``src_`` represent the actual input and output attributes at runtime.
     - The tool chain will automatically add operators according to the user's configuration to complete the conversion between runtime input and output and the original model input and output.

         - For example: when ``tensor_layout`` is ``NCHW`` and ``src_layout`` is ``NHWC``, the tool chain will automatically add a ``perm`` attribute of [0, 3, 1, 2] before the original model input of the ``Transpose`` operator.

- Color space conversion preprocessing

     - When ``csc_mode`` is ``LimitedRange`` or ``FullRange`` and ``src_format`` is ``YUV color space``, the toolchain will add it before the original input according to the built-in template parameters. A color space conversion operator, the ``csc_mat`` configuration is invalid at this time;
     - When ``csc_mode`` is ``Matrix`` and ``src_format`` is ``YUV color space``, the toolchain will add a ``csc_mat`` matrix before the original input according to the user-configured ``csc_mat`` matrix color space conversion operator to convert input YUV data into BGR or RGB data required for model calculation at runtime;
     - When ``csc_mode`` is ``Matrix``, the calculation process is to first uniformly convert the ``YUV / YVU color space`` input into ``YUV444`` format, and then multiply by ``csc_mat`` coefficient matrix.
     - When ``csc_mode`` is ``Matrix``, the value range of ``bias`` (csc_mat[3] / csc_mat[7] / csc_mat[11]) is (-9, 8). The remaining parameters (csc_mat[0-2] / csc_mat[4-6] / csc_mat[8-10]) have a value range of (-524289, 524288).

- Normalization preprocessing

     - The ``mean`` / ``std`` parameters in ``input_processors`` default to the value configured by the user in the ``calibration_mean`` / ``calibration_std`` parameter in the quantization configuration.
     - If the user wishes to use different normalization parameters at runtime, the ``mean`` / ``std`` parameters in the explicit configuration can be used to override the default values.

- Pad and Slice operations in data preprocessing

    Configuration example:

    .. code-block:: shell

        {
          ...
          "input_processors": [
            {
              "slicing": [0, 0, 0, 0, 0, 1, 0, 1]
            }
          ],
          ...
        }

    - ``padding`` This field indicates the length of the start and end of each axis that should be padded when data preprocessing pads a specific axis. It is represented in the form of a 32-bit integer array. If not set, the default value, which is an empty list, is used, indicating no padding.
    - ``padding_mode`` This field specifies the mode of padding. It is a string type, and the possible values ​​determine how the padding value is generated. The default value is "constant", indicating that a constant value is used for padding. Currently, only "constant" mode padding is supported.
    - ``padding_constant_value`` This field specifies the constant value used when the padding mode is "constant". It is a 32-bit integer. It represents a fixed value used for padding. The default value is 0.
    - ``slicing`` This field indicates the length of the start and end of each axis that should be sliced ​​when data preprocessing slices a specific axis. It is represented in the form of a 32-bit integer array. If not set, the default value, which is an empty list, is used, indicating no slicing.

------------------------------------
proto configuration definition
------------------------------------

.. code-block:: shell

    syntax = "proto3";
    
    package common;
    
    enum ColorSpace {
      AutoColorSpace = 0;
      GRAY = 1;
      BGR = 2;
      RGB = 3;
      RGBA = 4;
      YUV420SP = 6;   // Semi-Planner, NV12
      YVU420SP = 7;   // Semi-Planner, NV21
      YUYV422 = 8;     // Planner, YUYV
      UYVY422 = 9;     // Planner, UYVY
    }
    
    enum Layout {
      DefaultLayout = 0;
      NHWC = 1;
      NCHW = 2;
    }
    
    enum DataType {
      DefaultDataType = 0;
      U8 = 1;
      S8 = 2;
      U16 = 3;
      S16 = 4;
      U32 = 5;
      S32 = 6;
      U64 = 7;
      S64 = 8;
      FP16 = 9;
      FP32 = 10;
    }
    
    enum NPUMode {
      NPU1 = 0;
      NPU2 = 1;
      NPU3 = 2;
    }
    
    enum HardwareType {
      AX650 = 0;
      AX620E = 1;
      M76H = 2;
    }

.. code-block:: shell

    syntax = "proto3";
    
    import "path/to/common.proto";
    import "google/protobuf/struct.proto";
    
    package pulsar2.build;
    
    enum ModelType {
      ONNX = 0;
      QuantAxModel = 1;
      QuantONNX = 3;
    }
    
    enum QuantMethod {
      MinMax = 0;
      Percentile = 1;
      MSE = 2;
    }
    
    enum PrecisionAnalysisMethod {
      PerLayer = 0;
      EndToEnd = 1;
    }
    
    enum PrecisionAnalysisMode {
      Reference = 0;
      NPUBackend = 1;
    }
    
    enum CheckMode {
      CheckOutput = 0;
      CheckPerLayer = 1;
    }
    
    enum DataFormat {
      Image = 0;
      Numpy = 1;
      Binary = 2;
      NumpyObject = 3;
    }
    
    enum CSCMode {
      NoCSC = 0;
      Matrix = 1;
      FullRange = 2;
      LimitedRange = 3;
    }

    enum ScheduleStrategy {
      Tile = 0;
      Slice = 1;
    }
    
    enum MatchMode {
      Op = 0;
      Tensor = 1;
    }
    
    message Node {
      repeated string types = 1;
      repeated string inputs = 2;
      repeated string outputs = 3;
      .google.protobuf.Struct properties = 4;
    }
    
    message Graph {
      repeated Node nodes = 1;
      ScheduleStrategy type = 2;
      MatchMode match_mode = 3;
    }
    
    message InputQuantConfig {
      // input tensor name in origin model. "DEFAULT" means input config for all input tensors. type: string. required: true.
      string tensor_name = 1;
      // quantize calibration dataset archive file path. type: string. required: true. limitation: tar, tar.gz, zip.
      string calibration_dataset = 2;
      // quantize calibration data format. type: enum. required: false. default: Image. option: Image, Numpy, Binary, NumpyObject.
      DataFormat calibration_format = 3;
      // quantize calibration data size is min(${calibration_size}, size of ${calibration_dataset}), "-1" means load all dataset. type: int. required: false. default: 32.
      int32 calibration_size = 4;
      // quantize mean parameter of normlization. type: float array. required: false. default: [].
      repeated float calibration_mean = 5;
      // quantize std parameter of normlization. type: float array. required: false. default: [].
      repeated float calibration_std = 6;
    }
    
    message LayerConfig {
      // set layer quantize precision. type: string. required: must choose between `layer_name` and `op_type` and `layer_names` and `op_types`. default: .
      string layer_name = 1;
    
      // set quantize precision by operator type. type: string. required: must choose between `layer_name` and `op_type` and `layer_names` and `op_types`. default: .
      string op_type = 2;
    
      // start tensor names of subgraph quantization config. type: string array. required: false. default: [].
      repeated string start_tensor_names = 3;
      // end tensor names of subgraph quantization config. type: string array. required: false. default: [].
      repeated string end_tensor_names = 4;
    
      // quantize data type. type: enum. required: false. default: U8. option: U8, S8, U16, S16, FP32.
      common.DataType data_type = 5;
    
      // quantize weight type for Conv. type: enum. required: false. default: S8. option: S8, FP32.
      common.DataType weight_data_type = 6;
    
      // set layer quantize precision by layers name. type: enum. required: must choose between `layer_name` and `op_type` and `layer_names` and `op_types`. default: [].
      repeated string layer_names = 7;
    
      // set quantize precision by operator types. type: enum. required: must choose between `layer_name` and `op_type` and `layer_names` and `op_types`. default: [].
      repeated string op_types = 8;
    
      // quantize data type for Conv. type: enum. required: false. default: U8. option: U8, S8, U16, S16, FP32.
      common.DataType output_data_type = 10;
    }
    
    message OnnxOptimizeOption {
      // disable onnx optimization. type: bool. required: false. default: false.
      bool disable_onnx_optimization = 1;
      // enable onnx simplify by https://github.com/daquexian/onnx-simplifier. type: bool. required: false. default: false.
      bool enable_onnxsim = 2;
      // enable model check. type: bool. required: false. default: false.
      bool model_check = 3;
      // disable transformation check. type: bool. required: false. default: false.
      bool disable_transformation_check = 4;
      // save tensors data to optimize memory footprint. type: bool. required: false. default: false.
      bool save_tensors_data = 5;
    }
    
    message QuantConfig {
      repeated InputQuantConfig input_configs = 1;
      repeated LayerConfig layer_configs = 2;
    
      // quantize calibration method. type: enum. required: false. default: MinMax. option: MinMax, Percentile, MSE, KL.
      QuantMethod calibration_method = 3;
      // enable quantization precision analysis. type: bool. required: false. default: false.
      bool precision_analysis = 4;
      // precision analysis method. type: enum. required: false. default: PerLayer. option: PerLayer, EndToEnd.
      PrecisionAnalysisMethod precision_analysis_method = 5;
      // precision analysis mode. type: enum. required: false. default: Reference. option: Reference, NPUBackend.
      PrecisionAnalysisMode precision_analysis_mode = 6;
      // enable highest mix precision quantization. type: bool. required: false. default: false.
      bool highest_mix_precision = 7;
      // conv bias data type. type: enum. required: false. default: S32. option: S32, FP32.
      common.DataType conv_bias_data_type = 8;
      // refine weight threshold, should be a legal float number, like 1e-6. -1 means disable this feature. type: float. required: false. default: 1e-6. limitation: 0 or less than 0.0001.
      float refine_weight_threshold = 9;
      // enalbe smooth quant strategy. type: bool. required: false. default: false.
      bool enable_smooth_quant = 10;
      // smooth quant threshold. The larger the threshold, the more operators will be involved in performing SmoothQuant. limitation: 0~1.
      float smooth_quant_threshold = 20;
      // smooth quant strength, a well-balanced point to evenly split the quantization difficulty.
      float smooth_quant_strength = 30;
      // tranformer opt level. type: int. required: false. default: 0. limitation: 0~2.
      int32 transformer_opt_level = 40;
      // input sample data dir for precision analysis. type: string. required: false. default: .
      string input_sample_dir = 50;
      // LayerNormalization scale data type. type: enum. required: false. default: FP32. option: FP32, S32, U32.
      common.DataType ln_scale_data_type = 60;
      // quant check level, 0: no check; 1: check node dtype. type: int. required: false. default: 0.
      int32 check = 70;
      // refine weight scale and input scale, type: bool. required: false. default: false.
      bool disable_auto_refine_scale = 80;
      // enable easyquant; type bool. required: false. default: false.
      bool enable_easy_quant = 90;
      // disable quant optimization; type bool. required: false. default: false.
      bool disable_quant_optimization = 100;
      // enable brecq quantize strategy; type bool. required: false. default: false.
      bool enable_brecq = 110;
      // enable lsq quantize strategy; type bool. required: false. default: false.
      bool enable_lsq = 120;
      // enable adaround quantize strategy; type bool. required: false. default: false.
      bool enable_adaround = 130;
      // finetune epochs when enable finetune algorithm; type int32. required: false. default: 500.
      int32 finetune_epochs = 140;
      // finetune split block size when enable finetune algorithm; type int32. required: false. default: 4.
      int32 finetune_block_size = 150;
      // finetune batch size when enable finetune algorithm; type int32. required: false. default: 1.
      int32 finetune_batch_size = 160;
      // learning rate when enable finetune algorithm; type float. required: false. default: 1e-3.
      float finetune_lr = 170;
      // device for quant calibration. type: string. required: false. default: cpu. option: cpu, cuda:0, cuda:1, ..., cuda:7.
      string device = 180;
    }
    
    message InputProcessor {
      // input tensor name in origin model. "DEFAULT" means processor for all input tensors. type: string. required: true.
      string tensor_name = 1;
    
      // input tensor format in origin model. type: enum. required: false. default: AutoColorSpace. option: AutoColorSpace, BGR, RGB, GRAY.
      common.ColorSpace tensor_format = 2;
      // input tensor layout in origin model. type: enum. required: false. default: NCHW. option: NHWC, NCHW.
      common.Layout tensor_layout = 3;
    
      // input format in runtime. type: enum. required: false. default: AutoColorSpace. option: AutoColorSpace, GRAY, BGR, RGB, YUYV422, UYVY422, YUV420SP, YVU420SP, RAW.
      common.ColorSpace src_format = 4;
      // input layout in runtime; if `src_format` is YUV/YVU, `src_layout` will be changed to NHWC. type: enum. required: false. default: NCHW. option: NHWC, NCHW.
      common.Layout src_layout = 5;
      // input data type in runtime. type: enum. required: false. default: FP32. option: U8, S8, U16, S16, U32, S32, FP16, FP32.
      common.DataType src_dtype = 6;
    
      // extra compiler shapes for this input. src_extra_shapes size of every input should be the same. shape at the same index of every input will be treated as a input group which can inference independently at runtime. type: list of Shape. required: false. default [].
      repeated common.Shape src_extra_shapes = 11;
    
      // color space mode. type: enum. required: false. default: NoCSC. option: NoCSC, Matrix, FullRange, LimitedRange.
      CSCMode csc_mode = 7;
      // color space conversion matrix, 12 elements array that represents a 3x4 matrix. type: float array. required: false. default: [].
      repeated float csc_mat = 8;
      // mean parameter of normlization in runtime. type: float array. required: false. default: same with ${quant.input_configs.calibration_mean}.
      repeated float mean = 9;
      // std parameter of normlization in runtime. type: float array. required: false. default: same with ${quant.input_configs.calibration_std}.
      repeated float std = 10;
      // list containing the number of start and end pad values for axis when padding. type: int32 array. required: false. default: [].
      repeated int32 padding = 20;
      // padding mode. type: string. required: false. default: constant.
      string padding_mode = 21;
      // padding constant value. type: int32. required: false. default: 0.
      int32 padding_constant_value = 22;
      // list containing the number of start and end pad values for axis when slicing. type: int32 array. required: false. default: [].
      repeated int32 slicing = 30;
    }
    
    message OutputProcessor {
      // output tensor name in origin model. "DEFAULT" means processor for all output tensors. type: string. required: true.
      string tensor_name = 1;
    
      common.Layout tensor_layout = 2;
    
      // permute the output tensor. type: int32 array. required: false. default: [].
      repeated int32 dst_perm = 3;
    
      // output data type. type: enum. required: false. default: FP32. option: FP32, U8.
      common.DataType output_dtype = 4;
    }
    
    message OpProcessor {
      // operator name in origin model. type: string. required: true.
      string op_name = 1;
    
      // operator attributes to be patched. type: dict. default: {}. required: true.
      .google.protobuf.Struct attrs = 2;
    }
    
    message ConstProcessor {
      // const tensor name in origin model. type: string. required: true.
      string name = 1;
    
      // const tensor data array. type: list of double. required: false.
      repeated double data = 2;
    
      // const tensor data file path, support .bin / .npy / .txt. type: string. required: false.
      string data_path = 3;
    }
    
    message CompilerConfig {
      // static batch sizes. type: int array. required: false. default: [].
      repeated int32 static_batch_sizes = 1;
      // max dynamic batch. type: int, required: false. default: 0.
      optional int32 max_dynamic_batch_size = 2;
      // ddr bandwidth limit in GB, 0 means no limit. type: int. required: false. default: 0.
      optional float ddr_bw_limit = 12;
      // disable ir fix, only work in multi-batch compilation. type: bool. required: false. default: false.
      optional bool disable_ir_fix = 3;
      // compiler check level, 0: no check; 1: assert all close; 2: assert all equal; 3: check cosine simularity. type: int. required: false. default: 0.
      optional int32 check = 5;
      // dump npu perf information for profiling. type: bool. required: false. default: false.
      optional bool npu_perf = 6;
      // compiler check mode, CheckOutput: only check model output; CheckPerLayer: check model intermediate tensor and output. type: enum. required: false. default: CheckOutput. option: CheckOutput, CheckPerLayer.
      optional CheckMode check_mode = 7;
      // relative tolerance when check level is 1. type: float. required: false. default: 1e-5.
      optional float check_rtol = 8;
      // absolute tolerance when check level is 1. type: float. required: false. default: 0.
      optional float check_atol = 9;
      // cosine simularity threshold when check level is 3. type: float. required: false. default: 0.999.
      optional float check_cosine_simularity = 10;
      // tensor black list for per layer check, support regex. type: list of string. required: false. default: [].
      repeated string check_tensor_black_list = 11;
      // enable slice mode scheduler. type: bool. required: false. default: false.
      optional bool enable_slice_mode = 13;
      // enable tile mode scheduler. type: bool. required: false. default: false.
      optional bool enable_tile_mode = 52;
      // enable data soft compression. type: bool. required: false. default: false.
      optional bool enable_data_soft_compression = 14;
      // input sample data dir for compiler check. type: string. required: false. default: .
      optional string input_sample_dir = 30;
      repeated Graph compiler_group_patterns = 15;
    
      // sub compiler configs. type: CompilerConfig. required: false. default: [].
      repeated CompilerConfig sub_configs = 60;
      // start tensor names of subgraph compiler config, only can be configured in sub compiler config. type: string array. required: false. default: [].
      repeated string start_tensor_names = 50;
      // end tensor names of subgraph compiler config, only can be configured in sub compiler config. type: string array. required: false. default: [].
      repeated string end_tensor_names = 51;
    }
    
    
    message BuildConfig {
      // input model file path. type: string. required: true.
      string input = 1;
      // axmodel output directory. type: string. required: true.
      string output_dir = 2;
      // rename output axmodel. type: string. required: false. default: compiled.axmodel.
      string output_name = 3;
      // temporary data output directory. type: string. required: false. default: same with ${output_dir}.
      string work_dir = 4;
    
      // input model type. type: enum. required: false. default: ONNX. option: ONNX, QuantAxModel, QuantONNX.
      ModelType model_type = 5;
    
      // target hardware. type: enum. required: false. default: AX650. option: AX650, AX620E, M76H.
      common.HardwareType target_hardware = 6;
      // npu mode. while ${target_hardware} is AX650, npu mode can be NPU1 / NPU2 / NPU3. while ${target_hardware} is AX620E, npu mode can be NPU1 / NPU2. type: enum. required: false. default: NPU1.
      common.NPUMode npu_mode = 7;
    
      // modify model input shape of input model, this feature will take effect before the `input_processors` configuration. format: input1:1x3x224x224;input2:1x1x112x112. type: string. required: false. default: .
      string input_shapes = 8;
    
      OnnxOptimizeOption onnx_opt = 10;
    
      QuantConfig quant = 20;
    
      repeated InputProcessor input_processors = 31;
      repeated OutputProcessor output_processors = 32;
      repeated ConstProcessor const_processors = 33;
      repeated OpProcessor op_processors = 34;
      repeated OpProcessor quant_op_processors = 35;
    
      CompilerConfig compiler = 40;
    }
