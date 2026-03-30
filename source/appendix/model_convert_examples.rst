=========================
Model conversion examples
=========================

This section provides ``pulsar2 build`` conversion examples for typical models, including complete configuration files, conversion commands, real logs, and model input/output descriptions. All examples are based on the ``AX650`` platform and Pulsar2 5.1.

.. note::

    - The models and configuration files in this section are from `AXERA-TECH HuggingFace <https://huggingface.co/AXERA-TECH>`_
    - Before conversion, make sure the original model has been optimized using ``onnxsim``
    - The input/output tensor names must match the actual ONNX definitions. You can check them via ``onnx inspect --io model.onnx``

.. _convert_yolov5s:

----------------------------------------------
YOLOv5s (object detection)
----------------------------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``YOLOv5s`` is a real-time object detection model released by Ultralytics. It uses a CSPDarknet backbone and is suitable for real-time detection scenarios.

- **HuggingFace**: `AXERA-TECH/YOLOv5 <https://huggingface.co/AXERA-TECH/YOLOv5>`_
- **Model source**: `ultralytics/yolov5 <https://github.com/ultralytics/yolov5>`_
- **AxSamples**: `ax-samples <https://github.com/AXERA-TECH/ax-samples/blob/main/examples/ax650/ax_yolov5s_steps.cc>`__ / `axcl-samples <https://github.com/AXERA-TECH/axcl-samples/blob/main/examples/axcl/ax_yolov5s_steps.cc>`__

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``yolov5_build.json``:

.. code-block:: json

    {
      "model_type": "ONNX",
      "npu_mode": "NPU1",
      "quant": {
        "input_configs": [
          {
            "tensor_name": "images",
            "calibration_dataset": "calib-cocotest2017.tar",
            "calibration_size": 32,
            "calibration_mean": [0, 0, 0],
            "calibration_std": [255.0, 255.0, 255.0]
          }
        ],
        "calibration_method": "MinMax",
        "precision_analysis": false
      },
      "input_processors": [
        {
          "tensor_name": "images",
          "tensor_format": "RGB",
          "src_format": "BGR",
          "src_dtype": "U8",
          "src_layout": "NHWC"
        }
      ],
      "output_processors": [
        {
          "tensor_name": "/model.24/m.0/Conv_output_0",
          "dst_perm": [0, 2, 3, 1]
        },
        {
          "tensor_name": "/model.24/m.1/Conv_output_0",
          "dst_perm": [0, 2, 3, 1]
        },
        {
          "tensor_name": "/model.24/m.2/Conv_output_0",
          "dst_perm": [0, 2, 3, 1]
        }
      ],
      "compiler": {
        "check": 0
      }
    }

.. attention::

    The ``tensor_name`` fields in ``output_processors`` are the output names of the three detection heads of YOLOv5s. They may vary across model versions. Please use ``onnx inspect --io model.onnx`` to check the actual tensor names. ``dst_perm`` converts outputs from ``NCHW`` to ``NHWC`` layout, which makes post-processing easier.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compile and execute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    pulsar2 build --target_hardware AX650 --input yolov5s-cut.onnx --output_dir output --config yolov5_build.json

^^^^^^^^^^^^^^^^^^^^^^^^^^^
log reference information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    +----------------------------------+----------------------------+
    |            Model Name            |         OnnxModel          |
    +----------------------------------+----------------------------+
    |            Model Info            | Op Set: 17 / IR Version: 8 |
    +----------------------------------+----------------------------+
    |            IN: images            | float32: (1, 3, 640, 640)  |
    | OUT: /model.24/m.0/Conv_output_0 | float32: (1, 255, 80, 80)  |
    | OUT: /model.24/m.1/Conv_output_0 | float32: (1, 255, 40, 40)  |
    | OUT: /model.24/m.2/Conv_output_0 | float32: (1, 255, 20, 20)  |
    +----------------------------------+----------------------------+
    |               Add                |             7              |
    |              Concat              |             13             |
    |               Conv               |             60             |
    |             MaxPool              |             3              |
    |               Mul                |             57             |
    |              Resize              |             2              |
    |             Sigmoid              |             57             |
    +----------------------------------+----------------------------+
    |            Model Size            |          27.56 MB          |
    +----------------------------------+----------------------------+
    ...
    Calibration Progress(Phase 1): 100%|██████████| 32/32 [00:17<00:00,  1.79it/s]
    ...
    --------- Network Snapshot ---------
    Num of Op:                    [142]
    Num of Quantized Op:          [142]
    Num of Variable:              [269]
    Num of Quantized Var:         [269]
    ------- Quantization Snapshot ------
    Num of Quant Config:          [432]
    BAKED:                        [60]
    OVERLAPPED:                   [168]
    SLAVE:                        [9]
    ACTIVATED:                    [129]
    SOI:                          [6]
    PASSIVE_BAKED:                [60]
    Network Quantization Finished.
    ...
    tiling op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 147/147 0:00:00
    build op serially...   ━━━━━━━━━━━━━━━━━━━━━━━━━━ 649/649 0:00:02
    build op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1139/1139 0:00:00
    ...
    2026-03-23 19:44:00.890 | INFO     | yamain.command.build:compile_ptq_model:1365 - fuse 1 subgraph(s)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model input/output description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Direction
     - Tensor name
     - Dtype
     - Shape
     - Notes
   * - Input
     - images
     - UINT8
     - (1, 640, 640, 3)
     - BGR image, NHWC layout, letterbox preprocessing is required
   * - Output
     - /model.24/m.0/Conv_output_0
     - FLOAT32
     - (1, 80, 80, 255)
     - Large-scale feature map (detect small objects)
   * - Output
     - /model.24/m.1/Conv_output_0
     - FLOAT32
     - (1, 40, 40, 255)
     - Medium-scale feature map (detect medium objects)
   * - Output
     - /model.24/m.2/Conv_output_0
     - FLOAT32
     - (1, 20, 20, 255)
     - Small-scale feature map (detect large objects)

.. hint::

    On-board inference latency is about ``6.32 ms`` (AX650). For a complete on-board runtime example, please refer to `AXERA-TECH/YOLOv5 <https://huggingface.co/AXERA-TECH/YOLOv5>`_.

.. _convert_yolo11s:

----------------------------------------------
YOLO11s (object detection)
----------------------------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``YOLO11s`` is the latest generation YOLO detection model released by Ultralytics. It adopts an improved backbone and detection head design and provides better accuracy and speed compared with the previous generation.

- **HuggingFace**: `AXERA-TECH/YOLO11 <https://huggingface.co/AXERA-TECH/YOLO11>`_
- **Model source**: `ultralytics/ultralytics <https://github.com/ultralytics/ultralytics>`_
- **AxSamples**: `ax-samples <https://github.com/AXERA-TECH/ax-samples/blob/main/examples/ax650/ax_yolo11_steps.cc>`__ / `axcl-samples <https://github.com/AXERA-TECH/axcl-samples/blob/main/examples/axcl/ax_yolo11_steps.cc>`__

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``yolo11_build.json``:

.. code-block:: json

    {
      "model_type": "ONNX",
      "npu_mode": "NPU1",
      "quant": {
        "input_configs": [
          {
            "tensor_name": "images",
            "calibration_dataset": "calib-cocotest2017.tar",
            "calibration_size": 32,
            "calibration_mean": [0, 0, 0],
            "calibration_std": [255.0, 255.0, 255.0]
          }
        ],
        "calibration_method": "MinMax",
        "precision_analysis": false
      },
      "input_processors": [
        {
          "tensor_name": "images",
          "tensor_format": "BGR",
          "src_format": "BGR",
          "src_dtype": "U8",
          "src_layout": "NHWC"
        }
      ],
      "output_processors": [
        {
          "tensor_name": "/model.23/Concat_output_0",
          "dst_perm": [0, 2, 3, 1]
        },
        {
          "tensor_name": "/model.23/Concat_1_output_0",
          "dst_perm": [0, 2, 3, 1]
        },
        {
          "tensor_name": "/model.23/Concat_2_output_0",
          "dst_perm": [0, 2, 3, 1]
        }
      ],
      "compiler": {
        "check": 0
      }
    }

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compile and execute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    pulsar2 build --target_hardware AX650 --input yolo11s-cut.onnx --output_dir output --config yolo11_build.json

^^^^^^^^^^^^^^^^^^^^^^^^^^^
log reference information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    +----------------------------------+----------------------------+
    |            Model Name            |         OnnxModel          |
    +----------------------------------+----------------------------+
    |            Model Info            | Op Set: 17 / IR Version: 9 |
    +----------------------------------+----------------------------+
    |            IN: images            | float32: (1, 3, 640, 640)  |
    |  OUT: /model.23/Concat_output_0  | float32: (1, 144, 80, 80)  |
    | OUT: /model.23/Concat_1_output_0 | float32: (1, 144, 40, 40)  |
    | OUT: /model.23/Concat_2_output_0 | float32: (1, 144, 20, 20)  |
    +----------------------------------+----------------------------+
    |               Add                |             14             |
    |              Concat              |             20             |
    |               Conv               |             87             |
    |              MatMul              |             2              |
    |             MaxPool              |             3              |
    |               Mul                |             78             |
    |             Reshape              |             3              |
    |              Resize              |             2              |
    |             Sigmoid              |             77             |
    |             Softmax              |             1              |
    |              Split               |             10             |
    |            Transpose             |             2              |
    +----------------------------------+----------------------------+
    |            Model Size            |          36.03 MB          |
    +----------------------------------+----------------------------+
    ...
    Calibration Progress(Phase 1): 100%|██████████| 32/32 [00:25<00:00,  1.25it/s]
    ...
    --------- Network Snapshot ---------
    Num of Op:                    [222]
    Num of Quantized Op:          [222]
    Num of Variable:              [426]
    Num of Quantized Var:         [426]
    ------- Quantization Snapshot ------
    Num of Quant Config:          [693]
    BAKED:                        [88]
    OVERLAPPED:                   [295]
    SLAVE:                        [16]
    ACTIVATED:                    [190]
    SOI:                          [17]
    PASSIVE_BAKED:                [87]
    Network Quantization Finished.
    ...
    tiling op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 235/235 0:00:01
    build op serially...   ━━━━━━━━━━━━━━━━━━━━━━━ 1033/1033 0:00:05
    build op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1689/1689 0:00:00
    ...
    2026-03-23 19:45:20.303 | INFO     | yamain.command.build:compile_ptq_model:1365 - fuse 1 subgraph(s)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model input/output description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Direction
     - Tensor name
     - Dtype
     - Shape
     - Notes
   * - Input
     - images
     - UINT8
     - (1, 640, 640, 3)
     - BGR image, NHWC layout, letterbox preprocessing is required
   * - Output
     - /model.23/Concat_output_0
     - FLOAT32
     - (1, 80, 80, 144)
     - Large-scale feature map (detect small objects)
   * - Output
     - /model.23/Concat_1_output_0
     - FLOAT32
     - (1, 40, 40, 144)
     - Medium-scale feature map (detect medium objects)
   * - Output
     - /model.23/Concat_2_output_0
     - FLOAT32
     - (1, 20, 20, 144)
     - Small-scale feature map (detect large objects)

.. hint::

    Compared with YOLOv5, YOLO11 adopts an attention mechanism (including ``MatMul`` and ``Softmax`` operators). The model is larger but provides higher detection accuracy. On-board inference latency is about ``25 ms`` (AX650). For a complete on-board runtime example, please refer to `AXERA-TECH/YOLO11 <https://huggingface.co/AXERA-TECH/YOLO11>`_.

.. _convert_depth_anything_v2:

----------------------------------------------
Depth-Anything-V2 (monocular depth estimation)
----------------------------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Depth-Anything-V2`` is a monocular depth estimation model based on DINOv2. It takes a single RGB image as input and outputs a per-pixel depth map. This example uses the ViT-Small variant.

- **HuggingFace**: `AXERA-TECH/Depth-Anything-V2 <https://huggingface.co/AXERA-TECH/Depth-Anything-V2>`_
- **Model source**: `depth-anything/Depth-Anything-V2-Small <https://huggingface.co/depth-anything/Depth-Anything-V2-Small>`_
- **ONNX export reference**: `DepthAnythingV2 <https://github.com/AXERA-TECH/DepthAnythingV2.axera/tree/main/model_convert>`_

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``config.json`` (some ``layer_configs`` entries are omitted; please refer to the HuggingFace repository for the full configuration):

.. code-block::

    {
      "model_type": "ONNX",
      "npu_mode": "NPU3",
      "quant": {
        "input_configs": [
          {
            "tensor_name": "DEFAULT",
            "calibration_dataset": "calib-cocotest2017.tar",
            "calibration_size": 32,
            "calibration_mean": [123.675, 116.28, 103.53],
            "calibration_std": [58.395, 57.12, 57.375]
          }
        ],
        "calibration_method": "MinMax",
        "precision_analysis": true,
        "precision_analysis_method": "EndToEnd",
        "conv_bias_data_type": "FP32",
        "enable_smooth_quant": true,
        "disable_auto_refine_scale": true,
        "layer_configs":  [
          {
            "layer_name": "op_173:onnx.Mul_1",
            "data_type": "U16"
          },
          {
            "layer_name": "op_173:onnx.Softmax_0",
            "data_type": "U16"
          },
          {
            "layer_name": "op_173:onnx.MatMul_qkv_0",
            "data_type": "U16"
          },
          ...
        ]
      },
      "input_processors": [
        {
          "tensor_name": "DEFAULT",
          "tensor_format": "RGB",
          "src_format": "BGR",
          "src_dtype": "U8",
          "src_layout": "NHWC"
        }
      ],
      "compiler": {
        "check": 0
      }
    }

.. attention::

    - This model uses ``NPU3`` mode (3 cores) to fully utilize the AX650 NPU compute capability.
    - ``enable_smooth_quant`` is enabled to reduce the impact of outliers in Transformer blocks.
    - Many operators such as Softmax and MatMul are configured as ``U16`` in ``layer_configs`` to ensure quantization accuracy for the ViT model.
    - ``conv_bias_data_type`` is set to ``FP32`` to improve accuracy.
    - For the complete ``layer_configs`` (~50 items), please refer to `config.json in the HuggingFace repository <https://huggingface.co/AXERA-TECH/Depth-Anything-V2/blob/main/config.json>`_.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compile and execute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    pulsar2 build --target_hardware AX650 --input depth_anything_v2_vits.onnx --output_dir output --config config.json

^^^^^^^^^^^^^^^^^^^^^^^^^^^
log reference information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    +---------------+----------------------------+
    |  Model Name   |         OnnxModel          |
    +---------------+----------------------------+
    |  Model Info   | Op Set: 12 / IR Version: 7 |
    +---------------+----------------------------+
    |   IN: input   | float32: (1, 3, 518, 518)  |
    |  OUT: output  | float32: (1, 1, 518, 518)  |
    +---------------+----------------------------+
    |      Add      |            148             |
    |    Concat     |             1              |
    |     Conv      |             31             |
    | ConvTranspose |             2              |
    |      Div      |             37             |
    |      Erf      |             12             |
    |    Gather     |             36             |
    |    MatMul     |             72             |
    |      Mul      |             88             |
    |      Pow      |             25             |
    |  ReduceMean   |             50             |
    |     Relu      |             16             |
    |    Reshape    |             29             |
    |    Resize     |             5              |
    |     Slice     |             4              |
    |    Softmax    |             12             |
    |     Sqrt      |             25             |
    |      Sub      |             25             |
    |   Transpose   |             41             |
    +---------------+----------------------------+
    |  Model Size   |          94.26 MB          |
    +---------------+----------------------------+
    ...
    Enable Smooth Quant, this pass is used for outlier activation.
    ...
    Analysing Smooth Quantization Error(Phrase 1): 100%|██████████| 32/32 [00:51<00:00,  1.62s/it]
    Get Outlier Progress: 100%|██████████| 32/32 [01:15<00:00,  2.35s/it]
    ...
    Analysing Smooth Quantization Error(Phrase 2): 100%|██████████| 32/32 [00:51<00:00,  1.62s/it]
    ...
    --------- Network Snapshot ---------
    Num of Op:                    [792]
    Num of Quantized Op:          [792]
    Num of Variable:              [1552]
    Num of Quantized Var:         [1552]
    ------- Quantization Snapshot ------
    Num of Quant Config:          [2581]
    ...
    Network Quantization Finished.
    ...
    tiling op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 762/762 0:00:02
    build op serially...   ━━━━━━━━━━━━━━━━━━━━━━━ 1178/1178 0:00:11
    build op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1734/1734 0:00:00
    add ddr swap...   ━━━━━━━━━━━━━━━━━━━━━━━━━ 15821/15821 0:00:01
    ...
    2026-03-23 19:42:38.553 | INFO     | yamain.command.build:compile_ptq_model:1365 - fuse 1 subgraph(s)

.. note::

    The end-to-end conversion takes about ``8 minutes``. Smooth Quant analysis and per-layer precision comparison account for most of the time. If precision analysis is not required, set ``precision_analysis`` to ``false`` to speed up conversion.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model input/output description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Direction
     - Tensor name
     - Dtype
     - Shape
     - Notes
   * - Input
     - input
     - UINT8
     - (1, 518, 518, 3)
     - BGR image (runtime BGR input will be automatically converted to RGB), NHWC layout
   * - Output
     - output
     - FLOAT32
     - (1, 1, 518, 518)
     - Per-pixel depth map; larger values indicate farther distance

.. hint::

    On-board inference latency is about ``33 ms`` (AX650, NPU3 3-core mode). For a Python inference example, please refer to `AXERA-TECH/Depth-Anything-V2 <https://huggingface.co/AXERA-TECH/Depth-Anything-V2>`_. `pyaxengine <https://github.com/AXERA-TECH/pyaxengine>`_ is required.

.. _convert_cnclip:

----------------------------------------------
CN-CLIP (Chinese multimodal text encoder)
----------------------------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model overview
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Chinese-CLIP`` is a Chinese multimodal pre-trained model based on the CLIP framework. This example uses the **BERT text encoder** part (paired with ViT-L/14), which encodes Chinese text into embedding vectors for similarity calculation with image embeddings.

- **HuggingFace**: `AXERA-TECH/cnclip <https://huggingface.co/AXERA-TECH/cnclip>`_
- **Model source**: `OFA-Sys/Chinese-CLIP <https://github.com/OFA-Sys/Chinese-CLIP>`_
- **ONNX export reference**: `cnclip.axera <https://github.com/AXERA-TECH/cnclip.axera?tab=readme-ov-file#%E5%AF%BC%E5%87%BA%E6%A8%A1%E5%9E%8Bpytorch---onnx>`_
- **AxSamples**: `CLIP-ONNX-AX650-CPP <https://github.com/AXERA-TECH/CLIP-ONNX-AX650-CPP>`_

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cnclip_build.json``:

.. code-block:: json

    {
      "model_type": "ONNX",
      "npu_mode": "NPU1",
      "quant": {
        "input_configs": [
          {
            "tensor_name": "text",
            "calibration_dataset": "calib_text.tar",
            "calibration_format": "Numpy",
            "calibration_size": 32,
            "calibration_mean": [0],
            "calibration_std": [1]
          }
        ],
        "calibration_method": "MinMax",
        "precision_analysis": false,
        "transformer_opt_level": 1
      },
      "input_processors": [
        {
          "tensor_name": "text",
          "src_dtype": "S32",
          "src_layout": "NCHW"
        }
      ],
      "compiler": {
        "check": 0
      }
    }

.. attention::

    - This model is a **text encoder**. The input is a token-id sequence after tokenization, not an image.
    - ``calibration_format`` is set to ``Numpy``. The calibration data is a pre-tokenized numpy array (shape ``(1, 52)``, dtype ``int64``).
    - ``src_dtype`` is set to ``S32`` (signed 32-bit integer) for token-id input.
    - ``transformer_opt_level`` is set to 1 to enable Transformer-specific quantization optimizations.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compile and execute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    pulsar2 build --target_hardware AX650 --input cnclip_vit_l14_336px_bert_encoder.onnx --output_dir output --config cnclip_build.json

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
log reference information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    +---------------------------+----------------------------+
    |        Model Name         |         OnnxModel          |
    +---------------------------+----------------------------+
    |        Model Info         | Op Set: 14 / IR Version: 7 |
    +---------------------------+----------------------------+
    |         IN: text          |       int64: (1, 52)       |
    | OUT: unnorm_text_features |     float32: (1, 768)      |
    +---------------------------+----------------------------+
    |            Add            |            172             |
    |           Cast            |             3              |
    |         Constant          |            154             |
    |            Div            |             49             |
    |            Erf            |             12             |
    |          Gather           |             4              |
    |          MatMul           |             97             |
    |            Mul            |             50             |
    |            Pow            |             25             |
    |        ReduceMean         |             50             |
    |          Reshape          |             48             |
    |          Softmax          |             12             |
    |           Sqrt            |             25             |
    |            Sub            |             26             |
    |         Transpose         |             48             |
    +---------------------------+----------------------------+
    |        Model Size         |         390.12 MB          |
    +---------------------------+----------------------------+
    ...
    Transformer optimize level: 1
    ...
    Calibration Progress(Phase 1): 100%|██████████| 32/32 [00:11<00:00,  2.81it/s]
    ...
    --------- Network Snapshot ---------
    Num of Op:                    [312]
    Num of Quantized Op:          [308]
    Num of Variable:              [588]
    Num of Quantized Var:         [583]
    ------- Quantization Snapshot ------
    Num of Quant Config:          [949]
    BAKED:                        [89]
    OVERLAPPED:                   [452]
    ACTIVATED:                    [224]
    SOI:                          [61]
    PASSIVE_BAKED:                [72]
    FP32:                         [51]
    Network Quantization Finished.
    ...
    tiling op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 340/340 0:00:02
    build op serially...   ━━━━━━━━━━━━━━━━━━━━━━━━━━ 300/300 0:00:04
    build op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 386/386 0:00:00
    add ddr swap...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3859/3859 0:00:00
    ...
    2026-03-23 19:47:45.563 | INFO     | yamain.command.build:compile_ptq_model:1365 - fuse 1 subgraph(s)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model input/output description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Direction
     - Tensor name
     - Dtype
     - Shape
     - Notes
   * - Input
     - text
     - S32
     - (1, 52)
     - Token IDs after tokenization, max length 52
   * - Output
     - unnorm_text_features
     - FLOAT32
     - (1, 768)
     - Unnormalized text embedding vector

.. hint::

    For deployment, this model should be used together with the visual encoder: the visual encoder extracts image embeddings, and the text encoder extracts text embeddings. Image-text matching is then performed by cosine similarity. The tokenizer uses ``cn_vocab.txt`` (provided with the model). For an on-board runtime example, please refer to `CLIP-ONNX-AX650-CPP <https://github.com/AXERA-TECH/CLIP-ONNX-AX650-CPP>`_.
