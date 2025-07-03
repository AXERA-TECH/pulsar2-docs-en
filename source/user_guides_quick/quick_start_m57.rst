======================
Quick Start(M57)
======================

**This section applies to the following platforms:**

- M57

This section introduces the basic operations of ``ONNX`` model conversion, and uses the ``pulsar2`` tool to compile the ``ONNX`` model into the ``axmodel`` model. Please refer to the :ref:`《Development Environment Preparation》 <dev_env_prepare>` section to complete the development environment setup.
The example model in this section is the open source model ``MobileNetv2``.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pulsar2 toolchain command description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function commands in the ``Pulsar2`` toolchain start with ``pulsar2``. The commands that are most relevant to users are ``pulsar2 build``, ``pulsar2 run`` and ``pulsar2 version``.

* ``pulsar2 build`` is used to convert the ``onnx`` model to the ``axmodel`` format model

* ``pulsar2 run`` is used to run the simulation after the model is converted

* ``pulsar2 version`` can be used to view the version information of the current toolchain, which is usually required when reporting issues

.. code-block:: shell

    root@xxx:/data# pulsar2 --help
    usage: pulsar2 [-h] {version,build,run} ...
    
    positional arguments:
      {version,build,run}
    
    optional arguments:
      -h, --help           show this help message and exit

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model compilation configuration file description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mobilenet_v2_build_config.json`` in the ``/data/config/`` path shows:

.. code-block:: shell

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

.. attention::

    The ``tensor_name`` field in ``input_processors``, ``output_processors`` and ``quant`` nodes under ``input_configs`` needs to be set according to the actual input/output node name of the model, or it can be set to ``DEFAULT`` to indicate that the current configuration applies to all inputs or outputs.

    .. figure:: ../media/tensor_name.png
        :alt: pipeline
        :align: center

For more details, please refer to :ref:`Configuration File Detailed Description <config_details>`.

.. _model_compile_M57:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compile and execute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Take ``mobilenetv2-sim.onnx`` as an example, execute the following ``pulsar2 build`` command to compile and generate ``compiled.axmodel``:

.. code-block:: shell

    pulsar2 build --target_hardware M57 --input model/mobilenetv2-sim.onnx --output_dir output --config config/mobilenet_v2_build_config.json

.. warning::

    Before compiling the model, you need to make sure that the original model has been optimized using the ``onnxsim`` tool. The main purpose is to convert the model into a static graph that is more conducive to ``Pulsar2`` compilation and obtain better inference performance. There are two ways:

    1. Directly execute the command inside the ``Pulsar2`` docker: ``onnxsim in.onnx out.onnx``.
    2. When using ``pulsar2 build`` to convert the model, add the parameter: ``--onnx_opt.enable_onnxsim true`` (the default value is false).

    If you want to learn more about ``onnxsim``, you can visit the `official website <https://github.com/daquexian/onnx-simplifier>`_.

^^^^^^^^^^^^^^^^^^^^^^^^^
log reference information
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    $ pulsar2 build --target_hardware M57 --input model/mobilenetv2-sim.onnx --output_dir output --config config/mobilenet_v2_build_config.json
    2025-03-14 14:45:26.362 | WARNING  | yamain.command.build:fill_default:265 - apply default output processor configuration to ['output']
    2025-03-14 14:45:26.362 | WARNING  | yamain.command.build:fill_default:340 - ignore input csc config because of src_format is AutoColorSpace or src_format and tensor_format are the same
    2025-03-14 14:45:26.363 | INFO     | yamain.common.util:extract_archive:217 - extract [dataset/imagenet-32-images.tar] to [output/quant/dataset/input]...
    32 File(s) Loaded.
    Building onnx ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    2025-03-14 14:45:27.529 | INFO     | yamain.command.build:quant:748 - save optimized onnx to [output/frontend/optimized.onnx]
                                                                                Quant Config Table                                                                             
    ┏━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
    ┃ Input ┃ Shape            ┃ Dataset Directory          ┃ Data Format ┃ Tensor Format ┃ Mean                                                         ┃ Std                ┃
    ┡━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
    │ input │ [1, 3, 224, 224] │ output/quant/dataset/input │ Image       │ BGR           │ [103.93900299072266, 116.77899932861328, 123.68000030517578] │ [58.0, 58.0, 58.0] │
    └───────┴──────────────────┴────────────────────────────┴─────────────┴───────────────┴──────────────────────────────────────────────────────────────┴────────────────────┘
    Transformer optimize level: 0
    32 File(s) Loaded.
    Stastic Inf tensor: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  9.41it/s]
    [14:45:28] AX Set Float Op Table Pass Running ...         
    [14:45:29] AX Set MixPrecision Pass Running ...           
    [14:45:29] AX Set LN Quant dtype Quant Pass Running ...   
    [14:45:29] AX Reset Mul Config Pass Running ...           
    [14:45:29] AX Refine Operation Config Pass Running ...    
    [14:45:29] AX Tanh Operation Format Pass Running ...      
    [14:45:29] AX Confused Op Refine Pass Running ...         
    [14:45:29] AX Quantization Fusion Pass Running ...        
    [14:45:29] AX Quantization Simplify Pass Running ...      
    [14:45:29] AX Parameter Quantization Pass Running ...     
    [14:45:29] AX Runtime Calibration Pass Running ...        
    Calibration Progress(Phase 1): 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:03<00:00,  8.86it/s]
    [14:45:33] AX Quantization Alignment Pass Running ...     
    [14:45:33] AX Refine Int Parameter Pass Running ...       
    [14:45:33] AX Refine Scale Pass Running ...               
    [14:45:33] AX Passive Parameter Quantization Running ...  
    [14:45:33] AX Parameter Baking Pass Running ...           
    --------- Network Snapshot ---------
    Num of Op:                    [100]
    Num of Quantized Op:          [100]
    Num of Variable:              [278]
    Num of Quantized Var:         [278]
    ------- Quantization Snapshot ------
    Num of Quant Config:          [387]
    BAKED:                        [53]
    OVERLAPPED:                   [145]
    ACTIVATED:                    [65]
    SOI:                          [1]
    PASSIVE_BAKED:                [53]
    FP32:                         [70]
    Network Quantization Finished.
    Do quant optimization
    quant.axmodel export success: 
            /opt/pulsar2/quick_start_example/output/quant/quant_axmodel.onnx
            /opt/pulsar2/quick_start_example/output/quant/quant_axmodel.data
    ===>export io data to folder: output/quant/debug/io
    Building native ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    2025-03-14 14:45:35.003 | INFO     | yamain.command.build:compile_ptq_model:1029 - group 0 compiler transformation
    2025-03-14 14:45:35.005 | WARNING  | yamain.command.load_model:pre_process:616 - preprocess tensor [input]
    2025-03-14 14:45:35.005 | INFO     | yamain.command.load_model:pre_process:617 - tensor: input, (1, 224, 224, 3), U8
    2025-03-14 14:45:35.005 | INFO     | yamain.command.load_model:pre_process:617 - op: op:pre_dequant_1, AxDequantizeLinear, {'const_inputs': {'x_zeropoint': array(0, dtype=int32), 'x_scale': array(1., dtype=float32)}, 'output_dtype': <class 'numpy.float32'>, 'quant_method': 0}
    2025-03-14 14:45:35.005 | INFO     | yamain.command.load_model:pre_process:617 - tensor: tensor:pre_norm_1, (1, 224, 224, 3), FP32
    2025-03-14 14:45:35.005 | INFO     | yamain.command.load_model:pre_process:617 - op: op:pre_norm_1, AxNormalize, {'dim': 3, 'mean': [103.93900299072266, 116.77899932861328, 123.68000030517578], 'std': [58.0, 58.0, 58.0], 'output_dtype': FP32}
    2025-03-14 14:45:35.006 | INFO     | yamain.command.load_model:pre_process:617 - tensor: tensor:pre_transpose_1, (1, 224, 224, 3), FP32
    2025-03-14 14:45:35.006 | INFO     | yamain.command.load_model:pre_process:617 - op: op:pre_transpose_1, AxTranspose, {'perm': [0, 3, 1, 2]}
    2025-03-14 14:45:35.006 | WARNING  | yamain.command.load_model:post_process:638 - postprocess tensor [output]
    2025-03-14 14:45:35.006 | INFO     | yamain.command.load_model:ir_compiler_transformation:824 - use random data as gt input: input, uint8, (1, 224, 224, 3)
    2025-03-14 14:45:35.209 | INFO     | yamain.command.build:compile_ptq_model:1052 - group 0 QuantAxModel macs: 300,774,272
    2025-03-14 14:45:35.220 | INFO     | yamain.command.build:compile_ptq_model:1182 - subgraph [0], group: 0, type: GraphType.NPU
    2025-03-14 14:45:35.221 | INFO     | yamain.command.npu_backend_compiler:compile:174 - compile npu subgraph [0]
    tiling op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 68/68 0:00:00
    new_ddr_tensor = []
    build op serially...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 149/149 0:00:00
    build op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 268/268 0:00:00
    add ddr swap...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 344/344 0:00:00
    calc input dependencies...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 576/576 0:00:00
    calc output dependencies...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 576/576 0:00:00
    assign eu heuristic   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 576/576 0:00:00
    assign eu onepass   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 576/576 0:00:00
    assign eu greedy   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 576/576 0:00:00
    2025-03-14 14:45:36.965 | INFO     | yasched.test_onepass:results2model:2682 - clear job deps
    2025-03-14 14:45:36.966 | INFO     | yasched.test_onepass:results2model:2683 - max_cycle = 684,124
    build jobs   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 576/576 0:00:00
    2025-03-14 14:45:37.179 | INFO     | yamain.command.npu_backend_compiler:compile:235 - assemble model [0] [subgraph_npu_0] b1
    2025-03-14 14:45:37.931 | INFO     | yamain.command.build:compile_ptq_model:1221 - fuse 1 subgraph(s)

.. note::

    The host configuration for this example is:

        - Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz
        - Memory 32G

    The whole process takes about ``12s``, and the conversion time of hosts with different configurations may vary slightly.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
模型编译输出文件说明
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell  

    root@xxx:/data# tree output/
    output/
    |-- build_context.json
    |-- compiled.axmodel               # Model will be run on the board
    |-- compiler                       # Compiler backend intermediate results and debug information
    |   `-- debug                      # Front-end graph optimization intermediate results and debug information
    |       `-- subgraph_npu_0
    |           `-- b1
    |-- frontend
    |   |-- optimized.data
    |   `-- optimized.onnx             # Input model: floating point ONNX model after graph optimization
    `-- quant                          # Quantization tool output and debug information directory
        |-- dataset
        |   `-- input
        |       |-- ILSVRC2012_val_00000001.JPEG
        |       |-- ......
        |       `-- ILSVRC2012_val_00000032.JPEG
        |-- debug
        |   `-- io
        |       |-- float
        |       |   |-- input.npy
        |       |   `-- output.npy
        |       `-- quant
        |           |-- input.npy
        |           `-- output.npy
        |-- quant_axmodel.data
        |-- quant_axmodel.json         # Quantitative configuration information
        `-- quant_axmodel.onnx         # Quantized model, QuantAxModel

Among them, ``compiled.axmodel`` is the ``.axmodel`` model file that can be run on the board generated by the final compilation

.. note::

Since ``.axmodel`` is developed based on the **ONNX** model storage format, changing the ``.axmodel`` file suffix to ``.axmodel.onnx`` can support being directly opened by the network model graphical tool **Netron**.

    .. figure:: ../media/axmodel-netron.png
        :alt: pipeline
        :align: center

-----------------------
Model information query
-----------------------

You can use onnx inspect --io ${axmodel/onnx_path} to view the input and output information of the compiled axmodel model. You can also use -m -n -t to view the meta / node / tensor information in the model.

.. code-block:: shell

    root@xxx:/data# onnx inspect -m -n -t output/compiled.axmodel
    Failed to check model output/compiled.axmodel, statistic could be inaccurate!
    Meta information
    --------------------------------------------------------------------------------
      IR Version: 10
      Opset Import: [domain: ""
    version: 18
    ]
      Producer name: Pulsar2
      Producer version: 
      Domain: 
      Doc string: Pulsar2 Version:  3.4
    Pulsar2 Commit: 3dfd5692
      meta.{} = {} extra_data CgsKBWlucHV0EAEYAgoICgZvdXRwdXQSATEaQwoOc3ViZ3JhcGhfbnB1XzBSMQoVc3ViZ3JhcGhfbnB1XzBfYjFfbmV1EAEaFgoGcGFyYW1zGgxucHVfMF9wYXJhbXMiACgE
    Node information
    --------------------------------------------------------------------------------
      Node type "neu mode" has: 1
    --------------------------------------------------------------------------------
      Node "subgraph_npu_0": type "neu mode", inputs "['input']", outputs "['output']"
    Tensor information
    --------------------------------------------------------------------------------
      ValueInfo "input": type UINT8, shape [1, 224, 224, 3],
      ValueInfo "npu_0_params": type UINT8, shape [4324276],
      ValueInfo "subgraph_npu_0_b1_neu": type UINT8, shape [122368],
      ValueInfo "output": type FLOAT, shape [1, 1000],
      Initializer "npu_0_params": type UINT8, shape [4324276],
      Initializer "subgraph_npu_0_b1_neu": type UINT8, shape [122368],

.. _model_simulator_M57:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Simulation Run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This chapter introduces the basic operations of ``axmodel`` simulation. The ``pulsar2 run`` command can be used to run the ``axmodel`` model generated by ``pulsar2 build`` directly on the ``PC``. The running results of the network model can be quickly obtained without running on the board.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Simulation run preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some models can only support specific input data formats, and the output data of the model is also output in a module-specific format. Before the model simulation is run, the input data needs to be converted into a data format supported by the model. This part of the data operation is called ``pre-processing``. After the model simulation is run, the output data needs to be converted into a data format that can be analyzed and viewed by the tool. This part of the data operation is called ``post-processing``. The ``pre-processing`` and ``post-processing`` tools required for the simulation run are already included in the ``pulsar2-run-helper`` folder.

The contents of the ``pulsar2-run-helper`` folder are as follows:

.. code-block:: shell

    root@xxx:/data# ll pulsar2-run-helper/
    drwxr-xr-x 2 root root 4.0K Dec  2 12:23 models/
    drwxr-xr-x 5 root root 4.0K Dec  2 12:23 pulsar2_run_helper/
    drwxr-xr-x 2 root root 4.0K Dec  2 12:23 sim_images/
    drwxr-xr-x 2 root root 4.0K Dec  2 12:23 sim_inputs/
    drwxr-xr-x 2 root root 4.0K Dec  2 12:23 sim_outputs/
    -rw-r--r-- 1 root root 3.0K Dec  2 12:23 cli_classification.py
    -rw-r--r-- 1 root root 4.6K Dec  2 12:23 cli_detection.py
    -rw-r--r-- 1 root root    2 Dec  2 12:23 list.txt
    -rw-r--r-- 1 root root   29 Dec  2 12:23 requirements.txt
    -rw-r--r-- 1 root root  308 Dec  2 12:23 setup.cfg

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Simulation run example ``mobilenetv2``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Copy the ``compiled.axmodel`` generated in the :ref:`《Compile and Execute》 <model_compile_M57> section to the ``pulsar2-run-helper/models`` path and rename it to ``mobilenetv2.axmodel``

.. code-block:: shell

    root@xxx:/data# cp output/compiled.axmodel pulsar2-run-helper/models/mobilenetv2.axmodel

----------------------
Input data preparation
----------------------

Enter the ``pulsar2-run-helper`` directory and use the ``cli_classification.py`` script to process ``cat.jpg`` into the input data format required by ``mobilenetv2.axmodel``.

.. code-block:: shell

    root@xxx:~/data# cd pulsar2-run-helper
    root@xxx:~/data/pulsar2-run-helper# python3 cli_classification.py --pre_processing --image_path sim_images/cat.jpg --axmodel_path models/mobilenetv2.axmodel --intermediate_path sim_inputs/0
    [I] Write [input] to 'sim_inputs/0/input.bin' successfully.

---------------------------
Simulation Model Reasoning
---------------------------

Run the ``pulsar2 run`` command, use ``input.bin`` as the input data of ``mobilenetv2.axmodel`` and perform inference calculations, and output ``output.bin`` inference results.

.. code-block:: shell

    root@xxx:~/data/pulsar2-run-helper# pulsar2 run --model models/mobilenetv2.axmodel --input_dir sim_inputs --output_dir sim_outputs --list list.txt
    Building native ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    2025-03-14 15:00:34.804 | INFO     | yamain.command.run:run:90 - >>> [0] start
    2025-03-14 15:00:34.805 | INFO     | frontend.npu_subgraph_op:pyrun:89 - running npu subgraph: subgraph_npu_0, version: 1, target batch: 0
    2025-03-14 15:00:43.900 | INFO     | yamain.command.run:write_output:55 - write [output] to [sim_outputs/0/output.bin] successfully, size: 4000

----------------------
Output data processing
----------------------

Use the ``cli_classification.py`` script to post-process the ``output.bin`` data output by the simulation model inference to obtain the final calculation results.

.. code-block:: shell

    root@xxx:/data/pulsar2-run-helper# python3 cli_classification.py --post_processing --axmodel_path models/mobilenetv2.axmodel --intermediate_path sim_outputs/0
    [I] The following are the predicted score index pair.
    [I] 8.8490, 283
    [I] 8.7169, 285
    [I] 8.4528, 282
    [I] 8.4528, 281
    [I] 7.6603, 463

.. _onboard_running_M57:
