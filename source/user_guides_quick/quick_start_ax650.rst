======================
Quick Start(AX650)
======================

**This section applies to the following platforms:**

- AX650A
- AX650N
- M76H

This section introduces the basic operations of ``ONNX`` model conversion, and uses the ``pulsar2`` tool to compile the ``ONNX`` model into the ``axmodel`` model. Please refer to the :ref:`《Development Environment Preparation》 <dev_env_prepare>` section to complete the development environment setup.
The example model in this section is the open source model ``MobileNetv2``.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pulsar2 toolchain command description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The function commands in the ``Pulsar2`` toolchain start with ``pulsar2``. The commands that are most relevant to users are ``pulsar2 build``, ``pulsar2 run`` and ``pulsar2 version``.

* ``pulsar2 build`` is used to convert ``onnx`` models to ``axmodel`` format models
* ``pulsar2 run`` is used to run simulations after model conversion
* ``pulsar2 version`` can be used to view the version information of the current toolchain, which is usually required when reporting issues

.. code-block:: shell

    root@xxx:/data# pulsar2 --help
    usage: pulsar2 [-h] {version,build,run} ...
    
    positional arguments:
      {version,build,run}
    
    optional arguments:
      -h, --help           show this help message and exit

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Model compilation configuration file description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. _model_compile:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compile and execute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Take ``mobilenetv2-sim.onnx`` as an example, execute the following ``pulsar2 build`` command to compile and generate ``compiled.axmodel``:

.. code-block:: shell

    pulsar2 build  --target_hardware AX650 --input model/mobilenetv2-sim.onnx --output_dir output --config config/mobilenet_v2_build_config.json

.. warning::

    Before compiling the model, you need to make sure that the original model has been optimized using the ``onnxsim`` tool. The main purpose is to convert the model into a static graph that is more conducive to ``Pulsar2`` compilation and obtain better inference performance. There are two methods:
    
    1. Execute the command directly inside the ``Pulsar2`` docker: ``onnxsim in.onnx out.onnx``.
    2. When using ``pulsar2 build`` to convert the model, add the parameter: ``--onnx_opt.enable_onnxsim true`` (the default value is false).

    If you want to learn more about ``onnxsim``, you can visit the `official website <https://github.com/daquexian/onnx-simplifier>`_.

^^^^^^^^^^^^^^^^^^^^^^^^^^
log reference information
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    2024-09-25 11:45:26.533 | WARNING  | yamain.command.build:fill_default:300 - apply default output processor configuration to ['output']
    2024-09-25 11:45:26.533 | WARNING  | yamain.command.build:fill_default:364 - ignore input csc config because of src_format is AutoColorSpace or src_format and tensor_format are the same
    2024-09-25 11:45:26.534 | INFO     | yamain.common.util:extract_archive:181 - extract [dataset/imagenet-32-images.tar] to [output/quant/dataset/input]...
    32 File(s) Loaded.
    Building onnx ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    2024-09-25 11:45:27.422 | INFO     | yamain.command.build:quant:797 - save optimized onnx to [output/frontend/optimized.onnx]
                                   Quant Config Table                               
    ┏━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┓
    ┃       ┃           ┃ Dataset   ┃ Data      ┃ Tensor    ┃           ┃          ┃
    ┃ Input ┃ Shape     ┃ Directory ┃ Format    ┃ Format    ┃ Mean      ┃ Std      ┃
    ┡━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━┩
    │ input │ [1, 3,    │ output/q… │ Image     │ BGR       │ [103.939… │ [58.0,   │
    │       │ 224, 224] │           │           │           │ 116.7789… │ 58.0,    │
    │       │           │           │           │           │ 123.6800… │ 58.0]    │
    └───────┴───────────┴───────────┴───────────┴───────────┴───────────┴──────────┘
    Transformer optimize level: 0
    32 File(s) Loaded.
    
    Stastic Inf tensor:   0%|          | 0/1 [00:00<?, ?it/s]
    Stastic Inf tensor: 100%|██████████| 1/1 [00:00<00:00,  9.09it/s]
    Stastic Inf tensor: 100%|██████████| 1/1 [00:00<00:00,  9.06it/s]
    [11:45:28] AX Set Float Op Table Pass Running ...         
    [11:45:28] AX Set MixPrecision Pass Running ...           
    [11:45:28] AX Set LN Quant dtype Quant Pass Running ...   
    [11:45:28] AX Reset Mul Config Pass Running ...           
    [11:45:28] AX Refine Operation Config Pass Running ...    
    [11:45:28] AX Tanh Operation Format Pass Running ...      
    [11:45:28] AX Confused Op Refine Pass Running ...         
    [11:45:28] AX Quantization Fusion Pass Running ...        
    [11:45:28] AX Quantization Simplify Pass Running ...      
    [11:45:28] AX Parameter Quantization Pass Running ...     
    [11:45:29] AX Runtime Calibration Pass Running ...        
    
    Calibration Progress(Phase 1):   0%|          | 0/32 [00:00<?, ?it/s]
    Calibration Progress(Phase 1):   3%|▎         | 1/32 [00:00<00:03,  9.10it/s]
    Calibration Progress(Phase 1):   6%|▋         | 2/32 [00:00<00:03,  9.09it/s]
    Calibration Progress(Phase 1):   9%|▉         | 3/32 [00:00<00:03,  9.05it/s]
    Calibration Progress(Phase 1):  12%|█▎        | 4/32 [00:00<00:03,  9.02it/s]
    Calibration Progress(Phase 1):  16%|█▌        | 5/32 [00:00<00:02,  9.00it/s]
    Calibration Progress(Phase 1):  19%|█▉        | 6/32 [00:00<00:02,  8.96it/s]
    Calibration Progress(Phase 1):  22%|██▏       | 7/32 [00:00<00:02,  9.03it/s]
    Calibration Progress(Phase 1):  25%|██▌       | 8/32 [00:00<00:02,  9.03it/s]
    Calibration Progress(Phase 1):  28%|██▊       | 9/32 [00:00<00:02,  9.03it/s]
    Calibration Progress(Phase 1):  31%|███▏      | 10/32 [00:01<00:02,  9.02it/s]
    Calibration Progress(Phase 1):  34%|███▍      | 11/32 [00:01<00:02,  9.00it/s]
    Calibration Progress(Phase 1):  38%|███▊      | 12/32 [00:01<00:02,  8.94it/s]
    Calibration Progress(Phase 1):  41%|████      | 13/32 [00:01<00:02,  8.95it/s]
    Calibration Progress(Phase 1):  44%|████▍     | 14/32 [00:01<00:02,  8.96it/s]
    Calibration Progress(Phase 1):  47%|████▋     | 15/32 [00:01<00:01,  8.92it/s]
    Calibration Progress(Phase 1):  50%|█████     | 16/32 [00:01<00:01,  8.89it/s]
    Calibration Progress(Phase 1):  53%|█████▎    | 17/32 [00:01<00:01,  8.90it/s]
    Calibration Progress(Phase 1):  56%|█████▋    | 18/32 [00:02<00:01,  8.89it/s]
    Calibration Progress(Phase 1):  59%|█████▉    | 19/32 [00:02<00:01,  8.86it/s]
    Calibration Progress(Phase 1):  62%|██████▎   | 20/32 [00:02<00:01,  8.93it/s]
    Calibration Progress(Phase 1):  66%|██████▌   | 21/32 [00:02<00:01,  8.90it/s]
    Calibration Progress(Phase 1):  69%|██████▉   | 22/32 [00:02<00:01,  8.93it/s]
    Calibration Progress(Phase 1):  72%|███████▏  | 23/32 [00:02<00:01,  8.91it/s]
    Calibration Progress(Phase 1):  75%|███████▌  | 24/32 [00:02<00:00,  8.89it/s]
    Calibration Progress(Phase 1):  78%|███████▊  | 25/32 [00:02<00:00,  8.91it/s]
    Calibration Progress(Phase 1):  81%|████████▏ | 26/32 [00:02<00:00,  8.87it/s]
    Calibration Progress(Phase 1):  84%|████████▍ | 27/32 [00:03<00:00,  8.89it/s]
    Calibration Progress(Phase 1):  88%|████████▊ | 28/32 [00:03<00:00,  8.91it/s]
    Calibration Progress(Phase 1):  91%|█████████ | 29/32 [00:03<00:00,  8.86it/s]
    Calibration Progress(Phase 1):  94%|█████████▍| 30/32 [00:03<00:00,  8.85it/s]
    Calibration Progress(Phase 1):  97%|█████████▋| 31/32 [00:03<00:00,  8.77it/s]
    Calibration Progress(Phase 1): 100%|██████████| 32/32 [00:03<00:00,  8.74it/s]
    Calibration Progress(Phase 1): 100%|██████████| 32/32 [00:03<00:00,  8.91it/s]
    [11:45:32] AX Quantization Alignment Pass Running ...     
    [11:45:32] AX Refine Int Parameter Pass Running ...       
    [11:45:33] AX Refine Scale Pass Running ...               
    [11:45:33] AX Passive Parameter Quantization Running ...  
    [11:45:33] AX Parameter Baking Pass Running ...           
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
    quant.axmodel export success: 
    	/data/deploy/data/quick_start_example/output/quant/quant_axmodel.onnx
    	/data/deploy/data/quick_start_example/output/quant/quant_axmodel.data
    ===>export pb data to folder: output/quant/debug/test_data_set_0
    ===>export io data to folder: output/quant/debug/io
    Building native ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    2024-09-25 11:45:33.944 | INFO     | yamain.command.build:compile_ptq_model:1035 - group 0 compiler transformation
    2024-09-25 11:45:33.946 | WARNING  | yamain.command.load_model:pre_process:608 - preprocess tensor [input]
    2024-09-25 11:45:33.946 | INFO     | yamain.command.load_model:pre_process:609 - tensor: input, (1, 224, 224, 3), U8
    2024-09-25 11:45:33.947 | INFO     | yamain.command.load_model:pre_process:609 - op: op:pre_dequant_1, AxDequantizeLinear, {'const_inputs': {'x_zeropoint': array(0, dtype=int32), 'x_scale': array(1., dtype=float32)}, 'output_dtype': <class 'numpy.float32'>, 'quant_method': 0}
    2024-09-25 11:45:33.947 | INFO     | yamain.command.load_model:pre_process:609 - tensor: tensor:pre_norm_1, (1, 224, 224, 3), FP32
    2024-09-25 11:45:33.947 | INFO     | yamain.command.load_model:pre_process:609 - op: op:pre_norm_1, AxNormalize, {'dim': 3, 'mean': [103.93900299072266, 116.77899932861328, 123.68000030517578], 'std': [58.0, 58.0, 58.0], 'output_dtype': FP32}
    2024-09-25 11:45:33.947 | INFO     | yamain.command.load_model:pre_process:609 - tensor: tensor:pre_transpose_1, (1, 224, 224, 3), FP32
    2024-09-25 11:45:33.947 | INFO     | yamain.command.load_model:pre_process:609 - op: op:pre_transpose_1, AxTranspose, {'perm': [0, 3, 1, 2]}
    2024-09-25 11:45:33.947 | WARNING  | yamain.command.load_model:post_process:630 - postprocess tensor [output]
    2024-09-25 11:45:34.159 | INFO     | yamain.command.build:compile_ptq_model:1060 - QuantAxModel macs: 280,262,480
    2024-09-25 11:45:34.169 | INFO     | yamain.command.build:compile_ptq_model:1132 - subgraph [0], group: 0, type: GraphType.NPU
    2024-09-25 11:45:34.187 | INFO     | yasched.test_onepass:test_onepass_ir:3221 - schedule npu subgraph [0]
    tiling op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 68/68 0:00:00
    <frozen backend.ax650npu.oprimpl.normalize>:186: RuntimeWarning: divide by zero encountered in divide
    <frozen backend.ax650npu.oprimpl.normalize>:187: RuntimeWarning: invalid value encountered in divide
    new_ddr_tensor = []
    build op serially...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 103/103 0:00:00
    build op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 188/188 0:00:00
    add ddr swap...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 497/497 0:00:00
    calc input dependencies...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 921/921 0:00:00
    calc output dependencies...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 921/921 0:00:00
    assign eu heuristic   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 921/921 0:00:00
    assign eu onepass   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 921/921 0:00:00
    assign eu greedy   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 921/921 0:00:00
    2024-09-25 11:45:36.467 | INFO     | yasched.test_onepass:results2model:2541 - clear job deps
    2024-09-25 11:45:36.467 | INFO     | yasched.test_onepass:results2model:2542 - max_cycle = 450,154
    build jobs   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 921/921 0:00:00
    2024-09-25 11:45:36.796 | INFO     | yamain.command.build:compile_npu_subgraph:1332 - assembel model [subgraph_npu_0]
    2024-09-25 11:45:38.075 | INFO     | yamain.command.build:compile_ptq_model:1142 - fuse 1 subgraph(s)

.. note::

    The host configuration on which this example runs is:

        - Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz
        - Memory 32G

    The whole process takes about ``11s``, and the host conversion time varies slightly with different configurations.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Model compilation output file description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell  

    root@xxx:/data# tree output/
    output/
    ├── build_context.json
    ├── compiled.axmodel            # Model will be run on the board
    ├── compiler                    # Compiler backend intermediate results and debug information
    ├── frontend                    # Front-end graph optimization intermediate results and debug information
    │   └── optimized.onnx          # Input model: floating point ONNX model after graph optimization
    └── quant                       # Quantization tool output and debug information directory
        ├── dataset                 # The decompressed calibration set data directory
        │   └── input
        │       ├── ILSVRC2012_val_00000001.JPEG
        │       ├── ......
        │       └── ILSVRC2012_val_00000032.JPEG
        ├── debug
        ├── quant_axmodel.json      # Quantitative configuration information
        └── quant_axmodel.onnx      # Quantized model, QuantAxModel

``compiled.axmodel`` is the ``.axmodel`` model file that can be run on the board generated by the final compilation

.. note::

    Because ``.axmodel`` is developed based on the **ONNX** model storage format, changing the ``.axmodel`` file suffix to ``.axmodel.onnx`` can support being directly opened by the network model graphical tool **Netron**.

    .. figure:: ../media/axmodel-netron.png
        :alt: pipeline
        :align: center

--------------------------
Model information query
--------------------------

By using ``onnx inspect --io ${axmodel/onnx_path}`` to view the input and output information of compiled ``axmodel``, and other parameter ``-m -n -t`` to view model's information of ``meta / node / tensor`` 

.. code-block:: shell

    root@xxx:/data# onnx inspect -m -n -t output/compiled.axmodel
    Failed to check model output/compiled.axmodel, statistic could be inaccurate!
    Inpect of model output/compiled.axmodel
    ================================================================================
      Graph name: 8
      Graph inputs: 1
      Graph outputs: 1
      Nodes in total: 1
      ValueInfo in total: 4
      Initializers in total: 2
      Sparse Initializers in total: 0
      Quantization in total: 0
    
    Meta information:
    --------------------------------------------------------------------------------
      IR Version: 8
      Opset Import: [domain: ""
    version: 16
    ]
      Producer name: Pulsar2
      Producer version: 
      Domain: 
      Doc string: Pulsar2 Version:  2.4
    Pulsar2 Commit: 2064a8ee
      meta.{} = {} extra_data CgsKBWlucHV0EAEYAgoICgZvdXRwdXQSATEaQQoOc3ViZ3JhcGhfbnB1XzBSLwoVc3ViZ3JhcGhfbnB1XzBfYjFfbmV1EAEaFAoGcGFyYW1zGgpucHVfcGFyYW1zIgA=
    
    Node information:
    --------------------------------------------------------------------------------
      Node type "neu mode" has: 1
    --------------------------------------------------------------------------------
      Node "subgraph_npu_0": type "neu mode", inputs "['input']", outputs "['output']"
    
    Tensor information:
    --------------------------------------------------------------------------------
      ValueInfo "input": type UINT8, shape [1, 224, 224, 3],
      ValueInfo "npu_params": type UINT8, shape [4085516],
      ValueInfo "subgraph_npu_0_b1_neu": type UINT8, shape [56592],
      ValueInfo "output": type FLOAT, shape [1, 1000],
      Initializer "npu_params": type UINT8, shape [4085516],
      Initializer "subgraph_npu_0_b1_neu": type UINT8, shape [56592],

.. _model_simulator:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Simulation Run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This chapter introduces the basic operations of ``axmodel`` simulation. The ``pulsar2 run`` command can be used to run the ``axmodel`` model generated by ``pulsar2 build`` directly on the ``PC``. The running results of the network model can be quickly obtained without running on the board.

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Simulation run preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Some models can only support specific input data formats, and the output data of the model is also output in a module-specific format. Before the model simulation is run, the input data needs to be converted into a data format supported by the model. This part of the data operation is called ``pre-processing``. After the model simulation is run, the output data needs to be converted into a data format that can be analyzed and viewed by the tool. This part of the data operation is called ``post-processing``. The ``pre-processing`` and ``post-processing`` tools required for the simulation run are already included in the ``pulsar2-run-helper`` folder.

``pulsar2-run-helper`` folder contents are as follows:

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

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Simulation run example ``mobilenetv2``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Copy the ``compiled.axmodel`` generated in the :ref:`《Compile and Execute》 <model_compile>` section to the ``pulsar2-run-helper/models`` path and rename it to ``mobilenetv2.axmodel``

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
    >>> [0] start
    write [output] to [sim_outputs/0/output.bin] successfully
    >>> [0] finish

----------------------
Output data processing
----------------------

Use the ``cli_classification.py`` script to post-process the ``output.bin`` data output by the simulation model inference to obtain the final calculation results.

.. code-block:: shell

    root@xxx:/data/pulsar2-run-helper# python3 cli_classification.py --post_processing --axmodel_path models/mobilenetv2.axmodel --intermediate_path sim_outputs/0
    [I] The following are the predicted score index pair.
    [I] 9.5094, 285
    [I] 9.3773, 282
    [I] 9.2452, 281
    [I] 8.5849, 283
    [I] 7.6603, 287

.. _onboard_running:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Development board running
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This section describes how to run the ``compiled.axmodel`` model obtained through the :ref:`《Compile and Execute》 <model_compile_20e>` section on the ``AX630C`` ``AX620Q`` development board.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Development Board Acquisition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Get the **AX630C DEMO Board** after signing an NDA with AXera through the enterprise channel.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use the ax_run_model tool to quickly test the model inference speed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to facilitate users to evaluate the model, the :ref:`ax_run_model <ax_run_model>` tool is pre-made on the development board. This tool has several parameters that can easily test the model speed and accuracy.

Copy ``mobilenetv2.axmodel`` to the development board and execute the following command to quickly test the model inference performance (first perform 3 inferences for warm-up to eliminate statistical errors caused by resource initialization, then perform 10 inferences to calculate the average inference speed).

.. code-block:: shell

    /root # ax_run_model -m mobilenetv2.axmodel -w 3 -r 10
      Run AxModel:
            model: mobilenetv2.axmodel
             type: 1 Core
             vnpu: Disable
         affinity: 0b001
           warmup: 3
           repeat: 10
            batch: { auto: 1 }
         parallel: false
      pulsar2 ver: 1.2-patch2 7e6b2b5f
       engine ver: 2.3.0a
         tool ver: 2.1.2c
         cmm size: 4428624 Bytes
      ------------------------------------------------------
      min =   0.719 ms   max =   0.726 ms   avg =   0.721 ms
      ------------------------------------------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Use the sample_npu_classification example to test the inference results of a single image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. hint::

    The running example has been pre-installed in the file system of the development board, and its source files are located in the folder under the SDK path ``msp/sample/npu``. Copy ``mobilennetv2.axmodel`` to the development board and use ``sample_npu_classification`` for testing.

``sample_npu_classification`` Input parameter description:

.. code-block:: shell

    /root # sample_npu_classification --help
    usage: sample_npu_classification --model=string --image=string [options] ...
    options:
      -m, --model     joint file(a.k.a. joint model) (string)
      -i, --image     image file (string)
      -g, --size      input_h, input_w (string [=224,224])
      -r, --repeat    repeat count (int [=1])
      -?, --help      print this message

By executing the ``sample_npu_classification`` program, the classification model is run on the board. The running results are as follows:

.. code-block:: shell

    /root # sample_npu_classification -m mobilenetv2.axmodel -i /opt/data/npu/images/cat.jpg -r 10
    --------------------------------------
    model file : mobilenetv2.axmodel
    image file : /opt/data/npu/images/cat.jpg
    img_h, img_w : 224 224
    --------------------------------------
    Engine creating handle is done.
    Engine creating context is done.
    Engine get io info is done.
    Engine alloc io is done.
    Engine push input is done.
    --------------------------------------
    topk cost time:0.07 ms
    9.5094, 285
    9.3773, 282
    9.2452, 281
    8.5849, 283
    7.6603, 287
    --------------------------------------
    Repeat 10 times, avg time 0.72 ms, max_time 0.72 ms, min_time 0.72 ms
    --------------------------------------

- From here, we can see that the results of running the same ``mobilenetv2.axmodel`` model on the development board are consistent with the results of :ref:`《Simulation Run》 <model_simulator>`;
- For details on the source code and compilation generation of the executable program ``ax_classification`` on the board, please refer to :ref:`《Model Deployment Advanced Guide》 <model_deploy_advanced>`.
