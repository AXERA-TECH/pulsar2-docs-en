======================
Quick Start(AX620E)
======================

**This chapter applies to the following platforms:**

- AX630C
- AX620Q

This chapter will introduce in order:

* How to install ``Docker`` and start the container in different system environments
* How to use the ``Pulsar2 Docker`` tool chain to convert the ``onnx`` model to the ``axmodel`` model
* How to use the ``axmodel`` model to simulate and run on the ``x86`` platform and measure the difference between the inference results and the ``onnx`` inference results (internally called ``bisection``)
* How to run ``axmodel`` on the board

.. note::

    The ``bisection`` means comparing the errors between the inference results of different versions (file types) of the same model before and after tool chain compilation.

.. _dev_env_prepare_20e_en:

------------------------------------
Development environment preparation
------------------------------------

This section introduces the preparation of the development environment before using the ``Pulsar2`` tool chain.

``Pulsar2`` uses the ``Docker`` container for tool chain integration. Users can load the ``Pulsar2`` image file through ``Docker``, and then perform model conversion, compilation, simulation, etc., so the development environment preparation phase Just install the ``Docker`` environment correctly. Supported systems are ``MacOS``, ``Linux``, ``Windows``.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Install Docker development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- `MacOS Installation Docker Environment <https://docs.docker.com/desktop/mac/install/>`_

- `Linux installation Docker environment <https://docs.docker.com/engine/install/##server>`_

- `Windows installation Docker environment <https://docs.docker.com/desktop/windows/install/>`_

After ``Docker`` is installed successfully, enter ``sudo docker -v``

.. code-block:: shell

     $ sudo docker -v
     Docker version 20.10.7, build f0df350

The above content is displayed, indicating that ``Docker`` has been successfully installed. The following will introduce the installation and startup of the ``Pulsar2`` tool chain ``Image``.

~~~~~~~~~~~~~~~~~~~~~~~~~~
Install Pulsar2 toolchain
~~~~~~~~~~~~~~~~~~~~~~~~~~

Taking the system version ``Ubuntu 20.04`` and the tool chain ``ax_pulsar2_${version}.tar.gz`` as an example to illustrate the installation method of the ``Pulsar2`` tool chain.

.. hint::

     In actual operation, be sure to replace ${version} with the corresponding tool chain version number.

How to obtain the tool chain:

- `BaiDu Pan <https://pan.baidu.com/s/1FazlPdW79wQWVY-Qn--qVQ?pwd=sbru>`_
- `Google Drive <https://drive.google.com/drive/folders/1gJFkHw2gyW-7B9xTdpH_w72Ly2PQ7nsi?usp=drive_link>`_

^^^^^^^^^^^^^^^^^^^^^^^
Load Docker Image
^^^^^^^^^^^^^^^^^^^^^^^

Execute ``sudo docker load -i ax_pulsar2_${version}.tar.gz`` to import the docker image file. Correctly importing the image file will print the following log:

.. code-block:: shell

    $ sudo docker load -i ax_pulsar2_${version}.tar.gz
    Loaded image: pulsar2:${version}

Once completed, execute ``sudo docker image ls``

.. code-block:: shell

    $ sudo docker image ls
    REPOSITORY   TAG          IMAGE ID       CREATED         SIZE
    pulsar2      ${version}   xxxxxxxxxxxx   9 seconds ago   3.27GB

You can see that the tool chain image has been successfully loaded, and you can then start the container based on this image.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Start the tool chain image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Execute the following command to start the ``Docker`` container, and enter the ``bash`` environment after successful operation

.. code-block:: shell

    $ sudo docker run -it --net host --rm -v $PWD:/data pulsar2:${version}

----------------------
Version query
----------------------

``pulsar2 version`` is used to obtain the version information of the tool.

Example results

.. code-block:: bash

    root@xxx:/data# pulsar2 version
    version: ${version}
    commit: xxxxxxxx

.. _prepare_data_20e_en:

----------------------
Data preparation
----------------------

.. hint::

    The subsequent content of this chapter **《4.4. Model Compilation》**, **《4.6. Simulation Run》** required **original model**, **data**, **pictures**, **simulation Tool** has been provided in the ``quick_start_example`` folder :download:`Click to download the example file <https://github.com/xiguadong/assets/releases/download/v0.1/quick_start_example.zip>` and then unzip the downloaded file and copy it to the ``/data`` path of ``docker``.

.. code-block:: shell

    root@xxx:~/data# ls
    config  dataset  model  output  pulsar2-run-helper

* ``model``: stores the original ``ONNX`` model ``mobilenetv2-sim.onnx`` (the ``mobilenetv2.onnx`` has been optimized by using ``onnxsim`` in advance)
* ``dataset``: stores the data set compression package required for offline quantization calibration (PTQ Calibration) (supports common compression formats such as tar, tar.gz, gz, etc.)
* ``config``: Configuration file that stores running dependencies ``config.json``
* ``output``: stores the result output
* ``pulsar2-run-helper``: A tool that supports ``axmodel`` to run simulations in the X86 environment

After the data preparation is completed, the directory tree structure is as follows:

.. code-block:: shell

    root@xxx:/data# tree -L 2
    .
    ├── config
    │   ├── mobilenet_v2_build_config.json
    │   └── yolov5s_config.json
    ├── dataset
    │   ├── coco_4.tar
    │   └── imagenet-32-images.tar
    ├── model
    │   ├── mobilenetv2-sim.onnx
    │   └── yolov5s.onnx
    ├── output
    └── pulsar2-run-helper
        ├── cli_classification.py
        ├── cli_detection.py
        ├── models
        ├── pulsar2_run_helper
        ├── requirements.txt
        ├── setup.cfg
        ├── sim_images
        ├── sim_inputs
        └── sim_outputs

.. _model_compile_20e_en:

----------------------
Model compilation
----------------------

This chapter introduces the basic operations of ``ONNX`` model conversion. Use the ``pulsar2`` tool to compile the ``ONNX`` model into the ``axmodel`` model. Please refer to :ref:`《Development Environment Preparation》 <dev_env_prepare_20e_en>` Chapter completes the development environment setup.
The example model in this section is the open source model ``MobileNetv2``.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Command description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The functional instructions in the ``Pulsar2`` tool chain begin with ``pulsar2``, and the commands that are strongly related to users are ``pulsar2 build``, ``pulsar2 run`` and ``pulsar2 version``.

* ``pulsar2 build`` is used to convert ``onnx`` model to ``axmodel`` format model
* ``pulsar2 run`` is used for simulation running after model conversion
* ``pulsar2 version`` can be used to view the version information of the current tool chain. This information is usually required when reporting problems.

.. code-block:: shell

    root@xxx:/data# pulsar2 --help
    usage: pulsar2 [-h] {version,build,run} ...
    
    positional arguments:
      {version,build,run}
    
    optional arguments:
      -h, --help           show this help message and exit

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Configuration file description
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mobilenet_v2_build_config.json`` under the path ``/data/config/`` displays:

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
  
     The ``tensor_name`` field in ``input_configs`` under the ``input_processors``, ``output_processors`` and ``quant`` nodes needs to be set according to the actual input/output node name of the model, or it can be set to ` `DEFAULT` means that the current configuration applies to all inputs or outputs.

     .. figure:: ../media/tensor_name.png
         :alt: pipeline
         :align: center

For more details, please refer to :ref:`《Configuration File Detailed Description》 <config_details_en>`.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compile and execute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Taking ``mobilenetv2-sim.onnx`` as an example, execute the following ``pulsar2 build`` command to compile and generate ``compiled.axmodel``:

.. code-block:: shell

     pulsar2 build --input model/mobilenetv2-sim.onnx --output_dir output --config config/mobilenet_v2_build_config.json --target_hardware AX620E

.. warning::

     Before compiling the model, you need to ensure that the ``onnxsim`` tool has been used to optimize the original model. The main purpose is to transform the model into a static graph that is more conducive to ``Pulsar2`` compilation and obtain better inference performance. There are two methods:

     1. Directly execute the command inside ``Pulsar2`` docker: ``onnxsim in.onnx out.onnx``.
     2. When using ``pulsar2 build`` for model conversion, add the parameter: ``--onnx_opt.enable_onnxsim true`` (the default value is false).

     If you want to learn more about ``onnxsim``, you can visit the `official website <https://github.com/daquexian/onnx-simplifier>`_.

^^^^^^^^^^^^^^^^^^^^^^^^^^^
log reference information
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    $ pulsar2 build --input model/mobilenetv2-sim.onnx --output_dir output --config config/mobilenet_v2_build_config.json --target_hardware AX620E
    2023-07-29 14:23:01.757 | WARNING  | yamain.command.build:fill_default:313 - ignore input csc config because of src_format is AutoColorSpace or src_format and tensor_format are the same
    Building onnx ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    2023-07-29 14:23:07.806 | INFO     | yamain.command.build:build:424 - save optimized onnx to [output/frontend/optimized.onnx]
    patool: Extracting ./dataset/imagenet-32-images.tar ...
    patool: running /usr/bin/tar --extract --file ./dataset/imagenet-32-images.tar --directory output/quant/dataset/input
    patool: ... ./dataset/imagenet-32-images.tar extracted to `output/quant/dataset/input'.
                                                                            Quant Config Table
    ┏━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
    ┃ Input ┃ Shape            ┃ Dataset Directory ┃ Data Format ┃ Tensor Format ┃ Mean                                                         ┃ Std                ┃
    ┡━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
    │ input │ [1, 3, 224, 224] │ input             │ Image       │ BGR           │ [103.93900299072266, 116.77899932861328, 123.68000030517578] │ [58.0, 58.0, 58.0] │
    └───────┴──────────────────┴───────────────────┴─────────────┴───────────────┴──────────────────────────────────────────────────────────────┴────────────────────┘
    Transformer optimize level: 0
    32 File(s) Loaded.
    [14:23:09] AX LSTM Operation Format Pass Running ...      Finished.
    [14:23:09] AX Set MixPrecision Pass Running ...           Finished.
    [14:23:09] AX Refine Operation Config Pass Running ...    Finished.
    [14:23:09] AX Reset Mul Config Pass Running ...           Finished.
    [14:23:09] AX Tanh Operation Format Pass Running ...      Finished.
    [14:23:09] AX Confused Op Refine Pass Running ...         Finished.
    [14:23:09] AX Quantization Fusion Pass Running ...        Finished.
    [14:23:09] AX Quantization Simplify Pass Running ...      Finished.
    [14:23:09] AX Parameter Quantization Pass Running ...     Finished.
    Calibration Progress(Phase 1): 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:01<00:00, 18.07it/s]
    Finished.
    [14:23:11] AX Passive Parameter Quantization Running ...  Finished.
    [14:23:11] AX Parameter Baking Pass Running ...           Finished.
    [14:23:11] AX Refine Int Parameter Pass Running ...       Finished.
    [14:23:11] AX Refine Weight Parameter Pass Running ...    Finished.
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
    [Warning]File output/quant/quant_axmodel.onnx has already exist, quant exporter will overwrite it.
    [Warning]File output/quant/quant_axmodel.json has already exist, quant exporter will overwrite it.
    quant.axmodel export success: output/quant/quant_axmodel.onnx
    Building native ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    2023-07-29 14:23:18.332 | WARNING  | yamain.command.load_model:pre_process:454 - preprocess tensor [input]
    2023-07-29 14:23:18.332 | INFO     | yamain.command.load_model:pre_process:456 - tensor: input, (1, 224, 224, 3), U8
    2023-07-29 14:23:18.332 | INFO     | yamain.command.load_model:pre_process:459 - op: op:pre_dequant_1, AxDequantizeLinear, {'const_inputs': {'x_zeropoint': 0, 'x_scale': 1}, 'output_dtype': <class 'numpy.float32'>, 'quant_method': 0}
    2023-07-29 14:23:18.332 | INFO     | yamain.command.load_model:pre_process:456 - tensor: tensor:pre_norm_1, (1, 224, 224, 3), FP32
    2023-07-29 14:23:18.332 | INFO     | yamain.command.load_model:pre_process:459 - op: op:pre_norm_1, AxNormalize, {'dim': 3, 'mean': [103.93900299072266, 116.77899932861328, 123.68000030517578], 'std': [58.0, 58.0, 58.0]}
    2023-07-29 14:23:18.332 | INFO     | yamain.command.load_model:pre_process:456 - tensor: tensor:pre_transpose_1, (1, 224, 224, 3), FP32
    2023-07-29 14:23:18.332 | INFO     | yamain.command.load_model:pre_process:459 - op: op:pre_transpose_1, AxTranspose, {'perm': [0, 3, 1, 2]}
    tiling op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 174/174 0:00:00
    new_ddr_tensor = []
    build op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 440/440 0:00:00
    add ddr swap...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1606/1606 0:00:00
    calc input dependencies...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2279/2279 0:00:00
    calc output dependencies...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2279/2279 0:00:00
    assign eu heuristic   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2279/2279 0:00:00
    assign eu onepass   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2279/2279 0:00:00
    assign eu greedy   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2279/2279 0:00:00
    2023-07-29 14:23:21.762 | INFO     | yasched.test_onepass:results2model:1882 - max_cycle = 782,940
    2023-07-29 14:23:22.159 | INFO     | yamain.command.build:compile_npu_subgraph:1004 - QuantAxModel macs: 280,262,480
    2023-07-29 14:23:25.209 | INFO     | backend.ax620e.linker:link_with_dispatcher:1586 - DispatcherQueueType.IO: Generate 69 EU chunks, 7 Dispatcher Chunk
    2023-07-29 14:23:25.209 | INFO     | backend.ax620e.linker:link_with_dispatcher:1586 - DispatcherQueueType.Compute: Generate 161 EU chunks, 23 Dispatcher Chunk
    2023-07-29 14:23:25.209 | INFO     | backend.ax620e.linker:link_with_dispatcher:1587 - EU mcode size: 147 KiB
    2023-07-29 14:23:25.209 | INFO     | backend.ax620e.linker:link_with_dispatcher:1588 - Dispatcher mcode size: 21 KiB
    2023-07-29 14:23:25.209 | INFO     | backend.ax620e.linker:link_with_dispatcher:1589 - Total mcode size: 168 KiB
    2023-07-29 14:23:26.928 | INFO     | yamain.command.build:compile_ptq_model:940 - fuse 1 subgraph(s)

.. note::

    The host configuration this example is running on is:

        - Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz
        - Memory 32G

    The whole process takes about ``11s``, and the host conversion time of different configurations is slightly different.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Output file description
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell  

    root@xxx:/data# tree output/
    output/
    ├── build_context.json
    ├── compiled.axmodel            # finally run the model on the board, AxModel
    ├── compiler                    # compiler backend intermediate results and debug information
    ├── frontend                    # front-end graph optimization intermediate results and debug information
    │   └── optimized.onnx          # the input model is a floating-point ONNX model after graph optimization.
    └── quant                       # quantification tool output and debug information directory
        ├── dataset                 # unzipped calibration set data directory
        │   └── input
        │       ├── ILSVRC2012_val_00000001.JPEG
        │       ├── ......
        │       └── ILSVRC2012_val_00000032.JPEG
        ├── debug
        ├── quant_axmodel.json      # quantify configuration information
        └── quant_axmodel.onnx      # quantized model, QuantAxModel

Among them, ``compiled.axmodel`` is the executable ``.axmodel`` model file generated on the board after final compilation.

.. note::

    Because ``.axmodel`` is developed based on the **ONNX** model storage format, changing the ``.axmodel`` file suffix to ``.axmodel.onnx`` can be supported by the network model graphical tool **Netron** Open directly.

    .. figure:: ../media/axmodel-netron.png
        :alt: pipeline
        :align: center

----------------------
Information inquiry
----------------------

You can view the input and output information of the ``axmodel`` model through ``onnx inspect --io ${axmodel/onnx_path}``, and there are other ``-m -n -t`` parameters to view ` in the model `meta/node/tensor` information.

.. code-block:: shell

    root@xxx:/data# onnx inspect -m -n -t output/compiled.axmodel
    Failed to check model output/compiled.axmodel, statistic could be inaccurate!
    Inpect of model output/compiled.axmodel
    ================================================================================
      Graph name: 8
      Graph inputs: 1
      Graph outputs: 1
      Nodes in total: 1
      ValueInfo in total: 2
      Initializers in total: 2
      Sparse Initializers in total: 0
      Quantization in total: 0

    Meta information:
    --------------------------------------------------------------------------------
      IR Version: 7
      Opset Import: [version: 13
    ]
      Producer name: Pulsar2
      Producer version:
      Domain:
      Doc string: Pulsar2 Version:  1.8-beta1
    Pulsar2 Commit: 6a7e59de
      meta.{} = {} extra_data CgsKBWlucHV0EAEYAgoICgZvdXRwdXQSATEaMgoFbnB1XzBSKQoNbnB1XzBfYjFfZGF0YRABGhYKBnBhcmFtcxoMbnB1XzBfcGFyYW1zIgAoAQ==

    Node information:
    --------------------------------------------------------------------------------
      Node type "neu mode" has: 1
    --------------------------------------------------------------------------------
      Node "npu_0": type "neu mode", inputs "['input']", outputs "['output']"

    Tensor information:
    --------------------------------------------------------------------------------
      ValueInfo "input": type UINT8, shape [1, 224, 224, 3],
      ValueInfo "output": type FLOAT, shape [1, 1000],
      Initializer "npu_0_params": type UINT8, shape [3740416],
      Initializer "npu_0_b1_data": type UINT8, shape [173256],

.. _model_simulator_20e_en:

----------------------
Simulation run
----------------------

This chapter introduces the basic operations of ``axmodel`` simulation operation. Using the ``pulsar2 run`` command, you can directly run the ``axmodel`` model generated by ``pulsar2 build`` on the ``PC`` without You can quickly get the results of the network model by running it on the board.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Preparing for simulation run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``preprocessing`` and ``postprocessing`` tools required for simulation runtime are included in the ``pulsar2-run-helper`` folder.

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

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Simulate running ``mobilenetv2``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Copy the ``compiled.axmodel`` generated in the :ref:`《Model Compilation》<model_compile_20e_en>` chapter to the ``pulsar2-run-helper/models`` path, and rename it to ``mobilenetv2.axmodel``

.. code-block:: shell

    root@xxx:/data# cp output/compiled.axmodel pulsar2-run-helper/models/mobilenetv2.axmodel

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Input data preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Enter the ``pulsar2-run-helper`` directory and use the ``cli_classification.py`` script to process ``cat.jpg`` into the input data format required by ``mobilenetv2.axmodel``.

.. code-block:: shell

    root@xxx:~/data# cd pulsar2-run-helper
    root@xxx:~/data/pulsar2-run-helper# python3 cli_classification.py --pre_processing --image_path sim_images/cat.jpg --axmodel_path models/mobilenetv2.axmodel --intermediate_path sim_inputs/0
    [I] Write [input] to 'sim_inputs/0/input.bin' successfully.

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Simulation model reasoning
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run the ``pulsar2 run`` command, use ``input.bin`` as the input data of ``mobilenetv2.axmodel`` and perform inference calculations, and output the inference results in ``output.bin``.

.. code-block:: shell

    root@xxx:~/data/pulsar2-run-helper# pulsar2 run --model models/mobilenetv2.axmodel --input_dir sim_inputs --output_dir sim_outputs --list list.txt
    Building native ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:00
    >>> [0] start
    write [output] to [sim_outputs/0/output.bin] successfully
    >>> [0] finish

^^^^^^^^^^^^^^^^^^^^^^^^
Output data processing
^^^^^^^^^^^^^^^^^^^^^^^^

Use the ``cli_classification.py`` script to post-process the ``output.bin`` data output by the simulation model inference to obtain the final calculation result.

.. code-block:: shell

    root@xxx:/data/pulsar2-run-helper# python3 cli_classification.py --post_processing --axmodel_path models/mobilenetv2.axmodel --intermediate_path sim_outputs/0
    [I] The following are the predicted score index pair.
    [I] 9.1132, 285
    [I] 8.8490, 281
    [I] 8.7169, 282
    [I] 8.0566, 283
    [I] 6.8679, 463

.. _onboard_running_20e_en:

----------------------------
Development board running
----------------------------

This chapter introduces how to obtain the ``compiled.axmodel`` model through the :ref:`《Model Compilation》<model_compile_20e_en>` chapter on the ``AX630C`` ``AX620AV200`` ``AX620Q`` development board.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Development board acquisition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Get the **AX630C DEMO Board** after signing an NDA with AXera through the corporate channel.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use the ax_run_model tool to quickly test model inference speed
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In order to facilitate users to evaluate the model, the :ref:`ax_run_model <ax_run_model_en>` tool is pre-built on the development board. This tool has several parameters and can easily test the model speed and accuracy.

Copy ``mobilenetv2.axmodel`` to the development board and execute the following command to quickly test the model inference performance (first infer 3 times to warm up to eliminate statistical errors caused by resource initialization, then infer 10 times, and calculate the average inference speed).

.. code-block:: shell

    /root # ax_run_model -m mobilenetv2.axmodel -w 3 -r 10
    Run AxModel:
          model: mobilenetv2.axmodel
            type: Half
            vnpu: Enable
        affinity: 0b11
          warmup: 3
          repeat: 10
          batch: { auto: 1 }
    pulsar2 ver: 1.8-beta1 6a7e59de
      engine ver: 2.5.1a
        tool ver: 2.2.1
        cmm size: 4414192 Bytes
    ------------------------------------------------------
    min =   1.933 ms   max =   1.999 ms   avg =   1.946 ms
    ------------------------------------------------------

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use the ax_classification tool to test single image inference results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. hint::

     The onboard running example has been packaged and placed in the ``demo_onboard_ax620e`` folder :download:`Click to download the example file <../examples/demo_onboard_ax620e.zip>`
     Unzip the downloaded file. ``ax_classification`` is a pre-cross-compiled classification model executable program that can be run on **AX630C DEMO Board** and ``mobilennetv2.axmodel`` is a compiled classification model. ``cat.jpg`` is the test image.

Copy ``ax_classification``, ``mobilennetv2.axmodel``, ``cat.jpg`` to the development board. If ``ax_classification`` lacks executable permissions, you can add it through the following command

.. code-block:: shell

    /root/sample # chmod a+x ax_classification  # add execution permissions
    /root/sample # ls -l
    total 15344
    -rwx--x--x    1 root     root       5704472 Aug 25 11:04 ax_classification
    -rw-------    1 root     root        140391 Aug 25 11:05 cat.jpg
    -rw-------    1 root     root       3922461 Aug 25 11:05 mobilenetv2.axmodel

``ax_classification`` input parameter description: 

.. code-block:: shell

    /root/sample # ./ ax_classification --help
    usage: ax_classification --model=string --image=string [options] ...
    options:
      -m, --model     joint file(a.k.a. joint model) (string)
      -i, --image     image file (string)
      -g, --size      input_h, input_w (string [=224,224])
      -r, --repeat    repeat count (int [=1])
      -?, --help      print this message

By executing the ``ax_classification`` program, the classification model can be run on the board. The results are as follows:

.. code-block:: shell

    /root/sample # ./ax_classification -m mobilenetv2.axmodel -i cat.jpg -r 100
    --------------------------------------
    model file : mobilenetv2.axmodel
    image file : ../images/cat.jpg
    img_h, img_w : 224 224
    --------------------------------------
    Engine creating handle is done.
    Engine creating context is done.
    Engine get io info is done.
    Engine alloc io is done.
    Engine push input is done.
    --------------------------------------
    topk cost time:0.10 ms
    9.1132, 285
    8.8490, 281
    8.7169, 282
    8.0566, 283
    6.8679, 463
    --------------------------------------
    Repeat 100 times, avg time 1.93 ms, max_time 1.95 ms, min_time 1.93 ms
    --------------------------------------

- It can be seen from here that the results of running the same ``mobilenetv2.axmodel`` model on the development board are consistent with the results of :ref:`《Simulation Run》<model_simulator_20e_en>`;
- For the relevant source code and compilation and generation details of the on-board executable program ``ax_classification``, please refer to :ref:`《Model Deployment Advanced Guide》 <model_deploy_advanced_en>`.
