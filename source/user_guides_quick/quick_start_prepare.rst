=====================================
Development environment preparation
=====================================

This section describes the development environment preparations before using the ``Pulsar2`` toolchain.

``Pulsar2`` uses ``Docker`` container for toolchain integration. Users can load ``Pulsar2`` image files through ``Docker``, and then perform model conversion, compilation, simulation, etc. Therefore, in the development environment preparation stage, you only need to correctly install the ``Docker`` environment. Supported systems are ``MacOS``, ``Linux``, ``Windows``.

.. _dev_env_prepare:

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Install the Docker development environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Docker development environment can be installed on MacOS, Linux, and Windows operating systems. For the minimum configuration requirements and specific installation procedures for the installation environment under different operating systems, please refer to the following links:

- `MacOS install Docker environment <https://docs.docker.com/desktop/mac/install/>`_

- `Linux install Docker environment <https://docs.docker.com/engine/install/##server>`_

- `Windows install Docker environment <https://docs.docker.com/desktop/windows/install/>`_

After ``Docker`` is successfully installed, enter ``sudo docker -v``

.. code-block:: shell

    $ sudo docker -v
    Docker version 20.10.7, build f0df350

The above content is displayed, indicating that ``Docker`` has been installed successfully. The following will introduce the installation and startup of the ``Pulsar2`` toolchain ``Image``.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Install Pulsar2 toolchain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Take the system version of ``Ubuntu 20.04`` and the tool chain ``ax_pulsar2_${version}.tar.gz`` as an example to explain how to install the ``Pulsar2`` tool chain.

.. hint::

    In actual operation, be sure to replace ${version} with the corresponding toolchain version number.

How to obtain the tool chain:

- `BaiDu Pan <https://pan.baidu.com/s/1FazlPdW79wQWVY-Qn--qVQ?pwd=sbru>`_
- `Google Drive <https://drive.google.com/drive/folders/10rfQIAm5ktjJ1bRMsHbUanbAplIn3ium?usp=sharing>`_

^^^^^^^^^^^^^^^^^^^^^^^
Load Docker Image
^^^^^^^^^^^^^^^^^^^^^^^

Execute ``sudo docker load -i ax_pulsar2_${version}.tar.gz`` to import the docker image file. The following log will be printed if the image file is imported correctly:

.. code-block:: shell

    $ sudo docker load -i ax_pulsar2_${version}.tar.gz
    Loaded image: pulsar2:${version}

Once completed, execute ``sudo docker image ls``

.. code-block:: shell

    $ sudo docker image ls
    REPOSITORY   TAG          IMAGE ID       CREATED         SIZE
    pulsar2      ${version}   xxxxxxxxxxxx   9 seconds ago   3.27GB

You can see that the toolchain image has been successfully loaded, and then you can start the container based on this image.

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Start the toolchain image
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Execute the following command to start the ``Docker`` container. After successful operation, enter the ``bash`` environment

.. code-block:: shell

    $ sudo docker run -it --net host --rm -v $PWD:/data pulsar2:${version}

----------------------
Version Query
----------------------

``pulsar2 version`` is used to get the version information of the tool.

Example Results

.. code-block:: bash

    root@xxx:/data# pulsar2 version
    version: ${version}
    commit: xxxxxxxx

.. _prepare_data:

----------------------
Data preparation
----------------------

.. hint::

    The **original model** , **data** , **image** , **simulation tool** required for **model compilation** and **simulation running** are already provided in the ``quick_start_example`` folder :download:`Click to download the example file <https://github.com/xiguadong/assets/releases/download/v0.1/quick_start_example.zip>` Then unzip the downloaded file and copy it to the ``/data`` path of ``docker``.

.. code-block:: shell

    root@xxx:~/data# ls
    config  dataset  model  output  pulsar2-run-helper

* ``model``: stores the original ``ONNX`` model ``mobilenetv2-sim.onnx`` (the calculation graph of ``onnxsim`` has been optimized in advance)
* ``dataset``: stores the compressed package of the dataset required for offline quantitative calibration (PTQ Calibration) (supports common compression formats such as tar, tar.gz, gz, etc.)
* ``config``: stores the configuration file ``config.json`` that depends on the operation
* ``output``: stores the result output
* ``pulsar2-run-helper``: a tool that supports ``axmodel`` to perform simulation in the X86 environment

After data preparation is completed, the directory tree structure is as follows:

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
