.. _model_deploy_advanced:

====================================
Advanced Guide to Model Deployment
====================================

--------------------
Overview
--------------------

This chapter introduces how to use the NPU-related sample programs on the development board. For the source code of the relevant sample programs, refer to the ``msp/sample/npu`` directory in the SDK. For how to compile the NPU sample code, please refer to the "AX SDK User Guide".

~~~~~~~~~~~~~~~~~~~~
Running the Example
~~~~~~~~~~~~~~~~~~~~

**Running Preparation**

For ``AX650A``, ``AX650N``, ``M76H``, ``AX630C`` development boards, NPU related examples have been pre-installed in the ``/opt/bin/`` path, namely ``sample_npu_classification`` and ``sample_npu_yolov5s``.

For the ``AX620Q`` development board, since the 16M NorFlash solution is adopted by default, the above two examples are not included in the file system. You can mount the ``msp/out/bin/`` path in the SDK to the file system of the development board through NFS network mounting to obtain the above examples.

If it is prompted that the board space is insufficient, you can solve it by mounting the folder.

**Example of mounting ARM development board on MacOS**

.. hint::

    Due to the limited space on the board, folder sharing is usually required during testing. At this time, it is necessary to share the ``ARM`` development board with the host. Here we only take ``MacOS`` as an example.

The development machine needs the ``NFS`` service to mount the ``ARM`` development board, and the ``MacOS`` system comes with the ``NFS`` service. You only need to create the ``/etc/exports`` folder, and ``nfsd`` will automatically start and start using ``exports``.

``/etc/exports`` can be configured as follows:

.. code-block:: shell

    /path/your/sharing/directory -alldirs -maproot=root:wheel -rw -network xxx.xxx.xxx.xxx -mask 255.255.255.0

Parameter Explanation

.. list-table::
    :widths: 15 40
    :header-rows: 1

    * - Parameter name
      - Meaning
    * - alldirs
      - Share all files under the ``/Users`` directory. If you only want to share a folder, you can omit it.
    * - network
      - Mount the ARM development board IP address, which can be a network segment address.
    * - mask
      - Subnet mask, usually 255.255.255.0.
    * - maproot
      - Mapping rule. When ``maproot=root:wheel``, it means mapping the ``root`` user of the ``ARM`` board to the ``root`` user on the development machine, and the ``root`` group of ``ARM`` to the ``wheel`` (gid=0) group on ``MacOS``.
        If it is not specified, a ``nfsroot`` link failure error may occur.
    * - rw
      - Read and write operations, enabled by default

Modifying ``/etc/exports`` requires restarting the ``nfsd`` service

.. code-block:: bash

    sudo nfsd restart

If the configuration is successful, you can use

.. code-block:: bash

    sudo showmount -e

Command to view the mount information, for example, the output is ``/Users/skylake/board_nfs 10.168.21.xx``. After configuring the development machine, you need to execute the ``mount`` command on the ``ARM`` side.

.. code-block:: bash

    mount -t nfs -o nolock,tcp macos_ip:/your/shared/directory /mnt/directory

If there is a permission problem, you need to check whether the ``maproot`` parameter is correct.

.. hint::

    The ``network`` parameter can be configured in the form of a network segment, such as: ``10.168.21.0``. If ``Permission denied`` appears when mounting a single IP, you can try mounting within the network segment.

**Classification Model**

The following print information is based on the output of the AX650N development board. The print information of non-AX650N development boards shall be subject to the actual print.

.. code-block:: bash

    /root # sample_npu_classification -m /opt/data/npu/models/mobilenetv2.axmodel -i /opt/data/npu/images/cat.jpg -r 10
    --------------------------------------
    model file : /opt/data/npu/models/mobilenetv2.axmodel
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

**Detection Model**

.. code-block:: bash

    /root # sample_npu_yolov5s -m /opt/data/npu/models/yolov5s.axmodel -i /opt/data/npu/images/dog.jpg -r 10
    --------------------------------------
    model file : /opt/data/npu/models/yolov5s.axmodel
    image file : /opt/data/npu/images/dog.jpg
    img_h, img_w : 640 640
    --------------------------------------
    Engine creating handle is done.
    Engine creating context is done.
    Engine get io info is done.
    Engine alloc io is done.
    Engine push input is done.
    --------------------------------------
    post process cost time:2.25 ms
    --------------------------------------
    Repeat 10 times, avg time 7.65 ms, max_time 7.66 ms, min_time 7.65 ms
    --------------------------------------
    detection num: 3
    16:  91%, [ 138,  218,  310,  541], dog
    2:  69%, [ 470,   76,  690,  173], car
    1:  56%, [ 158,  120,  569,  420], bicycle
    --------------------------------------

--------------------
Other Examples
--------------------

Please refer to our open source projects on github:

- `AX-Samples <https://github.com/AXERA-TECH/ax-samples>`_
