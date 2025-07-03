======================================
Tool usage instructions of caffe2onnx 
======================================

This chapter introduces the AX version of the caffe2onnx conversion tool, which is used to convert Caffe floating-point models into ONNX floating-point models.

.. note::

   The model semantics below are all floating point models.

----------------------------------
Convert Caffe model to ONNX model
----------------------------------

We provide command line tools to convert Caffe models to ONNX models. On X86 platforms, you can pass a folder to convert all Caffe models in it:

.. code:: bash

   caffe2onnx --convert --checkpoint_path /path/to/your/model/zoo

This will recursively find all files with the suffix ".caffemodel" and their corresponding ".prototxt" files in the specified folder. This is a Caffe model, convert it to an ONNX model, and save it with the suffix ".onnx" using the Caffe model's prefix.

On the ARM platform, the interface is as follows:

.. code:: bash

   caffe2onnx_cli --convert --checkpoint_path /path/to/your/model/zoo

.. note::

   The ".prototxt" and ".caffemodel" files corresponding to the Caffe model need to be in the same folder and share a prefix.

----------------------------------
Validate the converted ONNX model
----------------------------------

On the X86 platform, you can use the following command line tool to split the original Caffe model and the converted ONNX model:

.. code:: bash

   caffe2onnx --validate --checkpoint_path /path/to/your/model/zoo

First, this will recursively find all files with the ".onnx" suffix in the specified folder, then match the corresponding ".prototxt" and ".caffemodel" files according to their prefixes, generate a random dataset, use ONNXRuntime and Caffe inference tools for inference, and calculate the "Correlation", "Standard Deviation", "Cosine Similarity", "Normalized Relative Error", "Max Difference" and "Mean Difference" of the two.

.. note::

   ".prototxt" and ".caffemodel" corresponding to the Caffe model
   The file and the converted ".onnx" file need to be in the same folder and share a prefix.

.. note::

   This step requires caffe to be installed.

.. note::

   Since the compatibility of Caffe ARM platform is not very good, this function is not currently supported on ARM platform.