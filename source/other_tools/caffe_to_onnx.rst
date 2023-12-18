======================================
Tool usage instructions of caffe2onnx 
======================================

This chapter introduces the AX version of the caffe2onnx conversion tool, which is used to convert Caffe floating-point models into ONNX floating-point models.

.. note::

   The model semantics below are all floating point models.

----------------------------------
Convert Caffe model to ONNX model
----------------------------------

We provide three ways to convert Caffe models into ONNX models.

1. You can pass in a Caffe file to convert a specific Caffe model you specify:

   .. code:: bash

      python3 /opt/pulsar2/tools/convert_caffe_to_onnx.py
            --prototxt_path /path/to/your/model.prototxt
            --caffemodel_path /path/to/your/model.caffemodel
            --onnx_path /path/to/your/model.onnx  # optional
            --opset_version OPSET_VERSION  # default to ONNX opset 13

   A ".caffemodel" and its matching ".prototxt" file together form a Caffe model.
   You need to specify both ``--caffemodel_path`` and ``--prototxt_path`` to determine a
   Caffe model. The ``--onnx_path`` and ``--opset_version`` parameters are optional.
   The default value of ``--opset_version`` is 13.

   .. note::

      If you do not specify the ``--onnx_path`` command line argument, the generated ONNX model will
      Use the ".caffemodel" model file (specified by ``--caffemodel_path``)
      prefix and store it in the same directory as the ".caffemodel" file.

2. Or you can pass in a folder to convert all Caffe models in it:

   .. code:: bash

      python3 /opt/pulsar2/tools/convert_caffe_to_onnx.py
            --checkpoint_path /path/to/your/model/zoo
            --opset_version OPSET_VERSION  # default to ONNX opset 13

    This will recursively find all files with the suffix ".caffemodel" in the specified folder and their corresponding
    ".prototxt" file, this is a Caffe model, convert it to an ONNX model,
    And use the prefix of the Caffe model and save it with the suffix ".onnx".

   .. note::

      ".prototxt" and ".caffemodel" corresponding to the Caffe model
      The files need to be in the same folder and share a prefix.

3. Command line tools of caffe2onnx 

   The new version of the tool chain provides the caffe2onnx command line tool, and you can also use the following methods to convert the model.

   .. code:: bash

      caffe2onnx --convert --checkpoint_path /path/to/your/model/zoo

----------------------------------
Validate the converted ONNX model
----------------------------------

You can use the following command to split the original Caffe model and the converted ONNX model:

.. code:: bash

   python3 /opt/pulsar2/tools/validate_caffe_onnx.py
         --checkpoint_path /path/to/your/model/zoo

First, this will recursively find all files with ".onnx" as the suffix in the specified folder, and then match the corresponding files according to their prefixes.
".prototxt" and ".caffemodel" files to generate a random data set using ONNX Runtime. And
the Caffe inference tool performs inference and calculates the "Correlation", "Standard Deviation", "Cosine Similarity", "Normalized Relative Error",
"Max Difference" and "Mean Difference"

.. note::

   ".prototxt" and ".caffemodel" corresponding to the Caffe model
   The file and the converted ".onnx" file need to be in the same folder and share a prefix.

.. note::

   This step requires caffe to be installed.

.. note::

   The new version of the tool chain provides the caffe2onnx command line tool, and you can also use the following method to verify the converted model.
   
.. code:: bash

   caffe2onnx --validate --checkpoint_path /path/to/your/model/zoo
