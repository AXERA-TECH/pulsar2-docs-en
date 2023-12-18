.. _ax_run_model_en:

==============================================
Instructions for using model evaluation tools
==============================================

In order to facilitate users to evaluate the model, the ``ax_run_model`` tool is pre-built on the development board. This tool has several parameters and can easily test the model speed and accuracy.
   .. code:: bash

      root@~# ax_run_model
      usage: ax_run_model --model=string [options] ...
         options:
         -m, --model                path to a model file (string)
         -r, --repeat               repeat times running a model (int [=1])
         -w, --warmup               repeat times before running a model to warming up (int [=1])
         -a, --affinity             npu affinity when running a model (int [=1])
         -v, --vnpu                 type of Visual-NPU inited {0=Disable, 1=STD, 2=BigLittle} (int [=0])
         -b, --batch                the batch will running (int [=0])
         -i, --input-folder         the folder of each inputs (folders) located (string [=])
         -o, --output-folder        the folder of each outputs (folders) will saved in (string [=])
         -l, --list                 the list of inputs which will test (string [=])
               --inputs-is-folder     each time model running needs inputs stored in each standalone input folders
               --outputs-is-folder    each time model running saved outputs stored in each standalone output folders
               --use-tensor-name      using tensor names instead of using tensor indexes when loading & saving io files
               --verify               verify outputs after running model
               --save-benchmark       save benchmark result(min, max, avg) as a json file
         -?, --help                 print this message


-----------------------------
Parameter Description
-----------------------------

There are two main parts to the evaluation tool parameters:

The first part is the parameters related to speed measurement:

.. data:: ax_run_model Parameter explanation

  --model

    - type of data: string
    - required or not: yes
    - description: specify the path to the test model

  --repeat

    - type of data: int
    - required or not: no
    - description: specify the number of loops to test and then display the speed in min/max/avg

  --warmup 
  
    - type of data: int
    - required or not: no
    - description: number of times to preheat before cycle test

  --affinity
  
    - type of data: int
    - required or not: no
    - description: affinity mask value, greater than 1 (0b001), less than 7 (0b111)

  --vnpu
  
    - type of data: int
    - required or not: no
    - description: virtual npu mode; 0 disables virtual npu; 1 standard split mode; 2 large and small core mode

  --batch 
  
    - type of data: int
    - required or not: no
    - description: specify the test batch

  --input-folder
  
    - type of data: string
    - required or not: no
    - description: specify the input folder for accuracy testing
  
  --output-folder
  
    - type of data: string
    - required or not: no
    - description: specify the output folder for accuracy testing

  --list
  
    - type of data: string
    - required or not: no
    - description: specify test list

  --inputs-is-folder
  
    - type of data: string
    - required or not: no
    - description: specify the input path --input-folder is composed of folders. If the parameter is not specified, it will take effect by default and will be discarded later.

  --outputs-is-folder
  
    - type of data: string
    - required or not: no
    - description: the specified output path --out-folder is composed of folders. If the parameters are not specified, it will take effect by default and will be discarded later.

  --use-tensor-name
  
    - type of data: string
    - required or not: no
    - description: specify to search for stimulus files by model input and output names. If not set, search by index. If parameters are not specified, they will take effect by default and will be discarded later.

  --verify
  
    - type of data: string
    - required or not: no
    - description: specify not to save the model output and the specified directory output file already exists, perform byte-by-byte comparison

-----------------------------
Usage example
-----------------------------

Taking the speed measurement requirement as an example, assuming that a single-core ``YOLOv5s`` model has been converted, and now you want to know the running speed of the board, you can run the following command:

   .. code:: bash

      root@~# ax_run_model -m /opt/data/npu/models/yolov5s.axmodel -w 10 -r 100
      Run AxModel:
            model: /opt/data/npu/models/yolov5s.axmodel
             type: NPU1
             vnpu: Disable
         affinity: 0b001
           repeat: 100
           warmup: 10
            batch: 1
      pulsar2 ver: 1.2-patch2 7e6b2b5f
       engine ver: V1.13.0 Apr 26 2023 16:48:53 1.1.0
         tool ver: 1.0.0
         cmm size: 12730188 Bytes
      ------------------------------------------------------
      min =   7.658 ms   max =   7.672 ms   avg =   7.662 ms
      ------------------------------------------------------


It can be seen from the printed log that VNPU is initialized to standard mode. At this time, the NPU is divided into three parts; and during this speed test, the affinity is set to the model with the largest affinity number.

By setting affinity, you can easily run multiple models at the same time for speed testing without writing code.

For example, in an SSH terminal window, run model a tens of thousands of times, and then in another SSH terminal, set different affinities, and observe the speed decrease of model b compared to the speed when model a is not run, you can get Under extremely high load conditions, model b is affected by the operation of model a (which may be more severe than the real situation). It should be noted that the ``-v`` parameter needs to be consistent in the two SSHs.

Another very common requirement is that after converting the model, you want to know the accuracy of the board. This can be tested through the accuracy parameters.

Take the classification model as an example to illustrate the use of directory structure and parameters. Here is a typical directory structure example:

   .. code:: bash

      root@~# tree /opt/data/npu/temp/
      /opt/data/npu/temp/
      |-- input
      |   `-- 0
      |       `-- data.bin
      |-- list.txt
      |-- mobilenet_v1.axmodel
      `-- output
         `-- 0
            `-- prob.bin

      4 directories, 4 files

The necessary parameters when testing accuracy are ``-m -i -o -l``, which respectively specify the model, input folder, output folder, and input list to be tested.

In addition, the output folders of these three modes are not empty, and existing files in the output folder will be overwritten when running the command; but if it is the output ``golden`` file that has been obtained from the ``Pulsar2`` simulation ,
it can be added the ``--verify`` parameter not to overwrite the output file, but to read the existing file in the output folder, and compare it bit by bit with the output of the current model in memory. This mode is used in suspected simulation and Particularly useful when board accuracy is misaligned

Parameter ``-l`` specifies the incentive folder list:

   .. code:: bash

      root@~# cat /opt/data/npu/temp/list.txt
      0
      root@~#

That is, in the example, the only incentive folder is specified. This parameter is very useful when the data set is large. For example, the input folder is a complete ``ImageNet`` data set with a lot of files;
but for this test it only want to test 10 files for verification, if there are no abnormalities, run the full test. Then such a requirement can be completed by creating two ``list.txt``. Only 10 lines of incentives are saved in one list, and another list file contains all the incentives.
The following is an example of the requirements of ``verify``. An example of running the ``ax_run_model`` parameter is as follows:

   .. code:: bash

      root@~# ax_run_model -m /opt/data/npu/temp/mobilenet_v1.axmodel -i /opt/data/npu/temp/input/ -o /opt/data/npu/temp/output/ -l /opt/data/npu/temp/list.txt --verify
       total found {1} input drive folders.
       infer model, total 1/1. Done.
       ------------------------------------------------------
       min =   3.347 ms   max =   3.347 ms   avg =   3.347 ms
       ------------------------------------------------------

      root@~#

It can be seen that the output of this model is aligned bit by bit under this set of input and output binary files. If not aligned, printing will report unaligned byte offsets.