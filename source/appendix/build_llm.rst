====================================================
Large Model Compilation (Experimental Stage)
====================================================

**Platforms Applicable**

- AX650N
- AX630C

**Verified Models**

- Llama2、Llama3、Llama3.2
- TinyLlama-1.1B
- Qwen1.5、Qwen2、Qwen2.5
- Phi2、Phi3
- MiniCPM、MiniCPM-V 2.0
- SmolLM
- ChatGLM3
- OpenBuddy

This chapter introduces basic operations for converting models(``*.safetensor`` or ``pytorch_model.bin``) from Huggingface into ``axmodel`` using the ``pulsar2`` tool. Please first refer to the :ref:`《Development Environment Preparation》 <dev_env_prepare>` section to complete the setup of the development environment.
The example model in this section is ``Qwen2-0.5B-Instruct``.

**Version Constraints**

This document is written based on Pulsar2 version 3.2.

**LLM ModelZoo**

- `AX650N <https://pan.baidu.com/s/1_LG-sPKnLS_LTWF3Cmcr7A?pwd=ph0e>`_
- `AX630C <https://pan.baidu.com/s/1X0aJTQM0bl8wsraspHnDUw?pwd=ifg5>`_

**Related Project AX-LLM**

This project explores the feasibility and related capability boundaries of landing commonly used LLMs (Large Language Models) on existing chip platforms, making it convenient for community developers to quickly evaluate and develop their own LLM applications.

- `AX-LLM <https://github.com/AXERA-TECH/ax-llm>`_

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Command Explanation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the ``Pulsar2`` toolchain, the ``pulsar2 llm_build`` command is used to complete the conversion of LLM models.

.. code-block:: shell

    root@xxx:/data# pulsar2 llm_build --help
    usage: pulsar2 llm_build [-h] [--input_path INPUT_PATH] [--output_path OUTPUT_PATH] [--prefill_len PREFILL_LEN]
                            [--parallel PARALLEL] [--model_config MODEL_CONFIG] [--kv_cache_len KV_CACHE_LEN]
                            [--post_topk POST_TOPK] [--post_weight_type {bf16,s8}] [-t {fp16,bf16,fp32}]
                            [-w {fp16,bf16,fp32,s8,s4}] [-c CHECK_LEVEL] [--chip {AX620E,AX650}] [--prompt PROMPT]

    optional arguments:
    -h, --help            show this help message and exit
    --input_path INPUT_PATH
                            path of model or npy path
    --output_path OUTPUT_PATH
                            path of dumpped ax_model
    --prefill_len PREFILL_LEN
                            token length of prefill
    --parallel PARALLEL   build parallel
    --model_config MODEL_CONFIG
                            config file
    --kv_cache_len KV_CACHE_LEN
                            length of kv_cache
    --post_topk POST_TOPK
                            post model output indices and prob
    --post_weight_type {bf16,s8}
                            post weight type
    -t {fp16,bf16,fp32}, --hidden_state_type {fp16,bf16,fp32}
                            hidden_state dtype
    -w {fp16,bf16,fp32,s8,s4}, --weight_type {fp16,bf16,fp32,s8,s4}
                            weight dtype
    -c CHECK_LEVEL, --check_level CHECK_LEVEL
                            check level 0:run 1:layer_check 2: cal 1+1
    --chip {AX620E,AX650}
                            chip
    --prompt PROMPT       prompt for check_level==2


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Download ax-llm-build
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    git clone https://github.com/AXERA-TECH/ax-llm-build.git

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Download Qwen2-0.5B-Instruct
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    cd ax-llm-build
    pip install -U huggingface_hub
    huggingface-cli download --resume-download Qwen/Qwen2-0.5B-Instruct --local-dir Qwen/Qwen2-0.5B-Instruct

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compile it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    pulsar2 llm_build --input_path Qwen/Qwen2-0.5B-Instruct/ --output_path Qwen/Qwen2-0.5B-w8a16/ --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX650

^^^^^^^^^^^^^^^^^^^^^
Log Information
^^^^^^^^^^^^^^^^^^^^^

.. code-block::

    pulsar2 llm_build --input_path Qwen/Qwen2-0.5B-Instruct/ --output_path Qwen/Qwen2-0.5B-w8a16/ --kv_cache_len 1023 --model_config config/qwen2-0.5B.json --hidden_state_type bf16 --weight_type s8 --parallel 8
    Config(
        model_name='Qwen2-0.5B-Instruct',
        model_type='qwen2',
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        hidden_size=896,
        intermediate_size=4864,
        vocab_size=151936,
        rope_theta=1000000.0,
        max_position_embeddings=32768,
        rope_partial_factor=1.0,
        rms_norm_eps=1e-06,
        norm_type='rms_norm',
        hidden_act='silu',
        hidden_act_param=0.03,
        scale_depth=1.4,
        scale_emb=1
    )
    2024-08-22 16:16:04.364 | SUCCESS  | yamain.command.llm_build:llm_build:100 - prepare llm model done!
    building llm decode layers   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 24/24 0:05:03
    building llm post layer   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:01:25
    2024-08-22 16:22:33.485 | SUCCESS  | yamain.command.llm_build:llm_build:160 - build llm model done!
    2024-08-22 16:22:47.861 | SUCCESS  | yamain.command.llm_build:llm_build:337 - check llm model done!

.. note::

    The example runs on a host configured to:

        - Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz
        - Memory 32G

    The whole process takes about ``6min``, and the conversion time varies slightly with different host configurations.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Embed file extract and optimize
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell  

    chmod +x ./tools/fp32_to_bf16
    chmod +x ./tools/embed_process.sh
    ./tools/embed_process.sh Qwen/Qwen2-0.5B-Instruct/ Qwen/Qwen2-0.5B-w8a16/

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Output files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell  

    root@xxx:/data/ax-llm-build# tree Qwen/Qwen2-0.5B-w8a16
    Qwen/Qwen2-0.5B-w8a16
    ├── model.embed_tokens.weight.bfloat16.bin
    ├── model.embed_tokens.weight.float32.bin # temp file, it can be deleted
    ├── model.embed_tokens.weight.npy # temp file, it can be deleted
    ├── qwen2_p128_l0_together.axmodel
    ├── qwen2_p128_l10_together.axmodel
    ├── qwen2_p128_l11_together.axmodel
    ├── qwen2_p128_l12_together.axmodel
    ├── qwen2_p128_l13_together.axmodel
    ├── qwen2_p128_l14_together.axmodel
    ├── qwen2_p128_l15_together.axmodel
    ├── qwen2_p128_l16_together.axmodel
    ├── qwen2_p128_l17_together.axmodel
    ├── qwen2_p128_l18_together.axmodel
    ├── qwen2_p128_l19_together.axmodel
    ├── qwen2_p128_l1_together.axmodel
    ├── qwen2_p128_l20_together.axmodel
    ├── qwen2_p128_l21_together.axmodel
    ├── qwen2_p128_l22_together.axmodel
    ├── qwen2_p128_l23_together.axmodel
    ├── qwen2_p128_l2_together.axmodel
    ├── qwen2_p128_l3_together.axmodel
    ├── qwen2_p128_l4_together.axmodel
    ├── qwen2_p128_l5_together.axmodel
    ├── qwen2_p128_l6_together.axmodel
    ├── qwen2_p128_l7_together.axmodel
    ├── qwen2_p128_l8_together.axmodel
    ├── qwen2_p128_l9_together.axmodel
    └── qwen2_post.axmodel


The files ``model.embed_tokens.weight.bfloat16.bin``, ``qwen_p128_l0.axmodel ~ qwen_p128_l23.axmodel``, ``qwen_post.axmodel`` are required for running on the board.

~~~~~~~~~~~~~~~~~~~~~~~
Development board run
~~~~~~~~~~~~~~~~~~~~~~~

This section describes how to run the LLM model on the ``AX650`` development board.

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Run large models using ax-llm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The files related to running the example have been uploaded to the web disk. Please download and refer to them
  
  - `Baidu Netdisk(AX650N) <https://pan.baidu.com/s/1_LG-sPKnLS_LTWF3Cmcr7A?pwd=ph0e>`_
  - `Baidu Netdisk(AX630C) <https://pan.baidu.com/s/1X0aJTQM0bl8wsraspHnDUw?pwd=ifg5>`_

.. code-block:: shell

    root@ax650:/mnt/qtang/llama_axera_cpp# ./run_qwen2_0.5B.sh
    [I][                            Init][ 128]: LLM init start
    3% | ██                                |   1 /  27 [0.27s<7.29s, 3.70 count/s] tokenizer init ok
    [I][                            Init][  26]: LLaMaEmbedSelector use mmap
    100% | ████████████████████████████████ |  27 /  27 [6.88s<6.88s, 3.92 count/s] init post axmodel ok,remain_cmm(11317 MB)
    [I][                            Init][ 244]: max_token_len : 1023
    [I][                            Init][ 249]: kv_cache_size : 128, kv_cache_num: 1023
    [I][                            Init][ 257]: prefill_token_num : 128
    [I][                            Init][ 266]: LLM init ok
    Type "q" to exit, Ctrl+c to stop current running
    >> who are you?
    [I][                             Run][ 464]: ttft: 129.16 ms
    I am a large language model created by Alibaba Cloud. I am called Qwen.
    
    [N][                             Run][ 603]: hit eos,avg 27.22 token/s

For the board run program compilation process, please refer to our open source project on github `AX-LLM <https://github.com/AXERA-TECH/ax-llm>`_


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Tokenizer Parser Explanation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Tokenizer parser in the ax-llm project uses both local modules and an HTTP Server. The local solution has tried two schemes: sentencepiece and tiktoken.

However, during actual debugging, we found that sentencepiece does not support special tokens well for different LLM models, requiring users to handle the splitting of special tokens themselves, which can easily lead to differences between the token ids on the board and those obtained from the AutoTokenizer module in the transformers library, ultimately affecting the correctness of the LLM output results.

Therefore, we recommend using the Tokenizer HTTP Server method for initial debugging to directly call the AutoTokenizer module in the transformers library for testing.

Features of the Tokenizer HTTP Server:

* Ensures correct token ids
* Facilitates the addition of chat templates
* Supports local and remote deployment
* Supports multi-user access

Example with the provided files for Qwen2.5 3B on the netdisk:

.. code-block:: shell

    root@xxx:/data/ax-llm-build# tree qwen2.5-3b-prefill-ax650/
    qwen2.5-3b-prefill-ax650/
    ├── main_prefill
    ├── qwen2.5-3B-prefill-ax650
    │   ├── model.embed_tokens.weight.bfloat16.bin
    │   ├── qwen2_p128_l0_together.axmodel
        ...
    │   ├── qwen2_p128_l12_together.axmodel
    │   └── qwen2_post.axmodel
    ├── qwen2.5_tokenizer
    │   ├── merges.txt
    │   ├── tokenizer_config.json
    │   ├── tokenizer.json
    │   └── vocab.json
    ├── qwen2.5_tokenizer.py
    ├── qwen.tiktoken
    ├── readme.txt
    └── run_qwen2.5_3B_prefill_ax650.sh

* qwen2.5_tokenizer: file related to tokenizer, be extracted from Qwen/Qwen2.5-3B-Instruct/
* qwen2.5_tokenizer.py: Tokenizer HTTP Server implemented in python

The running instructions are as follows:

* python qwen2.5_tokenizer.py --host xxx.xxx.xxx.xxx --port 12345, where --host xxx.xxx.xxx.xxx sets the IP address of the tokenizer parsing server. Ensure that the AX650N can access this address properly. It can be run natively on AX650N with python environment
* Change the IP address of --filename_tokenizer_model in run_qwen2.5_3B_prefill_ax650.sh to the same as that in Step 1
* Run run_qwen2.5_3B_prefill_ax650.sh

.. code-block:: shell

    root@xxx:/data/ax-llm-build# cat qwen2.5-3b-prefill-ax650/run_qwen2.5_3B_prefill_ax650.sh
    ./main_prefill \
    --template_filename_axmodel "qwen2.5-3B-prefill-ax650/qwen2_p128_l%d_together.axmodel" \
    --axmodel_num 36 \
    --tokenizer_type 2 \
    --filename_tokenizer_model http://xxx.xxx.xxx.xxx:12345 \
    --bos 0 --eos 0 \
    --filename_post_axmodel "qwen2.5-3B-prefill-ax650/qwen2_post.axmodel" \
    --filename_tokens_embed "qwen2.5-3B-prefill-ax650/model.embed_tokens.weight.bfloat16.bin" \
    --tokens_embed_num 151936 \
    --tokens_embed_size 2048 \
    --use_mmap_load_embed 1 \
    --live_print 1 \
    --continue 1 \
    --prompt "$1"

~~~~~~~~~~~~~~~~~~~~~~~
Other examples
~~~~~~~~~~~~~~~~~~~~~~~

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
MiniCPM-V 2.0
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Download MiniCPM-V 2.0**


.. code-block:: shell

    cd ax-llm-build
    pip install -U huggingface_hub
    huggingface-cli download --resume-download openbmb/MiniCPM-V-2 --local-dir openbmb/MiniCPM-V-2


**Get axmodel**

.. code-block:: shell

    pulsar2 llm_build --input_path openbmb/MiniCPM-V-2/ --output_path openbmb/MiniCPM-V-2-ax650 --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX650

Log Information

.. code-block::

    pulsar2 llm_build --input_path openbmb/MiniCPM-V-2/ --output_path openbmb/MiniCPM-V-2-ax650 --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX650 --parallel 8
    Config(
        model_name='openbmb/MiniCPM-V-2',
        model_type='minicpmv',
        num_hidden_layers=40,
        num_attention_heads=36,
        num_key_value_heads=36,
        hidden_size=2304,
        intermediate_size=5760,
        vocab_size=122753,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        rope_partial_factor=1.0,
        rms_norm_eps=1e-05,
        norm_type='rms_norm',
        hidden_act='silu',
        hidden_act_param=0.03,
        scale_depth=1.4,
        scale_emb=12,
        dim_model_base=256
    )
    2024-10-07 15:18:38.605 | SUCCESS  | yamain.command.llm_build:llm_build:101 - prepare llm model done!
    tiling op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3287/3287 0:00:44
    build op serially...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7610/7610 0:04:09
    build op...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11485/11485 0:00:00
    add ddr swap...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 253160/253160 0:00:42
    calc input dependencies...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 289230/289230 0:00:31
    calc output dependencies...   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 289230/289230 0:00:42
    assign eu heuristic   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 289230/289230 0:00:51
    assign eu onepass   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 289230/289230 0:00:10
    assign eu greedy   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 289230/289230 0:00:12
    building vision model   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:14:51
    building llm decode layers   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 40/40 0:04:24
    building llm post layer   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:02:19
    2024-10-07 15:40:14.676 | SUCCESS  | yamain.command.llm_build:llm_build:170 - build llm model done!
    2024-10-07 15:40:48.246 | SUCCESS  | yamain.command.llm_build:llm_build:349 - check llm model done!


**Get embed file**

.. code-block:: shell

    chmod +x ./tools/fp32_to_bf16
    chmod +x ./tools/embed_process.sh
    ./tools/embed_process_vl.sh openbmb/MiniCPM-V-2 openbmb/MiniCPM-V-2-ax650

The output file is:

.. code-block:: shell

    root@xxx: tree openbmb/MiniCPM-V-2-ax650/
    openbmb/MiniCPM-V-2-ax650/
    ├── minicpmv_p128_l0_together.axmodel
    ├── minicpmv_p128_l10_together.axmodel
    ...
    ├── minicpmv_p128_l19_together.axmodel
    ├── minicpmv_p128_l1_together.axmodel
    ├── minicpmv_p128_l20_together.axmodel
    ...
    ├── minicpmv_p128_l29_together.axmodel
    ├── minicpmv_p128_l2_together.axmodel
    ├── minicpmv_p128_l30_together.axmodel
    ...
    ├── minicpmv_p128_l39_together.axmodel
    ├── minicpmv_p128_l3_together.axmodel
    ...
    ├── minicpmv_p128_l8_together.axmodel
    ├── minicpmv_p128_l9_together.axmodel
    ├── minicpmv_post.axmodel
    ├── model.embed_tokens.weight.bfloat16.bin
    └── vpm_resampler.axmodel


**上板运行**

The upboard deployment project for MiniCPM-V requires a branch of minicpmv using ax-llm

- `ax-llm/tree/minicpm-v <https://github.com/AXERA-TECH/ax-llm/tree/minicpm-v>`_

.. figure:: ../media/ssd_dog.jpg
    :alt: pipeline
    :align: center

.. code-block:: shell

    root@ax650:/llm-test/minicpm-v-2.0# ./run_minicpmv-2.sh
    [I][                            Init][ 125]: LLM init start
    2% | █                                 |   1 /  44 [0.21s<9.11s, 4.83 count/s] tokenizer init ok
    [I][                            Init][  26]: LLaMaEmbedSelector use mmap
    100% | ████████████████████████████████ |  44 /  44 [33.54s<33.54s, 1.31 count/s] init vpm axmodel ok,remain_cmm(8086 MB)
    [I][                            Init][ 284]: max_token_len : 1023
    [I][                            Init][ 289]: kv_cache_size : 2304, kv_cache_num: 1023
    [I][                            Init][ 297]: prefill_token_num : 128
    [I][                            Init][ 306]: LLM init ok
    Type "q" to exit, Ctrl+c to stop current running
    prompt >> 描述下图片
    image >> ssd_dog.jpg
    [I][                          Encode][ 365]: image encode time : 728.507019 ms
    [I][                             Run][ 589]: ttft: 520.94 ms
    这幅图片展示了一只大而毛茸茸的狗，可能是拉布拉多或类似品种，坐在黄色和红色相间的门廊上。这只狗看起来在休息，它的目光朝向相机，表情平静。在狗的后面，有一辆红色自行车，车架上有黑色的装饰，停放在门廊上。自行车上挂着几个行李袋，表明它可能用于旅行或运输。背景中，可以看到一辆白色车辆，可能是汽车，停在门廊的后面。整个场景暗示了一个家庭环境，可能是在住宅区。

    [N][                             Run][ 728]: hit eos,avg 5.55 token/s

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Debugging instruction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pulsar2 llm_build`` enables debug precision debugging by using ``--check_level`` in the compile command

* ``--check_level 1``: Tests the similarity of the first layer.
* ``--check_level 2``: Specifies the contents of the prompt input to simulate the model file generated by the run compilation.

^^^^^^^^^^^^^^^^^^^^^
--check_level 1
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    pulsar2 llm_build --check_level 1 --input_path Qwen/Qwen2-0.5B-Instruct/ --output_path Qwen/Qwen2-0.5B-w8a16/ --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX650 

LOG：

.. code-block:: shell

    pulsar2 llm_build --check_level 1 --input_path Qwen/Qwen2-0.5B-Instruct/ --output_path Qwen/Qwen2-0.5B-w8a16/ --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX650 --parallel 8
    Config(
        model_name='Qwen2-0.5B-Instruct',
        model_type='qwen2',
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        hidden_size=896,
        intermediate_size=4864,
        vocab_size=151936,
        rope_theta=1000000.0,
        max_position_embeddings=32768,
        rope_partial_factor=1.0,
        rms_norm_eps=1e-06,
        norm_type='rms_norm',
        hidden_act='silu',
        hidden_act_param=0.03,
        scale_depth=1.4,
        scale_emb=1,
        dim_model_base=256
    )
    2024-10-07 01:23:28.414 | SUCCESS  | yamain.command.llm_build:llm_build:101 - prepare llm model done!
    building llm decode layers   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 24/24 0:00:39
    building llm post layer   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:01:26
    2024-10-07 01:25:34.765 | SUCCESS  | yamain.command.llm_build:llm_build:170 - build llm model done!
    2024-10-07 01:25:38.740 | INFO     | yamain.command.llm_build:llm_build:294 - decode layer0_gt layer0_got cos_sim is: 0.9986067835921196
    2024-10-07 01:25:45.421 | INFO     | yamain.command.llm_build:llm_build:325 - prefill layer0_gt layer0_got cos_sim is: 0.9986067835921196
    2024-10-07 01:25:45.421 | SUCCESS  | yamain.command.llm_build:llm_build:349 - check llm model done!

^^^^^^^^^^^^^^^^^^^^^
--check_level 2
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: shell

    pulsar2 llm_build --check_level 2 --prompt "<|im_start|>user\n1+1=?<|im_end|>\n<|im_start|>assistant\n" --input_path Qwen/Qwen2-0.5B-Instruct/ --output_path Qwen/Qwen2-0.5B-w8a16/ --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX650 

Because the debugging information of each hidden_layer is printed, the amount of information is a bit large, and only the more critical content is displayed here.

.. code-block:: shell

    pulsar2 llm_build --check_level 2 --prompt "<|im_start|>user\n1+1=?<|im_end|>\n<|im_start|>assistant\n" --input_path Qwen/Qwen2-0.5B-Instruct/ --output_path Qwen/Qwen2-0.5B-w8a16/ --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 128 --chip AX650
    Config(
        model_name='Qwen2-0.5B-Instruct',
        model_type='qwen2',
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        hidden_size=896,
        intermediate_size=4864,
        vocab_size=151936,
        rope_theta=1000000.0,
        max_position_embeddings=32768,
        rope_partial_factor=1.0,
        rms_norm_eps=1e-06,
        norm_type='rms_norm',
        hidden_act='silu',
        hidden_act_param=0.03,
        scale_depth=1.4,
        scale_emb=1,
        dim_model_base=256
    )
    2024-10-07 01:04:57.881 | SUCCESS  | yamain.command.llm_build:llm_build:101 - prepare llm model done!
    building llm decode layers   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 24/24 0:00:39
    building llm post layer   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1 0:01:26
    2024-10-07 01:07:04.398 | SUCCESS  | yamain.command.llm_build:llm_build:170 - build llm model done!
    Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
    load Qwen/Qwen2-0.5B-w8a16/qwen2_p128_l0_together
    load Qwen/Qwen2-0.5B-w8a16/qwen2_p128_l1_together
    ...
    load Qwen/Qwen2-0.5B-w8a16/qwen2_p128_l22_together
    load Qwen/Qwen2-0.5B-w8a16/qwen2_p128_l23_together
    2024-10-07 01:07:05.499 | INFO     | yasched.llm_utils:run:497 - simulate layer 0
    2024-10-07 01:07:11.902 | INFO     | yasched.llm_utils:run:503 - end simulate
    [[[-0.24707 0.0883789 -0.232422 ... -0.294922 0.0644531 -0.65625]
    [0.0649414 -0.183594 -0.251953 ... -0.248047 -0.0231934 -0.138672]
    [0.0766602 -0.0961914 0.152344 ... -0.0125732 0.106445 0.15625]
    ...
    [-0.0737305 -0.210938 -0.455078 ... -0.640625 0.0429688 -0.263672]
    [-0.0737305 -0.210938 -0.455078 ... -0.640625 0.0429688 -0.263672]
    [-0.0737305 -0.210938 -0.455078 ... -0.640625 0.0429688 -0.263672]]]
    2024-10-07 01:07:11.903 | INFO     | yasched.llm_utils:run:497 - simulate layer 1
    ...
    2024-10-07 01:09:35.992 | INFO     | yasched.llm_utils:run:497 - simulate layer 23
    2024-10-07 01:09:42.591 | INFO     | yasched.llm_utils:run:503 - end simulate
    [[[-1.25 0.222656 2.375 ... 2.07812 -0.410156 1.84375]
    [-0.289062 -1.08594 0.234375 ... 1.07812 -0.257812 -1.96094]
    [-0.0839844 -0.542969 0.636719 ... 3.21875 -0.351562 -2.01562]
    ...
    [-3.21875 -0.478516 1.42188 ... 4.8125 1.21875 -0.294922]
    [-3.21875 -0.478516 1.42188 ... 4.8125 1.21875 -0.294922]
    [-3.21875 -0.478516 1.42188 ... 4.8125 1.21875 -0.294922]]]
    2
    posibile ('\n', 0.0),('答案', 0.0),('Result', 0.0),('0', 0.0),('3', 0.0),('2', 1.0),('1', 0.0),('Answer', 0.0),('\\', 0.0),('4', 0.0)
    load Qwen/Qwen2-0.5B-w8a16/qwen2_p128_l0_together
    load Qwen/Qwen2-0.5B-w8a16/qwen2_p128_l1_together
    load Qwen/Qwen2-0.5B-w8a16/qwen2_p128_l2_together
    ...
    start_indice = 12
    2024-10-07 01:10:37.005 | INFO     | yasched.llm_utils:run:556 - simulate layer 23
    2024-10-07 01:10:38.859 | INFO     | yasched.llm_utils:run:562 - end simulate
    [-0.310547 -2.21875 0.871094 -1.86719 -0.546875]
    start_indice = 12
    <|im_end|>
    posibile ('\n', 0.0),('\\t', 0.0),('<|im_start|>', 0.0),(' \\', 0.0),('.', 0.0),('\n\n', 0.0),(' ', 0.0),('\\', 0.0),('<|im_end|>', 1.0),('\\n', 0.0)
    ====================================================================================================
    <|im_start|>user\n1+1=?<|im_end|>\n<|im_start|>assistant\n2<|im_end|>
    ====================================================================================================
    hit eos!
    2024-10-07 01:10:51.637 | SUCCESS  | yamain.command.llm_build:llm_build:349 - check llm model done!

