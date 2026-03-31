# LLM Build (Compile)

**Supported platforms**

- AX650A/AX650N/AX8850
  \- SDK ≥ v3.6.2
- AX630C
  \- SDK ≥ v3.0.0

**Verified models**

- Qwen3, Qwen2.5
- DeepSeek-R1-Distill
- MiniCPM4
- InternVL2_5, InternVL3
- ChatGLM3
- OpenBuddy
- SmolLM2
- Llama3.2
- Gemma2
- Phi2, Phi3
- TinyLlama

This chapter explains the basic workflow for converting models from Hugging Face. With the `pulsar2` tool, you can compile `*.safetensor` or `pytorch_model.bin` from a Hugging Face project into an `axmodel` model. Please first follow {ref}`《Development environment preparation》 <dev_env_prepare>` to set up the development environment.

The example model in this chapter is `Qwen3-0.6B`.

**Version constraint**

This document is written based on Pulsar2 version 5.2.

**LLM ModelZoo**

We periodically adapt popular LLMs in the community, including prebuilt models and on-board running examples.

- [Huggingface](https://huggingface.co/AXERA-TECH)

**Related project: AX-LLM**

This project explores what common LLMs (Large Language Models) can do on existing chip platforms, so developers can quickly evaluate and build their own LLM applications.

- [AX-LLM](https://github.com/AXERA-TECH/ax-llm)

## Command reference

In the `Pulsar2` toolchain, use `pulsar2 llm_build` to convert an LLM model.

```shell
root@xxx:/data# pulsar2 llm_build --help
usage: pulsar2 llm_build [-h] [--input_path INPUT_PATH] [--output_path OUTPUT_PATH] [--prefill_len PREFILL_LEN] [--parallel PARALLEL]
                         [--model_config MODEL_CONFIG] [--model_type MODEL_TYPE] [--kv_cache_len KV_CACHE_LEN] [--post_topk POST_TOPK]
                         [--post_weight_type {bf16,s8,fp8_e5m2,fp8_e4m3}] [-t {fp16,bf16,fp32}] [-w {fp16,bf16,fp32,s8,s4,fp8_e5m2,fp8_e4m3}] [-c CHECK_LEVEL]
                         [--chip {AX620E,AX650,LAMBERT}] [--prompt PROMPT] [--image_size IMAGE_SIZE] [--last_kv_cache_len LAST_KV_CACHE_LEN]
                         [--tensor_parallel_size TENSOR_PARALLEL_SIZE] [--ret_postnorm] [--ld_param_opt] [--npu_mode {NPU1,NPU2,NPU3}]

options:
  -h, --help            show this help message and exit
  --input_path INPUT_PATH
                        path of model or npy path (default: )
  --output_path OUTPUT_PATH
                        path of dumpped ax_model (default: .)
  --prefill_len PREFILL_LEN
                        token length of prefill (default: 0)
  --parallel PARALLEL   build parallel (default: 1)
  --model_config MODEL_CONFIG
                        config file (default: )
  --model_type MODEL_TYPE
                        config file (default: )
  --kv_cache_len KV_CACHE_LEN
                        length of kv_cache (default: 127)
  --post_topk POST_TOPK
                        post model output indices and prob (default: 0)
  --post_weight_type {bf16,s8,fp8_e5m2,fp8_e4m3}
                        post weight type (default: s8)
  -t {fp16,bf16,fp32}, --hidden_state_type {fp16,bf16,fp32}
                        hidden_state dtype (default: bf16)
  -w {fp16,bf16,fp32,s8,s4,fp8_e5m2,fp8_e4m3}, --weight_type {fp16,bf16,fp32,s8,s4,fp8_e5m2,fp8_e4m3}
                        weight dtype (default: s8)
  -c CHECK_LEVEL, --check_level CHECK_LEVEL
                        check level 0:run 1:layer_check 2: cal 1+1 (default: 0)
  --chip {AX620E,AX650,LAMBERT}
                        chip (default: AX650)
  --prompt PROMPT       prompt for check_level==2 (default: 1+1=)
  --image_size IMAGE_SIZE
                        vlm vision_part input_size (default: 224)
  --last_kv_cache_len LAST_KV_CACHE_LEN
                        last kv cache len (default: None)
  --tensor_parallel_size TENSOR_PARALLEL_SIZE
                        tensor parallel size (default: 0)
  --ret_postnorm        weather to return post_norm value in post layer (default: False)
  --ld_param_opt        ld_param_opt (default: False)
  --npu_mode {NPU1,NPU2,NPU3}
```

## Download the `ax-llm-build` project

If you want to compile a raw Hugging Face model into an `axmodel` file yourself, you can use the helper scripts in `ax-llm-build` to download the model, process embeddings, and so on.
If you are running a prebuilt model from [AXERA-TECH](https://huggingface.co/AXERA-TECH) on the board directly, you can skip this step.

```shell
git clone https://github.com/AXERA-TECH/ax-llm-build.git
```

## Download Qwen3-0.6B

```shell
cd ax-llm-build
pip install -U huggingface_hub
hf download Qwen/Qwen3-0.6B --local-dir Qwen/Qwen3-0.6B
```

## Build (compile)

```shell
pulsar2 llm_build --input_path Qwen/Qwen3-0.6B/  --output_path Qwen/Qwen3-0.6B-ax650 --hidden_state_type bf16 --kv_cache_len 1023 --prefill_len 128 --last_kv_cache_len 128 --last_kv_cache_len 256 --last_kv_cache_len 384 --last_kv_cache_len 512  --chip AX650 -c 1 --parallel 8
```

### Log example

```
pulsar2 llm_build --input_path Qwen/Qwen3-0.6B/  --output_path Qwen/Qwen3-0.6B-ax650 --hidden_state_type bf16 --kv_cache_len 1023 --prefill_len 128 --last_kv_cache_len 128 --last_kv_cache_len 256 --last_kv_cache_len 384 --last_kv_cache_len 512  --chip AX650 -c 1 --parallel 8
Config(
    model_name='Qwen3-0.6B',
    model_type='qwen3',
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    hidden_size=1024,
    head_dim=128,
    intermediate_size=3072,
    vocab_size=151936,
    rope_theta=1000000,
    max_position_embeddings=40960,
    rope_partial_factor=1.0,
    rope_local_base_freq=None,
    rms_norm_eps=1e-06,
    norm_type='rms_norm',
    hidden_act='silu',
    hidden_act_param=0.03,
    scale_depth=1.4,
    scale_emb=1,
    dim_model_base=256,
    origin_model_type='',
    quant=False,
    quant_sym=False,
    quant_bits=4,
    quant_group_size=128,
    rs_factor=32,
    rs_high_freq_factor=4.0,
    rs_low_freq_factor=1.0,
    rs_original_max_position_embeddings=8192,
    rs_rope_type='',
    rs_alpha=None,
    rs_beta_fast=None,
    rs_beta_slow=None,
    rs_mscale=None,
    rs_mscale_all_dim=None,
    rs_mrope_section=[16, 24, 24],
    interleaved_mrope=False,
    use_qk_norm=False,
    qk_norm_after_rope=False,
    layer_types=[],
    kv_cache_len=1023
)
2026-03-23 21:05:42.252 | SUCCESS  | yamain.command.llm_build:llm_build:258 - prepare llm model done!
building llm decode layers   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 28/28     0:02:50
building llm post layer   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1/1     0:01:23
2026-03-23 21:09:56.131 | SUCCESS  | yamain.command.llm_build:llm_build:368 - build llm model done!
2026-03-23 21:10:01.591 | INFO     | yamain.command.llm_build:llm_build:519 - decode layer0_gt layer0_got cos_sim is: 1.0
2026-03-23 21:10:12.356 | INFO     | yamain.command.llm_build:llm_build:553 - prefill layer0_gt layer0_got cos_sim is: 1.0
2026-03-23 21:10:12.357 | SUCCESS  | yamain.command.llm_build:llm_build:578 - check llm model done!
```

:::{note}
The host configuration used in this example:

> - Intel(R) Xeon(R) Gold 6336Y CPU @ 2.40GHz
> - Memory 32G

The whole process takes about `5min`. The conversion time can vary slightly across different host machines.
:::

### Extract and optimize embeddings

```shell
chmod +x ./tools/fp32_to_bf16
chmod +x ./tools/embed_process.sh
./tools/embed_process.sh Qwen/Qwen3-0.6B/ Qwen/Qwen3-0.6B-ax650/
```

### Output files

```shell
root@xxx:/data/ax-llm-build# tree Qwen/Qwen3-0.6B-ax650/
Qwen/Qwen3-0.6B-ax650/
├── model.embed_tokens.weight.bfloat16.bin
├── model.embed_tokens.weight.float32.bin   # Temporary file, can be deleted
├── model.embed_tokens.weight.npy           # Temporary file, can be deleted
├── qwen3_p128_l0_together.axmodel
├── qwen3_p128_l10_together.axmodel
├── qwen3_p128_l11_together.axmodel
├── qwen3_p128_l12_together.axmodel
├── qwen3_p128_l13_together.axmodel
├── qwen3_p128_l14_together.axmodel
├── qwen3_p128_l15_together.axmodel
├── qwen3_p128_l16_together.axmodel
├── qwen3_p128_l17_together.axmodel
├── qwen3_p128_l18_together.axmodel
├── qwen3_p128_l19_together.axmodel
├── qwen3_p128_l1_together.axmodel
├── qwen3_p128_l20_together.axmodel
├── qwen3_p128_l21_together.axmodel
├── qwen3_p128_l22_together.axmodel
├── qwen3_p128_l23_together.axmodel
├── qwen3_p128_l24_together.axmodel
├── qwen3_p128_l25_together.axmodel
├── qwen3_p128_l26_together.axmodel
├── qwen3_p128_l27_together.axmodel
├── qwen3_p128_l2_together.axmodel
├── qwen3_p128_l3_together.axmodel
├── qwen3_p128_l4_together.axmodel
├── qwen3_p128_l5_together.axmodel
├── qwen3_p128_l6_together.axmodel
├── qwen3_p128_l7_together.axmodel
├── qwen3_p128_l8_together.axmodel
├── qwen3_p128_l9_together.axmodel
└── qwen3_post.axmodel

0 directories, 32 files
```

Among them, `model.embed_tokens.weight.bfloat16.bin`, `qwen3_p128_l0_together.axmodel ~ qwen3_p128_l27_together.axmodel`, and `qwen3_post.axmodel` are required for running on the board.

## Run on the development board

This section shows how to run an LLM model on an `AX650` development board.

### Install `axllm`

We recommend using the installation script provided by the `ax-llm` project to install directly on the board:

```shell
curl -fsSL https://raw.githubusercontent.com/AXERA-TECH/ax-llm/axllm/install.sh | bash
```

After installation, run the following command to confirm `axllm` is installed successfully:

```shell
root@ax650:~/llm-test# axllm --help
Usage:
  axllm run <model_path> [options]    Run interactive chat mode
  axllm serve <model_path> [options]  Run HTTP API server mode

Arguments:
  model_path    Path to model directory containing config.json and model files

Serve options:
  --port <port> Server port (default: 8080)

Model directory structure:
  model_path/
    ├── config.json          # Model configuration
    ├── tokenizer.txt        # Tokenizer model
    ├── *.axmodel            # AXera model files
    └── post_config.json     # Post-processing config (optional)
```

If you want to learn how to build manually, see the [AX-LLM](https://github.com/AXERA-TECH/ax-llm) project documentation.

### Run the LLM with `ax-llm`

You can download all files for this example directly from Hugging Face. The current ModelZoo already includes the `tokenizer` files in each model repo, for example:

> - [Qwen3-0.6B](https://huggingface.co/AXERA-TECH/Qwen3-0.6B)

So in the current version you do not need to run a separate tokenizer parser anymore. Just download the Hugging Face model directory and pass that directory to `axllm run` or `axllm serve`. It will load and run automatically, which is simpler than older versions.

Take `AXERA-TECH/Qwen3-0.6B` as an example:

```shell
pip install -U huggingface_hub
hf download AXERA-TECH/Qwen3-0.6B --local-dir Qwen3-0.6B
```

### Run in CLI

```shell
root@ax650:~/llm-test# axllm run Qwen3-0.6B/
[I][                            Init][ 138]: LLM init start
tokenizer_type = 1
 96% | ███████████████████████████████   |  30 /  31 [3.25s<3.36s, 9.23 count/s] init post axmodel ok,remain_cmm(8662 MB)
[I][                            Init][ 199]: max_token_len : 2559
[I][                            Init][ 202]: kv_cache_size : 1024, kv_cache_num: 2559
[I][                            Init][ 205]: prefill_token_num : 128
[I][                            Init][ 209]: grp: 1, prefill_max_kv_cache_num : 1
[I][                            Init][ 209]: grp: 2, prefill_max_kv_cache_num : 512
[I][                            Init][ 209]: grp: 3, prefill_max_kv_cache_num : 1024
[I][                            Init][ 209]: grp: 4, prefill_max_kv_cache_num : 1536
[I][                            Init][ 209]: grp: 5, prefill_max_kv_cache_num : 2048
[I][                            Init][ 214]: prefill_max_token_num : 2048
[I][                            Init][  27]: LLaMaEmbedSelector use mmap
100% | ████████████████████████████████ |  31 /  31 [3.25s<3.25s, 9.54 count/s] embed_selector init ok
[I][                     load_config][ 282]: load config:
{
    "enable_repetition_penalty": false,
    "enable_temperature": false,
    "enable_top_k_sampling": false,
    "enable_top_p_sampling": false,
    "penalty_window": 20,
    "repetition_penalty": 1.2,
    "temperature": 0.9,
    "top_k": 10,
    "top_p": 0.8
}

[I][                            Init][ 272]: LLM init ok
Type "q" to exit
Ctrl+c to stop current running
"reset" to reset kvcache
"dd" to remove last conversation.
"pp" to print history.
----------------------------------------
prompt >> who are you?
[I][                      SetKVCache][ 406]: prefill_grpid:2 kv_cache_num:512 precompute_len:0 input_num_token:23
[I][                      SetKVCache][ 408]: current prefill_max_token_num:2048
[I][                      SetKVCache][ 409]: first run
[I][                             Run][ 457]: input token num : 23, prefill_split_num : 1
[I][                             Run][ 497]: prefill chunk p=0 history_len=0 grpid=1 kv_cache_num=0 input_tokens=23
[I][                             Run][ 519]: prefill indices shape: p=0 idx_elems=128 idx_rows=1 pos_rows=0
[I][                             Run][ 627]: ttft: 173.71 ms
<think>
Okay, the user asked, "Who are you?" I need to respond appropriately. Let me start by acknowledging their question. I should mention that I'm an AI assistant     designed to help with various tasks. It's important to keep the response friendly and open-ended so they feel comfortable sharing more. I should make sure to     highlight that I'm here to assist and that I'm not a person. Let me check if there's any additional information I should include to make the response more     helpful. Alright, that should cover it.
</think>

I'm an AI assistant designed to help with a wide range of tasks and questions. I'm here to assist you with anything you need! Let me know how I can help!

[N][                             Run][ 709]: hit eos,avg 15.68 token/s

[I][                      GetKVCache][ 380]: precompute_len:168, remaining:1880
prompt >> q
```

### Run as a service

`axllm` can start a model directory directly as an OpenAI API-compatible service. This is convenient for integration and secondary development.

```shell
root@ax650:~/llm-test# axllm serve Qwen3-0.6B/
[I][                            Init][ 138]: LLM init start
tokenizer_type = 1
 96% | ███████████████████████████████   |  30 /  31 [2.65s<2.74s, 11.30 count/s] init post axmodel ok,remain_cmm(8662 MB)
[I][                            Init][ 199]: max_token_len : 2559
[I][                            Init][ 202]: kv_cache_size : 1024, kv_cache_num: 2559
[I][                            Init][ 205]: prefill_token_num : 128
[I][                            Init][ 209]: grp: 1, prefill_max_kv_cache_num : 1
[I][                            Init][ 209]: grp: 2, prefill_max_kv_cache_num : 512
[I][                            Init][ 209]: grp: 3, prefill_max_kv_cache_num : 1024
[I][                            Init][ 209]: grp: 4, prefill_max_kv_cache_num : 1536
[I][                            Init][ 209]: grp: 5, prefill_max_kv_cache_num : 2048
[I][                            Init][ 214]: prefill_max_token_num : 2048
[I][                            Init][  27]: LLaMaEmbedSelector use mmap
100% | ████████████████████████████████ |  31 /  31 [2.65s<2.65s, 11.68 count/s] embed_selector init ok
[I][                     load_config][ 282]: load config:
{
    "enable_repetition_penalty": false,
    "enable_temperature": false,
    "enable_top_k_sampling": false,
    "enable_top_p_sampling": false,
    "penalty_window": 20,
    "repetition_penalty": 1.2,
    "temperature": 0.9,
    "top_k": 10,
    "top_p": 0.8
}

[I][                            Init][ 272]: LLM init ok
Starting server on port 8000 with model 'AXERA-TECH/Qwen3-0.6B'...
OpenAI API Server starting on http://0.0.0.0:8000
Max concurrency: 1
Models: AXERA-TECH/Qwen3-0.6B
```

After the service starts, you can call it through the standard OpenAI-compatible API.

### API call example

After the service starts, you can send standard HTTP requests to the OpenAI-compatible endpoints. The simplest example is:

```shell
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"AXERA-TECH/Qwen3-0.6B","messages":[{"role":"user","content":"你好"}]}'
```

If you want to test with the example script in the `ax-llm` project, you can do the following:

```shell
root@ax650:~/llm-test# curl -sOL https://raw.githubusercontent.com/AXERA-TECH/ax-llm/refs/heads/axllm/scripts/openai_demo.py

root@ax650:~/llm-test# python openai_demo.py --model AXERA-TECH/Qwen3-0.6B --api_url http://127.0.0.1:8000/v1
assistant:
<think>
Okay, the user just said "hello". I need to respond appropriately. Since they're greeting me, I should acknowledge their greeting. Maybe say "Hello!" in a friendly way. Let me check if there's any specific context I should consider, but the user didn't mention anything else. I should keep it simple and welcoming. Alright, time to send a response.
</think>

Hello! How can I assist you today? 😊
```

To customize the prompt, refer to:

```shell
root@ax650:~/llm-test# python openai_demo.py --model AXERA-TECH/Qwen3-0.6B --api_url http://127.0.0.1:8000/v1 --prompt "Please introduce yourself."
```

Note: `openai_demo.py` is only an example for calling the API. In real applications, we recommend integrating directly according to the OpenAI API specification.

For the board-side build flow of the runtime program, and more details about `run` / `serve` / API usage, see our open-source project on GitHub: [AX-LLM](https://github.com/AXERA-TECH/ax-llm)
