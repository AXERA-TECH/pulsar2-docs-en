# Quick Start(AX8860)

**This section applies to the following platforms:**

- AX8860

This section introduces the basic operations for converting an `ONNX` model. It uses the `pulsar2` tool to compile the `ONNX` model into an `axmodel`. First, follow {ref}`Development Environment Preparation <dev_env_prepare>` to set up the development environment.

The `AX8860` platform uses the Neutron v7 architecture. For more hardware architecture information, see {ref}`Introduction to AXera NPU (Neutron) <soc_introduction>`.

The example in this section uses the open-source `MobileNetv2` model.

## Pulsar2 toolchain commands

Commands in the `Pulsar2` toolchain start with `pulsar2`. The commands most relevant to users are `pulsar2 build`, `pulsar2 run`, and `pulsar2 version`.

- `pulsar2 build` converts an `onnx` model to an `axmodel`.
- `pulsar2 run` runs a simulation after model conversion.
- `pulsar2 version` displays the current toolchain version, which is normally required when reporting an issue.

```shell
root@xxx:/data# pulsar2 --help
usage: pulsar2 [-h] {version,build,run} ...

positional arguments:
  {version,build,run}

optional arguments:
  -h, --help           show this help message and exit
```

## Model compilation configuration

The `mobilenet_v2_build_config.json` file under `/data/config/` contains:

```shell
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
```

::::{attention}
Set the `tensor_name` field in `input_processors`, `output_processors`, and `input_configs` under `quant` according to the actual input or output node names of the model. It can also be set to `DEFAULT` to apply the current configuration to all inputs or outputs.

:::{figure} ../media/tensor_name.png
:align: center
:alt: tensor name
:::
::::

For details, see {ref}`Configuration File Details <config_details>`.

On the `AX8860` platform, the `npu_mode` field specifies the number of NPU Cores used to compile the model:

```{eval-rst}
.. list-table::
    :header-rows: 1
    :align: center

    * - ``npu_mode``
      - Number of NPU Cores
    * - ``NPU1``
      - 1 NPU Core
    * - ``NPU2``
      - 2 NPU Cores
    * - ``NPU4``
      - 4 NPU Cores
```

:::{note}
`AX8860` supports the `NPU1`, `NPU2`, and `NPU4` compilation modes. `NPU4` uses all four NPU Cores. `npu_mode` indicates the number of Cores, not particular Core numbers.
:::

(model_compile_ax8860)=

## Compile the model

Using `mobilenetv2-sim.onnx` as an example, run the following `pulsar2 build` command to generate `compiled.axmodel`:

```shell
pulsar2 build --target_hardware AX8860 --input model/mobilenetv2-sim.onnx --output_dir output --config config/mobilenet_v2_build_config.json
```

:::{warning}
Before compiling a model, ensure that the original model has been optimized with `onnxslim`. This converts the model into a static graph that is more suitable for `Pulsar2` compilation and can provide better inference performance. Use either of the following methods:

1. Run `onnxslim in.onnx out.onnx` directly inside the `Pulsar2` Docker container.
2. Add `--onnx_opt.enable_onnxsim true` when using `pulsar2 build` to convert the model. The default value is `false`.

For more information about `onnxslim`, visit the [official website](https://github.com/inisis/OnnxSlim).
:::

### Model compilation output

```shell
root@xxx:/data# tree output/
output/
|-- build_context.json
|-- compiled.axmodel               # Final AxModel to run on the board
|-- compiler                       # Compiler backend intermediate results and debug information
|   `-- debug
|       `-- subgraph_npu_0
|           `-- b1
|-- frontend
|   |-- optimized.data
|   `-- optimized.onnx             # Floating-point ONNX model after graph optimization
`-- quant                          # Quantization output and debug information
    |-- dataset
    |   `-- input
    |-- debug
    |   `-- io
    |-- quant_axmodel.data
    |-- quant_axmodel.json         # Quantization configuration
    `-- quant_axmodel.onnx         # Quantized model, QuantAxModel
```

`compiled.axmodel` is the final `.axmodel` file that can run on the board.

::::{note}
Because `.axmodel` is based on the **ONNX** model storage format, you can rename the file extension from `.axmodel` to `.axmodel.onnx` and open it directly with the **Netron** model visualization tool.

:::{figure} ../media/axmodel-netron.png
:align: center
:alt: axmodel netron
:::
::::

(model_simulator_ax8860)=

## Run a simulation

This section introduces the basic operations for `axmodel` simulation. The `pulsar2 run` command runs an `axmodel` generated by `pulsar2 build` directly on a `PC`, so you can quickly obtain model results without running it on a board.

### Prepare the simulation

Some models support only specific input data formats and produce outputs in model-specific formats. Before simulation, convert the input data into a format supported by the model; this is called `pre-processing`. After simulation, convert the output into a format that can be analyzed and inspected; this is called `post-processing`. The required `pre-processing` and `post-processing` tools are included in the `pulsar2-run-helper` directory.

### Simulate `mobilenetv2`

Copy the `compiled.axmodel` generated in {ref}`Compile the model <model_compile_ax8860>` to `pulsar2-run-helper/models` and rename it to `mobilenetv2.axmodel`.

```shell
root@xxx:/data# cp output/compiled.axmodel pulsar2-run-helper/models/mobilenetv2.axmodel
```

Enter the `pulsar2-run-helper` directory and use `cli_classification.py` to convert `cat.jpg` into the input format required by `mobilenetv2.axmodel`.

```shell
root@xxx:~/data# cd pulsar2-run-helper
root@xxx:~/data/pulsar2-run-helper# python3 cli_classification.py --pre_processing --image_path sim_images/cat.jpg --axmodel_path models/mobilenetv2.axmodel --intermediate_path sim_inputs/0
```

Run `pulsar2 run` with `input.bin` as the input to `mobilenetv2.axmodel`. The inference result is written to `output.bin`.

```shell
root@xxx:~/data/pulsar2-run-helper# pulsar2 run --model models/mobilenetv2.axmodel --input_dir sim_inputs --output_dir sim_outputs --list list.txt
```

Use `cli_classification.py` to post-process the `output.bin` produced by the simulation and obtain the final result.

```shell
root@xxx:/data/pulsar2-run-helper# python3 cli_classification.py --post_processing --axmodel_path models/mobilenetv2.axmodel --intermediate_path sim_outputs/0
```

:::{note}
Running a model on an `AX8860` board depends on the corresponding SDK, AXEngine runtime environment, and development-board image version. After compiling `compiled.axmodel`, use the target platform SDK documentation and {ref}`Advanced Model Deployment Guide <model_deploy_advanced>` to integrate it on the board.
:::
