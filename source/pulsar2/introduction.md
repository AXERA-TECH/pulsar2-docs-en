# Pulsar2 Toolchain overview

## Introduction

**Pulsar2** is an `all-in-one` new generation neural network compiler **independently developed** by [Axera](https://www.axera-tech.com/),
That is, **conversion**, **quantification**, **compilation**, and **heterogeneous** are four-in-one to achieve the **fast** and **efficient** deployment requirements of deep learning neural network models.
It is deeply customized and optimized for the new generation of `AX6`, `AX88`, `M7`, and `M5` series chips, making full use of the on-chip heterogeneous compute units (CPU+NPU) to improve neural-network model deployment efficiency.

**Special Note:**

- Tips in the toolchain documentation
  : - **Note**: Note content, further explanation of certain professional terms
    - **Hint**: Hint content, reminding users to confirm relevant information
    - **Attention**: Attention content, reminding users of relevant precautions for tool configuration
    - **Warning**: Warning content, reminding users to pay attention to the correct use of the tool chain. If the customer does not use it according to the Warning prompt content, incorrect results may occur.
- The commands in the tool chain document are compatible with on-board chips, such as `Pulsar2` supports `M76H`
- The **example commands** and **example output** in the tool chain documentation are all based on `AX650`.
- The computing power configuration of the specific chip is subject to the chip SPEC.

The core function of the `Pulsar2` tool chain is to compile the `.onnx` model into an `.axmodel` model that the chip can parse and run.

**Deployment Process**

:::{figure} ../media/deploy-pipeline.png
:align: center
:alt: pipeline
:::

## Guide to the content of subsequent chapters

- **Quick Start**: Development environment preparation and basic workflows for each chip platform.
- **Advanced Model Conversion**: How to use the `Pulsar2 Docker` toolchain to convert an `onnx` model into an `axmodel`.
- **Advanced Model Simulation**: How to simulate an `axmodel` on an `x86` platform and measure the difference between its inference results and the `onnx` results (internally called `bisection`).
- **Advanced On-board Model Execution**: How to run an `axmodel` on a board and obtain inference results on AXera SoC hardware.
- **Configuration File Reference**: Details of the configuration file used during model conversion and compilation.
- **Caffe-to-ONNX Tool**: How to convert a model exported by the Caffe AI training platform into the `onnx` format supported by the NPU toolchain.
- **On-board Model Performance and Accuracy Tool**: How to test model speed and accuracy on a board.
- **QAT 4W8F**: A brief guide to QAT 4W8F.
- **Functional Safety Statement**: The NPU toolchain functional-safety compliance statement.
- **Appendix**: The document appendix includes a list of supported operators and accuracy tuning suggestions

:::{note}
The so-called `bisection` is to compare the error between the inference results of different versions (file types) of the same model before and after the toolchain is compiled.
:::
