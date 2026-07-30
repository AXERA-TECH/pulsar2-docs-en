# Neutron v4 Architecture

(neutron_v4_architecture)=

:::{figure} ../../media/vNPU-ax620e.png
:align: center
:alt: Neutron v4 vNPU
:::

**Neutron v4** consists of **1** NPU Core containing **1** `CONV`, **1** `TENG`, and **1** `SDMA`. Unlike Neutron v3, which forms vNPUs by combining physical resource groups, Neutron v4 uses hardware time-division multiplexing to virtualize **2** vNPUs on the same group of physical EUs.

- `CONV` and `TENG` are compute EUs responsible for convolution and tensor or vector computation during model inference. `SDMA` is a data-movement EU responsible for reads and writes between DDR and OCM and for on-chip data transfers.
- The hardware time-division multiplexing mechanism schedules compute and data-movement EUs between two logical vNPUs, allowing the computation and communication stages of different inference tasks to overlap. For example, while one vNPU performs `CONV` or `TENG` computation, the other can use `SDMA` to prepare inputs, load weights, or write results, improving overall utilization of the physical EUs. When time-division multiplexing is enabled, the two logical vNPUs share on-chip resources, so each vNPU has less OCM available than a complete NPU resource with multiplexing disabled.

(neutron_v4_platform_npu_mode)=

## Runtime modes and npu_mode

### AXEngine runtime modes

```{eval-rst}
.. list-table::
    :header-rows: 1
    :align: center
    :widths: 30 70

    * - AXEngine runtime mode
      - Compute resources
    * - ``AX_ENGINE_VIRTUAL_NPU_ENABLE``
      - Enables logical vNPU partitioning and virtualizes two vNPUs on the same group of physical EUs. The two logical vNPUs share physical EUs and OCM, reducing the OCM available to each vNPU.
    * - ``AX_ENGINE_VIRTUAL_NPU_DISABLE``
      - Disables logical vNPU partitioning and hardware time-division multiplexing. One complete NPU resource is exposed at runtime, and the model uses all NPU resources for inference.
```

### Offline npu_mode and hardware resources

```{eval-rst}
.. list-table::
    :header-rows: 1
    :align: center
    :widths: 18 18 64

    * - Chip platform
      - ``npu_mode``
      - Hardware compute resources
    * - ``AX620E``
      - ``NPU1``
      - Compilation resources for one logical vNPU in time-division multiplexing mode, with less OCM available than in complete NPU resource mode.
    * - ``AX620E``
      - ``NPU2``
      - Complete NPU compilation resources with time-division multiplexing disabled. All NPU resources are used for inference, with more OCM available than to one logical vNPU in multiplexing mode.
    * - ``AX630C``
      - ``NPU1``
      - Compilation resources for one logical vNPU in time-division multiplexing mode, with less OCM available than in complete NPU resource mode.
    * - ``AX630C``
      - ``NPU2``
      - Complete NPU compilation resources with time-division multiplexing disabled. All NPU resources are used for inference, with more OCM available than to one logical vNPU in multiplexing mode.
    * - ``AX620Q``
      - ``NPU1``
      - Compilation resources for one logical vNPU in time-division multiplexing mode, with less OCM available than in complete NPU resource mode.
    * - ``AX620Q``
      - ``NPU2``
      - Complete NPU compilation resources with time-division multiplexing disabled. All NPU resources are used for inference, with more OCM available than to one logical vNPU in multiplexing mode.
```
