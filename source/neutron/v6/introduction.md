# Neutron v6 Architecture

(neutron_v6_architecture)=

:::{figure} ../../media/vNPU-ax615.png
:align: center
:alt: Neutron v6 vNPU
:::

**Neutron v6** iteratively optimizes the power consumption and area of the **Neutron v4** architecture. It retains an overall organization of **1** NPU Core, **1** `CONV`, **1** `TENG`, **1** `SDMA`, and optional dual-vNPU time-division multiplexing.

- Like Neutron v4, Neutron v6 divides EUs into compute and data-movement types. With time-division multiplexing enabled, two logical vNPUs can overlap the computation and communication stages of different inference tasks. The two logical vNPUs share on-chip resources, so each vNPU has less OCM available than the complete NPU resource with multiplexing disabled.
- Some Neutron v6 platforms expose only the single-NPU resource mode. See the runtime modes and `npu_mode` descriptions below for the available options.

(neutron_v6_platform_npu_mode)=

## Runtime modes and npu_mode

### AXEngine runtime modes

```{eval-rst}
.. list-table::
    :header-rows: 1
    :align: center
    :widths: 18 30 52

    * - Chip platform
      - AXEngine runtime mode
      - Compute resources
    * - ``AX615``
      - ``AX_ENGINE_VIRTUAL_NPU_ENABLE``
      - Enables logical vNPU partitioning and schedules two vNPUs on the same NPU Core through hardware time-division multiplexing. The two logical vNPUs share on-chip resources, reducing the OCM available to each vNPU.
    * - ``AX615``
      - ``AX_ENGINE_VIRTUAL_NPU_DISABLE``
      - Disables logical vNPU partitioning and hardware time-division multiplexing. One complete NPU resource is exposed at runtime, and the model uses all NPU resources for inference.
    * - ``M57``
      - ``AX_ENGINE_VIRTUAL_NPU_DISABLE``
      - Only the single-NPU resource mode is supported.
    * - ``AX637``
      - ``AX_ENGINE_VIRTUAL_NPU_DISABLE``
      - Only the single-NPU resource mode is supported.
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
    * - ``AX615``
      - ``NPU1``
      - Compilation resources for one logical vNPU in time-division multiplexing mode, with less OCM available than in complete NPU resource mode.
    * - ``AX615``
      - ``NPU2``
      - Complete NPU compilation resources with time-division multiplexing disabled. All NPU resources are used for inference, with more OCM available than to one logical vNPU in multiplexing mode.
    * - ``M57``
      - ``NPU1``
      - One NPU Core containing 1 ``CONV``, 1 ``TENG``, 1 ``DAU``, and 1 ``SDMA``.
    * - ``AX637``
      - ``NPU1``
      - One NPU Core containing 1 ``CONV``, 1 ``TENG``, 1 ``DAU``, and 1 ``SDMA``.
```
