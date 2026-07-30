# Neutron v3 Architecture

(neutron_v3_architecture)=

:::{figure} ../../media/vNPU-ax650.png
:align: center
:alt: Neutron v3 vNPU
:::

**Neutron v3** consists of **1** NPU Core containing **6** `CONV`, **3** `TENG`, and **3** `SDMA` EUs. These physical EUs are organized into **3** basic resource groups, each containing **2** `CONV`, **1** `TENG`, and **1** `SDMA`.

At runtime, the virtual NPU mode can be set through `AX_ENGINE_NPU_ATTR_T.eHardMode` in `AX_ENGINE_Init` to determine how the three basic resource groups are partitioned into vNPUs. When converting a model with the `Pulsar2` compiler, specify `NPU1`, `NPU2`, or `NPU3` according to the required resource scale.

(neutron_v3_platform_npu_mode)=

## Runtime modes and npu_mode

### AXEngine runtime modes

```{eval-rst}
.. list-table::
    :header-rows: 1
    :align: center
    :widths: 30 70

    * - AXEngine runtime mode
      - Compute resources
    * - ``AX_ENGINE_VIRTUAL_NPU_STD``
      - A 1 + 1 + 1 partition that divides the three basic resource groups into three independent vNPUs. Each vNPU contains 2 ``CONV``, 1 ``TENG``, and 1 ``SDMA``.
    * - ``AX_ENGINE_VIRTUAL_NPU_BIG_LITTLE`` / ``AX_ENGINE_VIRTUAL_NPU_LITTLE_BIG``
      - A 2 + 1 or 1 + 2 partition. The large vNPU uses two basic resource groups, and the small vNPU uses one.
    * - ``AX_ENGINE_VIRTUAL_NPU_DISABLE``
      - A partition of 3. Multiple vNPUs are disabled, and all three basic resource groups are used as one complete NPU resource.
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
    * - ``AX650``
      - ``NPU1``
      - One Neutron v3 basic resource group containing 2 ``CONV``, 1 ``TENG``, and 1 ``SDMA``.
    * - ``AX650``
      - ``NPU2``
      - Two Neutron v3 basic resource groups containing 4 ``CONV``, 2 ``TENG``, and 2 ``SDMA``.
    * - ``AX650``
      - ``NPU3``
      - Three Neutron v3 basic resource groups containing 6 ``CONV``, 3 ``TENG``, and 3 ``SDMA``.
    * - ``M76H``
      - ``NPU1``
      - One Neutron v3 basic resource group containing 2 ``CONV``, 1 ``TENG``, and 1 ``SDMA``.
    * - ``M76H``
      - ``NPU2``
      - Two Neutron v3 basic resource groups containing 4 ``CONV``, 2 ``TENG``, and 2 ``SDMA``.
    * - ``M76H``
      - ``NPU3``
      - Three Neutron v3 basic resource groups containing 6 ``CONV``, 3 ``TENG``, and 3 ``SDMA``.
```
