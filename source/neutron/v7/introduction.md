# Neutron v7 Architecture

(neutron_v7_architecture)=

The **Neutron v7** architecture expands into an NPU system in which multiple NPU Cores work together. Unlike previous versions, which focus on vNPU partitioning within a single Core, Neutron v7 emphasizes multi-Core parallelism, shared on-chip memory, task-dependency synchronization, and coordinated scheduling of computation and data movement.

The compiler divides a model into executable tasks that are mapped to one or more NPU Cores. Each NPU Core contains compute EUs, data-movement EUs, Local OCM, and a SyncManager. Multiple Cores can exchange data and coordinate tasks through Shared OCM and SyncManager.

- **Multi-Core parallelism**: Model subgraphs or tasks can be scheduled across multiple NPU Cores to improve parallelism and throughput.
- **On-chip memory reuse**: Local OCM enables data reuse within a Core, while Shared OCM exchanges data across Cores and reduces DDR access pressure.

On the `AX8860` platform, the `--npu_mode` option of `pulsar2 build` specifies the number of NPU Cores used to compile the model. `NPU1`, `NPU2`, and `NPU4` correspond to 1, 2, and 4 NPU Cores respectively; they do not identify particular Core numbers.

(neutron_v7_platform_npu_mode)=

## Runtime modes and npu_mode

### AXEngine runtime modes

```{eval-rst}
.. list-table::
    :header-rows: 1
    :align: center
    :widths: 22 78

    * - Chip platform
      - Compute resources
    * - ``AX8860``
      - To be updated.
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
    * - ``AX8860``
      - ``NPU1``
      - One NPU Core. Each Core contains 4 ``CONV``, 4 ``TENG``, 2 ``SDMA``, and 4 ``DAU``.
    * - ``AX8860``
      - ``NPU2``
      - Two NPU Cores. Each Core contains 4 ``CONV``, 4 ``TENG``, 2 ``SDMA``, and 4 ``DAU``, with cross-Core ``Shared OCM`` support.
    * - ``AX8860``
      - ``NPU4``
      - Four NPU Cores. Each Core contains 4 ``CONV``, 4 ``TENG``, 2 ``SDMA``, and 4 ``DAU``, with cross-Core ``Shared OCM`` support.
```
