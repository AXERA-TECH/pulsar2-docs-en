# Common configuration examples

This section provides configuration examples for common `pulsar2 build` scenarios, which can be quickly referenced and reused. All examples are based on the `AX650` platform.

:::{note}
- For the complete definition of configuration fields, please refer to {ref}`Configuration File Detailed Description <config_details>`
- `tensor_name` must match the actual tensor names defined in the ONNX model. You can check them via `onnx inspect --io model.onnx`
:::

(rgb_input_config)=

## RGB input

This is the most common configuration for image models. `input_processors` declares the runtime input data attributes of `compiled.axmodel`. The toolchain automatically embeds preprocessing operators into the model based on the configuration (such as dtype conversion, normalization, and layout conversion).

:::{warning}
The combination of `tensor_format` and `src_format` does **not** support RGB ↔ BGR channel swapping. If you set `src_format` to `BGR` and `tensor_format` to `RGB` (or vice versa), the compiled model will **not** embed a channel-reorder operator. Color space conversion is only supported in the {ref}`YUV input <yuv_input_config>` scenario.
:::

### Preprocessing is done inside compiled.axmodel

Embed preprocessing (normalization and layout conversion) into `compiled.axmodel` with the following configuration:

```json
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
      "tensor_layout": "NHWC",
      "src_format": "BGR",
      "src_dtype": "U8",
      "src_layout": "NHWC"
    }
  ],
  "compiler": {
    "check": 0
  }
}
```

**Key configuration:**

- Set `src_dtype` to `U8`: the input of `compiled.axmodel` becomes U8, and the toolchain automatically inserts an `AxDequantizeLinear` dequantization operator in the frontend to convert U8 to FP32 as required by the model.
- Set `src_layout` to `NHWC`: the toolchain inserts an `AxTranspose` operator to convert NHWC to NCHW as required by the model.
- `calibration_mean` / `calibration_std`: the toolchain inserts an `AxNormalize` operator to perform normalization.

You can confirm that preprocessing operators are embedded from the build log (check the output after `Building native`):

```bash
... | WARNING  | yamain.command.load_model:pre_process:616 - preprocess tensor [input]
... | INFO     | yamain.command.load_model:pre_process:618 - tensor: input, (1, 224, 224, 3), U8
... | INFO     | yamain.command.load_model:pre_process:619 - op: op:pre_dequant_1, AxDequantizeLinear, {'const_inputs': {'x_zeropoint': array(0, dtype=int32), 'x_scale': array(1., dtype=float32)}, 'output_dtype': <class 'numpy.float32'>, 'quant_method': 0}
... | INFO     | yamain.command.load_model:pre_process:618 - tensor: tensor:pre_norm_1, (1, 224, 224, 3), FP32
... | INFO     | yamain.command.load_model:pre_process:619 - op: op:pre_norm_1, AxNormalize, {'dim': 3, 'mean': [103.93900299072266, 116.77899932861328, 123.68000030517578], 'std': [58.0, 58.0, 58.0], 'output_dtype': FP32}
... | INFO     | yamain.command.load_model:pre_process:618 - tensor: tensor:pre_transpose_1, (1, 224, 224, 3), FP32
... | INFO     | yamain.command.load_model:pre_process:619 - op: op:pre_transpose_1, AxTranspose, {'perm': [0, 3, 1, 2]}
... | WARNING  | yamain.command.load_model:post_process:627 - postprocess tensor [output]
```

Between `preprocess tensor [input]` and `postprocess tensor [output]`, the log shows three preprocessing operators:

- `AxDequantizeLinear`: U8 → FP32 dtype conversion
- `AxNormalize`: normalization (subtract mean, divide by std)
- `AxTranspose`: NHWC → NCHW layout conversion

### Preprocessing is NOT done inside compiled.axmodel

If you want to perform preprocessing on CPU side (normalization, layout conversion, etc.) and then feed it to NPU for inference, you should configure `input_processors` to be exactly the same as the floating-point model input:

```json
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
      "tensor_layout": "NCHW",
      "src_format": "BGR",
      "src_dtype": "FP32",
      "src_layout": "NCHW",
      "mean": [0, 0, 0],
      "std": [1, 1, 1]
    }
  ],
  "compiler": {
    "check": 0
  }
}
```

**Key configuration:**

- Set `src_dtype` to `FP32`: same as the model input type, no dtype-conversion operator is inserted.
- Set `src_layout` to `NCHW`: same as the model input layout, no layout-conversion operator is inserted.
- Explicitly set `mean` to `[0, 0, 0]` and `std` to `[1, 1, 1]`: override the default values from `calibration_mean` / `calibration_std` so that no normalization operator is inserted.

:::{attention}
You must explicitly configure `mean` and `std`. If they are not configured, the toolchain will use `calibration_mean` / `calibration_std` by default and will still embed a normalization operator into the model.
:::

You can confirm that no preprocessing operator is embedded from the build log (check the output after `Building native`):

```bash
... | WARNING  | yamain.command.load_model:pre_process:616 - preprocess tensor [input]
... | WARNING  | yamain.command.load_model:post_process:627 - postprocess tensor [output]
```

If there is no `op:` line between `preprocess tensor [input]` and `postprocess tensor [output]`, it means `compiled.axmodel` does not include preprocessing operators. At runtime, users need to do the following by themselves:

1. Image decode and resize
2. BGR channel normalization (subtract mean, divide by std)
3. NHWC → NCHW layout conversion
4. Convert dtype to FP32

### Field description

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - Field
     - Description
   * - ``tensor_format``
     - The channel order used during model training (``RGB`` or ``BGR``), used for color space conversion when reading calibration data.
   * - ``src_format``
     - The channel order of the runtime input, usually ``BGR`` (OpenCV default).
   * - ``src_dtype``
     - Runtime input dtype. When set to ``U8``, a dequantization operator will be embedded; when set to ``FP32``, it will not be embedded.
   * - ``src_layout``
     - Runtime input layout. When set to ``NHWC``, layout conversion is automatically embedded; when set to ``NCHW``, it will not be embedded.
   * - ``mean`` / ``std``
     - Normalization parameters. By default, ``calibration_mean`` / ``calibration_std`` are used. Setting them explicitly to ``[0,0,0]`` / ``[1,1,1]`` disables embedding normalization.
```

:::{note}
The combination of `tensor_format` and `src_format` does **not** support RGB ↔ BGR channel swapping, and the compiled model will not reorder channels. Color space conversion is only used in the {ref}`YUV input <yuv_input_config>` scenario.
:::

(yuv_input_config)=

## YUV input

Cameras usually output YUV formats such as NV12/NV21. `Pulsar2` supports embedding YUV → RGB/BGR color space conversion into the model to avoid additional runtime overhead.

### NV12 (YUV420SP)

```json
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
      "src_format": "YUV420SP",
      "src_dtype": "U8",
      "src_layout": "NHWC",
      "csc_mode": "FullRange"
    }
  ],
  "compiler": {
    "check": 0
  }
}
```

### NV21 (YVU420SP)

Just change `src_format` to `YVU420SP`:

```json
{
  "input_processors": [
    {
      "tensor_name": "input",
      "tensor_format": "BGR",
      "src_format": "YVU420SP",
      "src_dtype": "U8",
      "src_layout": "NHWC",
      "csc_mode": "FullRange"
    }
  ]
}
```

### YUYV422

```json
{
  "input_processors": [
    {
      "tensor_name": "input",
      "tensor_format": "BGR",
      "src_format": "YUYV422",
      "src_dtype": "U8",
      "src_layout": "NHWC",
      "csc_mode": "LimitedRange"
    }
  ]
}
```

### Parameter description

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
     - Options
   * - ``src_format``
     - The YUV format of the runtime input
     - ``YUV420SP`` (NV12), ``YVU420SP`` (NV21), ``YUYV422``, ``UYVY422``
   * - ``tensor_format``
     - The expected color space of the model
     - ``BGR``, ``RGB``
   * - ``csc_mode``
     - Color space conversion mode
     - ``FullRange``, ``LimitedRange``, ``Matrix``
```

**csc_mode details:**

- `FullRange`: Full-range YUV conversion coefficients, suitable for most cameras.
- `LimitedRange`: Limited-range (BT.601/BT.709) coefficients, suitable for video streams.
- `Matrix`: user-defined 3×4 conversion matrix, configured via the `csc_mat` field.

**Custom CSC matrix:**

```json
{
  "input_processors": [
    {
      "tensor_name": "input",
      "tensor_format": "BGR",
      "src_format": "YUV420SP",
      "src_dtype": "U8",
      "src_layout": "NHWC",
      "csc_mode": "Matrix",
      "csc_mat": [1.164, 0.0, 1.596, -0.871,
                  1.164, -0.392, -0.813, 0.529,
                  1.164, 2.017, 0.0, -1.082]
    }
  ]
}
```

:::{warning}
- After configuring YUV input, `src_layout` will be automatically changed to `NHWC`.
- For NV12/NV21 input, the height of the input shape is 1.5× the original height (Y + UV planes).
- In `csc_mat`, the bias values (indices 3, 7, 11) must be in (-9, 8). The other parameters must be in (-524289, 524288).
- When validating accuracy on board, if `src_format` is YUV, it is recommended to use **IVE TDP for resize**. This preprocessing is aligned with OpenCV bilinear interpolation.
:::

(static_batch_config)=

## Static batch configuration

The compiler builds the model for the user-specified batch sizes. Weights are shared between batches, so the output model size is much smaller than the sum of individual batch models.

**Config file approach** — add `static_batch_sizes` under `compiler`:

```json
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
    "calibration_method": "MinMax"
  },
  "input_processors": [
    {
      "tensor_name": "input",
      "tensor_format": "BGR",
      "src_format": "BGR",
      "src_dtype": "U8",
      "src_layout": "NHWC"
    }
  ],
  "compiler": {
    "check": 0,
    "static_batch_sizes": [1, 2, 4]
  }
}
```

**Command line approach:**

```shell
pulsar2 build --target_hardware AX650 --input model.onnx --output_dir output --config config.json --compiler.static_batch_sizes 1 2 4
```

:::{hint}
Take mobilenetv2 as an example. The original input shape is `[1, 224, 224, 3]`. After setting `static_batch_sizes` to `[1, 2, 4]`, the input shape of the compilation output becomes `[4, 224, 224, 3]`.
:::

:::{attention}
- Static batch and dynamic batch modes are **mutually exclusive** and cannot be configured at the same time.
- If the model contains the `Reshape` operator, you may need to use the {ref}`Constant Data Patch <const_patch>` feature to change the batch dimension of shapes to `-1` or `0`.
:::

(dynamic_batch_config)=

## Dynamic batch configuration

The compiler automatically derives a batch-size set that the NPU can run efficiently and that does not exceed `max_dynamic_batch_size`. At runtime, the inference framework splits the actual batch into multiple runs if needed.

**Config file approach** — add `max_dynamic_batch_size` under `compiler`:

```json
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
    "calibration_method": "MinMax"
  },
  "input_processors": [
    {
      "tensor_name": "input",
      "tensor_format": "BGR",
      "src_format": "BGR",
      "src_dtype": "U8",
      "src_layout": "NHWC"
    }
  ],
  "compiler": {
    "check": 0,
    "max_dynamic_batch_size": 4
  }
}
```

**Command line approach:**

```shell
pulsar2 build --target_hardware AX650 --input model.onnx --output_dir output --config config.json --compiler.max_dynamic_batch_size 4
```

**Derivation rules:**

- The compiler starts from batch 1 and doubles the batch size (1 → 2 → 4 → ...). It stops when the batch exceeds the configured value or when the theoretical inference efficiency decreases.
- Theoretical inference efficiency = theoretical inference time / batch_size.

:::{hint}
When `max_dynamic_batch_size` is set to 4, the compilation output may include three batches: [1, 2, 4].

At runtime, the inference framework automatically splits the workload:

- batch=3 → internally runs batch 2 + batch 1 (two inferences)
- batch=9 → internally runs batch 4 + batch 4 + batch 1 (three inferences)
:::

(multi_input_config)=

## Multi-input configuration

When an ONNX model has multiple inputs (such as stereo vision, image + mask, multi-sensor fusion, etc.), you need to configure `input_configs` and `input_processors` for each input separately.

```json
{
  "model_type": "ONNX",
  "npu_mode": "NPU1",
  "quant": {
    "input_configs": [
      {
        "tensor_name": "rgb_image",
        "calibration_dataset": "./dataset/rgb_images.tar",
        "calibration_size": 32,
        "calibration_mean": [103.939, 116.779, 123.68],
        "calibration_std": [58.0, 58.0, 58.0]
      },
      {
        "tensor_name": "depth_map",
        "calibration_dataset": "./dataset/depth_maps.tar",
        "calibration_format": "Numpy",
        "calibration_size": 32,
        "calibration_mean": [0],
        "calibration_std": [1]
      }
    ],
    "calibration_method": "MinMax",
    "precision_analysis": false
  },
  "input_processors": [
    {
      "tensor_name": "rgb_image",
      "tensor_format": "BGR",
      "src_format": "BGR",
      "src_dtype": "U8",
      "src_layout": "NHWC"
    },
    {
      "tensor_name": "depth_map",
      "tensor_format": "GRAY",
      "src_format": "GRAY",
      "src_dtype": "FP32",
      "src_layout": "NCHW"
    }
  ],
  "compiler": {
    "check": 0
  }
}
```

**Key points:**

- Each input needs an independent `input_configs` entry and an independent `input_processors` entry.
- Different inputs can use different calibration datasets, data formats, and normalization parameters.
- `calibration_format` supports four formats: `Image` (default), `Numpy`, `Binary`, and `NumpyObject`.
- If all inputs share the same configuration, you can set `tensor_name` to `DEFAULT`.

**Mapping between calibration data and inputs:**

```{eval-rst}
.. list-table::
   :header-rows: 1

   * - Input tensor
     - Calibration format
     - Dataset content
     - Notes
   * - rgb_image
     - Image (default)
     - JPEG/PNG packed into a tar file
     - The toolchain reads images and normalizes automatically
   * - depth_map
     - Numpy
     - .npy files packed into a tar file
     - Must be preprocessed into numpy arrays that match the model input shape
```

:::{warning}
`tensor_name` must match the actual input names in the ONNX model. For simulation run, you need to prepare a bin file for each input, and the file name must match the tensor name.
:::

(skip_onnxsim_config)=

## Skip onnxslim (onnxsim)

By default, `pulsar2 build` runs internal graph optimization on the ONNX model using the open source `onnxslim` tool. In some scenarios (for example, the model has already been manually optimized, it contains custom operators, or the optimization causes compilation failure), you may need to skip these optimization steps.

**Command line approach:**

```shell
pulsar2 build --target_hardware AX650 --input model.onnx --output_dir output --config config.json --onnx_opt.disable_onnx_optimization true
```

**Config file approach** — add `onnx_opt` at the top level:

```json
{
  "model_type": "ONNX",
  "npu_mode": "NPU1",
  "onnx_opt": {
    "disable_onnx_optimization": true
  },
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
    "calibration_method": "MinMax"
  },
  "input_processors": [
    {
      "tensor_name": "input",
      "tensor_format": "BGR",
      "src_format": "BGR",
      "src_dtype": "U8",
      "src_layout": "NHWC"
    }
  ],
  "compiler": {
    "check": 0
  }
}
```
