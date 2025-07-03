==================================
NPU Operators support list(M57)
==================================

This section introduces the **NPU** supports for the ``ONNX`` operator in ``M57``.

- Supported ONNX opset_version >= 11. For detailed operator description, please refer to `onnx Operators <https://github.com/onnx/onnx/blob/main/docs/Operators.md>`_.
- Some of the supported operators do not have standard ONNX definitions yet. If such operators are included in the model, please consult technical support.

 .. note:: 
    | "Not supported yet": Indicates that the current version of the operator implementation does not support it, but the NPU can theoretically support it, and subsequent versions may support it.
    | "Unlimited": Indicates that the current operator implementation can support it. Since the test may not necessarily cover the entire parameter space, if something unexpected happens, you can give us feedback and we will treat it as a BUG and fix it as soon as possible.
    | "Not supported": Indicates that the implementation of this attribute cannot be supported.

+-----------------------+---------------------------------------------+
| Operators              | Attrs limitation                           |
+=======================+=============================================+
| Abs                   | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Add                   | Unlimited                                   |
+-----------------------+---------------------------------------------+
| And                   | Unlimited                                   |
+-----------------------+---------------------------------------------+
| ArgMax                | | axis: Unlimited                           |
|                       | | keepdims: Unlimited                       |
|                       | | select_last_index: Only supports          |
|                       | |                    setting to 0           |
+-----------------------+---------------------------------------------+
| ArgMin                | | axis: Unlimited                           |
|                       | | keepdims: Unlimited                       |
|                       | | select_last_index: Only supports          |
|                       | |                    setting to 0           |
+-----------------------+---------------------------------------------+
| AveragePool           | | auto_pad: Only supports NOTSET            |
|                       | | count_include_pad: Only supports          |
|                       | |                    setting to 1           |
|                       | | kernel_shape: Unlimited                   |
|                       | | pads: Unlimited                           |
|                       | | strides: Unlimited                        |
+-----------------------+---------------------------------------------+
| BatchNormalization    | | epsilon: Unlimited                        |
|                       | | momentum: Not supported                   |
|                       | | training_mode: Not supported              |
+-----------------------+---------------------------------------------+
| Bevpool               | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Cast                  | to:                                         |
|                       |                                             |
|                       | uint8/int8/uint16/int16/uint32/int32/float32|
+-----------------------+---------------------------------------------+
| Ceil                  | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Clip                  | | min: Unlimited                            |
|                       | | max: Unlimited                            |
+-----------------------+---------------------------------------------+
| Concat                | axis: Unlimited                             |
+-----------------------+---------------------------------------------+
| Constant              | Unlimited                                   |
+-----------------------+---------------------------------------------+
| ConstantOfShape       | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Conv                  | | auto_pad: Only supports NOTSET            |
|                       | | dilations: Unlimited                      |
|                       | | group: Unlimited                          |
|                       | | kernel_shape: Unlimited                   |
|                       | | pads: Unlimited                           |
|                       | | strides: Unlimited                        |
|                       | | note: The efficiency would be lower when  |
|                       | |    using DepthWise/Group Conv and dilation|
|                       | |  not equals to 1                          |
+-----------------------+---------------------------------------------+
| ConvTranspose         | | auto_pad: Only supports NOTSET            |
|                       | | dilations:  currently only setting to 1   |
|                       | | group: Unlimited                          |
|                       | | kernel_shape: Unlimited                   |
|                       | | output_shape: Not supported yet           |
|                       | | pads: Unlimited                           |
|                       | | strides: Unlimited                        |
|                       | | note: The efficiency would be lower in    |
|                       | |          DepthWise ConvTranspose.         |
|                       |                                             |
|                       | output_padding: output_padding_h <=         |
|                       | pads_bottom, output_padding_w <=            |
|                       | pads_right                                  |
+-----------------------+---------------------------------------------+
| Cos                   | Unlimited                                   |
+-----------------------+---------------------------------------------+
| CumSum                | | exclusive: currently only supports 0      |
|                       | | reverse:  currently only supports 0       |
+-----------------------+---------------------------------------------+
| DepthToSpace          | | blocksize: Unlimited                      |
|                       | | mode: Unlimited                           |
+-----------------------+---------------------------------------------+
| Div                   | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Elu                   | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Equal                 | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Erf                   | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Exp                   | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Expand                | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Flatten               | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Floor                 | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Gather                | | axis: Unlimited                           |
|                       | | indices:  currently Only supports 1       |
|                       |             dimension                       |
+-----------------------+---------------------------------------------+
| GatherElements        | | axis: Unlimited                           |
+-----------------------+---------------------------------------------+
| GatherND              |   Unlimited                                 |
+-----------------------+---------------------------------------------+
| Gelu                  | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Gemm                  | | alpha: Not supported yet                  |
|                       | | beta: Not supported yet                   |
|                       | | transA: Unlimited                         |
|                       | | transB: Unlimited                         |
+-----------------------+---------------------------------------------+
| GlobalAveragePool     | Unlimited                                   |
+-----------------------+---------------------------------------------+
| GlobalMaxPool         | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Greater               | Unlimited                                   |
+-----------------------+---------------------------------------------+
| GreaterOrEqual        | Unlimited                                   |
+-----------------------+---------------------------------------------+
| GridSample            | Unlimited                                   |
+-----------------------+---------------------------------------------+
| GroupNormalization    | Unlimited                                   |
+-----------------------+---------------------------------------------+
| HardSigmoid           | Unlimited                                   |
+-----------------------+---------------------------------------------+
| HardSwish             | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Identity              | Unlimited                                   |
+-----------------------+---------------------------------------------+
| InstanceNormalization | epsilon:Unlimited                           |
+-----------------------+---------------------------------------------+
| InverseSigmoid        | Unlimited                                   |
+-----------------------+---------------------------------------------+
| LayerNormalization    | axis Only supports -1                       |
|                       | (i.e. the last dimension)                   |
+-----------------------+---------------------------------------------+
| LeakyRelu             | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Less                  | Unlimited                                   |
+-----------------------+---------------------------------------------+
| LessOrEqual           | Unlimited                                   |
+-----------------------+---------------------------------------------+
| LpNormalization       | | axis currently Only supports -1           |
|                       | | (i.e. the last dimension)                 |
|                       | | p only supports 1 or 2                    |
+-----------------------+---------------------------------------------+
| LSTM                  | | activation_alpha: Not supported yet       |
|                       | | activation_beta: Not supported yet        |
|                       | | activations: Not supported yet            |
|                       | | clip: Not supported yet                   |
|                       | | hidden_size: Unlimited                    |
|                       | | input_forget: Not supported yet           |
|                       | | layout: Only supports setting to 0        |
|                       | | B: Unlimited                              |
|                       | | sequence_lens: Not supported              |
|                       | | initial_h: Unlimited                      |
|                       | | initial_c: Unlimited                      |
|                       | | P: Not supported yet                      |
|                       | direction:                                  |
|                       | Supports "bidirectional","reverse","forward"|
+-----------------------+---------------------------------------------+
| LogSoftmax            | Unlimited                                   |
+-----------------------+---------------------------------------------+
| MatMul                | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Max                   | Unlimited                                   |
+-----------------------+---------------------------------------------+
| MaxPool               | | auto_pad: Only supports setting to NOTSET |
|                       | | ceil_mode: Unlimited                      |
|                       | | dilations: Only supports 为1              |
|                       | | kernel_shape: Unlimited                   |
|                       | | pads: Unlimited                           |
|                       | | storage_order: Only supports setting to 0 |
|                       | | strides: Unlimited                        |
+-----------------------+---------------------------------------------+
| Min                   | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Mish                  | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Mul                   | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Neg                   | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Not                   | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Pad                   | | pads: Unlimited                           |
|                       | | constant_value: Unlimited                 |
|                       | | mode: Only supports constant              |
|                       | | axes: Not supported yet                   |
+-----------------------+---------------------------------------------+
| Pow                   | not suppors elemwise's calculation,         |
|                       | exponent only supports initializer          |
|                       | form and is a scalar。                      |
+-----------------------+---------------------------------------------+
| PRelu                 | When 4D tensor is input, the channel        |
|                       |  dimension is in the second dimension, and  |
|                       |  slope shape currently Only supports        |
|                       |  (channel,) or (1, channel, 1, 1)           |
+-----------------------+---------------------------------------------+
| ReduceL2              | | axes: Unlimited                           |
|                       | | keepdims: Unlimited                       |
|                       | | noop_with_empty_axes: This parameter      |
|                       |   is not supported yet                      |
+-----------------------+---------------------------------------------+
| ReduceMax             | | axes: Unlimited                           |
|                       | | keepdims: Unlimited                       |
|                       | | noop_with_empty_axes: This parameter      |
|                       |   is not supported yet                      |
+-----------------------+---------------------------------------------+
| ReduceMean            | | axes: Unlimited                           |
|                       | | keepdims: Unlimited                       |
|                       | | noop_with_empty_axes: This parameter      |
|                       |   is not supported yet                      |
+-----------------------+---------------------------------------------+
| ReduceMin             | | axes: Unlimited                           |
|                       | | keepdims: Unlimited                       |
|                       | | noop_with_empty_axes: This parameter      |
|                       |   is not supported yet                      |
+-----------------------+---------------------------------------------+
| ReduceSum             | | axes: Unlimited                           |
|                       | | keepdims: Unlimited                       |
|                       | | noop_with_empty_axes: This parameter      |
|                       |   is not supported yet                      |
+-----------------------+---------------------------------------------+
| Relu                  | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Reshape               | shape: Unlimited                            |
+-----------------------+---------------------------------------------+
| Resize                | mode: supports "nearest"、"linear"          |
|                       | scales: Unlimited                           |
|                       | nearest_mode:                               |
|                       | Only supports setting to round_prefer_ceil  |
+-----------------------+---------------------------------------------+
| RMSNormalization      | Unlimited                                   |
+-----------------------+---------------------------------------------+
| RoiAlign              | sampling_ratio: only not support 0          |
+-----------------------+---------------------------------------------+
| RotaryEmbedding       | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Round                 | Unlimited                                   |
+-----------------------+---------------------------------------------+
| ScatterElements       | Unlimited                                   |
+-----------------------+---------------------------------------------+
| ScatterND             | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Sigmoid               | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Silu                  | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Sin                   | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Slice                 | | starts: Unlimited                         |
|                       | | ends: Unlimited                           |
|                       | | axes: Unlimited                           |
|                       | | steps: Unlimited                          |
+-----------------------+---------------------------------------------+
| Softmax               | axis: Unlimited                             |
+-----------------------+---------------------------------------------+
| Softplus              | Unlimited                                   |
+-----------------------+---------------------------------------------+
| SpaceToDepth          | blocksize: Unlimited                        |
+-----------------------+---------------------------------------------+
| SpatialTransformer    | The interpolation method is "bilinear",     |
|                       | The boundary processing method              |
|                       | is "constant" (value = 0)                   |
+-----------------------+---------------------------------------------+
| Split                 | | axis: Unlimited                           |
|                       | | num_outputs: Unlimited                    |
+-----------------------+---------------------------------------------+
| Sqrt                  | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Squeeze               | axes: Unlimited                             |
+-----------------------+---------------------------------------------+
| Sub                   | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Swish                 | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Tanh                  | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Tile                  | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Topk                  | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Transpose             | perm: Unlimited                             |
+-----------------------+---------------------------------------------+
| Unsqueeze             | axes: Unlimited                             |
+-----------------------+---------------------------------------------+
| Where                 | Unlimited                                   |
+-----------------------+---------------------------------------------+
| Xor                   | Unlimited                                   |
+-----------------------+---------------------------------------------+
