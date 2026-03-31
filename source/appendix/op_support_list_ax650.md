# NPU Operators support list (AX650)

This section introduces the **NPU** supports for the `ONNX` operator in `AX650` `M76H`.

- Supported ONNX opset_version >= 11. For detailed operator description, please refer to [onnx Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md) .
- Some of the supported operators do not have standard ONNX definitions yet. If such operators are included in the model, please consult technical support.

:::{note}
"Not supported yet": Indicates that the current version of the operator implementation does not support it, but the NPU can theoretically support it, and subsequent versions may support it.

"Unlimited": Indicates that the current operator implementation can support it. Since the test may not necessarily cover the entire parameter space, if something unexpected happens, you can give us feedback and we will treat it as a BUG and fix it as soon as possible.

"Not supported": Indicates that the implementation of this attribute cannot be supported.
:::

| Operators             | Attrs limitation                                                                                                                                    |
| --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Abs                   | Unlimited                                                                                                                                           |
| Add                   | Unlimited                                                                                                                                           |
| And                   | Unlimited                                                                                                                                           |
| ArgMax                | - axis: Unlimited <br> - keepdims: Unlimited <br> - select_last_index: Only supports setting to 0                                                                                                                                                     |
| ArgMin                | - axis: Unlimited <br> - keepdims: Unlimited <br> - select_last_index: Only supports setting to 0                                                                                                                                                     |
| AveragePool           | - auto_pad: Only supports NOTSET <br> - ceil_mode: Unlimited <br> - count_include_pad: Only supports setting to 1 <br> - kernel_shape: Unlimited <br> - pads: Unlimited <br> - strides: Unlimited                                                                                                                                                     |
| BatchNormalization    | - epsilon: Unlimited <br> - momentum: Not supported <br> - training_mode: Not supported                                                                                                                                                     |
| Cast                  | to:uint8/int8/uint16/int16/uint32/int32/float32                                                                                                     |
| Ceil                  | Unlimited                                                                                                                                           |
| Clip                  | - min: Unlimited <br> - max: Unlimited                                                                                                                                                     |
| Concat                | axis: Unlimited                                                                                                                                     |
| Constant              | Unlimited                                                                                                                                           |
| ConstantOfShape       | Unlimited                                                                                                                                           |
| Conv                  | - auto_pad: Only supports NOTSET <br> - dilations: Unlimited <br> - group: Unlimited <br> - kernel_shape: Unlimited <br> - pads: Unlimited <br> - strides: Unlimited <br> - note: The efficiency would be lower when using DepthWise/Group Conv and dilation not equals to 1                                                                                                                                                     |
| ConvTranspose         | - auto_pad: Only supports NOTSET <br> - dilations: currently only setting to 1 <br> - group: Unlimited <br> - kernel_shape: Unlimited <br> - output_shape: Not supported yet <br> - pads: Unlimited <br> - strides: Unlimited <br> - note: The efficiency would be lower in DepthWise ConvTranspose. <br> - output_padding: output_padding_h \<= pads_bottom, output_padding_w \<= pads_right                                                                                                                                                     |
| Cos                   | Unlimited                                                                                                                                           |
| DepthToSpace          | - blocksize: Unlimited <br> - mode: currently Only supports DCR                                                                                                                                                     |
| Div                   | Unlimited                                                                                                                                           |
| Elu                   | Unlimited                                                                                                                                           |
| Equal                 | Unlimited                                                                                                                                           |
| Erf                   | Unlimited                                                                                                                                           |
| Exp                   | Unlimited                                                                                                                                           |
| Expand                | Unlimited                                                                                                                                           |
| Flatten               | Unlimited                                                                                                                                           |
| Floor                 | Unlimited                                                                                                                                           |
| Gather                | - axis: Unlimited <br> - indices: currently Only supports 1 dimension                                                                                                                                                     |
| GatherElements        | - axis: Unlimited                                                                                                                                   |
| GatherND              | Unlimited                                                                                                                                           |
| Gelu                  | Unlimited                                                                                                                                           |
| Gemm                  | - alpha: Not supported yet <br> - beta: Not supported yet <br> - transA: Unlimited <br> - transB: Unlimited                                                                                                                                                     |
| GlobalAveragePool     | Unlimited                                                                                                                                           |
| GlobalMaxPool         | Unlimited                                                                                                                                           |
| Greater               | Unlimited                                                                                                                                           |
| GreaterOrEqual        | Unlimited                                                                                                                                           |
| GridSample            | Unlimited                                                                                                                                           |
| GroupNormalization    | Unlimited                                                                                                                                           |
| HardSigmoid           | Unlimited                                                                                                                                           |
| HardSwish             | Unlimited                                                                                                                                           |
| Identity              | Unlimited                                                                                                                                           |
| InstanceNormalization | epsilon:Unlimited                                                                                                                                   |
| InverseSigmoid        | Unlimited                                                                                                                                           |
| LayerNormalization    | axis Only supports -1 (i.e. the last dimension)                                                                                                     |
| LeakyRelu             | Unlimited                                                                                                                                           |
| Less                  | Unlimited                                                                                                                                           |
| LessOrEqual           | Unlimited                                                                                                                                           |
| LpNormalization       | - axis currently Only supports -1 (i.e. the last dimension) <br> - p only supports 1 or 2                                                                                                                                                     |
| LSTM                  | - activation_alpha: Not supported yet <br> - activation_beta: Not supported yet <br> - activations: Not supported yet <br> - clip: Not supported yet <br> - hidden_size: Unlimited <br> - input_forget: Not supported yet <br> - layout: Only supports setting to 0 <br> - B: Unlimited <br> - sequence_lens: Not supported <br> - initial_h: Unlimited <br> - initial_c: Unlimited <br> - P: Not supported yet <br> - direction: Supports "bidirectional", "reverse", "forward"                                                                                                                                                     |
| LogSoftmax            | Unlimited                                                                                                                                           |
| MatMul                | Unlimited                                                                                                                                           |
| Max                   | Unlimited                                                                                                                                           |
| MaxPool               | - auto_pad: only setting to NOTSET <br> - ceil_mode: Unlimited <br> - dilations: only support 1 <br> - kernel_shape: Unlimited <br> - pads: Unlimited <br> - storage_order: only setting to 0 <br> - strides: Unlimited                                                                                                                                                     |
| Min                   | Unlimited                                                                                                                                           |
| Mish                  | Unlimited                                                                                                                                           |
| Mul                   | Unlimited                                                                                                                                           |
| Not                   | Unlimited                                                                                                                                           |
| Pad                   | - pads: Unlimited <br> - constant_value: Unlimited <br> - mode: Only supports constant <br> - axes: Not supported yet                                                                                                                                                     |
| Pow                   | not suppors elemwise's calculation, exponent only supports initializer form and is a scalar。                                                       |
| PRelu                 | When 4D tensor is input, the channel dimension is in the second dimension, and slope shape currently Only supports (channel,) or (1, channel, 1, 1) |
| ReduceL2              | - axes: Unlimited <br> - keepdims: Unlimited <br> - noop_with_empty_axes: This parameter is not supported yet                                                                                                                                                     |
| ReduceMax             | - axes: Unlimited <br> - keepdims: Unlimited <br> - noop_with_empty_axes: This parameter is not supported yet                                                                                                                                                     |
| ReduceMean            | - axes: Unlimited <br> - keepdims: Unlimited <br> - noop_with_empty_axes: This parameter is not supported yet                                                                                                                                                     |
| ReduceMin             | - axes: Unlimited <br> - keepdims: Unlimited <br> - noop_with_empty_axes: This parameter is not supported yet                                                                                                                                                     |
| ReduceSum             | - axes: Unlimited <br> - keepdims: Unlimited <br> - noop_with_empty_axes: This parameter is not supported yet                                                                                                                                                     |
| Relu                  | Unlimited                                                                                                                                           |
| Reshape               | shape: Unlimited                                                                                                                                    |
| Resize                | - mode: supports "nearest"、"linear" <br> - scales: Unlimited <br> - nearest_mode: Only supports setting to round_prefer_ceil                                                                                                                                                     |
| RMSNormalization      | Unlimited                                                                                                                                           |
| RoiAlign              | sampling_ratio: only support not equals to 0                                                                                                        |
| RotaryEmbedding       | Unlimited                                                                                                                                           |
| Sigmoid               | Unlimited                                                                                                                                           |
| Round                 | Unlimited                                                                                                                                           |
| ScatterElements       | Unlimited                                                                                                                                           |
| ScatterND             | Unlimited                                                                                                                                           |
| Sigmoid               | Unlimited                                                                                                                                           |
| Silu                  | Unlimited                                                                                                                                           |
| Sin                   | Unlimited                                                                                                                                           |
| Slice                 | - starts: Unlimited <br> - ends: Unlimited <br> - axes: Unlimited <br> - steps: Unlimited                                                                                                                                                     |
| Softmax               | axis: Unlimited                                                                                                                                     |
| Softplus              | Unlimited                                                                                                                                           |
| SpaceToDepth          | blocksize: Unlimited                                                                                                                                |
| SpatialTransformer    | The interpolation method is "bilinear", The boundary processing method 、 is "constant" (value = 0)                                                 |
| Split                 | - axis: Unlimited <br> - num_outputs: Unlimited                                                                                                                                                     |
| Sqrt                  | Unlimited                                                                                                                                           |
| Squeeze               | axes: Unlimited                                                                                                                                     |
| Sub                   | Unlimited                                                                                                                                           |
| Swish                 | Unlimited                                                                                                                                           |
| Tanh                  | Unlimited                                                                                                                                           |
| Tile                  | Unlimited                                                                                                                                           |
| Topk                  | Unlimited                                                                                                                                           |
| Transpose             | perm: Unlimited                                                                                                                                     |
| Unsqueeze             | axes: Unlimited                                                                                                                                     |
| Where                 | Unlimited                                                                                                                                           |
| Xor                   | Unlimited                                                                                                                                           |
