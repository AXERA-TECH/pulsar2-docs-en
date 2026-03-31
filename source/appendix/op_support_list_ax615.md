# NPU Operators support list (AX615)

This section introduces the **NPU** supports for the `ONNX` operator in `AX615`.

- Supported ONNX opset_version >= 11. For detailed operator description, please refer to [onnx Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md) .
- Some of the supported operators do not have standard ONNX definitions yet. If such operators are included in the model, please consult technical support.

:::{note}
"Not supported yet": Indicates that the current version of the operator implementation does not support it, but the NPU can theoretically support it, and subsequent versions may support it.

"Unlimited": Indicates that the current operator implementation can support it. Since the test may not necessarily cover the entire parameter space, if something unexpected happens, you can give us feedback and we will treat it as a BUG and fix it as soon as possible.

"Not supported": Indicates that the implementation of this attribute cannot be supported.
:::

| Operators             | Attrs limitation                                                                                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Abs                   | Unlimited                                                                                                                                            |
| Add                   | Unlimited                                                                                                                                            |
| ArgMax                | - axis: Unlimited <br> - keepdims: Unlimited <br> - select_last_index: Only supports setting to 0                                                                                                                                                      |
| ArgMin                | - axis: Unlimited <br> - keepdims: Unlimited <br> - select_last_index: Only supports setting to 0                                                                                                                                                      |
| AveragePool           | - auto_pad: Only supports NOTSET <br> - ceil_mode: Unlimited <br> - count_include_pad: Only supports setting to 1 <br> - kernel_shape: Unlimited <br> - pads: Unlimited <br> - strides: Unlimited                                                                                                                                                      |
| BatchNormalization    | - epsilon: Unlimited <br> - momentum: Not supported <br> - training_mode: Not supported                                                                                                                                                      |
| Cast                  | to:uint8/int8/uint16/int16/uint32/int32/float32                                                                                                      |
| Ceil                  | Unlimited                                                                                                                                            |
| Clip                  | - min: Unlimited <br> - max: Unlimited                                                                                                                                                      |
| Concat                | axis: Unlimited                                                                                                                                      |
| Constant              | Unlimited                                                                                                                                            |
| Conv                  | - auto_pad: Only supports NOTSET <br> - dilations: Unlimited <br> - group: Unlimited <br> - kernel_shape: Unlimited <br> - pads: Unlimited <br> - strides: Unlimited <br> - note: The efficiency would be lower when <br> - using DepthWise/Group Conv and dilation <br> - not equals to 1                                                                                                                                                      |
| ConvTranspose         | - auto_pad: Only supports NOTSET <br> - dilations: currently only setting to 1 <br> - group: Unlimited <br> - kernel_shape: Unlimited <br> - output_shape: Not supported yet <br> - pads: Unlimited <br> - strides: Unlimited <br> - note: The efficiency would be lower in DepthWise ConvTranspose. <br> - output_padding: output_padding_h \<= pads_bottom, output_padding_w \<= pads_right                                                                                                                                                      |
| DepthToSpace          | - blocksize: Unlimited <br> - mode: Currently only supports DCR                                                                                                                                                      |
| Div                   | Unlimited                                                                                                                                            |
| Elu                   | Unlimited                                                                                                                                            |
| Equal                 | Unlimited                                                                                                                                            |
| Erf                   | Unlimited                                                                                                                                            |
| Exp                   | Unlimited                                                                                                                                            |
| Expand                | Unlimited                                                                                                                                            |
| Flatten               | Unlimited                                                                                                                                            |
| Gather                | - axis: Unlimited <br> - indices: Currently only supports 1-D                                                                                                                                                      |
| Gelu                  | Unlimited                                                                                                                                            |
| Gemm                  | - alpha: Not supported yet <br> - beta: Not supported yet <br> - transA: Unlimited <br> - transB: Unlimited                                                                                                                                                      |
| GlobalAveragePool     | Unlimited                                                                                                                                            |
| GlobalMaxPool         | Unlimited                                                                                                                                            |
| Greater               | Unlimited                                                                                                                                            |
| GreaterOrEqual        | Unlimited                                                                                                                                            |
| HardSigmoid           | Unlimited                                                                                                                                            |
| HardSwish             | Unlimited                                                                                                                                            |
| Identity              | Unlimited                                                                                                                                            |
| InstanceNormalization | epsilon:Unlimited Only for small sizes                                                                                                               |
| LayerNormalization    | axis: Currently only supports -1 (i.e., the last dimension) Only for small sizes                                                                     |
| Less                  | Unlimited                                                                                                                                            |
| LessOrEqual           | Unlimited                                                                                                                                            |
| LpNormalization       | - axis: Currently only supports -1 (i.e., the last dimension) <br> - p: Only supports 1 or 2 Only for small sizes                                                                                                                                                      |
| LSTM                  | - activation_alpha: Not supported yet <br> - activation_beta: Not supported yet <br> - activations: Not supported yet <br> - clip: Not supported yet <br> - hidden_size: Unlimited <br> - input_forget: Not supported yet <br> - layout: Only supports setting to 0 <br> - B: Unlimited <br> - sequence_lens: Not supported <br> - initial_h: Unlimited <br> - initial_c: Unlimited <br> - P: Not supported yet <br> - direction: Supports "bidirectional", "reverse", "forward"                                                                                                                                                      |
| LeakyRelu             | Unlimited                                                                                                                                            |
| MatMul                | Unlimited                                                                                                                                            |
| Max                   | Unlimited                                                                                                                                            |
| Min                   | Unlimited                                                                                                                                            |
| Mish                  | Unlimited                                                                                                                                            |
| MaxPool               | - auto_pad: Only supports setting to NOTSET <br> - ceil_mode: Unlimited <br> - dilations: Only supports setting to 1 <br> - kernel_shape: Unlimited <br> - pads: Unlimited <br> - storage_order: Only supports setting to 0 <br> - strides: Unlimited                                                                                                                                                      |
| Mul                   | Unlimited                                                                                                                                            |
| Neg                   | Unlimited                                                                                                                                            |
| PRelu                 | For 4D tensor input, the channel dimension must be the second dimension, and the slope shape Currently only supports (channel,) or(1, channel, 1, 1) |
| Pad                   | - pads: Unlimited <br> - constant_value: Unlimited <br> - mode: Only supports "constant" <br> - axes: Not supported yet                                                                                                                                                      |
| Pow                   | Elementwise computation is not supported, the exponent must be provided as an initializer and must be a scalar.                                      |
| ReduceL2              | - axes: Unlimited <br> - keepdims: Unlimited <br> - noop_with_empty_axes: This parameter is not supported yet                                                                                                                                                      |
| ReduceMax             | - axes: Unlimited <br> - keepdims: Unlimited <br> - noop_with_empty_axes: This parameter is not supported yet                                                                                                                                                      |
| ReduceMax             | - axes: Unlimited <br> - keepdims: Unlimited <br> - noop_with_empty_axes: This parameter is not supported yet                                                                                                                                                      |
| ReduceMean            | - axes: Unlimited <br> - keepdims: Unlimited <br> - noop_with_empty_axes: This parameter is not supported yet                                                                                                                                                      |
| Relu                  | Unlimited                                                                                                                                            |
| Reshape               | shape: Unlimited                                                                                                                                     |
| Resize                | - mode: Supported options: "nearest" and "linear" <br> - scales: Unlimitednearest_mode: Only supports setting to round_prefer_ceil                                                                                                                                                      |
| Sigmoid               | Unlimited                                                                                                                                            |
| Slice                 | - starts: Unlimited <br> - ends: Unlimited <br> - axes: Unlimited <br> - steps: Unlimited                                                                                                                                                      |
| SpatialTransformer    | Interpolation mode: "bilinear", Border mode: "constant"(value is 0) Only for small sizes                                                             |
| Split                 | - axis: Unlimited <br> - num_outputs: Unlimited                                                                                                                                                      |
| Sqrt                  | Unlimited                                                                                                                                            |
| Silu                  | Unlimited                                                                                                                                            |
| Sin                   | Unlimited                                                                                                                                            |
| Swish                 | Unlimited                                                                                                                                            |
| Squeeze               | axes: Unlimited                                                                                                                                      |
| Softmax               | axis: Unlimited                                                                                                                                      |
| Softplus              | Unlimited                                                                                                                                            |
| SpaceToDepth          | blocksize: Unlimited                                                                                                                                 |
| Sub                   | Unlimited                                                                                                                                            |
| Tanh                  | Unlimited                                                                                                                                            |
| Topk                  | Unlimited                                                                                                                                            |
| Transpose             | perm: Unlimited                                                                                                                                      |
| Unsqueeze             | axes: Unlimited                                                                                                                                      |
