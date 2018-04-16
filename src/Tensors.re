[@bs.module "@tensorflow/tfjs"]
external tensor : Types.FFI.t => Types.FFI.t = "tensor";

let tensor: Types.TfArrayInput.t => Types.Tensor.t =
  tensorLike =>
    tensorLike
    |> Types.TfArrayInput.sendToTfjs
    |> tensor
    |> Types.Tensor.unsafeCast;
