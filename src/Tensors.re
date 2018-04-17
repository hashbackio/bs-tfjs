[@bs.module "@tensorflow/tfjs"]
external scalar : (Types.FFI.t, string) => Types.Tensor.tensor = "";

let scalar: Types.TensorLikeScalar.t => Types.Tensor.t =
  scalarValue =>
    scalarValue
    |> Types.TensorLikeScalar.sendToTfjs
    |> (((ffi, dType)) => scalar(ffi, dType |> Types.dTypeToJs))
    |> (tensor => Types.Tensor.Scalar(tensor));

[@bs.module "@tensorflow/tfjs"]
external tensor1d : (Types.FFI.t, string) => Types.Tensor.tensor = "";

let tensor1d: Types.TensorLike1D.t => Types.Tensor.t =
  tensor1DValue =>
    tensor1DValue
    |> Types.TensorLike1D.sendToTfjs
    |> (((ffi, dType)) => tensor1d(ffi, dType |> Types.dTypeToJs))
    |> (tensor => Types.Tensor.Tensor1D(tensor));

[@bs.module "@tensorflow/tfjs"]
external tensor2d : (Types.FFI.t, string) => Types.Tensor.tensor = "";

let tensor2d: Types.TensorLike2D.t => Types.Tensor.t =
  tensor2DValue =>
    tensor2DValue
    |> Types.TensorLike2D.sendToTfjs
    |> (((ffi, dType)) => tensor2d(ffi, dType |> Types.dTypeToJs))
    |> (tensor => Types.Tensor.Tensor2D(tensor));

[@bs.module "@tensorflow/tfjs"]
external tensor3d : (Types.FFI.t, string) => Types.Tensor.tensor = "";

let tensor3d: Types.TensorLike3D.t => Types.Tensor.t =
  tensor3DValue =>
    tensor3DValue
    |> Types.TensorLike3D.sendToTfjs
    |> (((ffi, dType)) => tensor3d(ffi, dType |> Types.dTypeToJs))
    |> (tensor => Types.Tensor.Tensor3D(tensor));

[@bs.module "@tensorflow/tfjs"]
external tensor4d : (Types.FFI.t, string) => Types.Tensor.tensor = "";

let tensor4d: Types.TensorLike4D.t => Types.Tensor.t =
  tensor4DValue =>
    tensor4DValue
    |> Types.TensorLike4D.sendToTfjs
    |> (((ffi, dType)) => tensor4d(ffi, dType |> Types.dTypeToJs))
    |> (tensor => Types.Tensor.Tensor4D(tensor));
