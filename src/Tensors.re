[@bs.module "@tensorflow/tfjs"]
external scalar : (Types.FFI.t, string) => Types.Scalar.t = "";

let scalar = scalarValue =>
  scalarValue
  |> Types.TensorLikeScalar.sendToTfjs
  |> (((ffi, dType)) => scalar(ffi, dType |> Types.dTypeToJs));

[@bs.module "@tensorflow/tfjs"]
external tensor1d : (Types.FFI.t, string) => Types.Tensor1D.t = "";

let tensor1d = tensor1DValue =>
  tensor1DValue
  |> Types.TensorLike1D.sendToTfjs
  |> (((ffi, dType)) => tensor1d(ffi, dType |> Types.dTypeToJs));

[@bs.module "@tensorflow/tfjs"]
external tensor2d : (Types.FFI.t, string) => Types.Tensor2D.t = "";

let tensor2d = tensor2DValue =>
  tensor2DValue
  |> Types.TensorLike2D.sendToTfjs
  |> (((ffi, dType)) => tensor2d(ffi, dType |> Types.dTypeToJs));

[@bs.module "@tensorflow/tfjs"]
external tensor3d : (Types.FFI.t, string) => Types.Tensor3D.t = "";

let tensor3d = tensor3DValue =>
  tensor3DValue
  |> Types.TensorLike3D.sendToTfjs
  |> (((ffi, dType)) => tensor3d(ffi, dType |> Types.dTypeToJs));

[@bs.module "@tensorflow/tfjs"]
external tensor4d : (Types.FFI.t, string) => Types.Tensor4D.t = "";

let tensor4d = tensor4DValue =>
  tensor4DValue
  |> Types.TensorLike4D.sendToTfjs
  |> (((ffi, dType)) => tensor4d(ffi, dType |> Types.dTypeToJs));
