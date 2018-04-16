[@bs.module "@tensorflow/tfjs"]
external tensor : Types.FFI.t => Types.FFI.t = "tensor";

let tensor: Types.tensorLike => Types.Tensor.t =
  tensorLike =>
    switch (tensorLike) {
    | Typed(ta) =>
      ta |> Types.FFI.unsafeCastToFFI |> tensor |> Types.Tensor.unsafeCast
    };
