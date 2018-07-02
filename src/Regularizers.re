/* TODO: Convert all of the options in this file to bs.deriving abstract */
type t;

type ffi;

external _unsafeToFfi : 'a => ffi = "%identity";

type regularizerType =
  | L1L2
  | Regularizer(t);

let regularizerTypeToJs = regularizerType =>
  switch (regularizerType) {
  | L1L2 => "l1l2" |> _unsafeToFfi
  | Regularizer(t) => t |> _unsafeToFfi
  };

[@bs.module "@tensorflow/tfjs"] [@bs.scope "regularizers"]
external l1l2 : unit => t = "";

[@bs.module "@tensorflow/tfjs"] [@bs.scope "regularizers"]
external l1l2WithOptions :
  {
    .
    "l1": Js.Undefined.t(float),
    "l2": Js.Undefined.t(float),
  } =>
  t =
  "";

let l1l2WithOptions = (~l1=?, ~l2=?, ()) =>
  l1l2WithOptions({
    "l1": l1 |> Js.Undefined.fromOption,
    "l2": l2 |> Js.Undefined.fromOption,
  });

[@bs.module "@tensorflow/tfjs"] [@bs.scope "regularizers"]
external l1 : unit => t = "";

[@bs.module "@tensorflow/tfjs"] [@bs.scope "regularizers"]
external l1WithOptions : {. "l1": float} => t = "";

let l1WithOptions = l1 => l1WithOptions({"l1": l1});

[@bs.module "@tensorflow/tfjs"] [@bs.scope "regularizers"]
external l2 : unit => t = "";

[@bs.module "@tensorflow/tfjs"] [@bs.scope "regularizers"]
external l2WithOptions : {. "l2": float} => t = "";

let l2WithOptions = l2 => l2WithOptions({"l2": l2});
