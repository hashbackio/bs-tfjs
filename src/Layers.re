[@bs.deriving jsConverter]
type activationType = [
  | `elu
  | `hardsigmoid
  | `linear
  | `relu
  | `relu6
  | `selu
  | `sigmoid
  | `softmax
  | `softplus
  | `softsign
  | `tanh
];

module Layer = (R: Core.Rank, D: Core.DataType) => {
  module SymbolicTensor = Models.SymbolicTensor(R, D);
  type t;
  [@bs.send] external apply : (t, SymbolicTensor.t) => SymbolicTensor.t = "";
  [@bs.send]
  external applyMany :
    (t, array(SymbolicTensor.t)) => array(SymbolicTensor.t) =
    "apply";
};

module Activations = (R: Core.Rank, D: Core.DataType) => {
  module Layer = Layer(R, D);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external elu : unit => Layer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external eluWithAlpha : {. "alpha": float} => Layer.t = "elu";
  let eluWithAlpha = alpha =>
    {"alpha": alpha |> Js.Math.max_float(0.0)} |> eluWithAlpha;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external leakyReLU : unit => Layer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external leakyReLUWithAlpha : {. "alpha": float} => Layer.t = "leakyReLU";
  let leakyReLUWithAlpha = alpha =>
    {"alpha": alpha |> Js.Math.max_float(0.0)} |> leakyReLUWithAlpha;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external softmax : unit => Layer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external softmaxWithAxis : {. "axis": int} => Layer.t = "softmax";
  let softmaxWithAxis = axis =>
    {"axis": axis |> R.axisToNegOneDefaultRank |> R.axisToJs}
    |> softmaxWithAxis;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external thresohldedReLU : unit => Layer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external thresohldedReLUWithTheta : {. "theta": float} => Layer.t =
    "thresohldedReLU";
  let thresohldedReLUWithTheta = theta =>
    {"theta": theta |> Js.Math.max_float(0.0)} |> thresohldedReLUWithTheta;
};

module Basic = (R: Core.Rank, D: Core.DataType) => {
  module Layer = Layer(R, D);
  module Initializer = Initializers.Initializer(R, D);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external activation : {. "activation": string} => Layer.t = "";
  let activation = activationType =>
    {"activation": activationType |> activationTypeToJs} |> activation;
};
