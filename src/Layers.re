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

module Layer = (Rin: Core.Rank, Rout: Core.Rank, D: Core.DataType) => {
  module SymbolicTensorIn = Models.SymbolicTensor(Rin, D);
  module SymbolicTensorOut = Models.SymbolicTensor(Rout, D);
  type t;
  [@bs.send]
  external apply : (t, SymbolicTensorIn.t) => SymbolicTensorOut.t = "";
  [@bs.send]
  external applyMany :
    (t, array(SymbolicTensorIn.t)) => array(SymbolicTensorOut.t) =
    "apply";
};

module Activations = (R: Core.Rank, D: Core.DataType) => {
  module Layer = Layer(R, R, D);
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
  module LayerFunctor = Layer;
  module Layer = Layer(R, R, D);
  module Initializer = Initializers.Initializer(R, D);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external activation : {. "activation": string} => Layer.t = "";
  let activation = activationType =>
    {"activation": activationType |> activationTypeToJs} |> activation;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external dense :
    {
      .
      "units": int,
      "activation": Js.Undefined.t(string),
      "useBias": Js.Undefined.t(bool),
      "kernelInitializer": Js.Undefined.t(Initializer.ffi),
      "biasInitializer": Js.Undefined.t(Initializer.ffi),
      "inputDim": Js.Undefined.t(int),
      "kernelConstraint": Js.Undefined.t(Constraints.ffi),
      "biasConstraint": Js.Undefined.t(Constraints.ffi),
      "kernelRegularizer": Js.Undefined.t(Regularizers.ffi),
      "biasRegularizer": Js.Undefined.t(Regularizers.ffi),
      "activityRegularizer": Js.Undefined.t(Regularizers.ffi),
    } =>
    Layer.t =
    "";
  let dense =
      (
        units,
        ~activation=?,
        ~useBias=?,
        ~kernelInitializer=?,
        ~biasInitializer=?,
        ~inputDim=?,
        ~kernelConstraint=?,
        ~biasConstraint=?,
        ~kernelRegularizer=?,
        ~biasRegularizer=?,
        ~activityRegularizer=?,
        (),
      ) =>
    {
      "units": Js.Math.max_int(1, units),
      "activation":
        activation
        |. Belt.Option.map(activationTypeToJs)
        |> Js.Undefined.fromOption,
      "useBias": useBias |> Js.Undefined.fromOption,
      "kernelInitializer":
        kernelInitializer
        |. Belt.Option.map(Initializer.initializerTypeToJs)
        |> Js.Undefined.fromOption,
      "biasInitializer":
        biasInitializer
        |. Belt.Option.map(Initializer.initializerTypeToJs)
        |> Js.Undefined.fromOption,
      "inputDim": inputDim |> Js.Undefined.fromOption,
      "kernelConstraint":
        kernelConstraint
        |. Belt.Option.map(Constraints.constraintTypesToJs)
        |> Js.Undefined.fromOption,
      "biasConstraint":
        biasConstraint
        |. Belt.Option.map(Constraints.constraintTypesToJs)
        |> Js.Undefined.fromOption,
      "kernelRegularizer":
        kernelRegularizer
        |. Belt.Option.map(Regularizers.regularizerTypeToJs)
        |> Js.Undefined.fromOption,
      "biasRegularizer":
        biasRegularizer
        |. Belt.Option.map(Regularizers.regularizerTypeToJs)
        |> Js.Undefined.fromOption,
      "activityRegularizer":
        activityRegularizer
        |. Belt.Option.map(Regularizers.regularizerTypeToJs)
        |> Js.Undefined.fromOption,
    }
    |> dense;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external dropout :
    {
      .
      "rate": float,
      "noiseShape": Js.Undefined.t(array(int)),
      "seed": Js.Undefined.t(int),
    } =>
    Layer.t =
    "";
  let dropout = (rate, ~noiseShape=?, ~seed=?, ()) =>
    {
      "rate": rate |> Js.Math.max_float(0.0) |> Js.Math.min_float(1.0),
      "noiseShape":
        noiseShape
        |. Belt.Option.map(R.getShapeArray)
        |> Js.Undefined.fromOption,
      "seed": seed |> Js.Undefined.fromOption,
    }
    |> dropout;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external embedding :
    {
      .
      "inputDim": int,
      "outputDim": int,
      "embeddingsInitializer": Js.Undefined.t(Initializer.ffi),
      "embeddingsRegularizer": Js.Undefined.t(Regularizers.ffi),
      "activityRegularizer": Js.Undefined.t(Regularizers.ffi),
      "embeddingsConstraint": Js.Undefined.t(Constraints.ffi),
      "maskZero": Js.Undefined.t(bool),
      "inputLength": Js.Undefined.t(int),
    } =>
    LayerFunctor(Core.Rank3)(Core.Rank4)(D).t =
    "";
  let embedding =
      (
        inputDim,
        outputDim,
        ~embeddingsInitializer=?,
        ~embeddingsRegularizer=?,
        ~activityRegularizer=?,
        ~embeddingsConstraint=?,
        ~maskZero=?,
        ~inputLength=?,
        (),
      ) =>
    {
      "inputDim": inputDim |> Js.Math.max_int(1),
      "outputDim": outputDim |> Js.Math.max_int(0),
      "embeddingsInitializer":
        embeddingsInitializer
        |. Belt.Option.map(Initializer.initializerTypeToJs)
        |> Js.Undefined.fromOption,
      "embeddingsRegularizer":
        embeddingsRegularizer
        |. Belt.Option.map(Regularizers.regularizerTypeToJs)
        |> Js.Undefined.fromOption,
      "activityRegularizer":
        activityRegularizer
        |. Belt.Option.map(Regularizers.regularizerTypeToJs)
        |> Js.Undefined.fromOption,
      "embeddingsConstraint":
        embeddingsConstraint
        |. Belt.Option.map(Constraints.constraintTypesToJs)
        |> Js.Undefined.fromOption,
      "maskZero": maskZero |> Js.Undefined.fromOption,
      "inputLength": inputLength |> Js.Undefined.fromOption,
    }
    |> embedding;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external flatten :
    {
      .
      "inputShape": Js.Undefined.t(array(int)),
      "batchInputShape": Js.Undefined.t(array(int)),
      "batchSize": Js.Undefined.t(int),
      "dtype": Js.Undefined.t(string),
      "name": Js.Undefined.t(string),
      "trainable": Js.Undefined.t(bool),
      "updatable": Js.Undefined.t(bool),
      "weights": Js.Undefined.t(Core.Tensor(R)(D).t),
    } =>
    LayerFunctor(R)(Core.Rank2)(D).t =
    "";
  let flatten =
      (
        ~inputShape=?,
        ~batchInputShape=?,
        ~batchSize=?,
        ~name=?,
        ~trainable=?,
        ~updatable=?,
        ~weights=?,
        (),
      ) =>
    {
      "inputShape":
        inputShape
        |. Belt.Option.map(R.getShapeArray)
        |> Js.Undefined.fromOption,
      "batchInputShape":
        batchInputShape
        |. Belt.Option.map(R.getShapeArray)
        |> Js.Undefined.fromOption,
      "batchSize": batchSize |> Js.Undefined.fromOption,
      "dtype": D.dType |> Core.dTypeToJs |> Js.Undefined.return,
      "name": name |> Js.Undefined.fromOption,
      "trainable": trainable |> Js.Undefined.fromOption,
      "updatable": updatable |> Js.Undefined.fromOption,
      "weights": weights |> Js.Undefined.fromOption,
    }
    |> flatten;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external repeatVector : {. "n": int} => Layer.t = "";
  let repeatVector = n => {"n": n} |> repeatVector;
};
