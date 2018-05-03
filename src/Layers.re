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

[@bs.deriving jsConverter]
type padding = [ | `valid | `same | `casual];

[@bs.deriving jsConverter]
type dataFormat = [ | `channelsFirst | `channelsLast];

[@bs.deriving jsConverter]
type implementationType =
  | [@bs.as 1] Mode1
  | [@bs.as 2] Mode2;

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

module RnnCell = (Rin: Core.Rank, Rout: Core.Rank, D: Core.DataType) => {
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

module Configs = (R: Core.Rank, D: Core.DataType) => {
  module Initializer = Initializers.Initializer(R, D);
  type inputConfig = {
    .
    "inputShape": Js.Undefined.t(array(int)),
    "batchInputShape": Js.Undefined.t(array(int)),
    "batchSize": Js.Undefined.t(int),
    "dtype": Js.Undefined.t(string),
    "name": Js.Undefined.t(string),
    "trainable": Js.Undefined.t(bool),
    "updatable": Js.Undefined.t(bool),
  };
  type normalizeConfig = {
    .
    "axis": Js.Undefined.t(int),
    "momentum": Js.Undefined.t(float),
    "epsilon": Js.Undefined.t(float),
    "center": Js.Undefined.t(bool),
    "scale": Js.Undefined.t(bool),
    "betaInitializer": Js.Undefined.t(Initializer.ffi),
    "gammaInitializer": Js.Undefined.t(Initializer.ffi),
    "movingMeanInitializer": Js.Undefined.t(Initializer.ffi),
    "movingVarianceInitializer": Js.Undefined.t(Initializer.ffi),
    "betaConstraint": Js.Undefined.t(Constraints.ffi),
    "gammaConstraint": Js.Undefined.t(Constraints.ffi),
    "betaRegularizer": Js.Undefined.t(Regularizers.ffi),
    "gammaRegularizer": Js.Undefined.t(Regularizers.ffi),
  };
  type poolingConfig = {
    .
    "poolSize": Js.Undefined.t(int),
    "strides": Js.Undefined.t(int),
    "padding": Js.Undefined.t(string),
    "dataFormat": Js.Undefined.t(string),
  };
  type recurrentConfig = {
    .
    /* BaseRnnLayerConfig */
    "cell": Js.Undefined.t(array(RnnCell(R)(R)(D).t)),
    "returnSequences": Js.Undefined.t(bool),
    "returnState": Js.Undefined.t(bool),
    "goBackwards": Js.Undefined.t(bool),
    "stateful": Js.Undefined.t(bool),
    "unroll": Js.Undefined.t(bool),
    "inputDim": Js.Undefined.t(int),
    "inputLength": Js.Undefined.t(int),
    /* SimpleRNNLayerConfig  */
    "units": int,
    "activation": Js.Undefined.t(string),
    "useBias": Js.Undefined.t(bool),
    "kernelInitializer": Js.Undefined.t(Initializer.ffi),
    "recurrentInitializer": Js.Undefined.t(Initializer.ffi),
    "biasInitializer": Js.Undefined.t(Initializer.ffi),
    "kernelRegularizer": Js.Undefined.t(Regularizers.ffi),
    "recurrentRegularizer": Js.Undefined.t(Regularizers.ffi),
    "biasRegularizer": Js.Undefined.t(Regularizers.ffi),
    "kernelConstraint": Js.Undefined.t(Constraints.ffi),
    "recurrentConstraint": Js.Undefined.t(Constraints.ffi),
    "biasConstraint": Js.Undefined.t(Constraints.ffi),
    "dropout": Js.Undefined.t(float),
    "recurrentDropout": Js.Undefined.t(float),
    /* GRULayerConfig, GruCellLayerConfig, LSTMLayerConfig, LSTMCellLayerConfig */
    "implementation": Js.Undefined.t(int),
    /* GruCellLayerConfig, LSTMCellLayerConfig */
    "recurrentActivation": Js.Undefined.t(string),
    /* LSTMLayerConfig, LSTMCellLayerConfig */
    "unitForgetBias": Js.Undefined.t(bool),
  };
  type stackedRnnCellsConfig = {. "cells": array(RnnCell(R)(R)(D).t)};
  let callFnWithInputConfig =
      (
        fn,
        ~inputShape=?,
        ~batchInputShape=?,
        ~batchSize=?,
        ~dtype=?,
        ~name=?,
        ~trainable=?,
        ~updatable=?,
        (),
      ) =>
    {
      "inputShape":
        inputShape
        |. Belt.Option.map(R.getShapeArray)
        |> Js.Undefined.fromOption,
      "batchInputShape":
        batchInputShape
        |. Belt.Option.map(R.getInputShapeArray)
        |> Js.Undefined.fromOption,
      "batchSize": batchSize |> Js.Undefined.fromOption,
      "dtype":
        dtype |. Belt.Option.map(Core.dTypeToJs) |> Js.Undefined.fromOption,
      "name": name |> Js.Undefined.fromOption,
      "trainable": trainable |> Js.Undefined.fromOption,
      "updatable": updatable |> Js.Undefined.fromOption,
    }
    |> Js.Undefined.return
    |> fn;
  let callFnWithNormalizeConfig =
      (
        fn,
        ~axis=?,
        ~momentum=?,
        ~epsilon=?,
        ~center=?,
        ~scale=?,
        ~betaInitializer=?,
        ~gammaInitializer=?,
        ~movingMeanInitializer=?,
        ~movingVarianceInitializer=?,
        ~betaConstraint=?,
        ~gammaConstraint=?,
        ~betaRegularizer=?,
        ~gammaRegularizer=?,
        (),
      ) =>
    {
      "axis":
        axis
        |. Belt.Option.map(R.axisToNegOneDefaultRank)
        |. Belt.Option.map(R.axisToJs)
        |> Js.Undefined.fromOption,
      "momentum": momentum |> Js.Undefined.fromOption,
      "epsilon": epsilon |> Js.Undefined.fromOption,
      "center": center |> Js.Undefined.fromOption,
      "scale": scale |> Js.Undefined.fromOption,
      "betaInitializer":
        betaInitializer
        |. Belt.Option.map(Initializer.initializerTypeToJs)
        |> Js.Undefined.fromOption,
      "gammaInitializer":
        gammaInitializer
        |. Belt.Option.map(Initializer.initializerTypeToJs)
        |> Js.Undefined.fromOption,
      "movingMeanInitializer":
        movingMeanInitializer
        |. Belt.Option.map(Initializer.initializerTypeToJs)
        |> Js.Undefined.fromOption,
      "movingVarianceInitializer":
        movingVarianceInitializer
        |. Belt.Option.map(Initializer.initializerTypeToJs)
        |> Js.Undefined.fromOption,
      "betaConstraint":
        betaConstraint
        |. Belt.Option.map(Constraints.constraintTypesToJs)
        |> Js.Undefined.fromOption,
      "gammaConstraint":
        gammaConstraint
        |. Belt.Option.map(Constraints.constraintTypesToJs)
        |> Js.Undefined.fromOption,
      "betaRegularizer":
        betaRegularizer
        |. Belt.Option.map(Regularizers.regularizerTypeToJs)
        |> Js.Undefined.fromOption,
      "gammaRegularizer":
        gammaRegularizer
        |. Belt.Option.map(Regularizers.regularizerTypeToJs)
        |> Js.Undefined.fromOption,
    }
    |> Js.Undefined.return
    |> fn;
  let callFnWithPoolingConfig =
      (fn, ~poolSize=?, ~strides=?, ~padding=?, ~dataFormat=?, ()) =>
    {
      "poolSize": poolSize |> Js.Undefined.fromOption,
      "strides": strides |> Js.Undefined.fromOption,
      "padding":
        padding |. Belt.Option.map(paddingToJs) |> Js.Undefined.fromOption,
      "dataFormat":
        dataFormat
        |. Belt.Option.map(dataFormatToJs)
        |> Js.Undefined.fromOption,
    }
    |> Js.Undefined.return
    |> fn;
  let callFnWithRecurrentConfig =
      (
        fn,
        ~units,
        ~cell=?,
        ~returnSequences=?,
        ~returnState=?,
        ~goBackwards=?,
        ~stateful=?,
        ~unroll=?,
        ~inputDim=?,
        ~inputLength=?,
        ~activation=?,
        ~useBias=?,
        ~kernelInitializer=?,
        ~recurrentInitializer=?,
        ~biasInitializer=?,
        ~kernelRegularizer=?,
        ~recurrentRegularizer=?,
        ~biasRegularizer=?,
        ~kernelConstraint=?,
        ~recurrentConstraint=?,
        ~biasConstraint=?,
        ~dropout=?,
        ~recurrentDropout=?,
        ~recurrentActivation=?,
        ~unitForgetBias=?,
        ~implementation=?,
        (),
      ) =>
    (
      {
        "units": units,
        "cell": cell |> Js.Undefined.fromOption,
        "returnSequences": returnSequences |> Js.Undefined.fromOption,
        "returnState": returnState |> Js.Undefined.fromOption,
        "goBackwards": goBackwards |> Js.Undefined.fromOption,
        "stateful": stateful |> Js.Undefined.fromOption,
        "unroll": unroll |> Js.Undefined.fromOption,
        "inputDim": inputDim |> Js.Undefined.fromOption,
        "inputLength": inputLength |> Js.Undefined.fromOption,
        "activation":
          activation
          |. Belt.Option.map(activationTypeToJs)
          |> Js.Undefined.fromOption,
        "useBias": useBias |> Js.Undefined.fromOption,
        "kernelInitializer":
          kernelInitializer
          |. Belt.Option.map(Initializer.initializerTypeToJs)
          |> Js.Undefined.fromOption,
        "recurrentInitializer":
          recurrentInitializer
          |. Belt.Option.map(Initializer.initializerTypeToJs)
          |> Js.Undefined.fromOption,
        "biasInitializer":
          biasInitializer
          |. Belt.Option.map(Initializer.initializerTypeToJs)
          |> Js.Undefined.fromOption,
        "kernelRegularizer":
          kernelRegularizer
          |. Belt.Option.map(Regularizers.regularizerTypeToJs)
          |> Js.Undefined.fromOption,
        "recurrentRegularizer":
          recurrentRegularizer
          |. Belt.Option.map(Regularizers.regularizerTypeToJs)
          |> Js.Undefined.fromOption,
        "biasRegularizer":
          biasRegularizer
          |. Belt.Option.map(Regularizers.regularizerTypeToJs)
          |> Js.Undefined.fromOption,
        "kernelConstraint":
          kernelConstraint
          |. Belt.Option.map(Constraints.constraintTypesToJs)
          |> Js.Undefined.fromOption,
        "recurrentConstraint":
          recurrentConstraint
          |. Belt.Option.map(Constraints.constraintTypesToJs)
          |> Js.Undefined.fromOption,
        "biasConstraint":
          biasConstraint
          |. Belt.Option.map(Constraints.constraintTypesToJs)
          |> Js.Undefined.fromOption,
        "dropout": dropout |> Js.Undefined.fromOption,
        "recurrentDropout": recurrentDropout |> Js.Undefined.fromOption,
        "recurrentActivation":
          recurrentActivation
          |. Belt.Option.map(activationTypeToJs)
          |> Js.Undefined.fromOption,
        "unitForgetBias": unitForgetBias |> Js.Undefined.fromOption,
        "implementation":
          implementation
          |. Belt.Option.map(implementationTypeToJs)
          |> Js.Undefined.fromOption,
      }: recurrentConfig
    )
    |> fn;
  let callFnWithGruLayerConfig =
      (
        fn,
        ~units,
        ~cell=?,
        ~returnSequences=?,
        ~returnState=?,
        ~goBackwards=?,
        ~stateful=?,
        ~unroll=?,
        ~inputDim=?,
        ~inputLength=?,
        ~activation=?,
        ~useBias=?,
        ~kernelInitializer=?,
        ~recurrentInitializer=?,
        ~biasInitializer=?,
        ~kernelRegularizer=?,
        ~recurrentRegularizer=?,
        ~biasRegularizer=?,
        ~kernelConstraint=?,
        ~recurrentConstraint=?,
        ~biasConstraint=?,
        ~dropout=?,
        ~recurrentDropout=?,
        ~implementation=?,
        (),
      ) =>
    callFnWithRecurrentConfig(
      fn,
      ~units,
      ~cell?,
      ~returnSequences?,
      ~returnState?,
      ~goBackwards?,
      ~stateful?,
      ~unroll?,
      ~inputDim?,
      ~inputLength?,
      ~activation?,
      ~useBias?,
      ~kernelInitializer?,
      ~recurrentInitializer?,
      ~biasInitializer?,
      ~kernelRegularizer?,
      ~recurrentRegularizer?,
      ~biasRegularizer?,
      ~kernelConstraint?,
      ~recurrentConstraint?,
      ~biasConstraint?,
      ~dropout?,
      ~recurrentDropout?,
      ~implementation?,
      (),
    );
  let callFnWithGruCellLayerConfig =
      (
        fn,
        ~units,
        ~cell=?,
        ~returnSequences=?,
        ~returnState=?,
        ~goBackwards=?,
        ~stateful=?,
        ~unroll=?,
        ~inputDim=?,
        ~inputLength=?,
        ~activation=?,
        ~useBias=?,
        ~kernelInitializer=?,
        ~recurrentInitializer=?,
        ~biasInitializer=?,
        ~kernelRegularizer=?,
        ~recurrentRegularizer=?,
        ~biasRegularizer=?,
        ~kernelConstraint=?,
        ~recurrentConstraint=?,
        ~biasConstraint=?,
        ~dropout=?,
        ~recurrentDropout=?,
        ~recurrentActivation=?,
        ~implementation=?,
        (),
      ) =>
    callFnWithRecurrentConfig(
      fn,
      ~units,
      ~cell?,
      ~returnSequences?,
      ~returnState?,
      ~goBackwards?,
      ~stateful?,
      ~unroll?,
      ~inputDim?,
      ~inputLength?,
      ~activation?,
      ~useBias?,
      ~kernelInitializer?,
      ~recurrentInitializer?,
      ~biasInitializer?,
      ~kernelRegularizer?,
      ~recurrentRegularizer?,
      ~biasRegularizer?,
      ~kernelConstraint?,
      ~recurrentConstraint?,
      ~biasConstraint?,
      ~dropout?,
      ~recurrentDropout?,
      ~recurrentActivation?,
      ~implementation?,
      (),
    );
  let callFnWithLstmLayerConfig =
      (
        fn,
        ~units,
        ~cell=?,
        ~returnSequences=?,
        ~returnState=?,
        ~goBackwards=?,
        ~stateful=?,
        ~unroll=?,
        ~inputDim=?,
        ~inputLength=?,
        ~activation=?,
        ~useBias=?,
        ~kernelInitializer=?,
        ~recurrentInitializer=?,
        ~biasInitializer=?,
        ~kernelRegularizer=?,
        ~recurrentRegularizer=?,
        ~biasRegularizer=?,
        ~kernelConstraint=?,
        ~recurrentConstraint=?,
        ~biasConstraint=?,
        ~dropout=?,
        ~recurrentDropout=?,
        ~unitForgetBias=?,
        ~implementation=?,
        (),
      ) =>
    callFnWithRecurrentConfig(
      fn,
      ~units,
      ~cell?,
      ~returnSequences?,
      ~returnState?,
      ~goBackwards?,
      ~stateful?,
      ~unroll?,
      ~inputDim?,
      ~inputLength?,
      ~activation?,
      ~useBias?,
      ~kernelInitializer?,
      ~recurrentInitializer?,
      ~biasInitializer?,
      ~kernelRegularizer?,
      ~recurrentRegularizer?,
      ~biasRegularizer?,
      ~kernelConstraint?,
      ~recurrentConstraint?,
      ~biasConstraint?,
      ~dropout?,
      ~recurrentDropout?,
      ~unitForgetBias?,
      ~implementation?,
      (),
    );
  let callFnWithLstmCellLayerConfig =
      (
        fn,
        ~units,
        ~cell=?,
        ~returnSequences=?,
        ~returnState=?,
        ~goBackwards=?,
        ~stateful=?,
        ~unroll=?,
        ~inputDim=?,
        ~inputLength=?,
        ~activation=?,
        ~useBias=?,
        ~kernelInitializer=?,
        ~recurrentInitializer=?,
        ~biasInitializer=?,
        ~kernelRegularizer=?,
        ~recurrentRegularizer=?,
        ~biasRegularizer=?,
        ~kernelConstraint=?,
        ~recurrentConstraint=?,
        ~biasConstraint=?,
        ~dropout=?,
        ~recurrentDropout=?,
        ~recurrentActivation=?,
        ~unitForgetBias=?,
        ~implementation=?,
        (),
      ) =>
    callFnWithRecurrentConfig(
      fn,
      ~units,
      ~cell?,
      ~returnSequences?,
      ~returnState?,
      ~goBackwards?,
      ~stateful?,
      ~unroll?,
      ~inputDim?,
      ~inputLength?,
      ~activation?,
      ~useBias?,
      ~kernelInitializer?,
      ~recurrentInitializer?,
      ~biasInitializer?,
      ~kernelRegularizer?,
      ~recurrentRegularizer?,
      ~biasRegularizer?,
      ~kernelConstraint?,
      ~recurrentConstraint?,
      ~biasConstraint?,
      ~dropout?,
      ~recurrentDropout?,
      ~recurrentActivation?,
      ~unitForgetBias?,
      ~implementation?,
      (),
    );
  let callFnWithRnnLayerConfig =
      (
        fn,
        ~cell,
        ~returnSequences=?,
        ~returnState=?,
        ~goBackwards=?,
        ~stateful=?,
        ~unroll=?,
        ~inputDim=?,
        ~inputLength=?,
        (),
      ) =>
    callFnWithRecurrentConfig(
      fn,
      ~cell,
      ~returnSequences?,
      ~returnState?,
      ~goBackwards?,
      ~stateful?,
      ~unroll?,
      ~inputDim?,
      ~inputLength?,
      (),
    );
  let callFnWithSimpleLayerConfig =
      (
        fn,
        ~units,
        ~cell=?,
        ~returnSequences=?,
        ~returnState=?,
        ~goBackwards=?,
        ~stateful=?,
        ~unroll=?,
        ~inputDim=?,
        ~inputLength=?,
        ~activation=?,
        ~useBias=?,
        ~kernelInitializer=?,
        ~recurrentInitializer=?,
        ~biasInitializer=?,
        ~kernelRegularizer=?,
        ~recurrentRegularizer=?,
        ~biasRegularizer=?,
        ~kernelConstraint=?,
        ~recurrentConstraint=?,
        ~biasConstraint=?,
        ~dropout=?,
        ~recurrentDropout=?,
        (),
      ) =>
    callFnWithRecurrentConfig(
      fn,
      ~units,
      ~cell?,
      ~returnSequences?,
      ~returnState?,
      ~goBackwards?,
      ~stateful?,
      ~unroll?,
      ~inputDim?,
      ~inputLength?,
      ~activation?,
      ~useBias?,
      ~kernelInitializer?,
      ~recurrentInitializer?,
      ~biasInitializer?,
      ~kernelRegularizer?,
      ~recurrentRegularizer?,
      ~biasRegularizer?,
      ~kernelConstraint?,
      ~recurrentConstraint?,
      ~biasConstraint?,
      ~dropout?,
      ~recurrentDropout?,
      (),
    );
  let callFnWithSimpleCellLayerConfig =
      (
        fn,
        ~units,
        ~cell=?,
        ~returnSequences=?,
        ~returnState=?,
        ~goBackwards=?,
        ~stateful=?,
        ~unroll=?,
        ~inputDim=?,
        ~inputLength=?,
        ~activation=?,
        ~useBias=?,
        ~kernelInitializer=?,
        ~recurrentInitializer=?,
        ~biasInitializer=?,
        ~kernelRegularizer=?,
        ~recurrentRegularizer=?,
        ~biasRegularizer=?,
        ~kernelConstraint=?,
        ~recurrentConstraint=?,
        ~biasConstraint=?,
        ~dropout=?,
        ~recurrentDropout=?,
        (),
      ) =>
    callFnWithRecurrentConfig(
      fn,
      ~units,
      ~cell?,
      ~returnSequences?,
      ~returnState?,
      ~goBackwards?,
      ~stateful?,
      ~unroll?,
      ~inputDim?,
      ~inputLength?,
      ~activation?,
      ~useBias?,
      ~kernelInitializer?,
      ~recurrentInitializer?,
      ~biasInitializer?,
      ~kernelRegularizer?,
      ~recurrentRegularizer?,
      ~biasRegularizer?,
      ~kernelConstraint?,
      ~recurrentConstraint?,
      ~biasConstraint?,
      ~dropout?,
      ~recurrentDropout?,
      (),
    );
  let callFnWithStackedCellsLayerConfig = (fn, ~cells, ()) =>
    {"cells": cells} |> fn;
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
  module Configs = Configs(R, D);
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
    Js.Undefined.t(Configs.inputConfig) => LayerFunctor(R)(Core.Rank2)(D).t =
    "";
  let flatten = Configs.callFnWithInputConfig(flatten);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external repeatVector : {. "n": int} => Layer.t = "";
  let repeatVector = n => {"n": n} |> repeatVector;
};

module Convolutional = (D: Core.DataType) => {
  module Conv1dLayer = Layer(Core.Rank3, Core.Rank3, D);
  module Conv2dLayer = Layer(Core.Rank4, Core.Rank4, D);
  module Conv1dInitializer = Initializers.Initializer(Core.Rank3, D);
  module Conv2dInitializer = Initializers.Initializer(Core.Rank4, D);
  type kernelSizeFfi;
  type kernelSize1D =
    | Square(int)
    | Rectangle(int, int);
  type kernelSize2D =
    | Cube(int)
    | Box(int, int, int);
  external unsafeToKernelSizeFfi : 'a => kernelSizeFfi = "%identity";
  let kernelSize1DToFfi = kernelSize1D =>
    switch (kernelSize1D) {
    | Square(length) => length |> unsafeToKernelSizeFfi
    | Rectangle(length, width) => [|length, width|] |> unsafeToKernelSizeFfi
    };
  let kernelSize2DToFfi = kernelSize2D =>
    switch (kernelSize2D) {
    | Cube(length) => length |> unsafeToKernelSizeFfi
    | Box(length, width, height) =>
      [|length, width, height|] |> unsafeToKernelSizeFfi
    };
  type strideFfi;
  external unsafeToStrideFfi : 'a => strideFfi = "%identity";
  type dilationRateFfi;
  external unsafeToDilationRateFfi : 'a => dilationRateFfi = "%identity";
  type strideOrDilationRate1D =
    | EvenStride(int)
    | OddStride(int, int)
    | Dilation(int);
  let strideOrDilationRate1DToStrideFfi = strideOrDilationRate1D =>
    switch (strideOrDilationRate1D) {
    | EvenStride(stride) => stride |> unsafeToStrideFfi
    | OddStride(length, width) => [|length, width|] |> unsafeToStrideFfi
    | Dilation(_) => 1 |> unsafeToStrideFfi
    };
  let strideOrDilationRate1DToDilationRateFfi = strideOrDilationRate1D =>
    switch (strideOrDilationRate1D) {
    | EvenStride(_) => 1 |> unsafeToDilationRateFfi
    | OddStride(_, _) => 1 |> unsafeToDilationRateFfi
    | Dilation(length) => length |> unsafeToDilationRateFfi
    };
  type strideOrDilationRate2D =
    | EvenStride(int)
    | OddStride(int, int, int)
    | EqualDilation(int)
    | RectangularDilation(int, int);
  let strideOrDilationRate2DToStrideFfi = strideOrDilationRate2D =>
    switch (strideOrDilationRate2D) {
    | EvenStride(length) => length |> unsafeToStrideFfi
    | OddStride(length, width, height) =>
      [|length, width, height|] |> unsafeToStrideFfi
    | EqualDilation(_) => 1 |> unsafeToStrideFfi
    | RectangularDilation(_, _) => 1 |> unsafeToStrideFfi
    };
  let strideOrDilationRate2DToDilationRateFfi = strideOrDilationRate2D =>
    switch (strideOrDilationRate2D) {
    | EvenStride(_) => 1 |> unsafeToDilationRateFfi
    | OddStride(_, _, _) => 1 |> unsafeToDilationRateFfi
    | EqualDilation(length) => length |> unsafeToDilationRateFfi
    | RectangularDilation(length, width) =>
      [|length, width|] |> unsafeToDilationRateFfi
    };
  type convInitializerFfi;
  external unsafeToConvInitializerFfi : 'a => convInitializerFfi = "%identity";
  type convInitializer =
    | Conv1d(Conv1dInitializer.initializerType)
    | Conv2d(Conv2dInitializer.initializerType);
  let convInitializerToFfi = convInitializer =>
    switch (convInitializer) {
    | Conv1d(conv1dInitializer) =>
      conv1dInitializer
      |> Conv1dInitializer.initializerTypeToJs
      |> unsafeToConvInitializerFfi
    | Conv2d(conv2dInitializer) =>
      conv2dInitializer
      |> Conv2dInitializer.initializerTypeToJs
      |> unsafeToConvInitializerFfi
    };
  type configFfi = {
    .
    "kernelSize": Js.Undefined.t(kernelSizeFfi),
    "filters": Js.Undefined.t(int),
    "stride": Js.Undefined.t(strideFfi),
    "dilationRate": Js.Undefined.t(dilationRateFfi),
    "padding": Js.Undefined.t(string),
    "dataFormat": Js.Undefined.t(string),
    "activation": Js.Undefined.t(string),
    "useBias": Js.Undefined.t(bool),
    "kernelInitializer": Js.Undefined.t(convInitializerFfi),
    "biasInitializer": Js.Undefined.t(convInitializerFfi),
    "kernelConstraint": Js.Undefined.t(Constraints.ffi),
    "biasConstraint": Js.Undefined.t(Constraints.ffi),
    "kernelRegularizer": Js.Undefined.t(Regularizers.ffi),
    "biasRegularizer": Js.Undefined.t(Regularizers.ffi),
    "activityRegularizer": Js.Undefined.t(Regularizers.ffi),
  };
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external conv1d : configFfi => Conv1dLayer.t = "";
  let conv1d =
      (
        ~kernelSize=?,
        ~filters=?,
        ~strideOrDilationRate=?,
        ~padding=?,
        ~dataFormat=?,
        ~activation=?,
        ~useBias=?,
        ~kernelInitializer=?,
        ~biasInitializer=?,
        ~kernelConstraint=?,
        ~biasConstraint=?,
        ~kernelRegularizer=?,
        ~biasRegularizer=?,
        ~activityRegularizer=?,
        (),
      ) =>
    {
      "kernelSize":
        kernelSize
        |. Belt.Option.map(kernelSize1DToFfi)
        |> Js.Undefined.fromOption,
      "filters": filters |> Js.Undefined.fromOption,
      "stride":
        strideOrDilationRate
        |. Belt.Option.map(strideOrDilationRate1DToStrideFfi)
        |> Js.Undefined.fromOption,
      "dilationRate":
        strideOrDilationRate
        |. Belt.Option.map(strideOrDilationRate1DToDilationRateFfi)
        |> Js.Undefined.fromOption,
      "padding":
        padding |. Belt.Option.map(paddingToJs) |> Js.Undefined.fromOption,
      "dataFormat":
        dataFormat
        |. Belt.Option.map(dataFormatToJs)
        |> Js.Undefined.fromOption,
      "activation":
        activation
        |. Belt.Option.map(activationTypeToJs)
        |> Js.Undefined.fromOption,
      "useBias": useBias |> Js.Undefined.fromOption,
      "kernelInitializer":
        kernelInitializer
        |. Belt.Option.map(x => Conv1d(x))
        |. Belt.Option.map(convInitializerToFfi)
        |> Js.Undefined.fromOption,
      "biasInitializer":
        biasInitializer
        |. Belt.Option.map(x => Conv1d(x))
        |. Belt.Option.map(convInitializerToFfi)
        |> Js.Undefined.fromOption,
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
    |> conv1d;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external conv2d : configFfi => Conv2dLayer.t = "";
  let conv2d =
      (
        ~kernelSize=?,
        ~filters=?,
        ~strideOrDilationRate=?,
        ~padding=?,
        ~dataFormat=?,
        ~activation=?,
        ~useBias=?,
        ~kernelInitializer=?,
        ~biasInitializer=?,
        ~kernelConstraint=?,
        ~biasConstraint=?,
        ~kernelRegularizer=?,
        ~biasRegularizer=?,
        ~activityRegularizer=?,
        (),
      ) =>
    {
      "kernelSize":
        kernelSize
        |. Belt.Option.map(kernelSize2DToFfi)
        |> Js.Undefined.fromOption,
      "filters": filters |> Js.Undefined.fromOption,
      "stride":
        strideOrDilationRate
        |. Belt.Option.map(strideOrDilationRate2DToStrideFfi)
        |> Js.Undefined.fromOption,
      "dilationRate":
        strideOrDilationRate
        |. Belt.Option.map(strideOrDilationRate2DToDilationRateFfi)
        |> Js.Undefined.fromOption,
      "padding":
        padding |. Belt.Option.map(paddingToJs) |> Js.Undefined.fromOption,
      "dataFormat":
        dataFormat
        |. Belt.Option.map(dataFormatToJs)
        |> Js.Undefined.fromOption,
      "activation":
        activation
        |. Belt.Option.map(activationTypeToJs)
        |> Js.Undefined.fromOption,
      "useBias": useBias |> Js.Undefined.fromOption,
      "kernelInitializer":
        kernelInitializer
        |. Belt.Option.map(x => Conv2d(x))
        |. Belt.Option.map(convInitializerToFfi)
        |> Js.Undefined.fromOption,
      "biasInitializer":
        biasInitializer
        |. Belt.Option.map(x => Conv2d(x))
        |. Belt.Option.map(convInitializerToFfi)
        |> Js.Undefined.fromOption,
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
    |> conv2d;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external conv2dTranspose : configFfi => Conv2dLayer.t = "";
  let conv2dTranspose =
      (
        ~kernelSize=?,
        ~filters=?,
        ~strideOrDilationRate=?,
        ~padding=?,
        ~dataFormat=?,
        ~activation=?,
        ~useBias=?,
        ~kernelInitializer=?,
        ~biasInitializer=?,
        ~kernelConstraint=?,
        ~biasConstraint=?,
        ~kernelRegularizer=?,
        ~biasRegularizer=?,
        ~activityRegularizer=?,
        (),
      ) =>
    {
      "kernelSize":
        kernelSize
        |. Belt.Option.map(kernelSize2DToFfi)
        |> Js.Undefined.fromOption,
      "filters": filters |> Js.Undefined.fromOption,
      "stride":
        strideOrDilationRate
        |. Belt.Option.map(strideOrDilationRate2DToStrideFfi)
        |> Js.Undefined.fromOption,
      "dilationRate":
        strideOrDilationRate
        |. Belt.Option.map(strideOrDilationRate2DToDilationRateFfi)
        |> Js.Undefined.fromOption,
      "padding":
        padding |. Belt.Option.map(paddingToJs) |> Js.Undefined.fromOption,
      "dataFormat":
        dataFormat
        |. Belt.Option.map(dataFormatToJs)
        |> Js.Undefined.fromOption,
      "activation":
        activation
        |. Belt.Option.map(activationTypeToJs)
        |> Js.Undefined.fromOption,
      "useBias": useBias |> Js.Undefined.fromOption,
      "kernelInitializer":
        kernelInitializer
        |. Belt.Option.map(x => Conv2d(x))
        |. Belt.Option.map(convInitializerToFfi)
        |> Js.Undefined.fromOption,
      "biasInitializer":
        biasInitializer
        |. Belt.Option.map(x => Conv2d(x))
        |. Belt.Option.map(convInitializerToFfi)
        |> Js.Undefined.fromOption,
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
    |> conv2dTranspose;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external depthwiseConv2d :
    {
      .
      "kernelSize": kernelSizeFfi,
      "depthMultiplier": Js.Undefined.t(int),
      "depthwiseInitializer": Js.Undefined.t(convInitializerFfi),
      "depthwiseConstraint": Js.Undefined.t(Constraints.ffi),
      "depthwiseRegularizer": Js.Undefined.t(Regularizers.ffi),
    } =>
    Conv2dLayer.t =
    "";
  let depthwiseConv2d =
      (
        kernelSize,
        ~depthMultiplier=?,
        ~depthwiseInitializer=?,
        ~depthwiseConstraint=?,
        ~depthwiseRegularizer=?,
        (),
      ) =>
    {
      "kernelSize": kernelSize |> kernelSize2DToFfi,
      "depthMultiplier": depthMultiplier |> Js.Undefined.fromOption,
      "depthwiseInitializer":
        depthwiseInitializer
        |. Belt.Option.map(x => Conv2d(x))
        |. Belt.Option.map(convInitializerToFfi)
        |> Js.Undefined.fromOption,
      "depthwiseConstraint":
        depthwiseConstraint
        |. Belt.Option.map(Constraints.constraintTypesToJs)
        |> Js.Undefined.fromOption,
      "depthwiseRegularizer":
        depthwiseRegularizer
        |. Belt.Option.map(Regularizers.regularizerTypeToJs)
        |> Js.Undefined.fromOption,
    }
    |> depthwiseConv2d;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external separableConv2d :
    {
      .
      "depthMultiplier": Js.Undefined.t(int),
      "depthwiseInitializer": Js.Undefined.t(convInitializerFfi),
      "pointwiseInitializer": Js.Undefined.t(convInitializerFfi),
      "depthwiseConstraint": Js.Undefined.t(Constraints.ffi),
      "pointwiseConstraint": Js.Undefined.t(Constraints.ffi),
      "depthwiseRegularizer": Js.Undefined.t(Regularizers.ffi),
      "pointwiseRegularizer": Js.Undefined.t(Regularizers.ffi),
    } =>
    Conv2dLayer.t =
    "";
  let separableConv2d =
      (
        ~depthMultiplier=?,
        ~depthwiseInitializer=?,
        ~pointwiseInitializer=?,
        ~depthwiseConstraint=?,
        ~pointwiseConstraint=?,
        ~depthwiseRegularizer=?,
        ~pointwiseRegularizer=?,
        (),
      ) =>
    {
      "depthMultiplier": depthMultiplier |> Js.Undefined.fromOption,
      "depthwiseInitializer":
        depthwiseInitializer
        |. Belt.Option.map(x => Conv2d(x))
        |. Belt.Option.map(convInitializerToFfi)
        |> Js.Undefined.fromOption,
      "pointwiseInitializer":
        pointwiseInitializer
        |. Belt.Option.map(x => Conv2d(x))
        |. Belt.Option.map(convInitializerToFfi)
        |> Js.Undefined.fromOption,
      "depthwiseConstraint":
        depthwiseConstraint
        |. Belt.Option.map(Constraints.constraintTypesToJs)
        |> Js.Undefined.fromOption,
      "pointwiseConstraint":
        pointwiseConstraint
        |. Belt.Option.map(Constraints.constraintTypesToJs)
        |> Js.Undefined.fromOption,
      "depthwiseRegularizer":
        depthwiseRegularizer
        |. Belt.Option.map(Regularizers.regularizerTypeToJs)
        |> Js.Undefined.fromOption,
      "pointwiseRegularizer":
        pointwiseRegularizer
        |. Belt.Option.map(Regularizers.regularizerTypeToJs)
        |> Js.Undefined.fromOption,
    }
    |> separableConv2d;
};

module Merge = (R: Core.Rank, D: Core.DataType) => {
  module Layer = Layer(R, R, D);
  module Configs = Configs(R, D);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external add : Js.Undefined.t(Configs.inputConfig) => Layer.t = "";
  let add = Configs.callFnWithInputConfig(add);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external average : Js.Undefined.t(Configs.inputConfig) => Layer.t = "";
  let average = Configs.callFnWithInputConfig(average);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external concatenate :
    Js.Undefined.t({. "axis": Js.Undefined.t(int)}) => Layer.t =
    "";
  let concatenate = (~axis=?, ()) =>
    {"axis": axis |. Belt.Option.map(R.axisToJs) |> Js.Undefined.fromOption}
    |> Js.Undefined.return
    |> concatenate;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external maximum : Js.Undefined.t(Configs.inputConfig) => Layer.t = "";
  let maximum = Configs.callFnWithInputConfig(maximum);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external minimum : Js.Undefined.t(Configs.inputConfig) => Layer.t = "";
  let minimum = Configs.callFnWithInputConfig(minimum);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external multiply : Js.Undefined.t(Configs.inputConfig) => Layer.t = "";
  let multiply = Configs.callFnWithInputConfig(multiply);
};

module Normalization = (R: Core.Rank, D: Core.DataType) => {
  module Layer = Layer(R, R, D);
  module Configs = Configs(R, D);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external batchNormalization :
    Js.Undefined.t(Configs.normalizeConfig) => Layer.t =
    "";
  let batchNormalization =
    Configs.callFnWithNormalizeConfig(batchNormalization);
};

module Pooling = (D: Core.DataType) => {
  module Conv1dLayer = Layer(Core.Rank3, Core.Rank3, D);
  module Conv1dDownRankLayer = Layer(Core.Rank3, Core.Rank2, D);
  module Conv2dLayer = Layer(Core.Rank4, Core.Rank4, D);
  module Configs1dLayer = Configs(Core.Rank3, D);
  module Configs2dLayer = Configs(Core.Rank4, D);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external averagePooling1d :
    Js.Undefined.t(Configs1dLayer.poolingConfig) => Conv1dLayer.t =
    "";
  let averagePooling1d =
    Configs1dLayer.callFnWithPoolingConfig(averagePooling1d);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external averagePooling2d :
    Js.Undefined.t(Configs2dLayer.poolingConfig) => Conv2dLayer.t =
    "";
  let averagePooling2d =
    Configs2dLayer.callFnWithPoolingConfig(averagePooling2d);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external globalAveragePooling1d :
    Js.Undefined.t(Configs1dLayer.poolingConfig) => Conv1dDownRankLayer.t =
    "";
  let globalAveragePooling1d =
    Configs1dLayer.callFnWithPoolingConfig(globalAveragePooling1d);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external globalAveragePooling2d :
    Js.Undefined.t({. "dataFormat": Js.Undefined.t(string)}) =>
    Conv2dLayer.t =
    "";
  let globalAveragePooling2d = (~dataFormat=?, ()) =>
    {
      "dataFormat":
        dataFormat
        |. Belt.Option.map(dataFormatToJs)
        |> Js.Undefined.fromOption,
    }
    |> Js.Undefined.return
    |> globalAveragePooling2d;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external globalMaxPooling1d :
    Js.Undefined.t(Configs1dLayer.poolingConfig) => Conv1dDownRankLayer.t =
    "";
  let globalMaxPooling1d =
    Configs1dLayer.callFnWithPoolingConfig(globalMaxPooling1d);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external globalMaxPooling2d :
    Js.Undefined.t({. "dataFormat": Js.Undefined.t(string)}) =>
    Conv2dLayer.t =
    "";
  let globalMaxPooling2d = (~dataFormat=?, ()) =>
    {
      "dataFormat":
        dataFormat
        |. Belt.Option.map(dataFormatToJs)
        |> Js.Undefined.fromOption,
    }
    |> Js.Undefined.return
    |> globalMaxPooling2d;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external maxPooling1d :
    Js.Undefined.t(Configs1dLayer.poolingConfig) => Conv1dLayer.t =
    "";
  let maxPooling1d = Configs1dLayer.callFnWithPoolingConfig(maxPooling1d);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external maxPooling2d :
    Js.Undefined.t(Configs2dLayer.poolingConfig) => Conv2dLayer.t =
    "";
  let maxPooling2d = Configs2dLayer.callFnWithPoolingConfig(maxPooling2d);
};

module Recurrent = (D: Core.DataType) => {
  module Rnn2dLayer = Layer(Core.Rank3, Core.Rank3, D);
  module RnnCell = RnnCell(Core.Rank2, Core.Rank2, D);
  module Configs2dLayer = Configs(Core.Rank3, D);
  [@bs.module "tensorflow/tfjs"] [@bs.scope "layers"]
  external gru : Configs2dLayer.recurrentConfig => Rnn2dLayer.t = "";
  let gru = Configs2dLayer.callFnWithGruLayerConfig(gru);
  [@bs.module "tensorflow/tfjs"] [@bs.scope "layers"]
  external gruCell : Configs2dLayer.recurrentConfig => RnnCell.t = "";
  let gruCell = Configs2dLayer.callFnWithGruCellLayerConfig(gruCell);
  [@bs.module "tensorflow/tfjs"] [@bs.scope "layers"]
  external lstm : Configs2dLayer.recurrentConfig => Rnn2dLayer.t = "";
  let lstm = Configs2dLayer.callFnWithLstmLayerConfig(lstm);
  [@bs.module "tensorflow/tfjs"] [@bs.scope "layers"]
  external lstmCell : Configs2dLayer.recurrentConfig => RnnCell.t = "";
  let lstmCell = Configs2dLayer.callFnWithLstmCellLayerConfig(lstmCell);
  [@bs.module "tensorflow/tfjs"] [@bs.scope "layers"]
  external rnn : Configs2dLayer.recurrentConfig => Rnn2dLayer.t = "";
  let rnn = Configs2dLayer.callFnWithRnnLayerConfig(rnn);
  [@bs.module "tensorflow/tfjs"] [@bs.scope "layers"]
  external simpleRNN : Configs2dLayer.recurrentConfig => Rnn2dLayer.t = "";
  let simpleRNN = Configs2dLayer.callFnWithSimpleLayerConfig(simpleRNN);
  [@bs.module "tensorflow/tfjs"] [@bs.scope "layers"]
  external simpleRNNCell : Configs2dLayer.recurrentConfig => RnnCell.t = "";
  let simpleRNNCell =
    Configs2dLayer.callFnWithSimpleCellLayerConfig(simpleRNNCell);
  [@bs.module "tensorflow/tfjs"] [@bs.scope "layers"]
  external stackedRNNCells : Configs2dLayer.stackedRnnCellsConfig => RnnCell.t =
    "";
  let stackedRNNCells =
    Configs2dLayer.callFnWithStackedCellsLayerConfig(stackedRNNCells);
};

module Inputs = (D: Core.DataType) => {
  module Input1dLayer = Layer(Core.Rank1, Core.Rank2, D);
  module Input2dLayer = Layer(Core.Rank2, Core.Rank3, D);
  module Input3dLayer = Layer(Core.Rank3, Core.Rank4, D);
  module Configs1d = Configs(Core.Rank1, D);
  module Configs2d = Configs(Core.Rank2, D);
  module Configs3d = Configs(Core.Rank3, D);
  [@bs.module "tensorflow/tfjs"]
  external input1d : Js.Undefined.t(Configs1d.inputConfig) => Input1dLayer.t =
    "input";
  let input1d = Configs1d.callFnWithInputConfig(input1d);
  [@bs.module "tensorflow/tfjs"]
  external input2d : Js.Undefined.t(Configs2d.inputConfig) => Input2dLayer.t =
    "input";
  let input2d = Configs2d.callFnWithInputConfig(input2d);
  [@bs.module "tensorflow/tfjs"]
  external input3d : Js.Undefined.t(Configs3d.inputConfig) => Input3dLayer.t =
    "input";
  let input3d = Configs3d.callFnWithInputConfig(input3d);
};
