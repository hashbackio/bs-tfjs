/* TODO: Convert the convolutional layers to use bs.deriving abstract */
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

module Layer =
       (
         Rin: Core.Rank,
         Rout: Core.Rank,
         Din: Core.DataType,
         Dout: Core.DataType,
       ) => {
  module SymbolicTensorIn = Models.SymbolicTensor(Rin, Din);
  module SymbolicTensorOut = Models.SymbolicTensor(Rout, Dout);
  module TensorIn = Core.Tensor(Rin, Din);
  module TensorOut = Core.Tensor(Rout, Dout);
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
  module Tensor = Core.Tensor(R, D);
  [@bs.deriving abstract]
  type inputConfig = {
    [@bs.optional]
    shape: array(int),
    [@bs.optional]
    batchInputShape: array(int),
    [@bs.optional]
    batchSize: int,
    [@bs.optional]
    dtype: string,
    [@bs.optional]
    name: string,
    [@bs.optional]
    trainable: bool,
  };
  [@bs.deriving abstract]
  type activationConfig = {activation: string};
  [@bs.deriving abstract]
  type denseConfig = {
    units: int,
    [@bs.optional]
    activation: string,
    [@bs.optional]
    useBias: bool,
    [@bs.optional]
    kernelInitializer: Initializer.ffi,
    [@bs.optional]
    biasInitializer: Initializer.ffi,
    [@bs.optional]
    inputDim: int,
    [@bs.optional]
    kernelConstraint: Constraints.ffi,
    [@bs.optional]
    biasConstraint: Constraints.ffi,
    [@bs.optional]
    kernelRegularizer: Regularizers.ffi,
    [@bs.optional]
    biasRegularizer: Regularizers.ffi,
    [@bs.optional]
    activityRegularizer: Regularizers.ffi,
    [@bs.optional]
    name: string,
    [@bs.optional]
    trainable: bool,
    [@bs.optional]
    weights: array(Tensor.t),
  };
  [@bs.deriving abstract]
  type dropoutConfig = {
    rate: float,
    [@bs.optional]
    noiseShape: array(int),
    [@bs.optional]
    seed: int,
  };
  [@bs.deriving abstract]
  type embeddingConfig = {
    inputDim: int,
    outputDim: int,
    [@bs.optional]
    embeddingsInitializer: Initializer.ffi,
    [@bs.optional]
    embeddingsRegularizer: Regularizers.ffi,
    [@bs.optional]
    activityRegularizer: Regularizers.ffi,
    [@bs.optional]
    embeddingsConstraint: Constraints.ffi,
    [@bs.optional]
    maskZero: bool,
    [@bs.optional]
    inputLength: int,
    [@bs.optional]
    name: string,
    [@bs.optional]
    trainable: bool,
    [@bs.optional]
    weights: array(Tensor.t),
  };
  [@bs.deriving abstract]
  type repeatVectorConfig = {n: int};
  [@bs.deriving abstract]
  type normalizeConfig = {
    [@bs.optional]
    axis: int,
    [@bs.optional]
    momentum: float,
    [@bs.optional]
    epsilon: float,
    [@bs.optional]
    center: bool,
    [@bs.optional]
    scale: bool,
    [@bs.optional]
    betaInitializer: Initializer.ffi,
    [@bs.optional]
    gammaInitializer: Initializer.ffi,
    [@bs.optional]
    movingMeanInitializer: Initializer.ffi,
    [@bs.optional]
    movingVarianceInitializer: Initializer.ffi,
    [@bs.optional]
    betaConstraint: Constraints.ffi,
    [@bs.optional]
    gammaConstraint: Constraints.ffi,
    [@bs.optional]
    betaRegularizer: Regularizers.ffi,
    [@bs.optional]
    gammaRegularizer: Regularizers.ffi,
  };
  [@bs.deriving abstract]
  type poolingConfig = {
    [@bs.optional]
    poolSize: int,
    [@bs.optional]
    strides: int,
    [@bs.optional]
    padding: string,
    [@bs.optional]
    dataFormat: string,
  };
  [@bs.deriving abstract]
  type rnnConfig = {
    [@bs.optional]
    cell: array(RnnCell(R)(R)(D).t),
    [@bs.optional]
    returnSequences: bool,
    [@bs.optional]
    returnState: bool,
    [@bs.optional]
    goBackwards: bool,
    [@bs.optional]
    stateful: bool,
    [@bs.optional]
    unroll: bool,
    [@bs.optional]
    inputDim: int,
    [@bs.optional]
    inputLength: int,
  };
  [@bs.deriving abstract]
  type simpleRNNConfig = {
    [@bs.optional]
    cell: array(RnnCell(R)(R)(D).t),
    [@bs.optional]
    returnSequences: bool,
    [@bs.optional]
    returnState: bool,
    [@bs.optional]
    goBackwards: bool,
    [@bs.optional]
    stateful: bool,
    [@bs.optional]
    unroll: bool,
    [@bs.optional]
    inputDim: int,
    [@bs.optional]
    inputLength: int,
    units: int,
    [@bs.optional]
    activation: string,
    [@bs.optional]
    useBias: bool,
    [@bs.optional]
    kernelInitializer: Initializer.ffi,
    [@bs.optional]
    recurrentInitializer: Initializer.ffi,
    [@bs.optional]
    biasInitializer: Initializer.ffi,
    [@bs.optional]
    kernelRegularizer: Regularizers.ffi,
    [@bs.optional]
    recurrentRegularizer: Regularizers.ffi,
    [@bs.optional]
    biasRegularizer: Regularizers.ffi,
    [@bs.optional]
    kernelConstraint: Constraints.ffi,
    [@bs.optional]
    recurrentConstraint: Constraints.ffi,
    [@bs.optional]
    biasConstraint: Constraints.ffi,
    [@bs.optional]
    dropout: float,
    [@bs.optional]
    recurrentDropout: float,
  };
  [@bs.deriving abstract]
  type gruConfig = {
    [@bs.optional]
    cell: array(RnnCell(R)(R)(D).t),
    [@bs.optional]
    returnSequences: bool,
    [@bs.optional]
    returnState: bool,
    [@bs.optional]
    goBackwards: bool,
    [@bs.optional]
    stateful: bool,
    [@bs.optional]
    unroll: bool,
    [@bs.optional]
    inputDim: int,
    [@bs.optional]
    inputLength: int,
    units: int,
    [@bs.optional]
    activation: string,
    [@bs.optional]
    useBias: bool,
    [@bs.optional]
    kernelInitializer: Initializer.ffi,
    [@bs.optional]
    recurrentInitializer: Initializer.ffi,
    [@bs.optional]
    biasInitializer: Initializer.ffi,
    [@bs.optional]
    kernelRegularizer: Regularizers.ffi,
    [@bs.optional]
    recurrentRegularizer: Regularizers.ffi,
    [@bs.optional]
    biasRegularizer: Regularizers.ffi,
    [@bs.optional]
    kernelConstraint: Constraints.ffi,
    [@bs.optional]
    recurrentConstraint: Constraints.ffi,
    [@bs.optional]
    biasConstraint: Constraints.ffi,
    [@bs.optional]
    dropout: float,
    [@bs.optional]
    recurrentDropout: float,
    [@bs.optional]
    implementation: int,
  };
  [@bs.deriving abstract]
  type gruCellConfig = {
    [@bs.optional]
    cell: array(RnnCell(R)(R)(D).t),
    [@bs.optional]
    returnSequences: bool,
    [@bs.optional]
    returnState: bool,
    [@bs.optional]
    goBackwards: bool,
    [@bs.optional]
    stateful: bool,
    [@bs.optional]
    unroll: bool,
    [@bs.optional]
    inputDim: int,
    [@bs.optional]
    inputLength: int,
    units: int,
    [@bs.optional]
    activation: string,
    [@bs.optional]
    useBias: bool,
    [@bs.optional]
    kernelInitializer: Initializer.ffi,
    [@bs.optional]
    recurrentInitializer: Initializer.ffi,
    [@bs.optional]
    biasInitializer: Initializer.ffi,
    [@bs.optional]
    kernelRegularizer: Regularizers.ffi,
    [@bs.optional]
    recurrentRegularizer: Regularizers.ffi,
    [@bs.optional]
    biasRegularizer: Regularizers.ffi,
    [@bs.optional]
    kernelConstraint: Constraints.ffi,
    [@bs.optional]
    recurrentConstraint: Constraints.ffi,
    [@bs.optional]
    biasConstraint: Constraints.ffi,
    [@bs.optional]
    dropout: float,
    [@bs.optional]
    recurrentDropout: float,
    [@bs.optional]
    implementation: int,
    [@bs.optional]
    recurrentActivation: string,
  };
  [@bs.deriving abstract]
  type lstmConfig = {
    [@bs.optional]
    cell: array(RnnCell(R)(R)(D).t),
    [@bs.optional]
    returnSequences: bool,
    [@bs.optional]
    returnState: bool,
    [@bs.optional]
    goBackwards: bool,
    [@bs.optional]
    stateful: bool,
    [@bs.optional]
    unroll: bool,
    [@bs.optional]
    inputDim: int,
    [@bs.optional]
    inputLength: int,
    units: int,
    [@bs.optional]
    activation: string,
    [@bs.optional]
    useBias: bool,
    [@bs.optional]
    kernelInitializer: Initializer.ffi,
    [@bs.optional]
    recurrentInitializer: Initializer.ffi,
    [@bs.optional]
    biasInitializer: Initializer.ffi,
    [@bs.optional]
    kernelRegularizer: Regularizers.ffi,
    [@bs.optional]
    recurrentRegularizer: Regularizers.ffi,
    [@bs.optional]
    biasRegularizer: Regularizers.ffi,
    [@bs.optional]
    kernelConstraint: Constraints.ffi,
    [@bs.optional]
    recurrentConstraint: Constraints.ffi,
    [@bs.optional]
    biasConstraint: Constraints.ffi,
    [@bs.optional]
    dropout: float,
    [@bs.optional]
    recurrentDropout: float,
    [@bs.optional]
    implementation: int,
    [@bs.optional]
    unitForgetBias: bool,
  };
  [@bs.deriving abstract]
  type lstmCellConfig = {
    [@bs.optional]
    cell: array(RnnCell(R)(R)(D).t),
    [@bs.optional]
    returnSequences: bool,
    [@bs.optional]
    returnState: bool,
    [@bs.optional]
    goBackwards: bool,
    [@bs.optional]
    stateful: bool,
    [@bs.optional]
    unroll: bool,
    [@bs.optional]
    inputDim: int,
    [@bs.optional]
    inputLength: int,
    units: int,
    [@bs.optional]
    activation: string,
    [@bs.optional]
    useBias: bool,
    [@bs.optional]
    kernelInitializer: Initializer.ffi,
    [@bs.optional]
    recurrentInitializer: Initializer.ffi,
    [@bs.optional]
    biasInitializer: Initializer.ffi,
    [@bs.optional]
    kernelRegularizer: Regularizers.ffi,
    [@bs.optional]
    recurrentRegularizer: Regularizers.ffi,
    [@bs.optional]
    biasRegularizer: Regularizers.ffi,
    [@bs.optional]
    kernelConstraint: Constraints.ffi,
    [@bs.optional]
    recurrentConstraint: Constraints.ffi,
    [@bs.optional]
    biasConstraint: Constraints.ffi,
    [@bs.optional]
    dropout: float,
    [@bs.optional]
    recurrentDropout: float,
    [@bs.optional]
    implementation: int,
    [@bs.optional]
    recurrentActivation: string,
    [@bs.optional]
    unitForgetBias: bool,
  };
  [@bs.deriving abstract]
  type stackedRnnCellsConfig = {cells: array(RnnCell(R)(R)(D).t)};
};

module Activations = (R: Core.Rank, Din: Core.DataType, Dout: Core.DataType) => {
  module Layer = Layer(R, R, Din, Dout);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external elu : unit => Layer.t = "";
  [@bs.deriving abstract]
  type eluConfig = {alpha: float};
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external eluWithConfig : eluConfig => Layer.t = "elu";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external leakyReLU : unit => Layer.t = "";
  [@bs.deriving abstract]
  type leakyReLUConfig = {alpha: float};
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external leakyReLUWithConfig : leakyReLUConfig => Layer.t = "leakyReLU";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external softmax : unit => Layer.t = "";
  [@bs.deriving abstract]
  type softmaxConfig = {axis: int};
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external softmaxWithConfig : softmaxConfig => Layer.t = "softmax";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external thresholdedReLU : unit => Layer.t = "";
  [@bs.deriving abstract]
  type thresholdedReLUConfig = {theta: float};
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external thresholdedReLUWithConifg : thresholdedReLUConfig => Layer.t =
    "thresohldedReLU";
};

module Basic =
       (
         Rin: Core.Rank,
         Rout: Core.Rank,
         Din: Core.DataType,
         Dout: Core.DataType,
       ) => {
  module EmbeddingLayer = Layer(Core.Rank1, Core.Rank2, Din, Dout);
  module FlattenLayer = Layer(Rin, Core.Rank1, Din, Dout);
  module Layer = Layer(Rin, Rout, Din, Dout);
  module Initializer = Initializers.Initializer(Rin, Din);
  module Configs = Configs(Rin, Din);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external activation : Configs.activationConfig => Layer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external dense : Configs.denseConfig => Layer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external dropout : Configs.dropoutConfig => Layer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external embedding : Configs.embeddingConfig => EmbeddingLayer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external flatten : unit => FlattenLayer.t = "";
  external flattenWithConfig : Configs.inputConfig => FlattenLayer.t =
    "flatten";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external repeatVector : Configs.repeatVectorConfig => Layer.t = "";
};

module Convolutional = (D: Core.DataType) => {
  module Conv1dLayer = Layer(Core.Rank3, Core.Rank3, D, D);
  module Conv2dLayer = Layer(Core.Rank4, Core.Rank4, D, D);
  module Conv1dInitializer = Initializers.Initializer(Core.Rank3, D);
  module Conv2dInitializer = Initializers.Initializer(Core.Rank4, D);
  type kernelSizeFfi;
  type kernelSize1D =
    | Square(int)
    | Rectangle(int, int);
  type kernelSize2D =
    | Cube(int)
    | Box(int, int, int);
  external _unsafeToKernelSizeFfi : 'a => kernelSizeFfi = "%identity";
  let kernelSize1DToFfi = kernelSize1D =>
    switch (kernelSize1D) {
    | Square(length) => length |> _unsafeToKernelSizeFfi
    | Rectangle(length, width) => [|length, width|] |> _unsafeToKernelSizeFfi
    };
  let kernelSize2DToFfi = kernelSize2D =>
    switch (kernelSize2D) {
    | Cube(length) => length |> _unsafeToKernelSizeFfi
    | Box(length, width, height) =>
      [|length, width, height|] |> _unsafeToKernelSizeFfi
    };
  type strideFfi;
  external _unsafeToStrideFfi : 'a => strideFfi = "%identity";
  type dilationRateFfi;
  external _unsafeToDilationRateFfi : 'a => dilationRateFfi = "%identity";
  type strideOrDilationRate1D =
    | EvenStride(int)
    | OddStride(int, int)
    | Dilation(int);
  let strideOrDilationRate1DToStrideFfi = strideOrDilationRate1D =>
    switch (strideOrDilationRate1D) {
    | EvenStride(stride) => stride |> _unsafeToStrideFfi
    | OddStride(length, width) => [|length, width|] |> _unsafeToStrideFfi
    | Dilation(_) => 1 |> _unsafeToStrideFfi
    };
  let strideOrDilationRate1DToDilationRateFfi = strideOrDilationRate1D =>
    switch (strideOrDilationRate1D) {
    | EvenStride(_) => 1 |> _unsafeToDilationRateFfi
    | OddStride(_, _) => 1 |> _unsafeToDilationRateFfi
    | Dilation(length) => length |> _unsafeToDilationRateFfi
    };
  type strideOrDilationRate2D =
    | EvenStride(int)
    | OddStride(int, int, int)
    | EqualDilation(int)
    | RectangularDilation(int, int);
  let strideOrDilationRate2DToStrideFfi = strideOrDilationRate2D =>
    switch (strideOrDilationRate2D) {
    | EvenStride(length) => length |> _unsafeToStrideFfi
    | OddStride(length, width, height) =>
      [|length, width, height|] |> _unsafeToStrideFfi
    | EqualDilation(_) => 1 |> _unsafeToStrideFfi
    | RectangularDilation(_, _) => 1 |> _unsafeToStrideFfi
    };
  let strideOrDilationRate2DToDilationRateFfi = strideOrDilationRate2D =>
    switch (strideOrDilationRate2D) {
    | EvenStride(_) => 1 |> _unsafeToDilationRateFfi
    | OddStride(_, _, _) => 1 |> _unsafeToDilationRateFfi
    | EqualDilation(length) => length |> _unsafeToDilationRateFfi
    | RectangularDilation(length, width) =>
      [|length, width|] |> _unsafeToDilationRateFfi
    };
  type convInitializerFfi;
  external _unsafeToConvInitializerFfi : 'a => convInitializerFfi =
    "%identity";
  type convInitializer =
    | Conv1d(Conv1dInitializer.initializerType)
    | Conv2d(Conv2dInitializer.initializerType);
  let convInitializerToFfi = convInitializer =>
    switch (convInitializer) {
    | Conv1d(conv1dInitializer) =>
      conv1dInitializer
      |> Conv1dInitializer.initializerTypeToJs
      |> _unsafeToConvInitializerFfi
    | Conv2d(conv2dInitializer) =>
      conv2dInitializer
      |> Conv2dInitializer.initializerTypeToJs
      |> _unsafeToConvInitializerFfi
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
  module Layer = Layer(R, R, D, D);
  module Configs = Configs(R, D);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external add : unit => Layer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external addWithConfig : Configs.inputConfig => Layer.t = "add";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external average : unit => Layer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external averageWithConfig : Configs.inputConfig => Layer.t = "average";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external concatenate : unit => Layer.t = "";
  [@bs.deriving abstract]
  type concatenateConfig = {
    [@bs.optional]
    axis: int,
  };
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external concatenateWithConfig : concatenateConfig => Layer.t =
    "concatenate";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external maximum : unit => Layer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external maximumWithConfig : Configs.inputConfig => Layer.t = "maximum";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external minimum : unit => Layer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external minimumWithConfig : Configs.inputConfig => Layer.t = "minimum";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external multiply : unit => Layer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external multiplyWithConfig : Configs.inputConfig => Layer.t = "multiply";
};

module Normalization = (R: Core.Rank, D: Core.DataType) => {
  module Layer = Layer(R, R, D, D);
  module Configs = Configs(R, D);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external batchNormalization : unit => Layer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external batchNormalizationWithConfig : Configs.normalizeConfig => Layer.t =
    "batchNormalization";
};

module Pooling = (D: Core.DataType) => {
  module Conv1dLayer = Layer(Core.Rank3, Core.Rank3, D, D);
  module Conv1dDownRankLayer = Layer(Core.Rank3, Core.Rank2, D, D);
  module Conv2dLayer = Layer(Core.Rank4, Core.Rank4, D, D);
  module Configs1dLayer = Configs(Core.Rank3, D);
  module Configs2dLayer = Configs(Core.Rank4, D);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external averagePooling1d : unit => Conv1dLayer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external averagePooling1dWithConfig :
    Configs1dLayer.poolingConfig => Conv1dLayer.t =
    "averagePooling1d";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external averagePooling2d : unit => Conv2dLayer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external averagePooling2dWithConfig :
    Configs2dLayer.poolingConfig => Conv2dLayer.t =
    "averagePooling2d";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external globalAveragePooling1d : unit => Conv1dDownRankLayer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external globalAveragePooling1dWithConfig :
    Configs1dLayer.poolingConfig => Conv1dDownRankLayer.t =
    "globalAveragePooling1d";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external globalAveragePooling2d : unit => Conv2dLayer.t = "";
  [@bs.deriving abstract]
  type globalPooling2dConfig = {
    [@bs.optional]
    dataFormat: string,
  };
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external globalAveragePooling2dWithConfig :
    globalPooling2dConfig => Conv2dLayer.t =
    "globalAveragePooling2d";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external globalMaxPooling1d : unit => Conv1dDownRankLayer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external globalMaxPooling1dWithConfig :
    Configs1dLayer.poolingConfig => Conv1dDownRankLayer.t =
    "globalMaxPooling1d";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external globalMaxPooling2d : unit => Conv2dLayer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external globalMaxPooling2dWithConfig :
    globalPooling2dConfig => Conv2dLayer.t =
    "globalMaxPooling2d";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external maxPooling1d : unit => Conv1dLayer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external maxPooling1dWithConfig :
    Configs1dLayer.poolingConfig => Conv1dLayer.t =
    "maxPooling1d";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external maxPooling2d : unit => Conv2dLayer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external maxPooling2dWithConfig :
    Configs2dLayer.poolingConfig => Conv2dLayer.t =
    "maxPooling2d";
};

module Recurrent = (D: Core.DataType) => {
  module Rnn2dLayer = Layer(Core.Rank2, Core.Rank2, D, D);
  module RnnCell = RnnCell(Core.Rank1, Core.Rank1, D);
  module Configs2dLayer = Configs(Core.Rank2, D);
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external gru : Configs2dLayer.gruConfig => Rnn2dLayer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external gruCell : Configs2dLayer.gruCellConfig => RnnCell.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external lstm : Configs2dLayer.lstmConfig => Rnn2dLayer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external lstmCell : Configs2dLayer.lstmCellConfig => RnnCell.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external rnn : Configs2dLayer.rnnConfig => Rnn2dLayer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external simpleRNN : Configs2dLayer.simpleRNNConfig => Rnn2dLayer.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external simpleRNNCell : Configs2dLayer.simpleRNNConfig => RnnCell.t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external stackedRNNCells : Configs2dLayer.stackedRnnCellsConfig => RnnCell.t =
    "";
};

module Inputs = (D: Core.DataType) => {
  module Input1dTensor = Models.SymbolicTensor(Core.Rank1, D);
  module Input2dTensor = Models.SymbolicTensor(Core.Rank2, D);
  module Input3dTensor = Models.SymbolicTensor(Core.Rank3, D);
  module Configs1d = Configs(Core.Rank1, D);
  module Configs2d = Configs(Core.Rank2, D);
  module Configs3d = Configs(Core.Rank3, D);
  [@bs.module "@tensorflow/tfjs"]
  external input1d : unit => Input1dTensor.t = "input";
  [@bs.module "@tensorflow/tfjs"]
  external input1dWithConfig : Configs1d.inputConfig => Input1dTensor.t =
    "input";
  [@bs.module "@tensorflow/tfjs"]
  external input2d : unit => Input2dTensor.t = "input";
  [@bs.module "@tensorflow/tfjs"]
  external input2dWithConfig : Configs2d.inputConfig => Input2dTensor.t =
    "input";
  [@bs.module "@tensorflow/tfjs"]
  external input3d : unit => Input3dTensor.t = "input";
  [@bs.module "@tensorflow/tfjs"]
  external input3dWithConfig : Configs3d.inputConfig => Input3dTensor.t =
    "input";
};

module Wrapper = (D: Core.DataType) => {
  module Recurrent = Recurrent(D);
  module Layer = Recurrent.Rnn2dLayer;
  [@bs.deriving abstract]
  type bidirectionalConfig = {
    layer: Layer.t,
    mergeMode: string,
  };
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "layers"]
  external bidirectional : bidirectionalConfig => Layer.t = "";
};
