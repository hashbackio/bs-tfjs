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
  [@bs.deriving jsConverter]
  type padding = [ | `valid | `same | `casual];
  [@bs.deriving jsConverter]
  type dataFormat = [ | `channelsFirst | `channelsLast];
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
