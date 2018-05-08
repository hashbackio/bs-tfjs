module Optimizer = (R: Core.Rank, D: Core.DataType) => {
  type t;
  [@bs.send]
  external minimize :
    (t, unit => Core.Scalar(D).t) => Js.Nullable.t(Core.Scalar(D).t) =
    "";
  let minimize = (t, f) => minimize(t, f) |> Js.Nullable.toOption;
  [@bs.send]
  external minimizeWithOptions :
    (t, unit => Core.Scalar(D).t, bool, array(Core.Variable(R)(D).t)) =>
    Js.Nullable.t(Core.Scalar(D).t) =
    "minimize";
  let minimizeWithOptions = (t, f, returnCost, varList) =>
    minimizeWithOptions(t, f, returnCost, varList) |> Js.Nullable.toOption;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "train"]
  external sgd : float => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "train"]
  external momentum : (float, float) => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "train"]
  external momentumNesterov : (float, float, [@bs.as {json|true|json}] _) => t =
    "momentum";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "train"]
  external adagrad : float => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "train"]
  external adagradWithOptions : (float, float) => t = "adagrad";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "train"]
  external adadelta : unit => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "train"]
  external adadeltaWithOptions : (float, float, float) => t = "adadelta";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "train"]
  external adam : unit => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "train"]
  external adamWithOptions : (float, float, float, float) => t = "adam";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "train"]
  external adamax : unit => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "train"]
  external adamaxWithOptions : (float, float, float, float, float) => t =
    "adamax";
  external rmsprop : float => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "train"]
  external rmspropWithOptions : (float, float, float, float, bool) => t =
    "rmsprop";
};

module Losses = (R: Core.Rank, D: Core.DataType) => {
  type t;
  [@bs.deriving jsConverter]
  type reduction = [
    | [@bs.as "NONE"] `None
    | [@bs.as "MEAN"] `Mean
    | [@bs.as "SUM"] `Sum
    | [@bs.as "SUM_BY_NONZERO_WEIGHTS"] `SumByNonzeroWeights
  ];
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "losses"]
  external absoluteDifference : (Core.Tensor(R)(D).t, Core.Tensor(R)(D).t) => t =
    "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "losses"]
  external absoluteDifferenceWithScalarOptions :
    (
      Core.Tensor(R)(D).t,
      Core.Tensor(R)(D).t,
      Core.Tensor(Core.Rank0)(D).t,
      string
    ) =>
    t =
    "";
  let absoluteDifferenceWithScalarOptions =
      (labels, predictions, weights, reduction) =>
    absoluteDifferenceWithScalarOptions(
      labels,
      predictions,
      weights,
      reduction |> reductionToJs,
    );
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "losses"]
  external absoluteDifferenceWithBroadcastOptions :
    (
      Core.Tensor(R)(D).t,
      Core.Tensor(R)(D).t,
      Core.Tensor(Core.Rank1)(D).t,
      string
    ) =>
    t =
    "";
  let absoluteDifferenceWithBroadcastOptions =
      (labels, predictions, weights, reduction) =>
    absoluteDifferenceWithBroadcastOptions(
      labels,
      predictions,
      weights,
      reduction |> reductionToJs,
    );
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "losses"]
  external absoluteDifferenceWithOptions :
    (Core.Tensor(R)(D).t, Core.Tensor(R)(D).t, Core.Tensor(R)(D).t, string) =>
    t =
    "";
  let absoluteDifferenceWithOptions =
      (labels, predictions, weights, reduction) =>
    absoluteDifferenceWithOptions(
      labels,
      predictions,
      weights,
      reduction |> reductionToJs,
    );
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "losses"]
  external computeWeightedLoss : Core.Tensor(R)(D).t => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "losses"]
  external computeWeightedLossWithScalarOptions :
    (Core.Tensor(R)(D).t, Core.Tensor(Core.Rank0)(D).t, string) => t =
    "";
  let computeWeightedLossWithScalarOptions = (labels, weights, reduction) =>
    computeWeightedLossWithScalarOptions(
      labels,
      weights,
      reduction |> reductionToJs,
    );
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "losses"]
  external computeWeightedLossWithBroadcastOptions :
    (Core.Tensor(R)(D).t, Core.Tensor(Core.Rank1)(D).t, string) => t =
    "";
  let computeWeightedLossWithBroadcastOptions = (labels, weights, reduction) =>
    computeWeightedLossWithBroadcastOptions(
      labels,
      weights,
      reduction |> reductionToJs,
    );
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "losses"]
  external computeWeightedLossWithOptions :
    (Core.Tensor(R)(D).t, Core.Tensor(R)(D).t, string) => t =
    "";
  let computeWeightedLossWithOptions = (labels, weights, reduction) =>
    computeWeightedLossWithOptions(
      labels,
      weights,
      reduction |> reductionToJs,
    );
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "losses"]
  external softmaxCrossEntropy :
    (Core.Tensor(R)(D).t, Core.Tensor(R)(D).t) => t =
    "";
};
