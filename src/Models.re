[@bs.deriving jsConverter]
type modelLoggingVerbosity =
  | Silent
  | Verbose;

module SymbolicTensor = (R: Core.Rank, D: Core.DataType) => {
  type t;
};

module Configs =
       (
         Rin: Core.Rank,
         Rout: Core.Rank,
         Din: Core.DataType,
         Dout: Core.DataType,
       ) => {
  module SymbolicTensorIn = SymbolicTensor(Rin, Din);
  module SymbolicTensorOut = SymbolicTensor(Rout, Dout);
  module Optimizer = Training.Optimizer(Rin, Din);
  module Losses = Training.Losses(Rin, Din);
  type modelConfig = {
    .
    "inputs": array(SymbolicTensorIn.t),
    "outputs": array(SymbolicTensorOut.t),
    "name": Js.Undefined.t(string),
  };
  type inputConfig = {
    .
    "shape": Js.Undefined.t(array(int)),
    "batchShape": Js.Undefined.t(array(int)),
    "name": Js.Undefined.t(string),
    "dtype": string,
    "sparse": Js.Undefined.t(bool),
  };
  type evaluateOrPredictConfig = {
    .
    "batchSize": Js.Undefined.t(int),
    "verbose": Js.Undefined.t(int),
    /* TODO: add in sampleWeights */
    "steps": Js.Undefined.t(int),
  };
  type fitConfig = {
    .
    /* TODO callbacks, classWeight, sampleWeight, validationData */
    "batchSize": Js.Undefined.t(int),
    "epochs": Js.Undefined.t(int),
    "verbose": Js.Undefined.t(int),
    "validationSplit": Js.Undefined.t(float),
    "shuffle": Js.Undefined.t(bool),
    "initialEpoch": Js.Undefined.t(int),
    "stepsPerEpoch": Js.Undefined.t(int),
    "validationSteps": Js.Undefined.t(int),
  };
  type compileConfig = {
    .
    /* TODO metrics */
    "optimizer": Optimizer.t,
    "loss": Losses.t,
  };
  let callFnWithModelConfig =
      (fn: modelConfig => 'a, ~inputs, ~outputs, ~name=?, ()) =>
    {
      "inputs": inputs,
      "outputs": outputs,
      "name": name |> Js.Undefined.fromOption,
    }
    |> fn;
  let callFnWithInputConfig =
      (fn: inputConfig => 'a, ~shape=?, ~batchShape=?, ~name=?, ~sparse=?, ()) =>
    {
      "shape":
        shape
        |. Belt.Option.map(Rin.getInputShapeArray)
        |> Js.Undefined.fromOption,
      "batchShape":
        batchShape
        |. Belt.Option.map(Rin.getShapeArray)
        |> Js.Undefined.fromOption,
      "name": name |> Js.Undefined.fromOption,
      "sparse": sparse |> Js.Undefined.fromOption,
      "dtype": Din.dType |> Core.dTypeToJs,
    }
    |> fn;
  let callFnWithEvaluateConfig =
      (
        fn: evaluateOrPredictConfig => 'a,
        ~batchSize=?,
        ~verbose=?,
        ~steps=?,
        (),
      ) =>
    {
      "batchSize": batchSize |> Js.Undefined.fromOption,
      "verbose":
        verbose
        |. Belt.Option.map(modelLoggingVerbosityToJs)
        |> Js.Undefined.fromOption,
      "steps": steps |> Js.Undefined.fromOption,
    }
    |> fn;
  let callFnWithPredictConfig =
      (fn: evaluateOrPredictConfig => 'a, ~batchSize=?, ~verbose=?, ()) =>
    {
      "batchSize": batchSize |> Js.Undefined.fromOption,
      "verbose":
        verbose
        |. Belt.Option.map(modelLoggingVerbosityToJs)
        |> Js.Undefined.fromOption,
      "steps": Js.Undefined.empty,
    }
    |> fn;
  let callFnWithFitConfig =
      (
        fn: fitConfig => 'a,
        ~batchSize=?,
        ~epochs=?,
        ~verbose=?,
        ~validationSplit=?,
        ~shuffle=?,
        ~initialEpoch=?,
        ~stepsPerEpoch=?,
        ~validationSteps=?,
      ) =>
    {
      "batchSize": batchSize |> Js.Undefined.fromOption,
      "epochs": epochs |> Js.Undefined.fromOption,
      "verbose":
        verbose
        |. Belt.Option.map(modelLoggingVerbosityToJs)
        |> Js.Undefined.fromOption,
      "validationSplit": validationSplit |> Js.Undefined.fromOption,
      "shuffle": shuffle |> Js.Undefined.fromOption,
      "initialEpoch": initialEpoch |> Js.Undefined.fromOption,
      "stepsPerEpoch": stepsPerEpoch |> Js.Undefined.fromOption,
      "validationSteps": validationSteps |> Js.Undefined.fromOption,
    }
    |> fn;
  let callFnWithCompileConfig = (fn: compileConfig => 'a, ~optimizer, ~loss) =>
    {"optimizer": optimizer, "loss": loss} |> fn;
};

module Model =
       (
         Rin: Core.Rank,
         Rout: Core.Rank,
         Din: Core.DataType,
         Dout: Core.DataType,
       ) => {
  module Configs = Configs(Rin, Rout, Din, Dout);
  module SymbolicTensorIn = Configs.SymbolicTensorIn;
  module SymbolicTensorOut = Configs.SymbolicTensorIn;
  module Optimizer = Training.Optimizer(Rin, Din);
  module Losses = Training.Losses(Rin, Din);
  module TensorIn = Core.Tensor(Rin, Din);
  module TensorOut = Core.Tensor(Rout, Dout);
  type model;
  type compiledModel;
  [@bs.module "@tensorflow/tfjs"]
  external make : Configs.modelConfig => model = "model";
  let make = Configs.callFnWithModelConfig(make);
  [@bs.module "@tensorflow/tfjs"]
  external loadModel : string => Js.Promise.t(model) = "";
  module Input = {
    [@bs.module "@tensorflow/tfjs"]
    external make : Configs.inputConfig => SymbolicTensorIn.t = "input";
    let make = Configs.callFnWithInputConfig(make);
  };
  [@bs.send]
  external compile :
    (
      model,
      {
        .
        "optimizer": Optimizer.t,
        "loss": Losses.t,
      }
    ) =>
    compiledModel =
    "";
  let compile = Configs.callFnWithCompileConfig(config => compile(_, config));
  [@bs.send]
  external evaluate :
    (
      compiledModel,
      array(TensorIn.t),
      array(TensorOut.t),
      Configs.evaluateOrPredictConfig
    ) =>
    Core.Scalar(Dout).t =
    "";
  let evaluate = (compiledModel, x, y) =>
    Configs.callFnWithEvaluateConfig(evaluate(compiledModel, x, y));
  [@bs.send]
  external predict :
    (model, array(TensorIn.t), Configs.evaluateOrPredictConfig) => TensorOut.t =
    "";
  let predict = (compiledModel, x) =>
    Configs.callFnWithPredictConfig(predict(compiledModel, x));
  [@bs.send]
  external predictOnBatch : (compiledModel, array(TensorIn.t)) => TensorOut.t =
    "";
  [@bs.send]
  external fit :
    (
      compiledModel,
      array(TensorIn.t),
      array(TensorOut.t),
      Configs.fitConfig
    ) =>
    Core.Scalar(Dout).t =
    "";
  let fit = (compiledModel, x, y) =>
    Configs.callFnWithFitConfig(fit(compiledModel, x, y));
};
