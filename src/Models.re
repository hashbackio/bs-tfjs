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
  [@bs.deriving abstract]
  type modelConfig = {
    inputs: SymbolicTensorIn.t,
    outputs: SymbolicTensorOut.t,
    [@bs.optional]
    name: string,
  };
  [@bs.deriving abstract]
  type modelConfigWithArrays = {
    inputs: array(SymbolicTensorIn.t),
    outputs: array(SymbolicTensorOut.t),
    [@bs.optional]
    name: string,
  };
  [@bs.deriving abstract]
  type inputConfig = {
    [@bs.optional]
    shape: array(int),
    [@bs.optional]
    batchShape: array(int),
    [@bs.optional]
    name: string,
    dtype: string,
    [@bs.optional]
    sparse: bool,
  };
  [@bs.deriving abstract]
  type evaluateOrPredictConfig = {
    [@bs.optional]
    batchSize: int,
    [@bs.optional]
    verbose: int,
    /* TODO: add in sampleWeights */
    [@bs.optional]
    steps: int,
  };
  [@bs.deriving abstract]
  type fitConfig = {
    /* TODO callbacks, classWeight, sampleWeight, validationData */
    [@bs.optional]
    batchSize: int,
    [@bs.optional]
    epochs: int,
    [@bs.optional]
    verbose: int,
    [@bs.optional]
    validationSplit: float,
    [@bs.optional]
    shuffle: bool,
    [@bs.optional]
    initialEpoch: int,
    [@bs.optional]
    stepsPerEpoch: int,
    [@bs.optional]
    validationSteps: int,
  };
  [@bs.deriving abstract]
  type compileConfig = {
    optimizer: Optimizer.t,
    loss: Losses.t,
    metrics: array(string),
  };
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
  [@bs.module "@tensorflow/tfjs"]
  external loadModel : string => Js.Promise.t(model) = "";
  module Input = {
    [@bs.module "@tensorflow/tfjs"]
    external make : Configs.inputConfig => SymbolicTensorIn.t = "input";
  };
  external _unsafeToCompiledModel : 'a => compiledModel = "%identity";
  [@bs.send] external compile : (model, Configs.compileConfig) => unit = "";
  let compile: (model, Configs.compileConfig) => compiledModel =
    (model, config) => {
      compile(model, config);
      model |> _unsafeToCompiledModel;
    };
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
  [@bs.send]
  external predict :
    (compiledModel, array(TensorIn.t), Configs.evaluateOrPredictConfig) =>
    TensorOut.t =
    "";
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
    Js.Promise.t(unit) =
    "";
};
