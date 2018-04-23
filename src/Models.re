module SymbolicTensor = {
  type t;
  type ts =
    | Single(t)
    | Many(list(t));
  external unsafeToJson : t => Js.Json.t = "%identity";
  let encode = ts =>
    switch (ts) {
    | Single(t) => t |> unsafeToJson
    | Many(ts) => ts |> Json.Encode.list(unsafeToJson)
    };
};

module ContainerConfig = {
  type t = {
    inputs: SymbolicTensor.ts,
    outputs: SymbolicTensor.ts,
    name: option(string),
  };
  let encode = ({inputs, outputs, name}) =>
    Json.Encode.(
      object_([
        ("inputs", inputs |> SymbolicTensor.encode),
        ("outputs", outputs |> SymbolicTensor.encode),
        ("name", name |> nullable(string)),
      ])
    );
};

module InputConfig = (R: Core.Rank, D: Core.DataType) => {
  type t = {
    shape: option(R.inputShape),
    batchShape: option(R.shape),
    name: option(string),
    sparse: option(bool),
  };
  let encode = ({shape, batchShape, name, sparse}) =>
    Json.Encode.(
      object_([
        (
          "shape",
          shape
          |> Js.Option.map((. x) => x |> R.getInputShapeArray)
          |> nullable(array(int)),
        ),
        (
          "batchShape",
          batchShape
          |> Js.Option.map((. x) => x |> R.getShapeArray)
          |> nullable(array(int)),
        ),
        ("name", name |> nullable(string)),
        ("dtype", D.dType |> Core.dTypeToJs |> string),
        ("sparse", sparse |> nullable(bool)),
      ])
    );
};

type model;

[@bs.module "@tensorflow/tfjs"] external make : Js.Json.t => model = "model";

let make = config => config |> ContainerConfig.encode |> make;

[@bs.module "@tensorflow/tfjs"]
external loadModel : string => Js.Promise.t(model) = "";

module Input = (R: Core.Rank, D: Core.DataType) => {
  module InputConfig = InputConfig(R, D);
  [@bs.module "@tensorflow/tfjs"]
  external input : Js.Json.t => SymbolicTensor.t = "";
  let input = inputConfig => inputConfig |> InputConfig.encode |> input;
};
