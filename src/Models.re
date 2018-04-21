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

type model;

[@bs.module "@tensorflow/tfjs"] external make : Js.Json.t => model = "model";

let make = config => config |> ContainerConfig.encode |> make;
