[@bs.deriving jsConverter]
type varianceScalingMode = [ | `fanIn | `fanOut | `fanAvg];

[@bs.deriving jsConverter]
type varianceScalingDistribution = [ | `normal | `uniform];

module Initializer = (R: Core.Rank, D: Core.DataType) => {
  type t;
  type ffi;
  external unsafeToFfi : 'a => ffi = "%identity";
  type initializerType =
    | Constant
    | GlorotNormal
    | GlorotUniform
    | HeNormal
    | Identity
    | LeCunNormal
    | Ones
    | Orthogonal
    | RandomNormal
    | RandomUniform
    | TruncatedNormal
    | VarianceScaling
    | Zeros
    | Initializer(t);
  let initializerTypeToJs = initializerType =>
    switch (initializerType) {
    | Constant => "constant" |> unsafeToFfi
    | GlorotNormal => "glorotNormal" |> unsafeToFfi
    | GlorotUniform => "glorotUniform" |> unsafeToFfi
    | HeNormal => "heNormal" |> unsafeToFfi
    | Identity => "identity" |> unsafeToFfi
    | LeCunNormal => "leCunNormal" |> unsafeToFfi
    | Ones => "ones" |> unsafeToFfi
    | Orthogonal => "orthogonal" |> unsafeToFfi
    | RandomNormal => "randomNormal" |> unsafeToFfi
    | RandomUniform => "randomUniform" |> unsafeToFfi
    | TruncatedNormal => "truncatedNormal" |> unsafeToFfi
    | VarianceScaling => "varianceScaling" |> unsafeToFfi
    | Zeros => "zeros" |> unsafeToFfi
    | Initializer(t) => t |> unsafeToFfi
    };
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external constant : D.t => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external glorotNormal : unit => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external glorotNormalWithOptions : {. "seed": float} => t = "glorotNormal";
  let glorotNormalWithOptions = seed =>
    {"seed": seed} |> glorotNormalWithOptions;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external glorotUniform : unit => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external glorotUniformWithOptions : {. "seed": float} => t =
    "glorotUniform";
  let glorotUniformWithOptions = seed =>
    {"seed": seed} |> glorotUniformWithOptions;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external heNormal : unit => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external heNormalWithOptions : {. "seed": float} => t = "heNormal";
  let heNormalWithOptions = seed => {"seed": seed} |> heNormalWithOptions;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external identity : unit => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external identityWithOptions : {. "gain": float} => t = "identity";
  let identityWithOptions = gain => {"gain": gain} |> identityWithOptions;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external leCunNormal : unit => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external leCunNormalWithOptions : {. "seed": float} => t = "leCunNormal";
  let leCunNormalWithOptions = seed =>
    {"seed": seed} |> leCunNormalWithOptions;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external ones : unit => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external orthogonal : unit => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external orthogonalWithOptions : {. "gain": float} => t = "orthogonal";
  let orthogonalWithOptions = gain => {"gain": gain} |> orthogonalWithOptions;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external randomNormal : unit => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external randomNormalWithOptions :
    {
      .
      "mean": float,
      "stddev": float,
      "seed": float,
    } =>
    t =
    "randomNormal";
  let randomNormalWithOptions = (mean, stddev, seed) =>
    {"mean": mean, "stddev": stddev, "seed": seed} |> randomNormalWithOptions;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external randomUniform : unit => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external randomUniformWithOptions :
    {
      .
      "minval": float,
      "maxval": float,
      "seed": float,
    } =>
    t =
    "randomUniform";
  let randomUniformWithOptions = (minval, maxval, seed) =>
    {"minval": minval, "maxval": maxval, "seed": seed}
    |> randomUniformWithOptions;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external truncatedNormal : unit => t = "";
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external truncatedNormalWithOptions :
    {
      .
      "mean": float,
      "stddev": float,
      "seed": float,
    } =>
    t =
    "truncatedNormal";
  let truncatedNormalWithOptions = (mean, stddev, seed) =>
    {"mean": mean, "stddev": stddev, "seed": seed}
    |> truncatedNormalWithOptions;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external varianceScaling :
    {
      .
      "scale": float,
      "mode": string,
      "distribution": string,
      "seed": Js.Undefined.t(float),
    } =>
    t =
    "";
  let varianceScaling = (scale, mode, distribution, ~seed=?, ()) =>
    {
      "scale": Js.Math.max_float(0.0, scale),
      "mode": mode |> varianceScalingModeToJs,
      "distribution": distribution |> varianceScalingDistributionToJs,
      "seed": seed |> Js.Undefined.fromOption,
    }
    |> varianceScaling;
  [@bs.module "@tensorflow/tfjs"] [@bs.scope "initializer"]
  external zeros : unit => t = "";
};
