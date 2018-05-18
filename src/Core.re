let init: unit => unit = [%raw
  {|
  function() {
    require('@tensorflow/tfjs-node');
    require('@tensorflow/tfjs').setBackend('tensorflow');
  }
|}
];

[@bs.deriving jsConverter]
type rank = [ | `R0 | `R1 | `R2 | `R3 | `R4];

module type Rank = {
  let rank: rank;
  type shape;
  type inputShape;
  type padding;
  type axis;
  let axisToJs: axis => int;
  let axisFromJs: int => option(axis);
  let axisToInclusiveNegRankExclusiveRank: axis => axis;
  let axisToNonNegativeRank: axis => axis;
  let axisToNegOneDefaultRank: axis => axis;
  let getShapeArray: shape => array(int);
  let getInputShapeArray: inputShape => array(int);
  let getPaddingArray: padding => array(array(int));
};

[@bs.deriving jsConverter]
type _rank0Axis =
  | [@bs.as 0] Default;

module Rank0:
  Rank with type shape = int with type inputShape = int with
    type padding = unit with
    type axis = _rank0Axis = {
  let rank = `R0;
  type shape = int;
  type inputShape = int;
  type padding = unit;
  type axis = _rank0Axis;
  let axisToJs = _rank0AxisToJs;
  let axisFromJs = _rank0AxisFromJs;
  let axisToInclusiveNegRankExclusiveRank = axis =>
    switch (axis) {
    | Default => Default
    };
  let axisToNonNegativeRank = axis =>
    switch (axis) {
    | Default => Default
    };
  /* TODO: I don't really know what to do here,  */
  let axisToNegOneDefaultRank = axis =>
    switch (axis) {
    | Default => Default
    };
  let getShapeArray = samples => [|samples|];
  let getInputShapeArray = _shape => [|1|];
  let getPaddingArray = () => [||];
};

[@bs.deriving jsConverter]
type _rank1Axis =
  | [@bs.as (-1)] ReversedX
  | [@bs.as 0] Default
  | [@bs.as 1] X;

module Rank1:
  Rank with type shape = (int, int) with type inputShape = int with
    type padding = (int, int) with
    type axis = _rank1Axis = {
  let rank = `R1;
  type shape = (int, int);
  type inputShape = int;
  type padding = (int, int);
  type axis = _rank1Axis;
  let axisToJs = _rank1AxisToJs;
  let axisFromJs = _rank1AxisFromJs;
  let axisToInclusiveNegRankExclusiveRank = axis =>
    switch (axis) {
    | X => Default
    | Default => Default
    | ReversedX => ReversedX
    };
  let axisToNonNegativeRank = axis =>
    switch (axis) {
    | X => X
    | Default => Default
    | ReversedX => X
    };
  let axisToNegOneDefaultRank = axis =>
    switch (axis) {
    | X => X
    | Default => ReversedX
    | ReversedX => ReversedX
    };
  let getShapeArray = ((samples, xs)) => [|samples, xs|];
  let getInputShapeArray = xs => [|xs|];
  let getPaddingArray = ((paddingBefore, paddingAfter)) => [|
    [|paddingBefore, paddingAfter|],
  |];
};

[@bs.deriving jsConverter]
type _rank2Axis =
  | [@bs.as (-2)] ReversedY
  | [@bs.as (-1)] ReversedX
  | [@bs.as 0] Default
  | [@bs.as 1] X
  | [@bs.as 2] Y;

module Rank2:
  Rank with type shape = (int, int, int) with type inputShape = (int, int) with
    type padding = ((int, int), (int, int)) with
    type axis = _rank2Axis = {
  let rank = `R2;
  type shape = (int, int, int);
  type inputShape = (int, int);
  type padding = ((int, int), (int, int));
  type axis = _rank2Axis;
  let axisToJs = _rank2AxisToJs;
  let axisFromJs = _rank2AxisFromJs;
  let axisToInclusiveNegRankExclusiveRank = axis =>
    switch (axis) {
    | Y => X
    | X => X
    | Default => Default
    | ReversedX => ReversedX
    | ReversedY => ReversedY
    };
  let axisToNonNegativeRank = axis =>
    switch (axis) {
    | Y => Y
    | X => X
    | Default => Default
    | ReversedX => X
    | ReversedY => Y
    };
  let axisToNegOneDefaultRank = axis =>
    switch (axis) {
    | Y => Y
    | X => X
    | Default => ReversedX
    | ReversedX => ReversedX
    | ReversedY => ReversedX
    };
  let getShapeArray = ((samples, xs, ys)) => [|samples, xs, ys|];
  let getInputShapeArray = ((xs, ys)) => [|xs, ys|];
  let getPaddingArray =
      (((xPaddingBefore, xPaddingAfter), (yPaddingBefore, yPaddingAfter))) => [|
    [|xPaddingBefore, xPaddingAfter|],
    [|yPaddingBefore, yPaddingAfter|],
  |];
};

[@bs.deriving jsConverter]
type _rank3Axis =
  | [@bs.as (-3)] ReversedZ
  | [@bs.as (-2)] ReversedY
  | [@bs.as (-1)] ReversedX
  | [@bs.as 0] Default
  | [@bs.as 1] X
  | [@bs.as 2] Y
  | [@bs.as 3] Z;

module Rank3:
  Rank with type shape = (int, int, int, int) with
    type inputShape = (int, int, int) with
    type padding = ((int, int), (int, int), (int, int)) with
    type axis = _rank3Axis = {
  let rank = `R3;
  type shape = (int, int, int, int);
  type inputShape = (int, int, int);
  type padding = ((int, int), (int, int), (int, int));
  type axis = _rank3Axis;
  let axisToJs = _rank3AxisToJs;
  let axisFromJs = _rank3AxisFromJs;
  let axisToInclusiveNegRankExclusiveRank = axis =>
    switch (axis) {
    | Z => Y
    | Y => Y
    | X => X
    | Default => Default
    | ReversedX => ReversedX
    | ReversedY => ReversedY
    | ReversedZ => ReversedZ
    };
  let axisToNonNegativeRank = axis =>
    switch (axis) {
    | Z => Z
    | Y => Y
    | X => X
    | Default => Default
    | ReversedX => X
    | ReversedY => Y
    | ReversedZ => Z
    };
  let axisToNegOneDefaultRank = axis =>
    switch (axis) {
    | Z => Z
    | Y => Y
    | X => X
    | Default => ReversedX
    | ReversedX => ReversedX
    | ReversedY => ReversedX
    | ReversedZ => ReversedX
    };
  let getInputShapeArray = ((xs, ys, zs)) => [|xs, ys, zs|];
  let getShapeArray = ((samples, xs, ys, zs)) => [|samples, xs, ys, zs|];
  let getPaddingArray =
      (
        (
          (xPaddingBefore, xPaddingAfter),
          (yPaddingBefore, yPaddingAfter),
          (zPaddingBefore, zPaddingAfter),
        ),
      ) => [|
    [|xPaddingBefore, xPaddingAfter|],
    [|yPaddingBefore, yPaddingAfter|],
    [|zPaddingBefore, zPaddingAfter|],
  |];
};

[@bs.deriving jsConverter]
type _rank4Axis =
  | [@bs.as (-4)] ReversedT
  | [@bs.as (-3)] ReversedZ
  | [@bs.as (-2)] ReversedY
  | [@bs.as (-1)] ReversedX
  | [@bs.as 0] Default
  | [@bs.as 1] X
  | [@bs.as 2] Y
  | [@bs.as 3] Z
  | [@bs.as 4] T;

module Rank4:
  Rank with type shape = (int, int, int, int, int) with
    type inputShape = (int, int, int, int) with
    type padding = ((int, int), (int, int), (int, int), (int, int)) with
    type axis = _rank4Axis = {
  let rank = `R4;
  type inputShape = (int, int, int, int);
  type shape = (int, int, int, int, int);
  type padding = ((int, int), (int, int), (int, int), (int, int));
  type axis = _rank4Axis;
  let axisToJs = _rank4AxisToJs;
  let axisFromJs = _rank4AxisFromJs;
  let axisToInclusiveNegRankExclusiveRank = axis =>
    switch (axis) {
    | T => Z
    | Z => Z
    | Y => Y
    | X => X
    | Default => Default
    | ReversedX => ReversedX
    | ReversedY => ReversedY
    | ReversedZ => ReversedZ
    | ReversedT => ReversedT
    };
  let axisToNonNegativeRank = axis =>
    switch (axis) {
    | T => T
    | Z => Z
    | Y => Y
    | X => X
    | Default => Default
    | ReversedX => X
    | ReversedY => Y
    | ReversedZ => Z
    | ReversedT => T
    };
  let axisToNegOneDefaultRank = axis =>
    switch (axis) {
    | T => T
    | Z => Z
    | Y => Y
    | X => X
    | Default => ReversedX
    | ReversedX => ReversedX
    | ReversedY => ReversedX
    | ReversedZ => ReversedX
    | ReversedT => ReversedX
    };
  let getInputShapeArray = ((xs, ys, zs, ts)) => [|xs, ys, zs, ts|];
  let getShapeArray = ((samples, xs, ys, zs, ts)) => [|
    samples,
    xs,
    ys,
    zs,
    ts,
  |];
  let getPaddingArray =
      (
        (
          (xPaddingBefore, xPaddingAfter),
          (yPaddingBefore, yPaddingAfter),
          (zPaddingBefore, zPaddingAfter),
          (tPaddingBefore, tPaddingAfter),
        ),
      ) => [|
    [|xPaddingBefore, xPaddingAfter|],
    [|yPaddingBefore, yPaddingAfter|],
    [|zPaddingBefore, zPaddingAfter|],
    [|tPaddingBefore, tPaddingAfter|],
  |];
};

[@bs.deriving jsConverter]
type dType = [ | `float32 | `int32 | `bool];

module type DataType = {let dType: dType; type t; type typedArray;};

module FloatDataType: DataType = {
  let dType = `float32;
  type t = float;
  type typedArray = Js.Typed_array.Float32Array.t;
};

module IntDataType: DataType = {
  let dType = `int32;
  type t = int;
  type typedArray = Js.Typed_array.Int32Array.t;
};

module BoolDataType: DataType = {
  let dType = `bool;
  type t = int;
  type typedArray = Js.Typed_array.Uint32Array.t;
};

module Variable = (R: Rank, D: DataType) => {
  type t;
  type dataId;
  type typedArray = D.typedArray;
  type primitiveDataType = D.t;
};

module rec Tensor:
  (R: Rank, D: DataType) =>
  {
    type t;
    type dataId;
    type typedArray = D.typedArray;
    type primitiveDataType = D.t;
    module Create: {
      let tensor: typedArray => t;
      let clone: t => t;
      let fill: (R.shape, D.t) => t;
      let linspace: (float, float, float) => Tensor(Rank1)(FloatDataType).t;
      let oneHot:
        (Tensor(Rank1)(IntDataType).t, int) => Tensor(Rank2)(FloatDataType).t;
      let oneHotWithFloatOptions:
        (Tensor(Rank1)(IntDataType).t, int, float, float) =>
        Tensor(Rank2)(FloatDataType).t;
      let oneHotWithIntOptions:
        (Tensor(Rank1)(IntDataType).t, int, int, int) =>
        Tensor(Rank2)(FloatDataType).t;
      let ones: R.shape => t;
      let onesLike: t => t;
      let randomNormal: R.shape => Tensor(R)(FloatDataType).t;
      let randomNormalWithOptions:
        (R.shape, float, float, float) => Tensor(R)(FloatDataType).t;
      let randomUniform: R.shape => Tensor(R)(FloatDataType).t;
      let randomUniformWithOptions: (R.shape, D.t, D.t) => Tensor(R)(D).t;
      let range: (int, int, int) => Tensor(Rank1)(IntDataType).t;
      let rangeFloats: (float, float, float) => Tensor(Rank1)(FloatDataType).t;
      let truncatedNormal: R.shape => Tensor(R)(D).t;
      let truncatedNormalWithOptions:
        (R.shape, float, float, float) => Tensor(R)(D).t;
      let variable: t => Variable(R)(D).t;
      let variableWithOptions: (t, bool, string) => Variable(R)(D).t;
      let zeros: R.shape => Tensor(R)(D).t;
      let zerosLike: t => t;
      /* TODO:
         https://js.tensorflow.org/api/0.9.0/#fromPixels
         */
    };
    module Transform: {
      let asScalar: t => Tensor(Rank0)(D).t;
      let expand1dDims: Tensor(Rank1)(D).t => Tensor(Rank2)(D).t;
      let expand2dDimsOnXAxis: Tensor(Rank2)(D).t => Tensor(Rank3)(D).t;
      let expand2dDimsOnYAxis: Tensor(Rank2)(D).t => Tensor(Rank3)(D).t;
      let expand3dDimsOnXAxis: Tensor(Rank3)(D).t => Tensor(Rank4)(D).t;
      let expand3dDimsOnYAxis: Tensor(Rank3)(D).t => Tensor(Rank4)(D).t;
      let expand3dDimsOnZAxis: Tensor(Rank3)(D).t => Tensor(Rank4)(D).t;
      let expandScalarDims: Tensor(Rank0)(D).t => Tensor(Rank1)(D).t;
      let flatten: t => Tensor(Rank1)(D).t;
      let pad: (t, R.padding) => t;
      let padWithOptions: (t, R.padding, D.t) => t;
      let reshapeAs1D: (t, Tensor(Rank1)(D).t) => Tensor(Rank1)(D).t;
      let reshapeAs2D: (t, Tensor(Rank2)(D).t) => Tensor(Rank2)(D).t;
      let reshapeAs3D: (t, Tensor(Rank3)(D).t) => Tensor(Rank3)(D).t;
      let reshapeAs4D: (t, Tensor(Rank4)(D).t) => Tensor(Rank4)(D).t;
      let reshapeTo1D: (t, Rank1.shape) => Tensor(Rank1)(D).t;
      let reshapeTo2D: (t, Rank2.shape) => Tensor(Rank2)(D).t;
      let reshapeTo3D: (t, Rank3.shape) => Tensor(Rank3)(D).t;
      let reshapeTo4D: (t, Rank4.shape) => Tensor(Rank4)(D).t;
      let toFloat: t => Tensor(R)(FloatDataType).t;
      let toInt: t => Tensor(R)(IntDataType).t;
      let toBool: t => Tensor(R)(BoolDataType).t;
      /* TODO:
         https://js.tensorflow.org/api/0.9.0/#buffer
         https://js.tensorflow.org/api/0.9.0/#squeeze
         */
    };
    module Operation: {
      let concat: array(t) => t;
      let concatAlongAxis: (array(t), R.axis) => t;
      let gather: (t, Tensor(Rank1)(IntDataType).t) => t;
      let gatherAlongAxis: (t, Tensor(Rank1)(IntDataType).t, R.axis) => t;
      let reverse: t => t;
      let reverseAlongAxis: (t, R.axis) => t;
      let reverseAlongManyAxis: (t, list(R.axis)) => t;
      let slice: (t, R.shape, R.shape) => t;
      let split: (t, int) => array(t);
      let splitAlongAxis: (t, int, R.axis) => array(t);
      let splitMany: (t, array(int)) => array(t);
      let splitManyAlongAxis: (t, array(int), R.axis) => array(t);
      let tile: (t, R.shape) => t;
      /* TODO:
         https://js.tensorflow.org/api/0.9.0/#stack
         */
    };
    let data: t => Js.Promise.t(D.typedArray);
    let dataSync: t => D.typedArray;
    let dispose: t => unit;
    let print: t => unit;
    let printVerbose: t => unit;
    let toString: t => string;
    let toStringVerbose: t => string;
    /* TODO:
       https://js.tensorflow.org/api/0.9.0/#tf.Tensor.buffer
       */
  } =
  (R: Rank, D: DataType) => {
    type t;
    type dataId;
    type typedArray = D.typedArray;
    type primitiveDataType = D.t;
    module Create = {
      [@bs.module "@tensorflow/tfjs"] external tensor : typedArray => t = "";
      [@bs.module "@tensorflow/tfjs"] external clone : t => t = "";
      [@bs.module "@tensorflow/tfjs"]
      external fill : (array(int), D.t, string) => t = "";
      let fill = (shape, value) =>
        fill(shape |> R.getShapeArray, value, D.dType |> dTypeToJs);
      [@bs.module "@tensorflow/tfjs"]
      external linspace :
        (float, float, float) => Tensor(Rank1)(FloatDataType).t =
        "";
      [@bs.module "@tensorflow/tfjs"]
      external oneHot :
        (Tensor(Rank1)(IntDataType).t, int) => Tensor(Rank2)(FloatDataType).t =
        "";
      [@bs.module "@tensorflow/tfjs"]
      external oneHotWithFloatOptions :
        (Tensor(Rank1)(IntDataType).t, int, float, float) =>
        Tensor(Rank2)(FloatDataType).t =
        "oneHot";
      [@bs.module "@tensorflow/tfjs"]
      external oneHotWithIntOptions :
        (Tensor(Rank1)(IntDataType).t, int, int, int) =>
        Tensor(Rank2)(FloatDataType).t =
        "oneHot";
      [@bs.module "@tensorflow/tfjs"]
      external ones : (array(int), string) => t = "";
      let ones = shape =>
        ones(shape |> R.getShapeArray, D.dType |> dTypeToJs);
      [@bs.module "@tensorflow/tfjs"] external onesLike : t => t = "";
      [@bs.module "@tensorflow/tfjs"]
      external randomNormal : array(int) => Tensor(R)(FloatDataType).t = "";
      let randomNormal = shape => shape |> R.getShapeArray |> randomNormal;
      [@bs.module "@tensorflow/tfjs"]
      external randomNormalWithOptions :
        (array(int), float, float, string, float) =>
        Tensor(R)(FloatDataType).t =
        "randomNormal";
      let randomNormalWithOptions = (shape, mean, stdDev, seed) =>
        randomNormalWithOptions(
          shape |> R.getShapeArray,
          mean,
          stdDev,
          `float32 |> dTypeToJs,
          seed,
        );
      [@bs.module "@tensorflow/tfjs"]
      external randomUniform : array(int) => Tensor(R)(FloatDataType).t = "";
      let randomUniform = shape => shape |> R.getShapeArray |> randomUniform;
      [@bs.module "@tensorflow/tfjs"]
      external randomUniformWithOptions :
        (array(int), D.t, D.t, string) => Tensor(R)(D).t =
        "randomUniform";
      let randomUniformWithOptions = (shape, minval, maxval) =>
        randomUniformWithOptions(
          shape |> R.getShapeArray,
          minval,
          maxval,
          D.dType |> dTypeToJs,
        );
      [@bs.module "@tensorflow/tfjs"]
      external range : (int, int, int, string) => Tensor(Rank1)(IntDataType).t =
        "";
      let range = (start, stop, step) =>
        range(start, stop, step, `int32 |> dTypeToJs);
      [@bs.module "@tensorflow/tfjs"]
      external rangeFloats :
        (float, float, float, string) => Tensor(Rank1)(FloatDataType).t =
        "range";
      let rangeFloats = (start, stop, step) =>
        rangeFloats(start, stop, step, `float32 |> dTypeToJs);
      [@bs.module "@tensorflow/tfjs"]
      external truncatedNormal :
        (array(int), Js.Undefined.t(D.t), Js.Undefined.t(D.t), string) =>
        Tensor(R)(D).t =
        "";
      let truncatedNormal = shape =>
        truncatedNormal(
          shape |> R.getShapeArray,
          Js.Undefined.empty,
          Js.Undefined.empty,
          D.dType |> dTypeToJs,
        );
      [@bs.module "@tensorflow/tfjs"]
      external truncatedNormalWithOptions :
        (array(int), float, float, string, float) => Tensor(R)(D).t =
        "truncatedNormal";
      let truncatedNormalWithOptions = (shape, mean, stdDev, seed) =>
        truncatedNormalWithOptions(
          shape |> R.getShapeArray,
          mean,
          stdDev,
          D.dType |> dTypeToJs,
          seed,
        );
      [@bs.module "@tensorflow/tfjs"]
      external variable : t => Variable(R)(D).t = "";
      [@bs.module "@tensorflow/tfjs"]
      external variableWithOptions : (t, bool, string) => Variable(R)(D).t =
        "";
      [@bs.module "@tensorflow/tfjs"]
      external zeros : (array(int), string) => Tensor(R)(D).t = "";
      let zeros = shape =>
        zeros(shape |> R.getShapeArray, D.dType |> dTypeToJs);
      [@bs.module "@tensorflow/tfjs"] external zerosLike : t => t = "";
    };
    module Transform = {
      [@bs.send] external asScalar : t => Tensor(Rank0)(D).t = "";
      [@bs.send]
      external expand1dDims :
        (Tensor(Rank1)(D).t, [@bs.as 1] _) => Tensor(Rank2)(D).t =
        "expandDims";
      [@bs.send]
      external expand2dDimsOnXAxis :
        (Tensor(Rank2)(D).t, [@bs.as 1] _) => Tensor(Rank3)(D).t =
        "expandDims";
      [@bs.send]
      external expand2dDimsOnYAxis :
        (Tensor(Rank2)(D).t, [@bs.as 2] _) => Tensor(Rank3)(D).t =
        "expandDims";
      [@bs.send]
      external expand3dDimsOnXAxis :
        (Tensor(Rank3)(D).t, [@bs.as 1] _) => Tensor(Rank4)(D).t =
        "expandDims";
      [@bs.send]
      external expand3dDimsOnYAxis :
        (Tensor(Rank3)(D).t, [@bs.as 2] _) => Tensor(Rank4)(D).t =
        "expandDims";
      [@bs.send]
      external expand3dDimsOnZAxis :
        (Tensor(Rank3)(D).t, [@bs.as 3] _) => Tensor(Rank4)(D).t =
        "expandDims";
      [@bs.send]
      external expandScalarDims :
        (Tensor(Rank0)(D).t, [@bs.as 0] _) => Tensor(Rank1)(D).t =
        "expandDims";
      [@bs.send] external flatten : t => Tensor(Rank1)(D).t = "";
      [@bs.module "@tensorflow/tfjs"]
      external pad : (t, array(array(int))) => t = "";
      let pad = (t, padding) => padding |> R.getPaddingArray |> pad(t);
      [@bs.module "@tensorflow/tfjs"]
      external padWithOptions : (t, array(array(int)), D.t) => t = "pad";
      let padWithOptions = (t, padding, valueToPadWith) =>
        padding |> R.getPaddingArray |> padWithOptions(t, _, valueToPadWith);
      [@bs.send]
      external reshapeAs1D : (t, Tensor(Rank1)(D).t) => Tensor(Rank1)(D).t =
        "reshapeAs";
      [@bs.send]
      external reshapeAs2D : (t, Tensor(Rank2)(D).t) => Tensor(Rank2)(D).t =
        "reshapeAs";
      [@bs.send]
      external reshapeAs3D : (t, Tensor(Rank3)(D).t) => Tensor(Rank3)(D).t =
        "reshapeAs";
      [@bs.send]
      external reshapeAs4D : (t, Tensor(Rank4)(D).t) => Tensor(Rank4)(D).t =
        "reshapeAs";
      [@bs.send]
      external reshapeTo1D : (t, array(int)) => Tensor(Rank1)(D).t =
        "reshape";
      let reshapeTo1D = (t, shape) =>
        shape |> Rank1.getShapeArray |> reshapeTo1D(t);
      [@bs.send]
      external reshapeTo2D : (t, array(int)) => Tensor(Rank2)(D).t =
        "reshape";
      let reshapeTo2D = (t, shape) =>
        shape |> Rank2.getShapeArray |> reshapeTo2D(t);
      [@bs.send]
      external reshapeTo3D : (t, array(int)) => Tensor(Rank3)(D).t =
        "reshape";
      let reshapeTo3D = (t, shape) =>
        shape |> Rank3.getShapeArray |> reshapeTo3D(t);
      [@bs.send]
      external reshapeTo4D : (t, array(int)) => Tensor(Rank4)(D).t =
        "reshape";
      let reshapeTo4D = (t, shape) =>
        shape |> Rank4.getShapeArray |> reshapeTo4D(t);
      [@bs.send] external toFloat : t => Tensor(R)(FloatDataType).t = "";
      [@bs.send] external toInt : t => Tensor(R)(IntDataType).t = "";
      [@bs.send] external toBool : t => Tensor(R)(BoolDataType).t = "";
    };
    module Operation = {
      [@bs.module "@tensorflow/tfjs"] external concat : array(t) => t = "";
      [@bs.module "@tensorflow/tfjs"]
      external concatAlongAxis : (array(t), int) => t = "concat";
      let concatAlongAxis = (ts, axis) =>
        axis |> R.axisToJs |> concatAlongAxis(ts);
      [@bs.module "@tensorflow/tfjs"]
      external gather : (t, Tensor(Rank1)(IntDataType).t) => t = "";
      [@bs.module "@tensorflow/tfjs"]
      external gatherAlongAxis : (t, Tensor(Rank1)(IntDataType).t, int) => t =
        "gather";
      let gatherAlongAxis = (t, indices, axis) =>
        axis |> R.axisToJs |> gatherAlongAxis(t, indices);
      [@bs.module "@tensorflow/tfjs"] external reverse : t => t = "";
      [@bs.module "@tensorflow/tfjs"]
      external reverseAlongAxis : (t, int) => t = "reverse";
      let reverseAlongAxis = (t, axis) =>
        axis
        |> R.axisToInclusiveNegRankExclusiveRank
        |> R.axisToJs
        |> reverseAlongAxis(t);
      [@bs.module "@tensorflow/tfjs"]
      external reverseAlongManyAxis : (t, array(int)) => t = "reverse";
      let reverseAlongManyAxis = (t, manyAxis) =>
        manyAxis
        |. Belt.List.map(R.axisToInclusiveNegRankExclusiveRank)
        |. Belt.List.map(R.axisToJs)
        |> Belt.List.toArray
        |> reverseAlongManyAxis(t);
      [@bs.module "@tensorflow/tfjs"]
      external slice : (t, array(int), array(int)) => t = "";
      let slice = (t, start, size) =>
        slice(t, start |> R.getShapeArray, size |> R.getShapeArray);
      [@bs.module "@tensorflow/tfjs"]
      external split : (t, int) => array(t) = "";
      [@bs.module "@tensorflow/tfjs"]
      external splitAlongAxis : (t, int, int) => array(t) = "split";
      let splitAlongAxis = (t, numOfSplits, axis) =>
        axis
        |> R.axisToNonNegativeRank
        |> R.axisToJs
        |> splitAlongAxis(t, numOfSplits);
      [@bs.module "@tensorflow/tfjs"]
      external splitMany : (t, array(int)) => array(t) = "split";
      [@bs.module "@tensorflow/tfjs"]
      external splitManyAlongAxis : (t, array(int), int) => array(t) =
        "split";
      let splitManyAlongAxis = (t, sizeOfSplits, axis) =>
        axis
        |> R.axisToNonNegativeRank
        |> R.axisToJs
        |> splitManyAlongAxis(t, sizeOfSplits);
      [@bs.module "@tensorflow/tfjs"]
      external tile : (t, array(int)) => t = "";
      let tile = (t, shape) => shape |> R.getShapeArray |> tile(t);
    };
    [@bs.send] external data : t => Js.Promise.t(D.typedArray) = "";
    [@bs.send] external dataSync : t => D.typedArray = "";
    [@bs.send] external dispose : t => unit = "";
    [@bs.send] external print : t => unit = "";
    [@bs.send]
    external printVerbose : (t, [@bs.as {json|true|json}] _) => unit = "print";
    [@bs.send] external toString : t => string = "";
    [@bs.send]
    external toStringVerbose : (t, [@bs.as {json|true|json}] _) => string =
      "toString";
  };

module Scalar = Tensor(Rank0);

module FloatScalar = Scalar(FloatDataType);

module IntScalar = Scalar(IntDataType);

module BoolScalar = Scalar(BoolDataType);

module Tensor1D = Tensor(Rank1);

module FloatTensor1D = Tensor1D(FloatDataType);

module IntTensor1D = Tensor1D(IntDataType);

module BoolTensor1D = Tensor1D(BoolDataType);

module Tensor2D = Tensor(Rank2);

module FloatTensor2D = Tensor2D(FloatDataType);

module IntTensor2D = Tensor2D(IntDataType);

module BoolTensor2D = Tensor2D(BoolDataType);

module Tensor3D = Tensor(Rank3);

module FloatTensor3D = Tensor3D(FloatDataType);

module IntTensor3D = Tensor3D(IntDataType);

module BoolTensor3D = Tensor3D(BoolDataType);

module Tensor4D = Tensor(Rank4);

module FloatTensor4D = Tensor4D(FloatDataType);

module IntTensor4D = Tensor4D(IntDataType);

module BoolTensor4D = Tensor4D(BoolDataType);
