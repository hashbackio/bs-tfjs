[@bs.deriving jsConverter]
type rank = [ | `R0 | `R1 | `R2 | `R3 | `R4];

module type Rank = {let rank: rank;};

module Rank0: Rank = {
  let rank = `R0;
};

module Rank1: Rank = {
  let rank = `R1;
};

module Rank2: Rank = {
  let rank = `R2;
};

module Rank3: Rank = {
  let rank = `R3;
};

module Rank4: Rank = {
  let rank = `R4;
};

module ShapeRank = {
  type shapeFromTfjs;
  type t =
    | ShapeRank0(array(int))
    | ShapeRank1(int)
    | ShapeRank2(int, int)
    | ShapeRank3(int, int, int)
    | ShapeRank4(int, int, int, int);
  external shapeFromTfjsToRank0 : shapeFromTfjs => array(int) = "%identity";
  let shapeFromTfjsToRank1 = shapeFromTfjs =>
    shapeFromTfjs |> shapeFromTfjsToRank0 |. Belt.Array.getUnsafe(0);
  let shapeFromTfjsToRank2 = shapeFromTfjs =>
    shapeFromTfjs
    |> shapeFromTfjsToRank0
    |> (
      shapeArray => (
        Belt.Array.getUnsafe(shapeArray, 0),
        Belt.Array.getUnsafe(shapeArray, 1),
      )
    );
  let shapeFromTfjsToRank3 = shapeFromTfjs =>
    shapeFromTfjs
    |> shapeFromTfjsToRank0
    |> (
      shapeArray => (
        Belt.Array.getUnsafe(shapeArray, 0),
        Belt.Array.getUnsafe(shapeArray, 1),
        Belt.Array.getUnsafe(shapeArray, 2),
      )
    );
  let shapeFromTfjsToRank4 = shapeFromTfjs =>
    shapeFromTfjs
    |> shapeFromTfjsToRank0
    |> (
      shapeArray => (
        Belt.Array.getUnsafe(shapeArray, 0),
        Belt.Array.getUnsafe(shapeArray, 1),
        Belt.Array.getUnsafe(shapeArray, 2),
        Belt.Array.getUnsafe(shapeArray, 3),
      )
    );
  let getShapeRank = (shapeFromTfjs, rank: rank) =>
    switch (rank) {
    | `R0 => ShapeRank0(shapeFromTfjs |> shapeFromTfjsToRank0)
    | `R1 => ShapeRank1(shapeFromTfjs |> shapeFromTfjsToRank1)
    | `R2 =>
      shapeFromTfjs
      |> shapeFromTfjsToRank2
      |> (((d1, d2)) => ShapeRank2(d1, d2))
    | `R3 =>
      shapeFromTfjs
      |> shapeFromTfjsToRank3
      |> (((d1, d2, d3)) => ShapeRank3(d1, d2, d3))
    | `R4 =>
      shapeFromTfjs
      |> shapeFromTfjsToRank4
      |> (((d1, d2, d3, d4)) => ShapeRank4(d1, d2, d3, d4))
    };
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
  type t = bool;
  type typedArray = Js.Typed_array.Uint32Array.t;
};

module rec Tensor:
  (R: Rank, D: DataType) =>
  {
    type t;
    type dataId;
    type typedArray = D.typedArray;
    type primitiveDataType = D.t;
    let make: typedArray => t;
    let number: t => int;
    let shape: t => ShapeRank.t;
    let size: t => int;
    let dtype: t => dType;
    let rankType: t => rank;
    let strides: t => array(int);
    let dataId: t => dataId;
    let flatten: t => Tensor(Rank0)(D).t;
    let asScalar: t => Tensor(Rank0)(D).t;
    let as1D: t => Tensor(Rank1)(D).t;
    let as2D: (t, int, int) => Tensor(Rank2)(D).t;
    let as3D: (t, int, int, int) => Tensor(Rank3)(D).t;
    let as4D: (t, int, int, int, int) => Tensor(Rank4)(D).t;
    let data: t => Js.Promise.t(D.typedArray);
    let dataSync: t => D.typedArray;
    let dispose: t => unit;
    let toFloat: t => Tensor(R)(FloatDataType).t;
    let toInt: t => Tensor(R)(IntDataType).t;
    let toBool: t => Tensor(R)(BoolDataType).t;
    let print: t => unit;
    let printVerbose: t => unit;
    let reshapeTo1D: (t, int) => Tensor(Rank1)(D).t;
    let reshapeTo2D: (t, (int, int)) => Tensor(Rank2)(D).t;
    let reshapeTo3D: (t, (int, int, int)) => Tensor(Rank3)(D).t;
    let reshapeTo4D: (t, (int, int, int, int)) => Tensor(Rank4)(D).t;
    let reshapeAs1D: (t, Tensor(Rank1)(D).t) => Tensor(Rank1)(D).t;
    let reshapeAs2D: (t, Tensor(Rank2)(D).t) => Tensor(Rank2)(D).t;
    let reshapeAs3D: (t, Tensor(Rank3)(D).t) => Tensor(Rank3)(D).t;
    let reshapeAs4D: (t, Tensor(Rank4)(D).t) => Tensor(Rank4)(D).t;
    let expandScalarDims: Tensor(Rank0)(D).t => Tensor(Rank1)(D).t;
    let expand1dDims: Tensor(Rank1)(D).t => Tensor(Rank2)(D).t;
    let expand2dDimsOnXAxis: Tensor(Rank2)(D).t => Tensor(Rank3)(D).t;
    let expand2dDimsOnYAxis: Tensor(Rank2)(D).t => Tensor(Rank3)(D).t;
    let expand3dDimsOnXAxis: Tensor(Rank3)(D).t => Tensor(Rank4)(D).t;
    let expand3dDimsOnYAxis: Tensor(Rank3)(D).t => Tensor(Rank4)(D).t;
    let expand3dDimsOnZAxis: Tensor(Rank3)(D).t => Tensor(Rank4)(D).t;
    let clone: t => t;
    let toString: t => string;
    let toStringVerbose: t => string;
    /* TODO:
        ------------------------------------------------------------------------------------------------------
        buffer () method source
        Returns a tf.TensorBuffer that holds the underlying data.

        Returns: tf.TensorBuffer

        ------------------------------------------------------------------------------------------------------
        squeeze (axis?) method source
        Returns a tf.Tensor with dimensions of size 1 removed from the shape. See tf.squeeze() for more details.

        Parameters:
        axis (number[]) A list of numbers. If specified, only squeezes the dimensions listed. The dimension
        index starts at 0. It is an error to squeeze a dimension that is not 1. Optional
        Returns: tf.Tensor
       */
  } =
  (R: Rank, D: DataType) => {
    type t;
    type dataId;
    type typedArray = D.typedArray;
    type primitiveDataType = D.t;
    [@bs.module "@tensorflow/tfjs"] external make : typedArray => t = "tensor";
    [@bs.send] external number : t => int = "";
    [@bs.send] external shape : t => ShapeRank.shapeFromTfjs = "";
    let shape = t => t |> shape |. ShapeRank.getShapeRank(R.rank);
    [@bs.send] external size : t => int = "";
    [@bs.send] external dtype : t => string = "";
    let dtype = t => t |> dtype |> dTypeFromJs |> Belt.Option.getExn;
    [@bs.send] external rankType : t => string = "";
    let rankType = t => t |> rankType |> rankFromJs |> Belt.Option.getExn;
    [@bs.send] external strides : t => array(int) = "";
    [@bs.send] external dataId : t => dataId = "";
    [@bs.send] external flatten : t => Tensor(Rank0)(D).t = "";
    [@bs.send] external asScalar : t => Tensor(Rank0)(D).t = "";
    [@bs.send] external as1D : t => Tensor(Rank1)(D).t = "";
    [@bs.send] external as2D : (t, int, int) => Tensor(Rank2)(D).t = "";
    [@bs.send] external as3D : (t, int, int, int) => Tensor(Rank3)(D).t = "";
    [@bs.send]
    external as4D : (t, int, int, int, int) => Tensor(Rank4)(D).t = "";
    [@bs.send] external data : t => Js.Promise.t(D.typedArray) = "";
    [@bs.send] external dataSync : t => D.typedArray = "";
    [@bs.send] external dispose : t => unit = "";
    [@bs.send] external toFloat : t => Tensor(R)(FloatDataType).t = "";
    [@bs.send] external toInt : t => Tensor(R)(IntDataType).t = "";
    [@bs.send] external toBool : t => Tensor(R)(BoolDataType).t = "";
    [@bs.send] external print : t => unit = "";
    [@bs.send]
    external printVerbose : (t, [@bs.as {json|true|json}] _) => unit = "print";
    [@bs.send]
    external reshapeAs1D : (t, array(int)) => Tensor(Rank1)(D).t = "reshape";
    let reshapeTo1D = (t, x) => reshapeAs1D(t, [|x|]);
    [@bs.send]
    external reshapeAs2D : (t, array(int)) => Tensor(Rank2)(D).t = "reshape";
    let reshapeTo2D = (t, (x, y)) => reshapeAs2D(t, [|x, y|]);
    [@bs.send]
    external reshapeAs3D : (t, array(int)) => Tensor(Rank3)(D).t = "reshape";
    let reshapeTo3D = (t, (x, y, z)) => reshapeAs3D(t, [|x, y, z|]);
    [@bs.send]
    external reshapeAs4D : (t, array(int)) => Tensor(Rank4)(D).t = "reshape";
    let reshapeTo4D = (tensor, (x, y, z, t)) =>
      reshapeAs4D(tensor, [|x, y, z, t|]);
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
    external expandScalarDims :
      (Tensor(Rank0)(D).t, [@bs.as 0] _) => Tensor(Rank1)(D).t =
      "expandDims";
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
    [@bs.send] external clone : t => t = "";
    [@bs.send] external toString : t => string = "";
    [@bs.send]
    external toStringVerbose : (t, [@bs.as {json|true|json}] _) => string =
      "toString";
  };

module FloatScalar = Tensor(Rank0, FloatDataType);

module IntScalar = Tensor(Rank0, IntDataType);

module BoolScalar = Tensor(Rank0, BoolDataType);

module FloatTensor1D = Tensor(Rank1, FloatDataType);

module IntTensor1D = Tensor(Rank1, IntDataType);

module BoolTensor1D = Tensor(Rank1, BoolDataType);

module FloatTensor2D = Tensor(Rank2, FloatDataType);

module IntTensor2D = Tensor(Rank2, IntDataType);

module BoolTensor2D = Tensor(Rank2, BoolDataType);

module FloatTensor3D = Tensor(Rank3, FloatDataType);

module IntTensor3D = Tensor(Rank3, IntDataType);

module BoolTensor3D = Tensor(Rank3, BoolDataType);

module FloatTensor4D = Tensor(Rank4, FloatDataType);

module IntTensor4D = Tensor(Rank4, IntDataType);

module BoolTensor4D = Tensor(Rank4, BoolDataType);
