module FFI = {
  type t;
  external unsafeCastToFFI : 'a => t = "%identity";
};

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

module TypedArray = {
  type maybeT;
  type t =
    | Float32(Js.Typed_array.Float32Array.t)
    | Int32(Js.Typed_array.Int32Array.t)
    | Bool(Js.Typed_array.Uint8Array.t);
  external castToMaybeT : 'a => maybeT = "%identity";
  let isFloat32Array: 'a => bool = [%bs.raw
    {|
      function(a) {
        return a instanceof Float32Array;
      }
    |}
  ];
  let isFloat32Array = a => a |> isFloat32Array;
  external unsafeCastToFloat32Array : 'a => Js.Typed_array.Float32Array.t =
    "%identity";
  let isInt32Array: 'a => bool = [%bs.raw
    {|
      function(a) {
        return a instanceof Int32Array;
      }
    |}
  ];
  let isInt32Array = a => a |> isInt32Array;
  external unsafeCastToInt32Array : 'a => Js.Typed_array.Int32Array.t =
    "%identity";
  let isUint8Array: 'a => bool = [%bs.raw
    {|
      function(a) {
        return a instanceof Uint8Array;
      }
    |}
  ];
  let isUint8Array = a => a |> isUint8Array;
  external unsafeCastToUint8Array : 'a => Js.Typed_array.Uint8Array.t =
    "%identity";
  let cast = a =>
    switch (a |> castToMaybeT) {
    | a when a |> isFloat32Array =>
      Some(Float32(a |> unsafeCastToFloat32Array))
    | a when a |> isInt32Array => Some(Int32(a |> unsafeCastToInt32Array))
    | a when a |> isUint8Array => Some(Bool(a |> unsafeCastToUint8Array))
    | _ => None
    };
  let sendToTfjs = t =>
    switch (t) {
    | Float32(f) => (FFI.unsafeCastToFFI(f), `float32)
    | Int32(i) => (FFI.unsafeCastToFFI(i), `int32)
    | Bool(b) => (FFI.unsafeCastToFFI(b), `bool)
    };
};

module TensorLikeScalar = {
  type t =
    | Float(float)
    | Int(int)
    | Bool(bool);
  let sendToTfjs = t =>
    switch (t) {
    | Float(f) => (FFI.unsafeCastToFFI(f), `float32)
    | Int(i) => (FFI.unsafeCastToFFI(i), `int32)
    | Bool(b) => (FFI.unsafeCastToFFI(b), `bool)
    };
};

module TensorLike1D = {
  type t =
    | Typed(TypedArray.t)
    | Int(array(int))
    | Float(array(float))
    | Bool(array(bool));
  let sendToTfjs = t =>
    switch (t) {
    | Typed(ta) => ta |> TypedArray.sendToTfjs
    | Float(f) => (FFI.unsafeCastToFFI(f), `float32)
    | Int(i) => (FFI.unsafeCastToFFI(i), `int32)
    | Bool(b) => (FFI.unsafeCastToFFI(b), `bool)
    };
};

module TensorLike2D = {
  type t =
    | Typed(TypedArray.t)
    | FlatInt(array(int))
    | FlatFloat(array(float))
    | FlatBool(array(bool))
    | Int(array(array(int)))
    | Float(array(array(float)))
    | Bool(array(array(bool)));
  let sendToTfjs = t =>
    switch (t) {
    | Typed(t) => t |> TypedArray.sendToTfjs
    | FlatFloat(f) => (FFI.unsafeCastToFFI(f), `float32)
    | FlatInt(i) => (FFI.unsafeCastToFFI(i), `int32)
    | FlatBool(b) => (FFI.unsafeCastToFFI(b), `bool)
    | Float(f) => (FFI.unsafeCastToFFI(f), `float32)
    | Int(i) => (FFI.unsafeCastToFFI(i), `int32)
    | Bool(b) => (FFI.unsafeCastToFFI(b), `bool)
    };
};

module TensorLike3D = {
  type t =
    | Typed(TypedArray.t)
    | FlatInt(array(int))
    | FlatFloat(array(float))
    | FlatBool(array(bool))
    | Int(array(array(array(int))))
    | Float(array(array(array(float))))
    | Bool(array(array(array(bool))));
  let sendToTfjs = t =>
    switch (t) {
    | Typed(a) => a |> TypedArray.sendToTfjs
    | FlatFloat(f) => (FFI.unsafeCastToFFI(f), `float32)
    | FlatInt(i) => (FFI.unsafeCastToFFI(i), `int32)
    | FlatBool(b) => (FFI.unsafeCastToFFI(b), `bool)
    | Float(f) => (FFI.unsafeCastToFFI(f), `float32)
    | Int(i) => (FFI.unsafeCastToFFI(i), `int32)
    | Bool(b) => (FFI.unsafeCastToFFI(b), `bool)
    };
};

module TensorLike4D = {
  type t =
    | Typed(TypedArray.t)
    | FlatInt(array(int))
    | FlatFloat(array(float))
    | FlatBool(array(bool))
    | Int(array(array(array(array(int)))))
    | Float(array(array(array(array(float)))))
    | Bool(array(array(array(array(bool)))));
  let sendToTfjs = t =>
    switch (t) {
    | Typed(a) => a |> TypedArray.sendToTfjs
    | FlatFloat(f) => (FFI.unsafeCastToFFI(f), `float32)
    | FlatInt(i) => (FFI.unsafeCastToFFI(i), `int32)
    | FlatBool(b) => (FFI.unsafeCastToFFI(b), `bool)
    | Float(f) => (FFI.unsafeCastToFFI(f), `float32)
    | Int(i) => (FFI.unsafeCastToFFI(i), `int32)
    | Bool(b) => (FFI.unsafeCastToFFI(b), `bool)
    };
};

type flatVector =
  | Float(array(float))
  | Int(array(int))
  | Bool(array(bool))
  | Typed(TypedArray.t);

module rec Tensor:
  (R: Rank) =>
  {
    type t;
    type dataId;
    let number: t => int;
    let shape: t => ShapeRank.t;
    let size: t => int;
    let dtype: t => dType;
    let rankType: t => rank;
    let strides: t => array(int);
    let dataId: t => dataId;
    let flatten: t => Tensor(Rank0).t;
    let asScalar: t => Tensor(Rank0).t;
    let as1D: t => Tensor(Rank1).t;
    let as2D: (t, int, int) => Tensor(Rank2).t;
    let as3D: (t, int, int, int) => Tensor(Rank3).t;
    let as4D: (t, int, int, int, int) => Tensor(Rank4).t;
    let asType: (t, dType) => t;
    let data: t => Js.Promise.t(TypedArray.t);
    let dataSync: t => TypedArray.t;
    let dispose: t => unit;
    let toFloat: t => t;
    let toInt: t => t;
    let toBool: t => t;
    let print: t => unit;
    let printVerbose: t => unit;
    let reshapeTo1D: (t, int) => Tensor(Rank1).t;
    let reshapeTo2D: (t, (int, int)) => Tensor(Rank2).t;
    let reshapeTo3D: (t, (int, int, int)) => Tensor(Rank3).t;
    let reshapeTo4D: (t, (int, int, int, int)) => Tensor(Rank4).t;
    let reshapeAs1D: (t, Tensor(Rank1).t) => Tensor(Rank1).t;
    let reshapeAs2D: (t, Tensor(Rank2).t) => Tensor(Rank2).t;
    let reshapeAs3D: (t, Tensor(Rank3).t) => Tensor(Rank3).t;
    let reshapeAs4D: (t, Tensor(Rank4).t) => Tensor(Rank4).t;
    let expandScalarDims: Tensor(Rank0).t => Tensor(Rank1).t;
    let expand1dDims: Tensor(Rank1).t => Tensor(Rank2).t;
    let expand2dDimsOnXAxis: Tensor(Rank2).t => Tensor(Rank3).t;
    let expand2dDimsOnYAxis: Tensor(Rank2).t => Tensor(Rank3).t;
    let expand3dDimsOnXAxis: Tensor(Rank3).t => Tensor(Rank4).t;
    let expand3dDimsOnYAxis: Tensor(Rank3).t => Tensor(Rank4).t;
    let expand3dDimsOnZAxis: Tensor(Rank3).t => Tensor(Rank4).t;
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
  (R: Rank) => {
    type t;
    type dataId;
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
    [@bs.send] external flatten : t => Tensor(Rank0).t = "";
    [@bs.send] external asScalar : t => Tensor(Rank0).t = "";
    [@bs.send] external as1D : t => Tensor(Rank1).t = "";
    [@bs.send] external as2D : (t, int, int) => Tensor(Rank2).t = "";
    [@bs.send] external as3D : (t, int, int, int) => Tensor(Rank3).t = "";
    [@bs.send] external as4D : (t, int, int, int, int) => Tensor(Rank4).t = "";
    [@bs.send] external asType : (t, dType) => t = "";
    [@bs.send] external data : t => Js.Promise.t(TypedArray.maybeT) = "";
    let data = t =>
      t
      |> data
      |> Js.Promise.(
           then_(maybeT =>
             maybeT |> TypedArray.cast |> Belt.Option.getExn |> resolve
           )
         );
    [@bs.send] external dataSync : t => TypedArray.maybeT = "";
    let dataSync = t => t |> dataSync |> TypedArray.cast |> Belt.Option.getExn;
    [@bs.send] external dispose : t => unit = "";
    [@bs.send] external toFloat : t => t = "";
    [@bs.send] external toInt : t => t = "";
    [@bs.send] external toBool : t => t = "";
    [@bs.send] external print : t => unit = "";
    [@bs.send]
    external printVerbose : (t, [@bs.as {json|true|json}] _) => unit = "print";
    [@bs.send]
    external reshapeAs1D : (t, array(int)) => Tensor(Rank1).t = "reshape";
    let reshapeTo1D = (t, x) => reshapeAs1D(t, [|x|]);
    [@bs.send]
    external reshapeAs2D : (t, array(int)) => Tensor(Rank2).t = "reshape";
    let reshapeTo2D = (t, (x, y)) => reshapeAs2D(t, [|x, y|]);
    [@bs.send]
    external reshapeAs3D : (t, array(int)) => Tensor(Rank3).t = "reshape";
    let reshapeTo3D = (t, (x, y, z)) => reshapeAs3D(t, [|x, y, z|]);
    [@bs.send]
    external reshapeAs4D : (t, array(int)) => Tensor(Rank4).t = "reshape";
    let reshapeTo4D = (tensor, (x, y, z, t)) =>
      reshapeAs4D(tensor, [|x, y, z, t|]);
    [@bs.send]
    external reshapeAs1D : (t, Tensor(Rank1).t) => Tensor(Rank1).t =
      "reshapeAs";
    [@bs.send]
    external reshapeAs2D : (t, Tensor(Rank2).t) => Tensor(Rank2).t =
      "reshapeAs";
    [@bs.send]
    external reshapeAs3D : (t, Tensor(Rank3).t) => Tensor(Rank3).t =
      "reshapeAs";
    [@bs.send]
    external reshapeAs4D : (t, Tensor(Rank4).t) => Tensor(Rank4).t =
      "reshapeAs";
    [@bs.send]
    external expandScalarDims :
      (Tensor(Rank0).t, [@bs.as 0] _) => Tensor(Rank1).t =
      "expandDims";
    [@bs.send]
    external expand1dDims : (Tensor(Rank1).t, [@bs.as 1] _) => Tensor(Rank2).t =
      "expandDims";
    [@bs.send]
    external expand2dDimsOnXAxis :
      (Tensor(Rank2).t, [@bs.as 1] _) => Tensor(Rank3).t =
      "expandDims";
    [@bs.send]
    external expand2dDimsOnYAxis :
      (Tensor(Rank2).t, [@bs.as 2] _) => Tensor(Rank3).t =
      "expandDims";
    [@bs.send]
    external expand3dDimsOnXAxis :
      (Tensor(Rank3).t, [@bs.as 1] _) => Tensor(Rank4).t =
      "expandDims";
    [@bs.send]
    external expand3dDimsOnYAxis :
      (Tensor(Rank3).t, [@bs.as 2] _) => Tensor(Rank4).t =
      "expandDims";
    [@bs.send]
    external expand3dDimsOnZAxis :
      (Tensor(Rank3).t, [@bs.as 3] _) => Tensor(Rank4).t =
      "expandDims";
    [@bs.send] external clone : t => t = "";
    [@bs.send] external toString : t => string = "";
    [@bs.send]
    external toStringVerbose : (t, [@bs.as {json|true|json}] _) => string =
      "toString";
  };

module Scalar = Tensor(Rank0);

module Tensor1D = Tensor(Rank1);

module Tensor2D = Tensor(Rank2);

module Tensor3D = Tensor(Rank3);

module Tensor4D = Tensor(Rank4);
