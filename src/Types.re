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

module Tensor = (R: Rank) => {
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
  /* [@bs.send] external flatten : t =>  */
  /*
   dataId: DataId; */
};

module Scalar = Tensor(Rank0);

module Tensor1D = Tensor(Rank1);

module Tensor2D = Tensor(Rank2);

module Tensor3D = Tensor(Rank3);

module Tensor4D = Tensor(Rank4);
