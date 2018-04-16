module FFI = {
  type t;
  external unsafeCastToFFI : 'a => t = "%identity";
};

type shapeRank0 = array(int);

/* TODO: Come back and see if this messes things up. TFJS is expecting (int) but
   you can't make a tuple of 1 value */
type shapeRank1 = int;

type shapeRank2 = (int, int);

type shapeRank3 = (int, int, int);

type shapeRank4 = (int, int, int, int);

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
    | Float32(f) => f |> FFI.unsafeCastToFFI
    | Int32(i) => i |> FFI.unsafeCastToFFI
    | Bool(b) => b |> FFI.unsafeCastToFFI
    };
};

module TensorLike = {
  type t =
    | Typed(TypedArray.t)
    | Int(int)
    | Float(float)
    | Bool(bool)
    | Int1D(array(int))
    | Float1D(array(float))
    | Bool1D(array(bool))
    | Int2D(array(array(int)))
    | Float2D(array(array(float)))
    | Bool2D(array(array(bool)))
    | Int3D(array(array(array(int))))
    | Float3D(array(array(array(float))))
    | Bool3D(array(array(array(bool))))
    | Int4D(array(array(array(array(int)))))
    | Float4D(array(array(array(array(float)))))
    | Bool4D(array(array(array(array(bool)))));
  let sendToTfjs = tensorLike =>
    switch (tensorLike) {
    | Typed(a) => a |> TypedArray.sendToTfjs
    | Int(i) => i |> FFI.unsafeCastToFFI
    | Float(f) => f |> FFI.unsafeCastToFFI
    | Bool(b) => b |> FFI.unsafeCastToFFI
    | Int1D(i) => i |> FFI.unsafeCastToFFI
    | Float1D(f) => f |> FFI.unsafeCastToFFI
    | Bool1D(b) => b |> FFI.unsafeCastToFFI
    | Int2D(i) => i |> FFI.unsafeCastToFFI
    | Float2D(f) => f |> FFI.unsafeCastToFFI
    | Bool2D(b) => b |> FFI.unsafeCastToFFI
    | Int3D(i) => i |> FFI.unsafeCastToFFI
    | Float3D(f) => f |> FFI.unsafeCastToFFI
    | Bool3D(b) => b |> FFI.unsafeCastToFFI
    | Int4D(i) => i |> FFI.unsafeCastToFFI
    | Float4D(f) => f |> FFI.unsafeCastToFFI
    | Bool4D(b) => b |> FFI.unsafeCastToFFI
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
    | Int(i) => i |> FFI.unsafeCastToFFI
    | Float(f) => f |> FFI.unsafeCastToFFI
    | Bool(b) => b |> FFI.unsafeCastToFFI
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
    | FlatInt(a) => a |> FFI.unsafeCastToFFI
    | FlatFloat(a) => a |> FFI.unsafeCastToFFI
    | FlatBool(a) => a |> FFI.unsafeCastToFFI
    | Int(a) => a |> FFI.unsafeCastToFFI
    | Float(a) => a |> FFI.unsafeCastToFFI
    | Bool(a) => a |> FFI.unsafeCastToFFI
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
    | FlatInt(a) => a |> FFI.unsafeCastToFFI
    | FlatFloat(a) => a |> FFI.unsafeCastToFFI
    | FlatBool(a) => a |> FFI.unsafeCastToFFI
    | Int(a) => a |> FFI.unsafeCastToFFI
    | Float(a) => a |> FFI.unsafeCastToFFI
    | Bool(a) => a |> FFI.unsafeCastToFFI
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
    | FlatInt(a) => a |> FFI.unsafeCastToFFI
    | FlatFloat(a) => a |> FFI.unsafeCastToFFI
    | FlatBool(a) => a |> FFI.unsafeCastToFFI
    | Int(a) => a |> FFI.unsafeCastToFFI
    | Float(a) => a |> FFI.unsafeCastToFFI
    | Bool(a) => a |> FFI.unsafeCastToFFI
    };
};

type flatVector =
  | Float(array(float))
  | Int(array(int))
  | Bool(array(bool))
  | Typed(TypedArray.t);

module Tensor = {
  type t;
  external unsafeCast : 'a => t = "%identity";
};

module TfArrayInput = {
  type t =
    | TypedArray(TypedArray.t)
    | TensorLike(TensorLike.t)
    | TensorLike1D(TensorLike1D.t)
    | TensorLike2D(TensorLike2D.t)
    | TensorLike3D(TensorLike3D.t)
    | TensorLike4D(TensorLike4D.t);
  let sendToTfjs = input =>
    switch (input) {
    | TypedArray(a) => TypedArray.sendToTfjs(a)
    | TensorLike(t) => TensorLike.sendToTfjs(t)
    | TensorLike1D(a) => a |> TensorLike1D.sendToTfjs
    | TensorLike2D(a) => a |> TensorLike2D.sendToTfjs
    | TensorLike3D(a) => a |> TensorLike3D.sendToTfjs
    | TensorLike4D(a) => a |> TensorLike4D.sendToTfjs
    };
};
