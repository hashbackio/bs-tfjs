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
};

type flatVector =
  | Float(array(float))
  | Int(array(int))
  | Bool(array(bool))
  | Typed(TypedArray.t);

module RegularArray = {
  type t('a) =
    | OneDimensional(array('a))
    | TwoDimensional(array(array('a)))
    | ThreeDimensional(array(array(array('a))))
    | FourDimensional(array(array(array(array('a)))));
  external unsafeCastToArray : 'a => Js.Array.t('b) = "%identity";
  let rec getArrayDimensions = maybeArray =>
    maybeArray |> Js.Array.isArray ?
      switch (maybeArray |> unsafeCastToArray |> Belt.Array.get(_, 0)) {
      | Some(a) => getArrayDimensions(a) + 1
      | None => 1
      } :
      0;
  let cast = a =>
    switch (a |> getArrayDimensions) {
    | 4 => Some(FourDimensional(a |> unsafeCastToArray))
    | 3 => Some(ThreeDimensional(a |> unsafeCastToArray))
    | 2 => Some(TwoDimensional(a |> unsafeCastToArray))
    | 1 => Some(OneDimensional(a |> unsafeCastToArray))
    | _ => None
    };
};
/* module Convertible = {
     type t =
       | Float(float)
       | Int(int)
       | RegularFloatArray(RegularArray.t(float))
       | RegularIntArray(RegularArray.t(int))
       | TypedArray(TypedArray.t);
     let cast = a =>
       switch (a |> Js.Types.classify) {
       | Js.Types.JSNumber(num) => Some(Float(num))
       | Js.Types.JSObject(obj) when obj |> Js.Array.isArray =>
         Some(obj |> Js.Array.from)
       | Js.Types.JSObject(obj) =>
         obj
         |> TypedArray.cast
         |> Belt.Option.flatMap(_, ta => Some(TypedArray(ta)))
       | _ => None
       };
   }; */
