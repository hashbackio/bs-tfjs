/* type shape = array(int);

   [@bs.deriving jsConverter]
   type dtype = [ | `float32 | `int32 | `uint8 | `bool];

   module TypedArray = {
     type maybeT;
     type t =
       | Float32(Js.Typed_array.Float32Array.t)
       | Int32(Js.Typed_array.Int32Array.t)
       | Uint8(Js.Typed_array.Uint8Array.t);
     external castToMaybeT : 'a => maybeT = "%identity";
     let isFloat32Array: 'a => Js.boolean = [%bs.raw
       {|
         function(a) {
           return a instanceof Float32Array;
         }
       |}
     ];
     let isFloat32Array = a => a |> isFloat32Array |> Js.to_bool;
     external unsafeCastToFloat32Array : 'a => Js.Typed_array.Float32Array.t =
       "%identity";
     let isInt32Array: 'a => Js.boolean = [%bs.raw
       {|
         function(a) {
           return a instanceof Int32Array;
         }
       |}
     ];
     let isInt32Array = a => a |> isInt32Array |> Js.to_bool;
     external unsafeCastToInt32Array : 'a => Js.Typed_array.Int32Array.t =
       "%identity";
     let isUint8Array: 'a => Js.boolean = [%bs.raw
       {|
         function(a) {
           return a instanceof Uint8Array;
         }
       |}
     ];
     let isUint8Array = a => a |> isUint8Array |> Js.to_bool;
     external unsafeCastToUint8Array : 'a => Js.Typed_array.Uint8Array.t =
       "%identity";
     let cast = a =>
       switch (a |> castToMaybeT) {
       | a when a |> isFloat32Array =>
         Some(Float32(a |> unsafeCastToFloat32Array))
       | a when a |> isInt32Array => Some(Int32(a |> unsafeCastToInt32Array))
       | a when a |> isUint8Array => Some(Uint8(a |> unsafeCastToUint8Array))
       | _ => None
       };
   };

   type flatVector =
     | Float(array(float))
     | Int(array(int))
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

   type shapeDType = (shape, dtype);

   type shapeDTypeArray = array(Js.Nullable.t(shapeDType));

   module Convertible = {
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
   };

   module Storage = {
     type t;
     [@bs.send] external shape : t => shape = "";
     [@bs.send] external dtype : t => string = "";
     let dtype = t => t |> dtype |> dtypeFromJs;
     [@bs.send] external dataSync : t => TypedArray.t = "";
   }; */
/* export interface Storage {
     readonly shape: Shape;
     readonly dtype: DType;
     dataSync(): TypedArray;
     data(): Promise<TypedArray>;
     dispose(): void;
   } */
/*
 export type ShapeDType = [Shape, DType];
 export type ShapeDTypeList = Array<null | ShapeDType>;
 // JavaScript objects that can be generally converted to Tensors.
 export type Convertible = number | RegularArray<number> | TypedArray;
 export type TensorLike = Storage | Convertible;
 export type DeviceType = "CPU" | "GPU";
 export type Padding = "same" | "valid";
 export interface ConvOpts {
   stride?: number | [number, number];
   padding?: Padding;
 }
 export interface PoolOpts {
   size?: number | [number, number];
   stride?: number | [number, number];
   padding?: Padding;
 }

 // Storage does not use backprop.
 export interface Storage {
   readonly shape: Shape;
   readonly dtype: DType;
   dataSync(): TypedArray;
   data(): Promise<TypedArray>;
   dispose(): void;
 }

 // BackendOps do not use backprop.
 export interface BackendOps {
   copyToDevice(x: Storage, device: string): Storage;
   getDevice(x: Storage): string;
   listDevices(): string[];
   fromTypedArray(data: TypedArray, shape: Shape, dtype?: DType,
                  device?: string): Storage;
   add(x: Storage, y: Storage): Storage;
   sub(x: Storage, y: Storage): Storage;
   mul(x: Storage, y: Storage): Storage;
   div(x: Storage, y: Storage): Storage;
   neg(x: Storage): Storage;
   exp(x: Storage): Storage;
   log(x: Storage): Storage;
   setDiag(input: Storage, diag: Storage): Storage;
   onesLike(x: Storage): Storage;
   zerosLike(x: Storage): Storage;
   fill(value: Storage, shape: Shape): Storage;
   square(x: Storage): Storage;
   pow(x: Storage, exponent: Storage): Storage;
   sqrt(x: Storage): Storage;
   sin(x: Storage): Storage;
   cos(x: Storage): Storage;
   tan(x: Storage): Storage;
   sinh(x: Storage): Storage;
   cosh(x: Storage): Storage;
   tanh(x: Storage): Storage;
   relu(x: Storage): Storage;
   // reluGrad should not be exposed to the public API.
   // Generally Propel wants to express gradient functions in terms of other
   // known ops. However due to the ubiquity and performance necessities of
   // ReLU, we break this design goal and expose a special op for ReLU's
   // backward pass.
   reluGrad(grads: Storage, features: Storage): Storage;
   sigmoid(x: Storage): Storage;
   abs(x: Storage): Storage;
   randn(shape: Shape, seed?: number): Storage;
   linspace(start: number, stop: number, num: number): Storage;
   range(start: number, limit: number, delta: number): Storage;
   transpose(x: Storage, perm: Storage): Storage;
   reverse(x: Storage, dims: Storage): Storage;
   matmul(x: Storage, y: Storage, transposeA: boolean,
          transposeB: boolean): Storage;
   argmax(x: Storage, axis: number): Storage;
   argmin(x: Storage, axis: number): Storage;
   reduceSum(x: Storage, axes: number[], keepDims: boolean): Storage;
   reduceMean(x: Storage, axes: number[], keepDims: boolean): Storage;
   reduceMax(x: Storage, axes: number[], keepDims: boolean): Storage;
   reduceMin(x: Storage, axes: number[], keepDims: boolean): Storage;
   slice(x: Storage, begin: number[], size: number[]): Storage;
   gather(x: Storage, indices: Storage, axis: number): Storage;
   concat(axis: number, inputs: Storage[]): Storage;
   pad(x: Storage, paddings: Array<[number, number]>, padValue: number): Storage;
   reshape(x: Storage, newShape: Shape): Storage;
   equal(x: Storage, y: Storage): Storage;
   greater(x: Storage, y: Storage): Storage;
   greaterEqual(x: Storage, y: Storage): Storage;
   less(x: Storage, y: Storage): Storage;
   lessEqual(x: Storage, y: Storage): Storage;
   select(cond: Storage, t: Storage, f: Storage): Storage;
   sign(x: Storage): Storage;
   softmax(x: Storage): Storage;
   logSoftmax(x: Storage): Storage;
   cast(x: Storage, dtype: DType): Storage;
   oneHot(x: Storage, depth: number, onValue: number,
          offValue: number): Storage;

   conv2d(input: Storage, filter: Storage, opts: ConvOpts): Storage;
   conv2dGradFilter(grad: Storage, input: Storage,
                    filterShape: Shape, opts: ConvOpts): Storage;
   conv2dGradInput(gradient: Storage, inputShape: Shape,
                   filter: Storage, opts: ConvOpts): Storage;
   maxPool(input: Storage, opts: PoolOpts): Storage;
   maxPoolGrad(grad: Storage, origInput: Storage, origOutput: Storage,
               opts: PoolOpts): Storage;
 }

 // A TapeEntry is created every time an op is executed. It is the bookkeeping
 // entry for backpropigation.
 export interface TapeEntry {
   name: string;
   oid: number;
   inputIds: number[];
   inputShapeDTypes: ShapeDTypeList;
   outputIds: number[];
   savedForBackward: any[];
 }

 /** TensorOpts are used to build Tensors in functions like tensor() and zeros().
  * Note that Tensors themselves implement the TensorOpts interface, so existing
  * tensors can be used to construct similiarly typed and located tensors.
  */
 export interface TensorOpts {
   dtype: DType;
   device?: string;
 }

 export type Mode = "RGBA" | "RGB" | "L"; */
