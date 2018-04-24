type constraintType =
  | MaxNorm
  | MinMaxNorm
  | NonNeg
  | UnitNorm;

type ffi;

external unsafeToFfi : 'a => ffi = "%identity";

let constraintTypesToJs = constraintType =>
  switch (constraintType) {
  | MaxNorm => "maxNorm" |> unsafeToFfi
  | MinMaxNorm => "minMaxNorm" |> unsafeToFfi
  | NonNeg => "nonNeg" |> unsafeToFfi
  | UnitNorm => "unitNorm" |> unsafeToFfi
  };
/* TODO: Expose the functions to create customer constraints */
