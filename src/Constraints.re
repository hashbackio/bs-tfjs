type constraintType =
  | MaxNorm
  | MinMaxNorm
  | NonNeg
  | UnitNorm;

type ffi;

external _unsafeToFfi : 'a => ffi = "%identity";

let constraintTypesToJs = constraintType =>
  switch (constraintType) {
  | MaxNorm => "maxNorm" |> _unsafeToFfi
  | MinMaxNorm => "minMaxNorm" |> _unsafeToFfi
  | NonNeg => "nonNeg" |> _unsafeToFfi
  | UnitNorm => "unitNorm" |> _unsafeToFfi
  };
/* TODO: Expose the functions to create customer constraints */
