type constraintType =
  | MaxNorm
  | MinMaxNorm
  | NonNeg
  | UnitNorm;

let constraintTypesToJs = constraintType =>
  switch (constraintType) {
  | MaxNorm => "maxNorm"
  | MinMaxNorm => "minMaxNorm"
  | NonNeg => "nonNeg"
  | UnitNorm => "unitNorm"
  };
/* TODO: Expose the functions to create customer constraints */
