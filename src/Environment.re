[@bs.deriving jsConverter]
type backend = [ | `webgl | `cpu | `tensorflow];

[@bs.module "@tensorflow/tfjs"] external disposeVariables: unit => unit = "";

[@bs.module "@tensorflow/tfjs"] external getBackend: unit => string = "";

let getBackend = () => getBackend()->backendFromJs;

[@bs.module "@tensorflow/tfjs"] external setBackend: string => unit = "";

let setBackend = backend => backend->backendToJs->setBackend;
