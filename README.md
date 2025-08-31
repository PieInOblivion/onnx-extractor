# onnx-extractor

A tiny, lightweight ONNX model parser for extracting tensor shapes, operations, and raw data.

## Model Loading

```rust
use onnx_extractor::OnnxModel;

// Load from file
let model = OnnxModel::load_from_file("model.onnx")?;

// Load from bytes
let bytes = std::fs::read("model.onnx")?;
let model = OnnxModel::load_from_bytes(&bytes)?;
```

## Model Functions

```rust
// Basic info
model.print_summary();
model.print_model_info();

// Tensor access
let tensor = model.get_tensor("input_name");
let tensor_names = model.tensor_names();
let inputs = model.get_input_tensors();
let outputs = model.get_output_tensors();
let weights = model.get_weight_tensors();

// Operation access
let operation = model.get_operation("op_name");
let conv_ops = model.get_operations_by_type("Conv");
let op_types = model.operation_types();
let op_counts = model.count_operations_by_type();

// Execution order
let topo_order = model.topological_order()?;
let exec_order = model.execution_order()?;
```

## Tensor Functions

```rust
let tensor = model.get_tensor("weight").unwrap();

// Shape and type info
println!("Shape: {:?}", tensor.shape);
println!("Data type: {:?}", tensor.data_type);
println!("Element count: {}", tensor.element_count());
println!("Rank: {}", tensor.rank());

// Data access (get_data assumes little-endian platform)
let raw_bytes = tensor.get_raw_data()?;
let float_data: Vec<f32> = tensor.get_data()?;

// Export as different types with same function
let as_f32: Vec<f32> = tensor.get_data()?;
let as_f64: Vec<f64> = tensor.get_data()?;
let as_i32: Vec<i32> = tensor.get_data()?;
let as_u8: Vec<u8> = tensor.get_data()?;

// Properties
let has_data = tensor.has_data();
let size_bytes = tensor.data_size_bytes();
let is_scalar = tensor.is_scalar();
let is_vector = tensor.is_vector();
let has_dynamic = tensor.has_dynamic_shape();
```

## Operation Functions

```rust
let op = model.get_operation("conv1").unwrap();

// Basic info
println!("Type: {}", op.op_type);
println!("Inputs: {:?}", op.inputs);
println!("Outputs: {:?}", op.outputs);

// Attribute access
let kernel_size = op.get_ints_attribute("kernel_shape");
let stride = op.get_int_attribute("stride");
let activation = op.get_string_attribute("activation");
let weight = op.get_float_attribute("alpha");

// Properties
let input_count = op.input_count();
let output_count = op.output_count();
let is_conv = op.is_op_type("Conv");
let has_bias = op.has_attribute("bias");
let attr_names = op.attribute_names();
```

## Data Types

Access the `DataType` enum for type checking:

```rust
use onnx_extractor::DataType;

let tensor = model.get_tensor("input").unwrap();
match tensor.data_type {
    DataType::Float => println!("32-bit float"),
    DataType::Double => println!("64-bit float"),
    DataType::Int32 => println!("32-bit int"),
    _ => println!("Other type"),
}

// Type properties
let size = tensor.data_type.size_in_bytes();
let is_float = tensor.data_type.is_float();
let is_int = tensor.data_type.is_integer();
```

## About the protobuf (`onnx.proto`)

This crate generates Rust types from the ONNX protobuf at build time using `prost-build`.

- `build.rs` will download `onnx.proto` from the ONNX repo if the file is missing.
- `prost-build` compiles the `.proto` into `onnx.rs` under `$OUT_DIR` and the crate includes it with:
	 `include!(concat!(env!("OUT_DIR"), "/onnx.rs"));`

Notes:
- The build step requires `curl` (only if `onnx.proto` isn't present) and network access.
- `prost_build::Config::bytes(["."])` configures `bytes` fields as `prost::bytes::Bytes`; the code converts these to `Vec<u8>` or `String` where needed.

## Troubleshooting

- If the build fails because `curl` is missing, either install `curl` or place `onnx.proto` at the repository root to avoid the download step.
- If you prefer to avoid code generation at build time, you can vendor the generated `onnx.rs` into `src/` and change `include!` accordingly.

## License

MIT
