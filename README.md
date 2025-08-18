 # onnx-extractor

 A tiny, lightweight ONNX model parser for extracting tensor shapes, operations, and raw data.

 ## Quick start

 ```rust
 use onnx_extractor::OnnxModel;

 let model = OnnxModel::load_from_file("model.onnx")?;
 model.print_summary();
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
