//! # onnx-extractor
//!
//! A lightweight ONNX model parser for extracting tensor shapes, operations, and data.
//!
//! This crate provides a simple interface to parse ONNX models and extract:
//! - Tensor information (shapes, data types, raw data)
//! - Operation details (inputs, outputs, attributes)
//! - Model structure (inputs, outputs, graph topology)
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use onnx_extractor::OnnxModel;
//!
//! let model = OnnxModel::load_from_file("model.onnx")?;
//! model.print_model_info();
//!
//! // Access tensor information
//! if let Some(tensor) = model.get_tensor("input") {
//!     println!("Input shape: {:?}", tensor.shape);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

// include generated protobuf code inside a small module so we can silence
// lints and doc warnings originating from the generated file only.
#[allow(clippy::all)]
#[allow(rustdoc::all)]
mod onnx_generated {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

pub use onnx_generated::*;

pub mod error;
pub mod model;
pub mod operation;
pub mod proto_adapter;
pub mod tensor;
pub mod types;

pub use error::{Error, Result};
pub use model::OnnxModel;
pub use operation::OperationInfo;
pub use tensor::TensorInfo;
pub use types::{AttributeValue, DataType};
