use crate::tensor::OnnxTensor;

pub use crate::tensor_proto::DataType;

/// ONNX tensor data types
impl DataType {
    /// Create DataType from ONNX type integer
    pub fn from_onnx_type(data_type: i32) -> Self {
        Self::try_from(data_type).unwrap_or(Self::Undefined)
    }

    /// Get the size in bytes for numeric types
    pub fn size_in_bytes(&self) -> Option<usize> {
        match self {
            DataType::Float | DataType::Int32 | DataType::Uint32 => Some(4),
            DataType::Double | DataType::Int64 | DataType::Uint64 => Some(8),
            DataType::Float16 | DataType::Bfloat16 | DataType::Int16 | DataType::Uint16 => Some(2),
            DataType::Int8 | DataType::Uint8 | DataType::Bool => Some(1),
            DataType::Complex64 => Some(8),
            DataType::Complex128 => Some(16),
            DataType::Float8e4m3fn
            | DataType::Float8e4m3fnuz
            | DataType::Float8e5m2
            | DataType::Float8e5m2fnuz
            | DataType::Float8e8m0 => Some(1),
            DataType::Uint4 | DataType::Int4 | DataType::Float4e2m1 => Some(1),
            DataType::String | DataType::Undefined => None,
        }
    }

    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            DataType::Float16
                | DataType::Float
                | DataType::Double
                | DataType::Bfloat16
                | DataType::Float8e4m3fn
                | DataType::Float8e4m3fnuz
                | DataType::Float8e5m2
                | DataType::Float8e5m2fnuz
                | DataType::Float8e8m0
                | DataType::Float4e2m1
        )
    }

    /// Check if this is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            DataType::Int8
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::Uint8
                | DataType::Uint16
                | DataType::Uint32
                | DataType::Uint64
                | DataType::Uint4
                | DataType::Int4
        )
    }
}

/// ONNX attribute values
#[derive(Debug, Clone)]
pub enum AttributeValue {
    Int(i64),
    Float(f32),
    String(String),
    Tensor(OnnxTensor),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Strings(Vec<String>),
}

impl AttributeValue {
    /// Try to get integer value
    pub fn as_int(&self) -> Option<i64> {
        match self {
            AttributeValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Try to get float value
    pub fn as_float(&self) -> Option<f32> {
        match self {
            AttributeValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Try to get string value
    pub fn as_string(&self) -> Option<&str> {
        match self {
            AttributeValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Try to get tensor value
    pub fn as_tensor(&self) -> Option<&OnnxTensor> {
        match self {
            AttributeValue::Tensor(t) => Some(t),
            _ => None,
        }
    }

    /// Try to get integer array value
    pub fn as_ints(&self) -> Option<&[i64]> {
        match self {
            AttributeValue::Ints(ints) => Some(ints),
            _ => None,
        }
    }

    /// Try to get float array value
    pub fn as_floats(&self) -> Option<&[f32]> {
        match self {
            AttributeValue::Floats(floats) => Some(floats),
            _ => None,
        }
    }

    /// Try to get string array value
    pub fn as_strings(&self) -> Option<&[String]> {
        match self {
            AttributeValue::Strings(strings) => Some(strings),
            _ => None,
        }
    }
}
