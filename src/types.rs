use crate::tensor::TensorInfo;

/// ONNX tensor data types
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Float32,
    Uint8,
    Int8,
    Uint16,
    Int16,
    Int32,
    Int64,
    String,
    Bool,
    Float16,
    Float64,
    Uint32,
    Uint64,
    Complex64,
    Complex128,
    BFloat16,
    Unknown(i32),
}

impl DataType {
    /// Create DataType from ONNX type integer
    pub fn from_onnx_type(data_type: i32) -> Self {
        match data_type {
            1 => DataType::Float32,
            2 => DataType::Uint8,
            3 => DataType::Int8,
            4 => DataType::Uint16,
            5 => DataType::Int16,
            6 => DataType::Int32,
            7 => DataType::Int64,
            8 => DataType::String,
            9 => DataType::Bool,
            10 => DataType::Float16,
            11 => DataType::Float64,
            12 => DataType::Uint32,
            13 => DataType::Uint64,
            14 => DataType::Complex64,
            15 => DataType::Complex128,
            16 => DataType::BFloat16,
            _ => DataType::Unknown(data_type),
        }
    }

    /// Get the size in bytes for numeric types
    pub fn size_in_bytes(&self) -> Option<usize> {
        match self {
            DataType::Float32 | DataType::Int32 | DataType::Uint32 => Some(4),
            DataType::Float64 | DataType::Int64 | DataType::Uint64 => Some(8),
            DataType::Float16 | DataType::BFloat16 | DataType::Int16 | DataType::Uint16 => Some(2),
            DataType::Int8 | DataType::Uint8 | DataType::Bool => Some(1),
            DataType::Complex64 => Some(8),
            DataType::Complex128 => Some(16),
            DataType::String | DataType::Unknown(_) => None,
        }
    }

    /// Check if this is a floating point type
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            DataType::Float16 | DataType::Float32 | DataType::Float64 | DataType::BFloat16
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
        )
    }
}

/// ONNX attribute values
#[derive(Debug, Clone)]
pub enum AttributeValue {
    Int(i64),
    Float(f32),
    String(String),
    Tensor(TensorInfo),
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
    pub fn as_tensor(&self) -> Option<&TensorInfo> {
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
