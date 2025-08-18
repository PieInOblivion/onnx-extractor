use crate::{DataType, Error, Result, TensorProto, tensor_shape_proto, type_proto};

/// Information about an ONNX tensor
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<i64>,
    pub data_type: DataType,
    pub data: Option<Vec<u8>>, // raw bytes of tensor data
}

impl TensorInfo {
    /// Create TensorInfo from ONNX TensorProto
    pub fn from_tensor_proto(tensor: &TensorProto) -> Result<Self> {
        crate::proto_adapter::tensor_from_proto(tensor)
    }

    /// Create TensorInfo from tensor type (for inputs/outputs/value_info)
    pub fn from_tensor_type(name: String, tensor_type: &type_proto::Tensor) -> Self {
        let shape = if let Some(shape_proto) = &tensor_type.shape {
            shape_proto
                .dim
                .iter()
                .map(|d| match &d.value {
                    Some(tensor_shape_proto::dimension::Value::DimValue(v)) => *v,
                    _ => -1,
                })
                .collect()
        } else {
            Vec::new()
        };

        TensorInfo {
            name,
            shape,
            data_type: DataType::from_onnx_type(tensor_type.elem_type.unwrap_or(0)),
            data: None,
        }
    }

    /// Get the raw tensor data bytes (if present)
    pub fn get_raw_data(&self) -> Result<Vec<u8>> {
        if let Some(data) = &self.data {
            Ok(data.clone())
        } else {
            Err(Error::MissingField("raw tensor data".to_string()))
        }
    }

    /// Get tensor data as f32 array
    pub fn get_f32_data(&self) -> Result<Vec<f32>> {
        match self.data_type {
            DataType::Float32 => {
                if let Some(bytes) = &self.data {
                    Ok(bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect())
                } else {
                    Err(Error::MissingField("tensor data".to_string()))
                }
            }
            _ => Err(Error::InvalidModel(format!(
                "Cannot convert {:?} to f32",
                self.data_type
            ))),
        }
    }

    /// Get tensor data as i32 array
    pub fn get_i32_data(&self) -> Result<Vec<i32>> {
        match self.data_type {
            DataType::Int32 => {
                if let Some(bytes) = &self.data {
                    Ok(bytes
                        .chunks_exact(4)
                        .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect())
                } else {
                    Err(Error::MissingField("tensor data".to_string()))
                }
            }
            _ => Err(Error::InvalidModel(format!(
                "Cannot convert {:?} to i32",
                self.data_type
            ))),
        }
    }

    /// Get tensor data as i64 array
    pub fn get_i64_data(&self) -> Result<Vec<i64>> {
        match self.data_type {
            DataType::Int64 => {
                if let Some(bytes) = &self.data {
                    Ok(bytes
                        .chunks_exact(8)
                        .map(|chunk| {
                            i64::from_le_bytes([
                                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5],
                                chunk[6], chunk[7],
                            ])
                        })
                        .collect())
                } else {
                    Err(Error::MissingField("tensor data".to_string()))
                }
            }
            _ => Err(Error::InvalidModel(format!(
                "Cannot convert {:?} to i64",
                self.data_type
            ))),
        }
    }

    /// Get the total number of elements in the tensor
    pub fn element_count(&self) -> i64 {
        self.shape.iter().filter(|&&dim| dim > 0).product()
    }

    /// Check if tensor has data
    pub fn has_data(&self) -> bool {
        self.data.is_some()
    }

    /// Get the size of tensor data in bytes
    pub fn data_size_bytes(&self) -> Option<usize> {
        self.data.as_ref().map(|d| d.len())
    }

    /// Check if tensor is scalar (0-dimensional)
    pub fn is_scalar(&self) -> bool {
        self.shape.is_empty()
    }

    /// Check if tensor is a vector (1-dimensional)
    pub fn is_vector(&self) -> bool {
        self.shape.len() == 1
    }

    /// Get tensor rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Check if tensor has dynamic dimensions
    pub fn has_dynamic_shape(&self) -> bool {
        self.shape.iter().any(|&dim| dim < 0)
    }
}
