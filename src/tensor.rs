use crate::{DataType, Error, TensorProto, tensor_shape_proto, type_proto};

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
    pub(crate) fn from_tensor_proto(tensor: &TensorProto) -> Result<Self, Error> {
        crate::proto_adapter::tensor_from_proto(tensor)
    }

    /// Create TensorInfo from tensor type (for inputs/outputs/value_info)
    pub(crate) fn from_tensor_type(name: String, tensor_type: &type_proto::Tensor) -> Self {
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
    pub fn get_raw_data(&self) -> Result<Vec<u8>, Error> {
        if let Some(data) = &self.data {
            Ok(data.clone())
        } else {
            Err(Error::MissingField("raw tensor data".to_string()))
        }
    }

    /// Extract tensor data as a typed array.
    ///
    /// This method interprets the raw tensor bytes as the specified type `T`.
    /// The tensor data is assumed to be stored in little-endian format as per
    /// the ONNX specification.
    pub fn get_data<T: Copy>(&self) -> std::result::Result<Vec<T>, Error> {
        let raw = self.get_raw_data()?;
        let type_size = std::mem::size_of::<T>();

        if raw.len() % type_size != 0 {
            return Err(Error::DataConversion(format!(
                "Data size {} is not aligned to type size {} (type: {})",
                raw.len(),
                type_size,
                std::any::type_name::<T>()
            )));
        }

        let mut result = Vec::with_capacity(raw.len() / type_size);
        for chunk in raw.chunks_exact(type_size) {
            // SAFETY: We know chunk has exactly type_size bytes
            // ONNX guarantees little-endian format matches most platforms
            unsafe {
                let value: T = std::ptr::read(chunk.as_ptr() as *const T);
                result.push(value);
            }
        }
        Ok(result)
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
