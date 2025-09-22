use prost::bytes::Bytes;
use std::mem::ManuallyDrop;
use std::sync::OnceLock;
use std::{any, mem, ptr, slice};

use crate::{
    DataType, Error, TensorProto, tensor_shape_proto::dimension::Value, type_proto::Tensor,
};

/// Information about an ONNX tensor
#[derive(Debug)]
pub struct OnnxTensor {
    pub name: String,
    pub shape: Vec<i64>,
    pub data_type: DataType,
    proto: Option<TensorProto>,
    // lazily built contiguous bytes for STRING tensors to support bytes() borrow
    cached_string_bytes: OnceLock<Box<[u8]>>,
}

impl OnnxTensor {
    pub(crate) fn new(
        name: String,
        shape: Vec<i64>,
        data_type: DataType,
        proto: Option<TensorProto>,
    ) -> Self {
        OnnxTensor {
            name,
            shape,
            data_type,
            proto,
            cached_string_bytes: OnceLock::new(),
        }
    }

    pub(crate) fn from_tensor_type(name: String, tensor_type: &Tensor) -> Result<Self, Error> {
        let shape = if let Some(shape_proto) = &tensor_type.shape {
            shape_proto
                .dim
                .iter()
                .map(|d| match &d.value {
                    Some(Value::DimValue(v)) => *v,
                    _ => -1,
                })
                .collect()
        } else {
            Vec::new()
        };

        let elem_type = tensor_type
            .elem_type
            .ok_or_else(|| Error::MissingField("tensor elem_type".to_string()))?;
        if elem_type == 0 {
            return Err(Error::InvalidModel(
                "tensor elem_type must not be UNDEFINED (0)".to_string(),
            ));
        }

        Ok(OnnxTensor::new(
            name,
            shape,
            DataType::from_onnx_type(elem_type),
            None,
        ))
    }

    /// Get a cloned copy of the tensor data bytes.
    ///
    /// Returns an error if the tensor has no data present.
    pub fn clone_bytes(&self) -> Result<Box<[u8]>, Error> {
        let b = self.bytes()?;
        Ok(b.into())
    }

    /// Borrow the tensor data as a byte slice without copying.
    ///
    /// Returns an error if the tensor has no data present.
    pub fn bytes(&self) -> Result<&[u8], Error> {
        let t = self
            .proto
            .as_ref()
            .ok_or_else(|| Error::MissingField("tensor data".to_string()))?;

        if let Some(raw) = &t.raw_data {
            if !raw.is_empty() {
                return Ok(raw.as_ref());
            }
        }

        let bytes_opt: Option<&[u8]> = match storage_backing(self.data_type) {
            Some(StorageBacking::F32) => Some(slice_bytes_as::<f32>(t.float_data.as_slice())),
            Some(StorageBacking::F64) => Some(slice_bytes_as::<f64>(t.double_data.as_slice())),
            Some(StorageBacking::I64) => Some(slice_bytes_as::<i64>(t.int64_data.as_slice())),
            Some(StorageBacking::U64) => Some(slice_bytes_as::<u64>(t.uint64_data.as_slice())),
            Some(StorageBacking::I32) => Some(slice_bytes_as::<i32>(t.int32_data.as_slice())),
            Some(StorageBacking::Strings) => {
                let buf = self
                    .cached_string_bytes
                    .get_or_init(|| concat_strings_to_boxed_bytes(&t.string_data));
                Some(buf.as_ref())
            }
            None => None,
        };

        let bytes = bytes_opt.ok_or_else(|| Error::MissingField("tensor data".to_string()))?;
        if bytes.is_empty() && self.data_type != DataType::String {
            return Err(Error::MissingField("tensor data".to_string()));
        }
        Ok(bytes)
    }

    /// Consume the tensor and return the owned data as a boxed byte slice.
    ///
    /// Rules for consistency:
    /// - If raw_data is present, we return a new contiguous Box<[u8]> with its content.
    /// - For numeric typed fields, we avoid extra copies by reinterpreting the owned Vec<T>
    ///   into a Vec<u8> using a zero-copy cast and then boxing it.
    /// - For strings, we concatenate all entries into a single Box<[u8]>.
    pub fn into_bytes(mut self) -> Result<Box<[u8]>, Error> {
        let t = self
            .proto
            .as_mut()
            .ok_or_else(|| Error::MissingField("tensor data".to_string()))?;

        if let Some(raw) = t.raw_data.take() {
            if !raw.is_empty() {
                return Ok(raw.to_vec().into_boxed_slice());
            }
        }

        let bytes_opt: Option<Box<[u8]>> = match storage_backing(self.data_type) {
            Some(StorageBacking::F32) => Some(into_box::<f32>(mem::take(&mut t.float_data))),
            Some(StorageBacking::F64) => Some(into_box::<f64>(mem::take(&mut t.double_data))),
            Some(StorageBacking::I64) => Some(into_box::<i64>(mem::take(&mut t.int64_data))),
            Some(StorageBacking::U64) => Some(into_box::<u64>(mem::take(&mut t.uint64_data))),
            Some(StorageBacking::I32) => Some(into_box::<i32>(mem::take(&mut t.int32_data))),
            Some(StorageBacking::Strings) => {
                // ensure cache is initialised then move it
                let _ = self
                    .cached_string_bytes
                    .get_or_init(|| concat_strings_to_boxed_bytes(&t.string_data));
                let taken = mem::take(self.cached_string_bytes.get_mut().unwrap());
                Some(taken)
            }
            None => None,
        };

        let bytes = bytes_opt.ok_or_else(|| Error::MissingField("tensor data".to_string()))?;
        if bytes.is_empty() && self.data_type != DataType::String {
            return Err(Error::MissingField("tensor data".to_string()));
        }
        Ok(bytes)
    }

    /// Extract tensor data as a new boxed typed slice.
    ///
    /// This method interprets the raw tensor bytes as the specified type `T`.
    /// The tensor data is assumed to be stored in little-endian format as per
    /// the ONNX specification.
    ///
    /// Note: This function assumes a little-endian platform. Multi-byte types
    /// (e.g., f32, i32, u64) may return incorrect values on big-endian platforms.
    /// Does not interpret non-native Rust types such as Complex128 correctly.
    pub fn copy_data_as<T: Copy>(&self) -> Result<Box<[T]>, Error> {
        let bytes = self.bytes()?;
        let type_size = mem::size_of::<T>();

        if bytes.len() % type_size != 0 {
            return Err(Error::DataConversion(format!(
                "Data size {} is not aligned to type size {} (type: {})",
                bytes.len(),
                type_size,
                any::type_name::<T>()
            )));
        }

        let count = bytes.len() / type_size;
        // allocate exact sized boxed slice
        let mut out_uninit: Box<[mem::MaybeUninit<T>]> =
            vec![mem::MaybeUninit::<T>::uninit(); count].into_boxed_slice();

        // fill using unaligned reads to avoid UB if raw_data isn't suitably aligned
        for (i, chunk) in bytes.chunks_exact(type_size).enumerate() {
            let value = unsafe { ptr::read_unaligned(chunk.as_ptr() as *const T) };
            out_uninit[i].write(value);
        }

        // all elements have been initialised
        let out: Box<[T]> = unsafe { std::mem::transmute(out_uninit) };
        Ok(out)
    }

    /// Get the total number of elements in the tensor
    pub fn element_count(&self) -> i64 {
        self.shape.iter().filter(|&&dim| dim > 0).product()
    }

    /// Check if tensor has data
    pub fn has_data(&self) -> bool {
        self.bytes().map(|b| !b.is_empty()).unwrap_or(false)
    }

    /// Get the size of tensor data in bytes
    pub fn data_size_bytes(&self) -> Option<usize> {
        self.bytes().map(|b| b.len()).ok()
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

// storage backing variants for tensor proto
enum StorageBacking {
    F32,
    F64,
    I64,
    U64,
    I32,
    Strings,
}

// map DataType to backing storage if any
fn storage_backing(dt: DataType) -> Option<StorageBacking> {
    match dt {
        DataType::Float | DataType::Complex64 => Some(StorageBacking::F32),
        DataType::Double | DataType::Complex128 => Some(StorageBacking::F64),
        DataType::Int64 => Some(StorageBacking::I64),
        DataType::Uint32 | DataType::Uint64 => Some(StorageBacking::U64),
        DataType::Int32
        | DataType::Int16
        | DataType::Int8
        | DataType::Int4
        | DataType::Uint16
        | DataType::Uint8
        | DataType::Uint4
        | DataType::Bool
        | DataType::Float16
        | DataType::Bfloat16
        | DataType::Float8e4m3fn
        | DataType::Float8e4m3fnuz
        | DataType::Float8e5m2
        | DataType::Float8e5m2fnuz
        | DataType::Float8e8m0
        | DataType::Float4e2m1 => Some(StorageBacking::I32),
        DataType::String => Some(StorageBacking::Strings),
        DataType::Undefined => None,
    }
}

// concatenate a slice of prost Bytes into a single owned boxed byte slice
fn concat_strings_to_boxed_bytes(parts: &[Bytes]) -> Box<[u8]> {
    let total: usize = parts.iter().map(|b| b.len()).sum();
    let mut out = Vec::with_capacity(total);
    for s in parts {
        out.extend_from_slice(s.as_ref());
    }
    out.into_boxed_slice()
}

fn slice_bytes_as<T: Copy>(slice: &[T]) -> &[u8] {
    unsafe {
        slice::from_raw_parts(
            slice.as_ptr() as *const u8,
            slice.len() * mem::size_of::<T>(),
        )
    }
}

fn into_box<T: Copy>(v: Vec<T>) -> Box<[u8]> {
    let mut v = ManuallyDrop::new(v);
    let len = v.len() * mem::size_of::<T>();
    let cap = v.capacity() * mem::size_of::<T>();
    let ptr = v.as_mut_ptr() as *mut u8;
    let vu8: Vec<u8> = unsafe { Vec::from_raw_parts(ptr, len, cap) };
    vu8.into_boxed_slice()
}
