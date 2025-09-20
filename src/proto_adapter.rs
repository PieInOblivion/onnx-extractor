use crate::{
    AttributeProto, AttributeValue, DataType, Error, NodeProto, OnnxOperation, OnnxTensor,
    TensorProto,
};
use std::collections::HashMap;

/// Centralised adapter functions that translate generated protobuf types into
/// crate-native types. Keep all direct proto-field usage here so future changes
/// to `onnx.proto` need only update this file.
/// Create OnnxTensor from ONNX TensorProto
pub(crate) fn tensor_from_proto(tensor: &TensorProto) -> Result<OnnxTensor, Error> {
    let shape: Vec<i64> = tensor.dims.clone();
    let data_type = DataType::from_onnx_type(tensor.data_type.unwrap_or(0));

    // Try raw_data first, then fall back to typed data fields
    let raw_bytes = if tensor.raw_data.as_ref().map_or(true, |d| d.is_empty()) {
        extract_typed_data(tensor)
    } else {
        Some(tensor.raw_data.as_ref().unwrap().to_vec())
    };

    Ok(OnnxTensor {
        name: tensor.name.clone().unwrap_or_default(),
        shape,
        data_type,
        raw_bytes,
    })
}

fn extract_typed_data(tensor: &TensorProto) -> Option<Vec<u8>> {
    let data_type =
        DataType::try_from(tensor.data_type.unwrap_or(0)).unwrap_or(DataType::Undefined);

    macro_rules! numeric {
        ($field:expr) => {
            if $field.is_empty() {
                None
            } else {
                Some($field.iter().flat_map(|&x| x.to_le_bytes()).collect())
            }
        };
    }

    macro_rules! strings {
        ($field:expr) => {
            if $field.is_empty() {
                None
            } else {
                Some($field.iter().flat_map(|s| s.iter().copied()).collect())
            }
        };
    }

    match data_type {
        DataType::Float | DataType::Complex64 => numeric!(tensor.float_data),
        DataType::Double | DataType::Complex128 => numeric!(tensor.double_data),
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
        | DataType::Float4e2m1 => numeric!(tensor.int32_data),
        DataType::Int64 => numeric!(tensor.int64_data),
        DataType::Uint32 | DataType::Uint64 => numeric!(tensor.uint64_data),
        DataType::String => strings!(tensor.string_data),
        DataType::Undefined => numeric!(tensor.float_data)
            .or_else(|| numeric!(tensor.double_data))
            .or_else(|| numeric!(tensor.int32_data))
            .or_else(|| numeric!(tensor.int64_data))
            .or_else(|| numeric!(tensor.uint64_data))
            .or_else(|| strings!(tensor.string_data)),
    }
}

/// Create OnnxOperation from ONNX NodeProto
pub(crate) fn operation_from_node_proto(node: &NodeProto) -> Result<OnnxOperation, Error> {
    let mut attributes = HashMap::new();

    for attr in &node.attribute {
        let value = parse_attribute_proto(attr)?;
        let attr_name = attr.name.clone().unwrap_or_default();
        if !attr_name.is_empty() {
            attributes.insert(attr_name, value);
        }
    }

    Ok(OnnxOperation {
        name: node.name.clone().unwrap_or_default(),
        op_type: node.op_type.clone().unwrap_or_default(),
        inputs: node.input.clone(),
        outputs: node.output.clone(),
        attributes,
    })
}

/// Parse ONNX attribute into AttributeValue
pub(crate) fn parse_attribute_proto(attr: &AttributeProto) -> Result<AttributeValue, Error> {
    let attr_type = attr.r#type.unwrap_or(0);
    match attr_type {
        1 => Ok(AttributeValue::Float(attr.f.unwrap_or(0.0))),
        2 => Ok(AttributeValue::Int(attr.i.unwrap_or(0))),
        3 => {
            let s = attr.s.clone().unwrap_or_default();
            Ok(AttributeValue::String(String::from_utf8(s.to_vec())?))
        }
        4 => {
            if let Some(tensor) = &attr.t {
                let tensor_info = tensor_from_proto(tensor)?;
                Ok(AttributeValue::Tensor(tensor_info))
            } else {
                Err(Error::MissingField("tensor attribute data".to_string()))
            }
        }
        6 => Ok(AttributeValue::Floats(attr.floats.clone())),
        7 => Ok(AttributeValue::Ints(attr.ints.clone())),
        8 => {
            let strings_bytes = attr.strings.clone();
            let strings: Result<Vec<String>, Error> = strings_bytes
                .iter()
                .map(|s| String::from_utf8(s.to_vec()).map_err(Error::from))
                .collect();
            Ok(AttributeValue::Strings(strings?))
        }
        _ => Err(Error::Unsupported(format!("attribute type: {}", attr_type))),
    }
}
