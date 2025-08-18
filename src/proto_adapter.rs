use crate::{AttributeProto, AttributeValue, Error, NodeProto, Result, TensorInfo, TensorProto};
use std::collections::HashMap;

/// Centralised adapter functions that translate generated protobuf types into
/// crate-native types. Keep all direct proto-field usage here so future changes
/// to `onnx.proto` need only update this file.
/// Create TensorInfo from ONNX TensorProto
pub fn tensor_from_proto(tensor: &TensorProto) -> Result<TensorInfo> {
    let shape: Vec<i64> = tensor.dims.clone();
    let data_type = crate::DataType::from_onnx_type(tensor.data_type.unwrap_or(0));

    // extract raw tensor data
    let data = if let Some(raw_data) = &tensor.raw_data {
        if !raw_data.is_empty() {
            Some(raw_data.to_vec())
        } else {
            None
        }
    } else {
        // handle different data type fields
        match data_type {
            crate::DataType::Float32 => {
                let float_data = tensor.float_data.clone();
                if !float_data.is_empty() {
                    let byte_data: Vec<u8> =
                        float_data.iter().flat_map(|&f| f.to_le_bytes()).collect();
                    Some(byte_data)
                } else {
                    None
                }
            }
            crate::DataType::Int32 => {
                let int_data = tensor.int32_data.clone();
                if !int_data.is_empty() {
                    let byte_data: Vec<u8> =
                        int_data.iter().flat_map(|&i| i.to_le_bytes()).collect();
                    Some(byte_data)
                } else {
                    None
                }
            }
            crate::DataType::Int64 => {
                let int64_data = tensor.int64_data.clone();
                if !int64_data.is_empty() {
                    let byte_data: Vec<u8> =
                        int64_data.iter().flat_map(|&i| i.to_le_bytes()).collect();
                    Some(byte_data)
                } else {
                    None
                }
            }
            _ => None,
        }
    };

    Ok(TensorInfo {
        name: tensor.name.clone().unwrap_or_default(),
        shape,
        data_type,
        data,
    })
}

/// Create OperationInfo from ONNX NodeProto
pub fn operation_from_node_proto(node: &NodeProto) -> Result<crate::OperationInfo> {
    let mut attributes = HashMap::new();

    for attr in &node.attribute {
        let value = parse_attribute_proto(attr)?;
        let attr_name = attr.name.clone().unwrap_or_default();
        if !attr_name.is_empty() {
            attributes.insert(attr_name, value);
        }
    }

    Ok(crate::OperationInfo {
        name: node.name.clone().unwrap_or_default(),
        op_type: node.op_type.clone().unwrap_or_default(),
        inputs: node.input.clone(),
        outputs: node.output.clone(),
        attributes,
    })
}

/// Parse ONNX attribute into AttributeValue
pub fn parse_attribute_proto(attr: &AttributeProto) -> Result<AttributeValue> {
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
            let strings: Result<Vec<String>> = strings_bytes
                .iter()
                .map(|s| String::from_utf8(s.to_vec()).map_err(Error::from))
                .collect();
            Ok(AttributeValue::Strings(strings?))
        }
        _ => Err(Error::Unsupported(format!("attribute type: {}", attr_type))),
    }
}
