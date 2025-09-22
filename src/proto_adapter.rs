use crate::{
    AttributeProto, AttributeValue, DataType, Error, NodeProto, OnnxOperation, OnnxTensor,
    TensorProto,
};
use std::{collections::HashMap, mem};

/// Centralised adapter functions that translate generated protobuf types into
/// crate-native types. Keep all direct proto-field usage here so future changes
/// to `onnx.proto` need only update this file.
/// Create OnnxTensor from ONNX TensorProto
pub(crate) fn tensor_from_proto(tensor: TensorProto) -> Result<OnnxTensor, Error> {
    let shape: Vec<i64> = tensor.dims.clone();
    let data_type = DataType::from_onnx_type(tensor.data_type.unwrap_or(0));
    let name = tensor.name.clone().unwrap_or_default();

    Ok(OnnxTensor::new(name, shape, data_type, Some(tensor)))
}

/// Create OnnxOperation from ONNX NodeProto
pub(crate) fn operation_from_node_proto(mut node: NodeProto) -> Result<OnnxOperation, Error> {
    let mut attributes = HashMap::new();

    for attr in node.attribute.drain(..) {
        let attr_name = attr.name.clone().unwrap_or_default();
        let value = parse_attribute_proto(attr)?;
        if !attr_name.is_empty() {
            attributes.insert(attr_name, value);
        }
    }

    Ok(OnnxOperation {
        name: node.name.take().unwrap_or_default(),
        op_type: node.op_type.take().unwrap_or_default(),
        inputs: node.input,
        outputs: node.output,
        attributes,
    })
}

/// Parse ONNX attribute into AttributeValue
pub(crate) fn parse_attribute_proto(mut attr: AttributeProto) -> Result<AttributeValue, Error> {
    let attr_type = attr.r#type.unwrap_or(0);
    match attr_type {
        1 => Ok(AttributeValue::Float(attr.f.take().unwrap_or(0.0))),
        2 => Ok(AttributeValue::Int(attr.i.take().unwrap_or(0))),
        3 => {
            let s = attr.s.take().unwrap_or_default();
            Ok(AttributeValue::String(String::from_utf8(s.to_vec())?))
        }
        4 => {
            if let Some(tensor) = attr.t.take() {
                let onnx_tensor = tensor_from_proto(tensor)?;
                Ok(AttributeValue::Tensor(onnx_tensor))
            } else {
                Err(Error::MissingField("tensor attribute data".to_string()))
            }
        }
        6 => Ok(AttributeValue::Floats(mem::take(&mut attr.floats))),
        7 => Ok(AttributeValue::Ints(mem::take(&mut attr.ints))),
        8 => {
            let strings_bytes = mem::take(&mut attr.strings);
            let strings: Result<Vec<String>, Error> = strings_bytes
                .iter()
                .map(|s| String::from_utf8(s.to_vec()).map_err(Error::from))
                .collect();
            Ok(AttributeValue::Strings(strings?))
        }
        _ => Err(Error::Unsupported(format!("attribute type: {}", attr_type))),
    }
}
