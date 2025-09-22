use crate::{AttributeValue, Error, NodeProto, proto_adapter};
use std::collections::HashMap;

/// Information about an ONNX operation/node
#[derive(Debug)]
pub struct OnnxOperation {
    pub name: String,
    pub op_type: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, AttributeValue>,
}

impl OnnxOperation {
    /// Create OnnxOperation from ONNX NodeProto
    pub(crate) fn from_node_proto(node: NodeProto) -> Result<Self, Error> {
        proto_adapter::operation_from_node_proto(node)
    }

    /// Get attribute by name
    pub fn get_attribute(&self, name: &str) -> Option<&AttributeValue> {
        self.attributes.get(name)
    }

    /// Get integer attribute by name
    pub fn get_int_attribute(&self, name: &str) -> Option<i64> {
        self.get_attribute(name)?.as_int()
    }

    /// Get float attribute by name
    pub fn get_float_attribute(&self, name: &str) -> Option<f32> {
        self.get_attribute(name)?.as_float()
    }

    /// Get string attribute by name
    pub fn get_string_attribute(&self, name: &str) -> Option<&str> {
        self.get_attribute(name)?.as_string()
    }

    /// Get integer array attribute by name
    pub fn get_ints_attribute(&self, name: &str) -> Option<&[i64]> {
        self.get_attribute(name)?.as_ints()
    }

    /// Get float array attribute by name
    pub fn get_floats_attribute(&self, name: &str) -> Option<&[f32]> {
        self.get_attribute(name)?.as_floats()
    }

    /// Check if operation has a specific attribute
    pub fn has_attribute(&self, name: &str) -> bool {
        self.attributes.contains_key(name)
    }

    /// Get input count
    pub fn input_count(&self) -> usize {
        self.inputs.len()
    }

    /// Get output count
    pub fn output_count(&self) -> usize {
        self.outputs.len()
    }

    /// Check if this is a specific operation type
    pub fn is_op_type(&self, op_type: &str) -> bool {
        self.op_type == op_type
    }

    /// Get all attribute names
    pub fn attribute_names(&self) -> Vec<&String> {
        self.attributes.keys().collect()
    }
}
