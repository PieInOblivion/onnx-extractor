use prost::Message;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use crate::{Error, ModelProto, OperationInfo, Result, TensorInfo, type_proto};

/// Main ONNX model container
pub struct OnnxModel {
    pub tensors: HashMap<String, TensorInfo>,
    pub operations: Vec<OperationInfo>,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub model_version: i64,
    pub producer_name: String,
    pub producer_version: String,
}

impl OnnxModel {
    /// Load ONNX model from file path
    pub fn load_from_file(path: &str) -> Result<Self> {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        Self::load_from_bytes(&buffer)
    }

    /// Load ONNX model from byte slice
    pub fn load_from_bytes(data: &[u8]) -> Result<Self> {
        let model = ModelProto::decode(data)?;
        let graph = model
            .graph
            .ok_or_else(|| Error::InvalidModel("No graph found in model".to_string()))?;

        let mut onnx_model = OnnxModel {
            tensors: HashMap::new(),
            operations: Vec::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            model_version: model.model_version.unwrap_or(0),
            producer_name: model.producer_name.unwrap_or_default(),
            producer_version: model.producer_version.unwrap_or_default(),
        };

        // pre-allocate based on graph sizes to avoid repeated reallocations
        onnx_model.tensors.reserve(
            graph.initializer.len()
                + graph.value_info.len()
                + graph.input.len()
                + graph.output.len(),
        );
        onnx_model.operations.reserve(graph.node.len());
        onnx_model.inputs.reserve(graph.input.len());
        onnx_model.outputs.reserve(graph.output.len());

        // collect initialiser names first
        let mut initializer_names = std::collections::HashSet::new();
        for tensor in &graph.initializer {
            if let Some(name) = &tensor.name {
                if !name.is_empty() {
                    initializer_names.insert(name.as_str());
                }
            }
        }

        // extract input names (excluding initialisers)
        for input in &graph.input {
            let name = input.name.clone().unwrap_or_default();
            if !name.is_empty() && !initializer_names.contains(name.as_str()) {
                onnx_model.inputs.push(name);
            }
        }

        // extract output names
        for output in &graph.output {
            onnx_model
                .outputs
                .push(output.name.clone().unwrap_or_default());
        }

        // parse initialiser tensors (weights/constants)
        for tensor in &graph.initializer {
            let tensor_info = TensorInfo::from_tensor_proto(tensor)?;
            let tensor_name = tensor.name.clone().unwrap_or_default();
            if !tensor_name.is_empty() {
                onnx_model.tensors.insert(tensor_name, tensor_info);
            }
        }

        // parse value_info for intermediate tensor shapes and types
        for value_info in &graph.value_info {
            if let Some(t) = &value_info.r#type
                && let Some(type_proto_value) = &t.value
                && let type_proto::Value::TensorType(tensor_type) = type_proto_value
            {
                let name = value_info.name.clone().unwrap_or_default();
                if !name.is_empty() {
                    let tensor_info = TensorInfo::from_tensor_type(name.clone(), tensor_type);
                    onnx_model.tensors.insert(name, tensor_info);
                }
            }
        }

        // parse input tensor info
        for input in &graph.input {
            if let Some(t) = &input.r#type
                && let Some(type_proto_value) = &t.value
                && let type_proto::Value::TensorType(tensor_type) = type_proto_value
            {
                let name = input.name.clone().unwrap_or_default();
                if !name.is_empty() {
                    let tensor_info = TensorInfo::from_tensor_type(name.clone(), tensor_type);
                    onnx_model.tensors.insert(name, tensor_info);
                }
            }
        }

        // parse output tensor info
        for output in &graph.output {
            if let Some(t) = &output.r#type
                && let Some(type_proto_value) = &t.value
                && let type_proto::Value::TensorType(tensor_type) = type_proto_value
            {
                let name = output.name.clone().unwrap_or_default();
                if !name.is_empty() {
                    let tensor_info = TensorInfo::from_tensor_type(name.clone(), tensor_type);
                    onnx_model.tensors.insert(name, tensor_info);
                }
            }
        }

        // parse operations/nodes
        for node in &graph.node {
            let operation = OperationInfo::from_node_proto(node)?;
            onnx_model.operations.push(operation);
        }

        Ok(onnx_model)
    }

    /// Get tensor information by name
    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Get all operations of a specific type
    pub fn get_operations_by_type(&self, op_type: &str) -> Vec<&OperationInfo> {
        self.operations
            .iter()
            .filter(|op| op.op_type == op_type)
            .collect()
    }

    /// Get operation by name
    pub fn get_operation(&self, name: &str) -> Option<&OperationInfo> {
        self.operations.iter().find(|op| op.name == name)
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<&String> {
        self.tensors.keys().collect()
    }

    /// Get all operation types in the model
    pub fn operation_types(&self) -> Vec<String> {
        // collect unique operation types using a hash set of &str to avoid
        // allocating intermediate owned Strings, then sort the resulting Vec
        use std::collections::HashSet;
        let mut set: HashSet<&str> = HashSet::with_capacity(self.operations.len());
        for op in &self.operations {
            set.insert(op.op_type.as_str());
        }
        let mut op_types: Vec<String> = set.into_iter().map(|s| s.to_string()).collect();
        op_types.sort();
        op_types
    }

    /// Count operations by type
    pub fn count_operations_by_type(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        counts.reserve(self.operations.len());
        for op in &self.operations {
            *counts.entry(op.op_type.clone()).or_insert(0) += 1;
        }
        counts
    }

    /// Get input tensors
    pub fn get_input_tensors(&self) -> Vec<&TensorInfo> {
        self.inputs
            .iter()
            .filter_map(|name| self.get_tensor(name))
            .collect()
    }

    /// Get output tensors
    pub fn get_output_tensors(&self) -> Vec<&TensorInfo> {
        self.outputs
            .iter()
            .filter_map(|name| self.get_tensor(name))
            .collect()
    }

    /// Get tensors with data (initialisers/weights)
    pub fn get_weight_tensors(&self) -> Vec<&TensorInfo> {
        self.tensors
            .values()
            .filter(|tensor| tensor.has_data())
            .collect()
    }

    /// Return operations in a simple topological order using Kahn's algorithm.
    ///
    /// The returned vector contains references into `self.operations` and
    /// represents an order such that producers appear before their consumers.
    /// Operations are processed in the order they become available with no
    /// additional prioritisation.
    ///
    /// See also [`execution_order`](Self::execution_order) for a version that
    /// prioritises operations consuming model inputs.
    ///
    /// If the graph contains cycles or there are unresolved dependencies,
    /// the function returns an `Error::InvalidModel`.
    pub fn topological_order(&self) -> Result<Vec<&OperationInfo>> {
        use std::collections::VecDeque;

        let op_count = self.operations.len();

        // map tensor name -> producer op index
        let mut producer: HashMap<&str, usize> = HashMap::new();
        for (idx, op) in self.operations.iter().enumerate() {
            for out in &op.outputs {
                if !out.is_empty() {
                    producer.entry(out.as_str()).or_insert(idx);
                }
            }
        }

        // map tensor name -> list of consumer op indices
        let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (idx, op) in self.operations.iter().enumerate() {
            for input in &op.inputs {
                if !input.is_empty() {
                    consumers.entry(input.as_str()).or_default().push(idx);
                }
            }
        }

        // indegree = number of inputs coming from other ops (i.e. produced by some op)
        let mut indegree: Vec<usize> = vec![0; op_count];
        for (idx, op) in self.operations.iter().enumerate() {
            let mut count = 0usize;
            for input in &op.inputs {
                if input.is_empty() {
                    continue;
                }
                if producer.contains_key(input.as_str()) {
                    count += 1;
                }
            }
            indegree[idx] = count;
        }

        // start with ops that have indegree 0
        let mut queue: VecDeque<usize> = VecDeque::new();
        for (idx, &d) in indegree.iter().enumerate() {
            if d == 0 {
                queue.push_back(idx);
            }
        }

        let mut ordered: Vec<&OperationInfo> = Vec::with_capacity(op_count);

        while let Some(idx) = queue.pop_front() {
            let op = &self.operations[idx];
            ordered.push(op);

            // mark outputs as available and reduce indegree of consumers
            for out in &op.outputs {
                if out.is_empty() {
                    continue;
                }
                if let Some(cons_list) = consumers.get(out.as_str()) {
                    for &cidx in cons_list {
                        // only decrease indegree if the dependency was counted from a producer
                        if indegree[cidx] > 0 {
                            indegree[cidx] -= 1;
                            if indegree[cidx] == 0 {
                                queue.push_back(cidx);
                            }
                        }
                    }
                }
            }
        }

        if ordered.len() != op_count {
            Err(Error::InvalidModel(
                "Graph has cycles or unresolved dependencies".to_string(),
            ))
        } else {
            Ok(ordered)
        }
    }

    /// Return operations in execution-optimised topological order.
    ///
    /// The returned vector contains references into `self.operations` and
    /// represents an order such that producers appear before their consumers.
    /// Operations that consume model inputs are prioritised over parameter-only
    /// operations.
    ///
    /// This uses Kahn's algorithm with prioritisation. See also
    /// [`topological_order`](Self::topological_order) for a simple version
    /// without prioritisation.
    ///
    /// If the graph contains cycles or there are unresolved dependencies,
    /// the function returns an `Error::InvalidModel`.
    pub fn execution_order(&self) -> Result<Vec<&OperationInfo>> {
        use std::collections::VecDeque;

        let op_count = self.operations.len();

        // map tensor name -> producer op index
        let mut producer: HashMap<&str, usize> = HashMap::new();
        for (idx, op) in self.operations.iter().enumerate() {
            for out in &op.outputs {
                if !out.is_empty() {
                    producer.entry(out.as_str()).or_insert(idx);
                }
            }
        }

        // map tensor name -> list of consumer op indices
        let mut consumers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (idx, op) in self.operations.iter().enumerate() {
            for input in &op.inputs {
                if !input.is_empty() {
                    consumers.entry(input.as_str()).or_default().push(idx);
                }
            }
        }

        // indegree = number of inputs coming from other ops
        let mut indegree: Vec<usize> = vec![0; op_count];
        for (idx, op) in self.operations.iter().enumerate() {
            let mut count = 0usize;
            for input in &op.inputs {
                if input.is_empty() {
                    continue;
                }
                if producer.contains_key(input.as_str()) {
                    count += 1;
                }
            }
            indegree[idx] = count;
        }

        // start with ops that have indegree 0, prioritizing those that consume model inputs
        let mut queue: VecDeque<usize> = VecDeque::new();
        let mut ready_ops: Vec<usize> = Vec::new();

        for (idx, &d) in indegree.iter().enumerate() {
            if d == 0 {
                ready_ops.push(idx);
            }
        }

        // sort ready ops: input consumers first, then parameter-only ops
        ready_ops.sort_by_key(|&idx| {
            let op = &self.operations[idx];
            let consumes_input = op.inputs.iter().any(|input| self.inputs.contains(input));
            !consumes_input // false sorts before true, so input consumers come first
        });

        for idx in ready_ops {
            queue.push_back(idx);
        }

        let mut ordered: Vec<&OperationInfo> = Vec::with_capacity(op_count);

        while let Some(idx) = queue.pop_front() {
            let op = &self.operations[idx];
            ordered.push(op);

            // collect newly ready operations
            let mut newly_ready: Vec<usize> = Vec::new();

            for out in &op.outputs {
                if out.is_empty() {
                    continue;
                }
                if let Some(cons_list) = consumers.get(out.as_str()) {
                    for &cidx in cons_list {
                        if indegree[cidx] > 0 {
                            indegree[cidx] -= 1;
                            if indegree[cidx] == 0 {
                                newly_ready.push(cidx);
                            }
                        }
                    }
                }
            }

            // sort newly ready ops: input consumers first
            newly_ready.sort_by_key(|&idx| {
                let op = &self.operations[idx];
                let consumes_input = op.inputs.iter().any(|input| self.inputs.contains(input));
                !consumes_input
            });

            // add to front of queue (input consumers) or back (parameter ops)
            for cidx in newly_ready {
                let consumer_op = &self.operations[cidx];
                let consumes_input = consumer_op
                    .inputs
                    .iter()
                    .any(|input| self.inputs.contains(input));

                if consumes_input {
                    queue.push_front(cidx);
                } else {
                    queue.push_back(cidx);
                }
            }
        }

        if ordered.len() != op_count {
            Err(Error::InvalidModel(
                "Graph has cycles or unresolved dependencies".to_string(),
            ))
        } else {
            Ok(ordered)
        }
    }

    /// Print comprehensive model information
    pub fn print_model_info(&self) {
        println!("=== ONNX Model Information ===");
        println!(
            "Producer: {} v{}",
            self.producer_name, self.producer_version
        );
        println!("Model Version: {}", self.model_version);
        println!("Inputs: {:?}", self.inputs);
        println!("Outputs: {:?}", self.outputs);

        println!("\n=== Tensors ({}) ===", self.tensors.len());
        for (name, tensor) in &self.tensors {
            println!(
                "  {}: {:?} ({:?}) [{}{}]",
                name,
                tensor.shape,
                tensor.data_type,
                if tensor.has_data() { "data" } else { "no data" },
                if self.inputs.contains(name) {
                    ", input"
                } else if self.outputs.contains(name) {
                    ", output"
                } else {
                    ""
                }
            );
        }

        println!("\n=== Operations ({}) ===", self.operations.len());
        let op_counts = self.count_operations_by_type();
        for (op_type, count) in &op_counts {
            println!("  {}: {} operations", op_type, count);
        }

        println!("\n=== Operation Details ===");
        for op in &self.operations {
            println!(
                "  {} ({}): {} -> {}",
                op.name,
                op.op_type,
                op.inputs.join(", "),
                op.outputs.join(", ")
            );
            if !op.attributes.is_empty() {
                println!("    Attributes: {:?}", op.attribute_names());
            }
        }
    }

    /// Print a summary of the model
    pub fn print_summary(&self) {
        println!("=== ONNX Model Summary ===");
        println!(
            "Inputs: {} | Outputs: {} | Operations: {} | Tensors: {}",
            self.inputs.len(),
            self.outputs.len(),
            self.operations.len(),
            self.tensors.len()
        );

        let op_counts = self.count_operations_by_type();
        println!("Operation types: {:?}", op_counts);

        let weight_count = self.get_weight_tensors().len();
        println!("Weight tensors: {}", weight_count);
    }
}
