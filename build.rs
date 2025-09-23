use std::path::Path;
use std::process::Command;
use std::{env, fs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = env::var("OUT_DIR")?;
    let out_dir_proto_path = Path::new(&out_dir).join("onnx.proto");

    // download onnx.proto if it doesn't exist in OUT_DIR
    if !out_dir_proto_path.exists() {
        let curl_res = Command::new("curl")
            .arg("-L") // follow redirects
            .arg("-s") // silent
            .arg("-f") // fail on http errors
            .arg("-o")
            .arg(&out_dir_proto_path)
            .arg("https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto")
            .output()?;

        // if curl failed, use local version
        if !curl_res.status.success() {
            let manifest = env::var("CARGO_MANIFEST_DIR")?;
            let repo_proto = Path::new(&manifest).join("proto/onnx.proto");
            fs::copy(&repo_proto, &out_dir_proto_path)?;
        }

        // compile the protobuf
        prost_build::Config::new()
            .bytes(["."])
            .compile_protos(&[&out_dir_proto_path], &[&out_dir])?;
    }

    Ok(())
}
