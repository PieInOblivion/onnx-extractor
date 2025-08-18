use std::env;
use std::path::Path;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = env::var("OUT_DIR")?;
    let proto_path = Path::new(&out_dir).join("onnx.proto");

    // download onnx.proto if it doesn't exist in OUT_DIR
    if !proto_path.exists() {
        let output = Command::new("curl")
            .arg("-L") // follow redirects
            .arg("-s") // silent
            .arg("-f") // fail on http errors
            .arg("-o")
            .arg(&proto_path)
            .arg("https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto")
            .output()?;

        if !output.status.success() {
            return Err(format!(
                "Failed to download onnx.proto. Please ensure curl is installed.\nError: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }
    }

    // compile the protobuf
    prost_build::Config::new()
        .bytes(["."])
        .compile_protos(&[&proto_path], &[&out_dir])?;

    Ok(())
}
