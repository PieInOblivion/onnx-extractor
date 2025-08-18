use std::path::Path;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_path = Path::new("onnx.proto");

    // download onnx.proto if it doesn't exist
    if !proto_path.exists() {
        println!("cargo:warning=Downloading onnx.proto...");

        let output = Command::new("curl")
            .arg("-L") // follow redirects
            .arg("-s") // silent
            .arg("-f") // fail on http errors
            .arg("-o")
            .arg(proto_path)
            .arg("https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto")
            .output()?;

        if !output.status.success() {
            return Err(format!(
                "Failed to download onnx.proto. Please ensure curl is installed.\nError: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }

        println!("cargo:warning=Successfully downloaded onnx.proto");
    }

    // compile the protobuf
    prost_build::Config::new()
        .bytes(["."])
        .compile_protos(&["onnx.proto"], &["."])?;

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=onnx.proto");

    Ok(())
}
