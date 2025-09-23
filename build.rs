use std::{env, path::PathBuf};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let generated_rs = out_dir.join("onnx.rs");

    if !generated_rs.exists() {
        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
        let repo_proto = manifest_dir.join("proto").join("onnx.proto");

        prost_build::Config::new()
            .bytes(["."])
            .compile_protos(&[&repo_proto], &[&manifest_dir])?;
    }

    Ok(())
}
