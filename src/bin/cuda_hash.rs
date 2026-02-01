use anyhow::Result;
use clap::Parser;

#[cfg(feature = "cuda")]
use anyhow::Context;
#[cfg(feature = "cuda")]
use std::fs;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::compile_ptx;

/// CUDA-accelerated hashing tool (FNV-1a 32-bit).
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Path to newline-delimited input strings
    #[arg(short, long)]
    input: String,

    /// Max number of lines to hash (default: all)
    #[arg(short, long)]
    limit: Option<usize>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "cuda"))]
fn run() -> Result<()> {
    anyhow::bail!("This tool requires the cuda feature. Rebuild with --features cuda.");
}

#[cfg(feature = "cuda")]
fn run() -> Result<()> {
    let args = Args::parse();
    let content = fs::read_to_string(&args.input)
        .with_context(|| format!("Failed to read {}", args.input))?;
    let mut lines: Vec<String> = content
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| line.to_string())
        .collect();
    if let Some(limit) = args.limit {
        lines.truncate(limit);
    }

    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(lines.len());
    let mut lengths = Vec::with_capacity(lines.len());
    for line in &lines {
        offsets.push(data.len() as i32);
        lengths.push(line.len() as i32);
        data.extend_from_slice(line.as_bytes());
    }

    let kernel = r#"
    extern "C" __global__ void fnv1a(const unsigned char* data,
                                     const int* offsets,
                                     const int* lengths,
                                     unsigned int* out,
                                     int count) {
        int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
        if (idx >= count) {
            return;
        }
        unsigned int hash = 2166136261u;
        int offset = offsets[idx];
        int len = lengths[idx];
        for (int i = 0; i < len; i++) {
            hash ^= (unsigned int)data[offset + i];
            hash *= 16777619u;
        }
        out[idx] = hash;
    }
    "#;

    let ptx = compile_ptx(kernel)?;
    let device = CudaDevice::new(0)?;
    let module = device.load_ptx(ptx, "cuda_hash", &["fnv1a"])?;
    let stream = device.fork_default_stream()?;

    let data_dev = device.htod_copy(data)?;
    let offsets_dev = device.htod_copy(offsets)?;
    let lengths_dev = device.htod_copy(lengths)?;
    let mut out_dev = device.alloc_zeros::<u32>(lines.len())?;

    let cfg = LaunchConfig::for_num_elems(lines.len() as u32);
    unsafe {
        let func = module.get_func("fnv1a")?;
        func.launch_on_stream(
            &stream,
            cfg,
            (
                &data_dev,
                &offsets_dev,
                &lengths_dev,
                &mut out_dev,
                lines.len() as i32,
            ),
        )?;
    }
    stream.synchronize()?;

    let hashes = device.dtoh_sync_copy(&out_dev)?;
    for (line, hash) in lines.iter().zip(hashes.iter()) {
        println!("{hash:08x}\t{line}");
    }

    Ok(())
}
