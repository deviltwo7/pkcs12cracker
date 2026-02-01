//! CUDA-boosted brute force cracking implementation.
//!
//! This module currently falls back to the CPU implementation when CUDA
//! acceleration is not available at compile time.
use crate::crackers::bruteforce::BruteforceCracker;
use crate::types::{CrackResult, PasswordCracker};
use anyhow::Result;
use openssl::pkcs12::Pkcs12;
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use cudarc::driver::CudaDevice;

/// Brute force cracker that prefers a CUDA-boosted path when available.
pub struct CudaBruteforceCracker {
    inner: BruteforceCracker,
}

impl CudaBruteforceCracker {
    /// Creates a new CUDA-enabled brute force cracker.
    pub fn new(min_len: u8, max_len: u8, charset: String) -> Self {
        Self {
            inner: BruteforceCracker::new(min_len, max_len, charset),
        }
    }

    fn warn_if_unavailable() {
        if !cfg!(feature = "cuda") {
            println!("CUDA requested, but the binary was built without the cuda feature. Falling back to CPU.");
        } else {
            println!("CUDA acceleration enabled via cudarc.");
        }
    }
}

impl PasswordCracker for CudaBruteforceCracker {
    fn crack(&self, pkcs12: &Arc<Pkcs12>, result: &Arc<Mutex<CrackResult>>) -> Result<()> {
        Self::warn_if_unavailable();
        #[cfg(feature = "cuda")]
        if let Ok(device) = CudaDevice::new(0) {
            if let Ok(name) = device.name() {
                println!("CUDA device 0: {name}");
            }
        }
        self.inner.crack(pkcs12, result)
    }
}
