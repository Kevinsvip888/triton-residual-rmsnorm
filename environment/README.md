# Recorded environment

`requirements-experiment.txt` is the unchanged `uv pip freeze` output from the environment reused for Project 1 serving and Project 2 GPU work. Its SHA256 matches the environment digest in all three archived Day 11 protocols.

The direct tensor-computation dependencies are PyTorch and Triton. The full freeze also includes vLLM and other serving packages; it is an experiment record, not a claim that every listed package is necessary for this kernel.

Recorded setup: Python 3.12.15, torch 2.13.0+cu132, Triton 3.7.1, PyTorch CUDA build 13.2, RTX 4090 / SM89, NVIDIA driver 580.76.05. The setup console showed Ubuntu 22.04.5 and `nvidia-smi` reported CUDA 13.0; PyTorch CUDA checks and GPU tests succeeded in the existing environment. These version strings describe different components. No package or driver upgrade was made for publication.

The package freeze does not archive wheel files, the driver, compiler binaries, or Compute Sanitizer. The remote console reported Compute Sanitizer 2025.1.0.0 (build 35583870). That console report is not a binary/toolchain lockfile.

The supported path for the experiment machine is to reactivate its existing environment. Creating a fresh environment from the freeze requires compatible downloadable wheels and an appropriate CUDA wheel index; this was not tested during publication preparation. An installation attempt is documented in the reproduction guide, separately from the already-used environment.
