# Denaro CUDA miner (pool)
## Usage

[Download the latest files](https://github.com/geiccobs/denaro-cuda-miner/releases/latest), you'll need `libkernel.so` and `cuda`.

Use `LD_LIBRARY_PATH=. ./cuda -help` to see the full list of arguments.  
Let's look at them:
- `-address` - your wallet address, you can get it from https://t.me/DenaroCoinBot
- `-device` - GPU device ID, you can get it from `nvidia-smi`
- `-blocks` - number of blocks related to GPU
- `-threads` - number of threads related to GPU
- `-node` - node address to connect to
- `-pool` - pool address to connect to
- `-share` - difficulty of shares, increase it if you see a lot of shares
- `-fee` - dev fee, means that 1 share every X is sent to dev (me <3)
- `-silent` - don't print anything to stdout  

### Platforms
This miner is tested on both Linux and Windows (WSL 2).  
Obviously is working only on NVIDIA GPUs.

## Installation

```bash
git clone https://github.com/geiccobs/denaro-cuda-miner
cd denaro-cuda-miner
```

### Compiling by source

You can skip this if you want to use pre-built binary.  
[Install golang first](https://go.dev/doc/install)  
[Install CUDA toolkit](https://developer.nvidia.com/cuda-downloads)
```bash
nvcc --ptxas-options=-v --compiler-options '-fPIC' -o libkernel.so --shared kernel.cu
LD_LIBRARY_PATH=. go build
```

## Discussed topics
### Dev fee
I've done this work basically for free, without having any idea about CUDA, nor an NVIDIA GPU.  
Dev fee can be obviously turned off, just by setting `-fee` parameter to huge values or 0.

### Hashrate, how is it calculated?
Honestly, I don't know.  
At the moment it's a value took from the pool so it isn't really your GPU hashrate, but all the miners pointed to your address combined.

### Multiple GPUs
I'll be working on it.  
Currently, you can select which GPU to use by setting `-device` parameter, but you can't use multiple GPUs at the same time.