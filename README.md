# Denaro CUDA miner (pool)
## Usage

[Download the latest files](https://github.com/geiccobs/denaro-cuda-miner/releases/latest), you'll need just `cuda`.

Use `./cuda --help` to see the full list of arguments.  
Let's look at them:
- `--address` - your wallet address, you can get it from https://t.me/DenaroCoinBot (learn about [multi-address support](#multi-address))
- `--node` - node address to connect to
- `--pool` - pool address to connect to
- `--silent` - don't print anything to stdout
- `--verbose` - don't clear stdout after each share (useful for debugging)
- `--device` - GPU device ID, you can get it from `nvidia-smi`
- `--threads` - number of threads related to GPU
- `--blocks` - number of blocks related to GPU
- `--share` - difficulty of shares, increase it if you see a lot of shares
- `--fee` - dev fee, means that 1 every X blocks are mined by the dev (me <3)

### Platforms
This miner is tested on both Linux and Windows (WSL 2).  
Obviously it is working only on NVIDIA GPUs.

## Installation

```bash
git clone https://github.com/geiccobs/denaro-cuda-miner
cd denaro-cuda-miner
```

### Compiling by source

You can skip this if you want to use pre-built binary.  
[Install CUDA toolkit](https://developer.nvidia.com/cuda-downloads)
```bash
sudo apt install libjson-c-dev libcurl4-openssl-dev
nvcc -std=c++11 -O3 -arch=sm_70 --ptxas-options=-v -use_fast_math --compiler-options '-fPIC' -lineinfo main.cu requests.cu kernel/kernel.cu crypto/base58.cu crypto/sha256.cu crypto/hex.cu -lcurl -ljson-c -o cuda
```

## Discussed topics
### Dev fee
I've done this work basically for free, without having any idea about CUDA, nor an NVIDIA GPU.  
Dev fee can be obviously turned off, just by setting `--fee` parameter to huge values or 0.

### Multiple GPUs
I'll be working on it.  
Currently, you can select which GPU to use by setting `--device` parameter, but you can't use multiple GPUs at the same time.

### Avoid crashing
To avoid your miner stopping after a crash (because yes, it can always happen) you can start it using the following command in your terminal:
```bash
while true; do ./cuda --address YOUR_ADDRESS; sleep 1; done
```
There you go, your miner won't need to be manually restarted after any issue.

### Multi-address
Multi-address support is available, you can use it by setting `--address` parameter to a comma-separated list of addresses.

Here's an example:
```bash
./cuda --address YOUR_ADDRESS,ADDRESS_2,ADDRESS_3
```