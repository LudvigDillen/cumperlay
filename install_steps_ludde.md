## ✅ Installing `cmp` with PyTorch and system CUDA (working procedure)

### 1. Create and activate the environment

```bash
conda env create -f environment.yml
conda activate cumperlay
```

---

### 2. Verify that system CUDA exists (must be 12.*)

```bash
ls /usr/local | grep cuda
```

Expected output example:

```
cuda
cuda-12.5
```

Check that CUDA headers are present:

```bash
ls /usr/local/cuda/include | grep cuda_fp16
```

Expected output:

```
cuda_fp16.h
cuda_fp16.hpp
```

---

### 3. Point the environment to system CUDA (is only temporary, see last point)

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```

---

### 4. Confirm that `nvcc` comes from system CUDA

```bash
which nvcc
```

Expected:

```
/usr/local/cuda/bin/nvcc
```

```bash
nvcc --version
```

Expected:

```
Cuda compilation tools, release 12.*
```

---

### 5. Ensure conda did **not** install its own CUDA headers

```bash
echo "$CONDA_PREFIX"
```

Example:

```
/home/lu2277di/sw/miniconda3/envs/cumperlay
```

```bash
ls "$CONDA_PREFIX/include" | grep cuda_fp16 || echo "no conda cuda_fp16 in env"
```

Expected:

```
no conda cuda_fp16 in env
```

---

### 6. Install `cmp` using PyTorch extension build

(arch `8.9` is correct for RTX 4090)

```bash
TORCH_CUDA_ARCH_LIST="8.9" pip install --no-build-isolation "git+https://github.com/circle-group/cmp.git#egg=cmp"
```

---

## ✅ Done — successful installation criteria

You know it worked if:

```bash
python -c "import cmp; print(cmp.__version__)"
```

returns a version and **does not error**, and:

```bash
python -c "import cmp.ops as ops; print(ops)"
```

prints CUDA extension symbols.

---

## Optional (but recommended to avoid re-exporting on every shell)

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
nano "$CONDA_PREFIX/etc/conda/activate.d/cuda_vars.sh"
```
and then add
```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
```
and deactivate script
```bash
mkdir -p "$CONDA_PREFIX/etc/conda/deactivate.d"
nano "$CONDA_PREFIX/etc/conda/deactivate.d/cuda_vars.sh"
```
and then add
```bash
export PATH=$(echo "$PATH" | sed -e "s|$CUDA_HOME/bin:||")
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed -e "s|$CUDA_HOME/lib64:||")
unset CUDA_HOME
```
