# Installation

Use Python 3.11, an NVIDIA CUDA-capable GPU, and a compatible CUDA-enabled PyTorch installation. The standard rational models depend on the separately installed `kat_rational` package.

1. Install the appropriate PyTorch build using the [official installation instructions](https://pytorch.org/get-started/locally/).
2. From this repository, install the numerical dependencies:

```powershell
python -m pip install -r requirements.txt
```

The requirements specify the Python package versions. Choose the PyTorch CUDA build for your platform before installing these requirements.

3. In a separate dependency directory **outside this repository**, download and install the upstream package:

```powershell
git clone https://github.com/Adamdad/rational_kat_cu.git
cd rational_kat_cu
python -m pip install -e . --no-build-isolation
```

Follow the upstream project's platform and compiler requirements. Disabling build isolation allows its build to access PyTorch already installed in the environment. For Windows, use a compatible Windows Triton distribution according to its platform instructions; the standard `triton` distribution may not provide a matching Windows wheel. Do not assume a CPU-only PyTorch installation can execute the KAT kernels.

4. Check the environment:

```powershell
python -c "import torch; from kat_rational import KAT_Group; print(torch.__version__, torch.cuda.is_available())"
```

The CUDA availability result must be `True` for training through the provided entry point. Use the same Python interpreter for package installation, terminal commands, and PyCharm.

This repository does not vendor `kat_rational`, `rational_kat_cu`, Triton kernels, or compiled dependency binaries. The local B-spline and ungrouped-rational modules are explicit ablation components; they do not provide a fallback implementation for the standard KAT backend. The current model construction still requires the external KAT dependency for all configurations.
