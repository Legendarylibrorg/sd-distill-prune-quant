# Windows setup

Windows 10/11 with PowerShell 7+ is supported. CUDA acceleration requires the
matching NVIDIA driver and PyTorch wheel.

## 1. Prerequisites

1. **Python 3.10 or newer.** Download from
   <https://www.python.org/downloads/windows/> and tick *"Add python.exe to
   PATH"* during the installer.
2. **Git for Windows** from <https://git-scm.com/download/win>.
3. **Windows Terminal + PowerShell 7+** is strongly recommended.
4. **NVIDIA driver** matching CUDA 11.8 or 12.x for GPU acceleration. Verify
   with `nvidia-smi` from PowerShell.

If PowerShell refuses to execute `.ps1` files, relax the policy for the current
user (one-time):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## 2. Clone

```powershell
git clone https://github.com/Legendarylibrorg/sd-distill-prune-quant.git
cd sd-distill-prune-quant
```

## 3. Virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 4. PyTorch with CUDA

Install the CUDA wheel **first**:

```powershell
# CUDA 12.1
pip install --index-url https://download.pytorch.org/whl/cu121 `
    torch torchvision

# CUDA 11.8 (older driver)
# pip install --index-url https://download.pytorch.org/whl/cu118 `
#     torch torchvision
```

## 5. Remaining requirements

```powershell
pip install -r requirements.txt
pip install "git+https://github.com/openai/CLIP.git"   # optional
# pip install xformers                                  # often unavailable on Windows
```

## 6. Verify

```powershell
python -m sd_compress info
```

## 7. Run

```powershell
.\run.ps1                                                # full pipeline + UI
.\run.ps1 --no-serve                                     # pipeline only
.\run.ps1 distill-progressive
.\run.ps1 evaluate --stage distilled --model-dir .\output\distilled
```

The Gradio UI listens on <http://localhost:8080>. To bind a different host or
port set the environment variables before running:

```powershell
$env:SERVER_HOST = "127.0.0.1"
$env:SERVER_PORT = 7860
.\run.ps1 serve
```

## 8. Tips

- **Enable long paths** to avoid checkpoint-saving errors:

  ```powershell
  New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
      -Name "LongPathsEnabled" -Value 1 -PropertyType DWord -Force
  ```

- **Defender slowdown.** Add the repository folder (and the model cache under
  `%USERPROFILE%\.cache\huggingface`) to Windows Defender's exclusion list to
  speed up the first run.
- **Use `python`, not `python3`.** The Microsoft Python launcher only exposes
  `python` and `py` on PATH.
- **Persistent environment variables.** Use `$env:VAR = "value"` inside the
  PowerShell session, or `setx VAR value` to persist across sessions.

## 9. Troubleshooting

- **`CUDA error: no kernel image is available for execution`.** Your driver is
  newer than the wheel; reinstall PyTorch with a matching `--index-url`.
- **`pip install xformers` errors.** Skip it. The pipeline tolerates missing
  xFormers and falls back to default attention.
- **Long install of `lpips`.** Pre-install scientific Python wheels via
  `pip install --prefer-binary lpips` or use `conda install -c conda-forge lpips`.
- **Gradio cannot open the firewall.** Approve the prompt or run as
  Administrator the first time.
