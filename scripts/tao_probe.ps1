param(
    [string]$Image = "nvcr.io/nvidia/tao/tao-toolkit:7.1.0-pyt",
    [string]$Output = "tao_probe.txt"
)

$ErrorActionPreference = "Stop"

Write-Host "[1/2] Verifying GPU access in $Image ..."
docker run --rm --gpus all --shm-size=16g `
    --ulimit memlock=-1 --ulimit stack=67108864 `
    $Image nvidia-smi
if ($LASTEXITCODE -ne 0) {
    throw "GPU verification failed with exit code $LASTEXITCODE"
}

Write-Host "[2/2] Inventorying TAO commands and anomaly-related content ..."
$Probe = @'
set +e
echo '=== IMAGE ==='
cat /etc/os-release 2>/dev/null
echo '=== TAO EXECUTABLE ==='
command -v tao
echo '=== TAO HELP ==='
tao --help 2>&1
python -m tao --help 2>&1
echo '=== INSTALLED PYTHON PACKAGES ==='
python -m pip list 2>/dev/null | grep -Ei 'tao|anomal|efficientad|patchcore|onnx'
echo '=== NVIDIA TAO PYTORCH TASK MODULES ==='
python - <<'PY'
import pkgutil
try:
    import nvidia_tao_pytorch
except Exception as exc:
    print(f'NVIDIA_TAO_IMPORT_ERROR: {exc}')
else:
    names = sorted(module.name for module in pkgutil.iter_modules(nvidia_tao_pytorch.__path__))
    for name in names:
        print(f'TAO_TASK_MODULE: {name}')
    candidates = [name for name in names if any(word in name.lower() for word in ('visual_changenet', 'changenet', 'anomal', 'efficientad', 'patchcore'))]
    for name in candidates:
        print(f'VISUAL_ANOMALY_CANDIDATE: {name}')
PY
echo '=== VISUAL-ANOMALY FILE CANDIDATES ==='
# PyTorch's torch/autograd/anomaly_mode.py is a NaN/gradient debugger, not an
# industrial visual-anomaly model. Exclude PyTorch internals and test fixtures.
find /opt /workspace /usr/local/lib/python* -maxdepth 7 \
  \( -iname '*visual_changenet*' -o -iname '*changenet*' -o -iname '*efficientad*' -o -iname '*patchcore*' -o -iname '*visual*anomal*' \) \
  ! -path '*/torch/*' ! -path '*/pytorch/test/*' -print 2>/dev/null | head -n 300
'@

$Result = docker run --rm --gpus all --shm-size=16g `
    --ulimit memlock=-1 --ulimit stack=67108864 `
    $Image /bin/bash -lc $Probe 2>&1
$Result | Tee-Object -FilePath $Output

$Report = $Result -join "`n"
$HasCandidate = $Report -match "VISUAL_ANOMALY_CANDIDATE:" -or `
    ($Report -match "=== VISUAL-ANOMALY FILE CANDIDATES ===\s*\r?\n\s*/")
if ($HasCandidate) {
    Write-Warning "A possible visual-anomaly component was found. Verify its official train/export documentation before use."
} else {
    Write-Warning "VERDICT: NO VISUAL-ANOMALY TRAINING TASK FOUND IN THIS IMAGE."
    Write-Warning "PyTorch autograd anomaly_mode files are numerical debugging utilities, not defect detection."
}

Write-Host ""
Write-Host "Probe saved to $((Resolve-Path $Output).Path)"
Write-Host "GPU access is only a prerequisite. Continue only if this report identifies"
Write-Host "a supported visual anomaly TRAIN and EXPORT task with a spatial map output."
