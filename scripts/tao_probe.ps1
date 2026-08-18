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
echo '=== ANOMALY-RELATED FILES ==='
find /opt /workspace /usr/local/lib/python* -maxdepth 5 \
  \( -iname '*anomal*' -o -iname '*efficientad*' -o -iname '*patchcore*' \) \
  -print 2>/dev/null | head -n 300
'@

$Result = docker run --rm --gpus all --shm-size=16g `
    --ulimit memlock=-1 --ulimit stack=67108864 `
    $Image /bin/bash -lc $Probe 2>&1
$Result | Tee-Object -FilePath $Output

Write-Host ""
Write-Host "Probe saved to $((Resolve-Path $Output).Path)"
Write-Host "GPU access is only a prerequisite. Continue only if this report identifies"
Write-Host "a supported visual anomaly TRAIN and EXPORT task with a spatial map output."
