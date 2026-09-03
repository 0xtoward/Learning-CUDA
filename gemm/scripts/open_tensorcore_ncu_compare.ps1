$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$naive = Join-Path $repoRoot 'profile\tensor_fp16_naive_gui\tensor_fp16_naive_detailed.ncu-rep'
$tuned = Join-Path $repoRoot 'profile\tensor_fp16_cutlass_gui\tensor_fp16_cutlass_detailed.ncu-rep'

$uiCandidates = @(
    'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.1\host\windows-desktop-win7-x64\ncu-ui.exe',
    'C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.3.0\host\windows-desktop-win7-x64\ncu-ui.exe'
)
$ui = $uiCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $ui) {
    throw 'Could not find ncu-ui.exe. Open both .ncu-rep files manually in Nsight Compute.'
}
if (-not (Test-Path -LiteralPath $naive) -or -not (Test-Path -LiteralPath $tuned)) {
    throw 'One or both Tensor Core NCU reports are missing.'
}

Start-Process -FilePath $ui -ArgumentList @($naive, $tuned)
