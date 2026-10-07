param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$RunArgs,
    [string]$Title = ""
)
$root = $PSScriptRoot
$log = Join-Path $root "results_r02\logs\dashboard_$Name.log"
if (-not $Title) { $Title = "DyF-ZTL Vong 2 - $Name" }
$inner = "Set-Location '$root'; `$host.UI.RawUI.WindowTitle = '$Title'; " +
         "Write-Host 'Luu y: click chuot vao cua so se lam Windows tam dung hien thi (che do Select). Neu lo click, bam Esc.' -ForegroundColor Yellow; " +
         "python -u run_parallel.py $RunArgs --log '$log'"
Start-Process powershell.exe -ArgumentList '-NoExit', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', $inner
Write-Host "Da mo cua so '$Title'. Log bang tien do: $log"

