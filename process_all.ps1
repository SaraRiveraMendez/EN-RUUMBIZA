# process_all.ps1
# Recorre todo el dataset y genera los CSVs para cada .wav

$dataset = "C:\Users\rsara\OneDrive\Documents\car_diagnostics_dataset"
$exe     = ".\sliding_window_pc.exe"

$wavFiles = Get-ChildItem -Path $dataset -Recurse -Filter "*.wav"

$total   = $wavFiles.Count
$current = 0
$errors  = @()

Write-Host "`n======================================"
Write-Host "  Dataset  : $dataset"
Write-Host "  Archivos : $total"
Write-Host "======================================`n"

foreach ($file in $wavFiles) {
    $current++
    $label = $file.Directory.Name   # carpeta hoja = etiqueta

    Write-Host "[$current/$total] $label \ $($file.Name)"

    $result = & $exe $file.FullName $label 2>&1
    if ($LASTEXITCODE -ne 0) {
        $errors += "$($file.FullName)"
        Write-Host "  [FAIL]" -ForegroundColor Red
    } else {
        # Mostrar solo la línea de ventanas generadas
        $result | Select-String "Ventanas CSV" | ForEach-Object { Write-Host "  $_" }
    }
}

Write-Host "`n======================================"
Write-Host "  RESUMEN FINAL"
Write-Host "======================================"
Write-Host "  Procesados : $total"
Write-Host "  Errores    : $($errors.Count)"
if ($errors.Count -gt 0) {
    Write-Host "`n  Archivos con error:"
    $errors | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
}
Write-Host ""