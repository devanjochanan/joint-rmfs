$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "C:\Users\admin\Documents\Dewa's Sandbox\netlogo-rmfs"
& "C:\Users\admin\Documents\Dewa's Sandbox\torch-gpu\Scripts\python.exe" "scripts\experiments\distributed_sensitivity_campaign.py" --manifest "data\runtime\distributed_sensitivity\sensitivity_full_kpi_v2_e7c1a5bb20e3\manifest.json" --machine-id "win_admin" --run-continuously --resume --progress
