$ErrorActionPreference = "Stop"
Set-Location -LiteralPath "D:\lukman-rmfs\Combinatrix"
& "D:\lukman-rmfs\.rmfs\Scripts\python.exe" "scripts\experiments\distributed_sensitivity_campaign.py" --manifest "data\runtime\distributed_sensitivity\sensitivity_full_kpi_v2_e7c1a5bb20e3\manifest.json" --machine-id "win_lukman" --run-continuously --resume --progress
