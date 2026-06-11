$logFile = "C:\Users\vlche\dev\RAG-L-R-Assistant\logs\bm25_auto.log"
$python  = "C:\Python314\python.exe"
$script  = "C:\Users\vlche\dev\RAG-L-R-Assistant\scripts\build_bm25_index.py"

function Log($msg) {
    $ts = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    "[$ts] $msg" | Tee-Object -FilePath $logFile -Append
}

Log "Watcher started. Monitoring keyword index processes (PIDs 15940, 32932)."

while ($true) {
    $active = Get-Process -Id 15940,32932 -ErrorAction SilentlyContinue
    if (-not $active) {
        Log "Keyword index processes finished. Checking free RAM..."
        $mem = Get-CimInstance Win32_OperatingSystem
        $freeGB = [math]::Round($mem.FreePhysicalMemory / 1MB, 1)
        Log "Free RAM: ${freeGB} GB. Launching BM25 with --max-papers 1000000 ..."
        $bm25Log = "C:\Users\vlche\dev\RAG-L-R-Assistant\logs\build_bm25.log"
        Start-Process -FilePath $python `
            -ArgumentList "$script --max-papers 1000000" `
            -RedirectStandardOutput $bm25Log `
            -RedirectStandardError "C:\Users\vlche\dev\RAG-L-R-Assistant\logs\build_bm25_err.log" `
            -NoNewWindow -Wait
        Log "BM25 process finished. Check $bm25Log for results."
        break
    }
    $cpus = $active | ForEach-Object { "$($_.Id):CPU=$([math]::Round($_.CPU,0))" }
    Log "Still running: $($cpus -join ', '). Waiting 60s..."
    Start-Sleep -Seconds 60
}
