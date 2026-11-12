# This script builds and runs all the examples. It exits as soon as any example exits with a
# nonzero exit code. If the `/headless` argument is given, each example is run for at most 15
# seconds, after which point it moves on to the next example. Otherwise, human input is required to
# close all the windows.

$headless = $false

foreach ($arg in $args) {
    if ($arg -eq '/headless') {
        $headless = $true
    } else {
        Echo "${PSCommandPath}: unknown argument: $arg"
        Exit 1
    }
}

cd "$PSScriptRoot"
cargo build --bins

foreach ($example in Get-ChildItem -Directory -Name) {
    # Continue without printing anything if the example is excluded in the manifest.
    cargo build --bin $example 1>$null 2>$null
    if ($LastExitCode -ne 0) {
        continue;
    }

    $proc = Start-Process "cargo" "run --bin $example" -NoNewWindow -PassThru

    $timeouted = $null

    if ($headless) {
        $proc | Wait-Process -Timeout 15 -ErrorAction SilentlyContinue -ErrorVariable timeouted
    } else {
        $proc | Wait-Process
    }

    if ($timeouted) {
        $proc | Stop-Process
    } elseif ($proc.ExitCode -ne 0) {
        Exit $proc.ExitCode
    }
}

rm pipeline-caching\pipeline_cache.bin
