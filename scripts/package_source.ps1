param(
    [string]$Destination = "dist"
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$versionFile = Join-Path $repositoryRoot "src/chromarecover/_version.py"
$versionMatch = Select-String -LiteralPath $versionFile -Pattern '__version__ = "([^"]+)"'
if (-not $versionMatch) {
    throw "Could not determine ChromaRecover version"
}
$version = $versionMatch.Matches[0].Groups[1].Value
$destinationPath = Join-Path $repositoryRoot $Destination
New-Item -ItemType Directory -Path $destinationPath -Force | Out-Null
$archivePath = Join-Path $destinationPath "chromarecover-$version-source.zip"

Push-Location -LiteralPath $repositoryRoot
try {
    $changes = & git status --porcelain --untracked-files=normal
    if ($LASTEXITCODE -ne 0) {
        throw "git status failed with exit code $LASTEXITCODE"
    }
    if ($changes) {
        throw "Commit or remove workspace changes before packaging; archives are built from HEAD"
    }
    & git archive --format=zip --output=$archivePath HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "git archive failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

Write-Output $archivePath
