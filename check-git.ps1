# Quick Git Verification Script
Write-Host "Checking Git installation..." -ForegroundColor Cyan

try {
    $gitVersion = git --version 2>&1
    Write-Host "✓ Git is installed!" -ForegroundColor Green
    Write-Host "  Version: $gitVersion" -ForegroundColor Gray
    
    # Check if git config is set
    $userName = git config --global user.name 2>&1
    $userEmail = git config --global user.email 2>&1
    
    if ($userName -and $userEmail) {
        Write-Host "✓ Git is configured:" -ForegroundColor Green
        Write-Host "  Name: $userName" -ForegroundColor Gray
        Write-Host "  Email: $userEmail" -ForegroundColor Gray
    } else {
        Write-Host "⚠ Git is not configured yet." -ForegroundColor Yellow
        Write-Host "  Run these commands to configure:" -ForegroundColor Yellow
        Write-Host "  git config --global user.name `"Your Name`"" -ForegroundColor White
        Write-Host "  git config --global user.email `"your.email@example.com`"" -ForegroundColor White
    }
    
    Write-Host "`n✓ Ready to connect to GitHub!" -ForegroundColor Green
    Write-Host "  Run: .\setup-github.ps1" -ForegroundColor Cyan
    
} catch {
    Write-Host "✗ Git is not installed or not in PATH." -ForegroundColor Red
    Write-Host "  Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "  After installation, restart your terminal and run this script again." -ForegroundColor Yellow
}

