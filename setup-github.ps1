# GitHub Repository Setup Script
# This script will connect your local repository to GitHub

Write-Host "Setting up GitHub connection..." -ForegroundColor Cyan

# Check if git is available
try {
    $gitVersion = git --version 2>&1
    Write-Host "Git found: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Git is not installed or not in your PATH." -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "After installing Git, restart your terminal and run this script again." -ForegroundColor Yellow
    exit 1
}

# Check if repository is initialized
if (-not (Test-Path .git)) {
    Write-Host "Initializing git repository..." -ForegroundColor Cyan
    git init
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to initialize repository" -ForegroundColor Red
        exit 1
    }
    Write-Host "Repository initialized successfully!" -ForegroundColor Green
} else {
    Write-Host "Repository already initialized" -ForegroundColor Green
}

# Check if remote already exists
$remoteExists = git remote get-url origin 2>&1
if ($LASTEXITCODE -eq 0) {
    Write-Host "Remote 'origin' already exists: $remoteExists" -ForegroundColor Yellow
    $update = Read-Host "Do you want to update it to https://github.com/SohamAdgatla/CEP-Project? (y/n)"
    if ($update -eq 'y' -or $update -eq 'Y') {
        git remote set-url origin https://github.com/SohamAdgatla/CEP-Project
        Write-Host "Remote updated successfully!" -ForegroundColor Green
    }
} else {
    Write-Host "Adding GitHub remote..." -ForegroundColor Cyan
    git remote add origin https://github.com/SohamAdgatla/CEP-Project
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Remote added successfully!" -ForegroundColor Green
    } else {
        Write-Host "Failed to add remote" -ForegroundColor Red
        exit 1
    }
}

# Verify remote
Write-Host "`nVerifying remote configuration..." -ForegroundColor Cyan
git remote -v

Write-Host "`n✓ GitHub connection setup complete!" -ForegroundColor Green
Write-Host "`nNext steps:" -ForegroundColor Cyan
Write-Host "1. Add your files: git add ." -ForegroundColor White
Write-Host "2. Commit your changes: git commit -m 'Initial commit'" -ForegroundColor White
Write-Host "3. Push to GitHub: git push -u origin main" -ForegroundColor White
Write-Host "   (or 'git push -u origin master' if your default branch is master)" -ForegroundColor Gray
Write-Host "`nNote: You may need to authenticate with GitHub when pushing." -ForegroundColor Yellow
Write-Host "Consider using a Personal Access Token or GitHub CLI for authentication." -ForegroundColor Yellow

