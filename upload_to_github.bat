@echo off
REM GitHub Upload Script for Cep-Project
REM Replace YOUR_USERNAME with your actual GitHub username

echo ========================================
echo GitHub Upload Script
echo ========================================
echo.

REM Check if remote already exists
git remote -v >nul 2>&1
if %errorlevel% == 0 (
    echo Remote repository already configured.
    echo.
    echo Current remotes:
    git remote -v
    echo.
    set /p push="Push to GitHub? (y/n): "
    if /i "%push%"=="y" (
        git push -u origin main
    )
) else (
    echo No remote repository configured.
    echo.
    echo Please follow these steps:
    echo 1. Create a repository named "Cep-Project" on GitHub
    echo 2. Copy the repository URL (https://github.com/YOUR_USERNAME/Cep-Project.git)
    echo 3. Run this command:
    echo    git remote add origin YOUR_REPOSITORY_URL
    echo 4. Then run this script again
    echo.
    set /p url="Or enter repository URL now: "
    if not "%url%"=="" (
        git remote add origin %url%
        echo Remote added!
        echo.
        set /p push="Push to GitHub now? (y/n): "
        if /i "%push%"=="y" (
            git push -u origin main
        )
    )
)

echo.
echo ========================================
pause

