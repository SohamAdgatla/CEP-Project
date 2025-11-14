#!/bin/bash
# GitHub Upload Script for Cep-Project
# Replace YOUR_USERNAME with your actual GitHub username

echo "========================================"
echo "GitHub Upload Script"
echo "========================================"
echo ""

# Check if remote already exists
if git remote -v | grep -q origin; then
    echo "Remote repository already configured."
    echo ""
    echo "Current remotes:"
    git remote -v
    echo ""
    read -p "Push to GitHub? (y/n): " push
    if [[ "$push" == "y" || "$push" == "Y" ]]; then
        git push -u origin main
    fi
else
    echo "No remote repository configured."
    echo ""
    echo "Please follow these steps:"
    echo "1. Create a repository named 'Cep-Project' on GitHub"
    echo "2. Copy the repository URL (https://github.com/YOUR_USERNAME/Cep-Project.git)"
    echo "3. Run this command:"
    echo "   git remote add origin YOUR_REPOSITORY_URL"
    echo "4. Then run this script again"
    echo ""
    read -p "Or enter repository URL now: " url
    if [ ! -z "$url" ]; then
        git remote add origin "$url"
        echo "Remote added!"
        echo ""
        read -p "Push to GitHub now? (y/n): " push
        if [[ "$push" == "y" || "$push" == "Y" ]]; then
            git push -u origin main
        fi
    fi
fi

echo ""
echo "========================================"

