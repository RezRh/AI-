#!/usr/bin/env python
"""
Setup GitHub repository for the Multi-Model AI Assistant project.
This script helps create a new GitHub repository and push the code.
"""

import os
import subprocess
import sys

def run_command(command):
    """Run a shell command and print output."""
    print(f"Running: {command}")
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    if process.stdout:
        print(process.stdout)
    if process.stderr:
        print(f"Error: {process.stderr}")
    return process.returncode == 0

def setup_github_repo(repo_name):
    """Set up a new GitHub repository and push the code."""
    # Check if git is installed
    if not run_command("git --version"):
        print("Git is not installed. Please install Git first.")
        return False
    
    # Check if the current directory is a git repository
    if not os.path.exists(".git"):
        # Initialize git repository
        if not run_command("git init"):
            print("Failed to initialize git repository.")
            return False
    
    # Add all files to git
    if not run_command("git add ."):
        print("Failed to add files to git.")
        return False
    
    # Commit changes
    if not run_command('git commit -m "Initial commit of Multi-Model AI Assistant"'):
        print("Failed to commit files.")
        return False
    
    # Create GitHub repository using GitHub CLI if available
    if run_command("gh --version"):
        print("\nGitHub CLI detected. Creating repository...")
        if run_command(f"gh repo create {repo_name} --public --source=. --push"):
            print(f"\nSuccess! Repository created at: https://github.com/{repo_name}")
            return True
    
    # If GitHub CLI is not available or failed, provide manual instructions
    print("\nTo push to GitHub manually:")
    print("1. Create a new repository on GitHub: https://github.com/new")
    print(f"2. Name it: {repo_name}")
    print("3. Run the following commands:")
    print(f"   git remote add origin https://github.com/yourusername/{repo_name}.git")
    print("   git branch -M main")
    print("   git push -u origin main")
    
    return True

def main():
    """Main function."""
    print("===== GitHub Repository Setup for Multi-Model AI Assistant =====\n")
    
    # Get repository name
    default_name = "multi-model-ai-assistant"
    repo_name = input(f"Enter repository name (default: {default_name}): ").strip()
    if not repo_name:
        repo_name = default_name
    
    # Get GitHub username
    github_username = input("Enter your GitHub username: ").strip()
    if github_username:
        full_repo_name = f"{github_username}/{repo_name}"
    else:
        full_repo_name = repo_name
    
    # Confirm
    print(f"\nSetting up repository: {full_repo_name}")
    confirm = input("Continue? (y/n): ").lower()
    if confirm != 'y':
        print("Setup cancelled.")
        return
    
    # Setup repository
    if setup_github_repo(full_repo_name):
        print("\nRepository setup process completed.")
    else:
        print("\nRepository setup failed.")

if __name__ == "__main__":
    main()