# Setting Up Your GitHub Repository

This guide will help you push the Multi-Model AI Assistant project to your GitHub account.

## Prerequisites

- A GitHub account
- Git installed on your computer
- Optional: GitHub CLI installed (for easier setup)

## Option 1: Using the Setup Script (Recommended)

This project includes a setup script that automates the GitHub repository creation process.

1. Run the setup script:
   ```
   python setup_github.py
   ```

2. Follow the prompts to enter your GitHub username and repository name.

3. The script will:
   - Initialize a Git repository (if not already initialized)
   - Add all project files
   - Make an initial commit
   - Create a GitHub repository (if GitHub CLI is installed)
   - Provide instructions for manual setup if needed

## Option 2: Manual Setup

If you prefer to set up the repository manually:

1. Initialize a Git repository in the project folder:
   ```
   git init
   ```

2. Add all project files:
   ```
   git add .
   ```

3. Commit the files:
   ```
   git commit -m "Initial commit of Multi-Model AI Assistant"
   ```

4. Create a new repository on GitHub by visiting:
   https://github.com/new

5. Connect your local repository to GitHub:
   ```
   git remote add origin https://github.com/yourusername/your-repo-name.git
   git branch -M main
   git push -u origin main
   ```

## Important Files to Check Before Pushing

- Make sure your `.env` file (if you created one with API keys) is not being tracked by Git
- Verify that `config.json` contains default values and no personal API keys
- Check that the `.gitignore` file is properly set up to exclude sensitive files

## After Pushing to GitHub

1. Verify that all files were uploaded correctly
2. Add a description to your repository
3. Consider adding topics/tags to make your repository discoverable

## Troubleshooting

- If you encounter authentication issues, make sure you've set up authentication with GitHub (using SSH keys or a personal access token)
- If some files are missing, check the `.gitignore` file to ensure they're not being excluded
- For large files that fail to push, consider using Git LFS or removing them from the repository