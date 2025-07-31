#!/usr/bin/env python3
"""
Security Verification Script
Checks that all environment variables are properly configured and secure.
"""

import os
import sys
from pathlib import Path

def check_gitignore():
    """Check that .gitignore properly excludes sensitive files."""
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        print("❌ .gitignore file not found!")
        return False
    
    gitignore_content = gitignore_path.read_text()
    required_patterns = [".env", ".env.local", "api_keys.py", "secrets.json"]
    
    missing_patterns = []
    for pattern in required_patterns:
        if pattern not in gitignore_content:
            missing_patterns.append(pattern)
    
    if missing_patterns:
        print(f"⚠️  .gitignore missing patterns: {missing_patterns}")
        return False
    
    print("✅ .gitignore properly configured")
    return True

def check_env_files():
    """Check environment file configuration."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_example.exists():
        print("❌ .env.example template not found!")
        return False
    
    if not env_file.exists():
        print("⚠️  .env file not found. Copy from .env.example and configure your API keys.")
        return False
    
    # Check if API keys are properly set
    env_content = env_file.read_text()
    if "your_openai_api_key_here" in env_content:
        print("⚠️  OpenAI API key not configured in .env")
        return False
    
    if "your_serpapi_key_here" in env_content:
        print("⚠️  SerpAPI key not configured in .env")
        return False
    
    print("✅ Environment files properly configured")
    return True

def check_frontend_config():
    """Check frontend environment configuration."""
    frontend_dir = Path("app")
    if not frontend_dir.exists():
        print("⚠️  Frontend app directory not found")
        return True  # Not critical
    
    frontend_env = frontend_dir / ".env.local"
    frontend_example = frontend_dir / ".env.example"
    
    if not frontend_example.exists():
        print("⚠️  Frontend .env.example not found")
        return False
    
    if frontend_env.exists():
        env_content = frontend_env.read_text()
        # Check that no API keys are exposed in frontend
        sensitive_patterns = ["OPENAI_API_KEY", "SERPAPI_API_KEY", "LANGCHAIN_API_KEY"]
        exposed_keys = [pattern for pattern in sensitive_patterns if pattern in env_content]
        
        if exposed_keys:
            print(f"🚨 SECURITY RISK: Sensitive keys found in frontend .env.local: {exposed_keys}")
            print("   Remove these keys from frontend environment files!")
            return False
    
    print("✅ Frontend environment secure")
    return True

def check_git_status():
    """Check if any sensitive files are staged for commit."""
    try:
        import subprocess
        result = subprocess.run(["git", "status", "--porcelain"], 
                              capture_output=True, text=True, cwd=".")
        
        if result.returncode != 0:
            print("⚠️  Not a git repository or git not available")
            return True
        
        staged_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
        sensitive_files = [f for f in staged_files if '.env' in f and not '.env.example' in f]
        
        if sensitive_files:
            print(f"🚨 SECURITY RISK: Sensitive files staged for commit: {sensitive_files}")
            print("   Use 'git reset' to unstage these files!")
            return False
            
        print("✅ No sensitive files staged for commit")
        return True
    except Exception as e:
        print(f"⚠️  Could not check git status: {e}")
        return True

def main():
    """Run all security checks."""
    print("🔐 Security Configuration Verification")
    print("=" * 40)
    
    checks = [
        check_gitignore,
        check_env_files,
        check_frontend_config,
        check_git_status,
    ]
    
    all_passed = True
    for check in checks:
        if not check():
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 All security checks passed!")
        print("Your API keys are properly configured and secure.")
    else:
        print("⚠️  Some security issues found. Please address them before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()
