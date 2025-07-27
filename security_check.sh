#!/bin/bash
# Security check script for GitHub repository preparation

echo "🔒 Security Check for Healthcare Expert AI Repository"
echo "=================================================="

# Check for actual API keys in tracked files (look for specific patterns)
echo "🔍 Checking for actual API keys in tracked files..."
suspicious_found=false

# Check for Cartesia API keys
if git ls-files | xargs grep "sk_car_" 2>/dev/null | grep -v ".env.example"; then
    echo "❌ Found Cartesia API key!"
    suspicious_found=true
fi

# Check for AssemblyAI keys (32+ alphanumeric)
if git ls-files | xargs grep -E "[a-f0-9]{32}" 2>/dev/null | grep -v ".env.example" | grep -v "voice_id"; then
    echo "❌ Found possible AssemblyAI API key!"
    suspicious_found=true
fi

# Check for Azure OpenAI keys (long alphanumeric strings)
if git ls-files | xargs grep -E "[A-Za-z0-9]{40,}" 2>/dev/null | grep -v ".env.example" | grep -v "a0e99841-438c-4a64-b679-ae501e7d6091"; then
    echo "❌ Found possible Azure API key!"
    suspicious_found=true
fi

if [ "$suspicious_found" = true ]; then
    echo "❌ FOUND POSSIBLE API KEYS IN TRACKED FILES! Review and remove them before pushing."
    exit 1
else
    echo "✅ No actual API keys found in tracked files"
fi

# Check if .env is properly ignored
echo "🔍 Checking if .env is ignored..."
if git check-ignore .env >/dev/null 2>&1; then
    echo "✅ .env file is properly ignored"
else
    echo "❌ .env file is NOT ignored! Add to .gitignore"
    exit 1
fi

# Check if database files are ignored
echo "🔍 Checking if database files are ignored..."
if git check-ignore conversation_memory.db >/dev/null 2>&1; then
    echo "✅ Database files are properly ignored"
else
    echo "❌ Database files are NOT ignored! Add to .gitignore"
    exit 1
fi

# Check if virtual environment is ignored
echo "🔍 Checking if virtual environment is ignored..."
if git check-ignore voice-agent/ >/dev/null 2>&1; then
    echo "✅ Virtual environment is properly ignored"
else
    echo "❌ Virtual environment is NOT ignored! Add to .gitignore"
    exit 1
fi

# Check for required files
echo "🔍 Checking for required repository files..."
required_files=("README.md" "LICENSE" ".gitignore" ".env.example" "requirements.txt")
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file exists"
    else
        echo "❌ $file is missing!"
        exit 1
    fi
done

echo ""
echo "🎉 All security checks passed! Repository is ready for GitHub."
echo "📋 Next steps:"
echo "   1. git commit -m 'Initial commit: Healthcare Expert AI Voice Agent'"
echo "   2. Create repository on GitHub"
echo "   3. git remote add origin <your-repo-url>"
echo "   4. git push -u origin main"
echo ""
echo "⚠️  REMEMBER: Users must create their own .env file with API keys!"
