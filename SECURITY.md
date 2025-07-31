# 🔐 API Keys and Security Setup

This guide explains how to securely configure API keys for your LangChain chatbot without exposing them in version control.

## 🚨 IMPORTANT SECURITY NOTICE

**NEVER commit API keys to version control!** All sensitive configuration is stored in `.env` files that are excluded from Git.

## 📋 Required API Keys

### 1. OpenAI API Key

- **Purpose**: Powers the main chat functionality
- **Get it from**: [OpenAI Platform](https://platform.openai.com/api-keys)
- **Cost**: Pay-per-use (check current pricing)

### 2. SerpAPI Key

- **Purpose**: Enables web search capabilities
- **Get it from**: [SerpAPI Dashboard](https://serpapi.com/dashboard)
- **Free tier**: 100 searches/month

### 3. LangChain API Key (Optional)

- **Purpose**: Enables tracing and monitoring
- **Get it from**: [LangSmith](https://smith.langchain.com/)
- **Free tier**: Available

## ⚙️ Setup Instructions

### Step 1: Copy Environment Templates

```bash
# For the backend
cp .env.example .env

# For the frontend
cd app
cp .env.example .env.local
```

### Step 2: Add Your API Keys

Edit the `.env` file and replace the placeholder values:

```bash
# Replace with your actual API keys
OPENAI_API_KEY=sk-your_actual_openai_key_here
SERPAPI_API_KEY=your_actual_serpapi_key_here
LANGCHAIN_API_KEY=ls__your_actual_langchain_key_here
```

### Step 3: Verify Configuration

Run the configuration validator:

```bash
python config.py
```

## 🔒 Security Best Practices

### ✅ What's Protected

- ✅ `.env` files are in `.gitignore`
- ✅ API keys are loaded from environment variables
- ✅ Configuration validation prevents invalid keys
- ✅ Safe logging (keys are masked in logs)

### ⚠️ Important Notes

- **Frontend**: Never put API keys in `NEXT_PUBLIC_` variables (they're exposed to browsers)
- **Backend**: All sensitive keys should only be in the backend `.env` file
- **Production**: Use proper secret management (Azure Key Vault, AWS Secrets Manager, etc.)

## 🚀 Deployment Security

### For Production Deployment:

1. **Environment Variables**: Set API keys as environment variables in your hosting platform
2. **Secret Management**: Use cloud secret management services
3. **HTTPS**: Always use HTTPS in production
4. **CORS**: Configure proper CORS origins

### Common Hosting Platforms:

- **Vercel**: Add environment variables in project settings
- **Netlify**: Configure in site settings > environment variables
- **Railway**: Set environment variables in project settings
- **Heroku**: Use `heroku config:set` command

## 🔍 Troubleshooting

### Configuration Issues:

```bash
# Check if environment variables are loaded
python -c "from config import config; print(config.get_safe_config())"

# Validate configuration
python config.py
```

### Common Errors:

- `OPENAI_API_KEY not set`: Check your `.env` file
- `Invalid API key`: Verify key format and permissions
- `Module not found`: Install missing dependencies with `pip install python-dotenv`

## 📁 File Structure

```
langchain-chatbot/
├── .env                 # Backend API keys (NOT in git)
├── .env.example         # Template for backend (safe to commit)
├── config.py           # Secure configuration loader
├── app/
│   ├── .env.local      # Frontend environment (NOT in git)
│   └── .env.example    # Template for frontend (safe to commit)
└── .gitignore          # Protects sensitive files
```

## 🆘 Need Help?

1. Check that all `.env` files exist and contain valid keys
2. Verify `.gitignore` includes environment files
3. Run the configuration validator
4. Check the application logs for specific error messages

Remember: **When in doubt, never commit API keys!** 🔐
