# 🤖 LangChain Chatbot

A modern, AI-powered chatbot application built with LangChain, OpenAI GPT, and Next.js. This project demonstrates the integration of advanced language models with web search capabilities and mathematical tools in a beautiful, responsive web interface.

## 🌟 What is this project?

This is a full-stack chatbot application that combines the power of LangChain's agent framework with OpenAI's GPT models to create an intelligent assistant capable of:

- Answering questions using AI reasoning
- Performing web searches for real-time information
- Executing mathematical calculations
- Providing streaming responses for better user experience

The project showcases modern web development practices with a FastAPI backend and a Next.js frontend, featuring a beautiful dark/light theme toggle and responsive design.

## ✨ Key Functionalities

### 🧠 **AI Capabilities**

- **Intelligent Conversations**: Powered by OpenAI GPT models (configurable)
- **Web Search Integration**: Real-time web search using SerpAPI
- **Mathematical Operations**: Built-in calculator functions (add, subtract, multiply, exponentiate)
- **Tool-based Reasoning**: LangChain agent that decides which tools to use
- **Streaming Responses**: Real-time response streaming for better UX

### 🎨 **User Interface**

- **Modern Design**: Beautiful gradient backgrounds with smooth transitions
- **Dark/Light Theme**: Toggle between elegant dark and warm light themes
- **Responsive Layout**: Works seamlessly on desktop and mobile devices
- **Markdown Support**: Rich text rendering for formatted responses
- **Real-time Typing**: See responses as they're generated

### 🔧 **Technical Features**

- **Async Processing**: Fast, non-blocking request handling
- **CORS Support**: Properly configured for frontend-backend communication
- **Environment Configuration**: Secure API key management
- **Error Handling**: Robust error handling and validation
- **Type Safety**: Full TypeScript support in frontend

## 🏗️ How it's Built

### **Backend Stack**

- **FastAPI**: Modern, fast web framework for building APIs
- **LangChain**: Framework for developing applications with language models
- **OpenAI API**: GPT models for natural language processing
- **SerpAPI**: Web search capabilities
- **Uvicorn**: ASGI server for running the FastAPI application
- **Pydantic**: Data validation and settings management
- **Python-dotenv**: Environment variable management

### **Frontend Stack**

- **Next.js 15**: React framework with App Router
- **React 19**: Latest React features and hooks
- **TypeScript**: Type-safe JavaScript development
- **Tailwind CSS**: Utility-first CSS framework
- **React Markdown**: Markdown rendering with GitHub Flavored Markdown

### **Development Tools**

- **Black**: Code formatting
- **Flake8**: Code linting
- **MyPy**: Static type checking
- **Pytest**: Testing framework

## 📁 Project Structure

```
langchain-chatbot/
├── api/                          # Backend API
│   ├── agent.py                  # LangChain agent implementation
│   └── main.py                   # FastAPI application
├── app/                          # Frontend Next.js application
│   ├── src/
│   │   ├── app/
│   │   │   ├── layout.tsx        # Root layout with theme provider
│   │   │   ├── page.tsx          # Main chat interface
│   │   │   └── globals.css       # Global styles
│   │   └── components/
│   │       ├── ThemeProvider.tsx # Theme context and management
│   │       ├── ThemeToggle.tsx   # Theme toggle component
│   │       ├── TextArea.tsx      # Chat input component
│   │       ├── Output.tsx        # Chat output component
│   │       └── MarkdownRenderer.tsx # Markdown rendering
│   ├── package.json              # Node.js dependencies
│   ├── tailwind.config.js        # Tailwind configuration
│   ├── next.config.ts           # Next.js configuration
│   └── tsconfig.json            # TypeScript configuration
├── config.py                     # Configuration management
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Python project configuration
└── README.md                    # Project documentation
```

## 🚀 How to Run This Project

### Prerequisites

- **Python 3.11+** installed on your system
- **Node.js 18+** and npm installed
- **OpenAI API Key** (get one from [OpenAI](https://platform.openai.com/api-keys))
- **SerpAPI Key** (optional, for web search - get from [SerpAPI](https://serpapi.com/))

### Step 1: Clone the Repository

```bash
git clone https://github.com/arnavkj11/langchain-chatbot.git
cd langchain-chatbot
```

### Step 2: Set Up Environment Variables

Create a `.env` file in the root directory:

```bash
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional (for web search functionality)
SERPAPI_API_KEY=your_serpapi_key_here

# Optional configurations
OPENAI_MODEL=gpt-4o-mini
OPENAI_TEMPERATURE=0.7
OPENAI_MAX_TOKENS=1000
HOST=localhost
PORT=8000
FRONTEND_URL=http://localhost:3000
```

### Step 3: Set Up the Backend

```bash
# Install Python dependencies
pip install -r requirements.txt

# Navigate to the API directory
cd api

# Run the FastAPI server
uvicorn main:app --reload
```

The backend will be available at `http://localhost:8000`

### Step 4: Set Up the Frontend

Open a new terminal window:

```bash
# Navigate to the app directory
cd app

# Install Node.js dependencies
npm install

# Run the Next.js development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

### Step 5: Start Chatting! 🎉

1. Open your browser and go to `http://localhost:3000`
2. Toggle between light and dark themes using the theme switcher
3. Start asking questions - try asking for calculations or current information!

## 🔑 API Endpoints

- `GET /` - Health check endpoint
- `POST /chat` - Main chat endpoint for streaming responses
- `POST /chat/simple` - Simple chat endpoint for non-streaming responses

## 🛠️ Development

### Running Tests

```bash
# Backend tests
pytest

# Frontend linting
cd app
npm run lint
```

### Code Formatting

```bash
# Python code formatting
black .
flake8 .

# TypeScript type checking
cd app
npm run build
```

## 🎯 Example Queries to Try

1. **Mathematical**: "What is 25 \* 43 + 156?"
2. **Web Search**: "What's the latest news about AI developments?"
3. **General**: "Explain quantum computing in simple terms"
4. **Complex**: "Search for the current stock price of Tesla and calculate what 100 shares would cost"

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the amazing framework
- [OpenAI](https://openai.com/) for the powerful language models
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- [Next.js](https://nextjs.org/) for the React framework
- [Tailwind CSS](https://tailwindcss.com/) for the utility-first CSS framework

---

**Happy Chatting!** 🚀✨
