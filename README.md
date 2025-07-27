# Healthcare Expert AI Voice Agent 🏥🤖

An advanced AI-powered voice agent specialized in healthcare information and medical advice. This system combines real-time speech recognition, retrieval-augmented generation (RAG), and natural text-to-speech to provide interactive healthcare consultations.

## ✨ Features

- **🏥 Healthcare Specialization**: Domain-specific AI trained for medical information and health advice
- **🎤 Real-time Voice Recognition**: Live audio streaming with AssemblyAI Universal-Streaming API v3
- **🧠 Enhanced RAG System**: Vector-based knowledge retrieval from healthcare documents
- **🔊 Natural Voice Synthesis**: High-quality text-to-speech with Cartesia AI
- **💾 Conversation Learning**: Persistent memory and user feedback integration
- **📊 Response Classification**: Intelligent routing between LLM, RAG, Memory, and hybrid responses
- **⚡ Low Latency**: Optimized for real-time voice interactions

## 🏗️ System Architecture

### Core Components

1. **Voice Processing Pipeline**
   - Real-time audio capture and streaming
   - Voice Activity Detection (VAD)
   - Live transcription with AssemblyAI

2. **AI Response Engine**
   - Azure OpenAI GPT-4 integration
   - Healthcare-focused system prompts
   - Response type classification (LLM/RAG/Memory/Hybrid)

3. **Knowledge Management**
   - Chroma vector database for healthcare documents
   - Conversation history and learning
   - User preference tracking

4. **Audio Output**
   - Cartesia AI for premium voice synthesis
   - Fallback to system TTS engines
   - Optimized for conversational flow

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- macOS (for system TTS fallback)
- Microphone and speakers
- Required API keys (see Setup section)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/healthcare-expert-ai.git
   cd healthcare-expert-ai
   ```

2. **Create virtual environment**
   ```bash
   python -m venv voice-agent
   source voice-agent/bin/activate  # On Windows: voice-agent\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (see Configuration section)
   ```

5. **Run the healthcare expert**
   ```bash
   python enhanced_rag_voice_agent.py
   ```

## ⚙️ Configuration

### Required API Keys

Create a `.env` file with the following configuration:

```env
# AssemblyAI (Speech Recognition)
ASSEMBLYAI_API_KEY=your_assemblyai_api_key

# Cartesia AI (Text-to-Speech)
CARTESIA_API_KEY=your_cartesia_api_key

# Azure OpenAI (LLM)
AZURE_OPENAI_API_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=your_azure_endpoint
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your_gpt4_deployment

# Azure OpenAI Embeddings (Optional)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=your_embedding_deployment
AZURE_OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

### API Key Setup

1. **AssemblyAI**: Get your API key from [AssemblyAI Console](https://www.assemblyai.com/)
2. **Cartesia AI**: Sign up at [Cartesia](https://cartesia.ai/) for voice synthesis
3. **Azure OpenAI**: Set up through [Azure Cognitive Services](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service)

## 📋 Usage

### Basic Operation

1. Start the application: `python enhanced_rag_voice_agent.py`
2. Wait for "Healthcare Expert AI ready! How can I help with your health questions?"
3. Speak your health-related questions
4. Listen to the AI's responses

### Example Interactions

- **Medical Information**: "What are the symptoms of diabetes?"
- **Health Advice**: "What should I know about high blood pressure?"
- **Medication Questions**: "Tell me about common side effects of aspirin"
- **Conversation Summary**: "What have we discussed so far?"

### Response Types

The system intelligently classifies and routes responses:

- **🧠 LLM**: Direct AI responses for math, general questions
- **📚 RAG**: Knowledge-base enhanced responses for medical queries
- **💭 Memory**: Conversation history and summaries
- **🔄 RAG+Memory**: Combined knowledge and conversation context

## 🗂️ Project Structure

```
healthcare-expert-ai/
├── enhanced_rag_voice_agent.py    # Main application
├── healthcare_domain_config.py    # Domain-specific configuration
├── conversation_memory.db         # SQLite conversation storage
├── chroma_db/                     # Vector database
├── inspect_conversation_db.py     # Database inspection tools
├── inspect_knowledge_base.py      # Knowledge base tools
├── clear_conversations.py         # Database management
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment template
└── README.md                     # This file
```

## 🛠️ Development

### Database Management

- **Inspect conversations**: `python inspect_conversation_db.py`
- **Clear conversation history**: `python clear_conversations.py`
- **Inspect knowledge base**: `python inspect_knowledge_base.py`

### Adding Healthcare Knowledge

```python
from enhanced_rag_voice_agent import EnhancedRAGVoiceAgent

agent = EnhancedRAGVoiceAgent()
documents = ["Your healthcare document content..."]
agent.add_knowledge_to_rag(documents)
```

### Customization

- Modify system prompts in `generate_enhanced_response()`
- Adjust response classification in `_is_math_or_general_question()`
- Update healthcare domain config in `healthcare_domain_config.py`

## 🎯 Performance Optimization

- **Low Latency**: Optimized for real-time voice interactions
- **Smart Caching**: Conversation memory and RAG result caching
- **Fallback Systems**: Multiple TTS and embedding providers
- **Resource Management**: Efficient thread and memory management

## 🔒 Security & Privacy

- **API Key Protection**: Environment variable configuration
- **Local Processing**: Conversation data stored locally
- **No Data Sharing**: Healthcare information remains private
- **Secure Connections**: HTTPS/WSS for all API communications

## 🚨 Important Notes

- **Medical Disclaimer**: This AI is for informational purposes only and should not replace professional medical advice
- **API Costs**: Monitor usage of paid APIs (AssemblyAI, Cartesia, Azure OpenAI)
- **Privacy**: Healthcare conversations are stored locally in SQLite database
- **Accuracy**: AI responses should be verified with healthcare professionals

## 📊 Features in Detail

### Voice Recognition
- Real-time audio streaming
- Voice Activity Detection (VAD)
- High-accuracy medical term recognition

### Knowledge Base
- Vector similarity search
- Healthcare document embedding
- Context-aware retrieval

### Conversation Learning
- User feedback integration
- Response time optimization
- Preference learning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [AssemblyAI](https://www.assemblyai.com/) for speech recognition
- [Cartesia AI](https://cartesia.ai/) for voice synthesis
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/cognitive-services/openai-service) for language models
- [LangChain](https://langchain.com/) for AI application framework
- [Chroma](https://www.trychroma.com/) for vector database

## 📞 Support

For questions or support, please open an issue on GitHub or contact the maintainers.

---

**⚠️ Medical Disclaimer**: This AI healthcare expert is designed for informational and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
