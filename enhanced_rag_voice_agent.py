import asyncio
import threading  # Enhanced thread management with global stop_event pattern
import queue
import sounddevice as sd
import numpy as np
import webrtcvad
from dotenv import load_dotenv
import assemblyai as aai
from cartesia import Cartesia
import pyttsx3
import io
import wave
import tempfile
import websocket
import json
import os
import sqlite3
import uuid
import time
import subprocess
import shlex
import base64
import re  # For regex pattern matching in question classification
from urllib.parse import urlencode
from langchain.agents import initialize_agent, AgentType
from langchain.tools import BaseTool, Tool
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.callbacks.manager import get_openai_callback
from datetime import datetime
import logging
import pickle
from typing import List, Dict, Any

load_dotenv()

# Enhanced RAG and Learning Components
class ConversationLearner:
    """Learns from conversation patterns and user feedback"""
    
    def __init__(self, db_path="conversation_memory.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for conversation learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversation history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                timestamp DATETIME,
                user_input TEXT,
                agent_response TEXT,
                user_feedback TEXT,
                response_time_ms REAL,
                context_used TEXT
            )
        ''')
        
        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                preference_type TEXT,
                preference_value TEXT,
                confidence_score REAL,
                last_updated DATETIME
            )
        ''')
        
        # Common patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_patterns (
                pattern_type TEXT,
                pattern_data BLOB,
                frequency INTEGER,
                last_seen DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_conversation(self, user_input: str, agent_response: str, 
                        response_time: float, context_used: str = ""):
        """Log conversation for learning"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        conversation_id = str(uuid.uuid4())
        cursor.execute('''
            INSERT INTO conversations 
            (id, timestamp, user_input, agent_response, response_time_ms, context_used)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (conversation_id, datetime.now(), user_input, agent_response, 
              response_time, context_used))
        
        conn.commit()
        conn.close()
        return conversation_id
    
    def learn_from_feedback(self, conversation_id: str, feedback: str):
        """Learn from user feedback"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE conversations 
            SET user_feedback = ? 
            WHERE id = ?
        ''', (feedback, conversation_id))
        
        conn.commit()
        conn.close()
    
    def get_conversation_summary(self, last_n=5) -> str:
        """Get summary of recent conversations for context"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_input, agent_response, timestamp 
            FROM conversations 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (last_n,))
        
        conversations = cursor.fetchall()
        conn.close()
        
        if conversations:
            summary = "Recent conversation context:\n"
            for user_input, agent_response, timestamp in reversed(conversations):
                summary += f"User: {user_input}\nHealthcare Expert: {agent_response}\n\n"
            return summary.strip()
        return ""

    def detect_feedback(self, user_input: str) -> str:
        """Detect feedback phrases in user input"""
        positive_feedback = ['good', 'helpful', 'great', 'excellent', 'perfect', 'thanks', 'thank you']
        negative_feedback = ['bad', 'wrong', 'incorrect', 'not helpful', 'useless', 'terrible']
        
        user_lower = user_input.lower()
        
        for phrase in positive_feedback:
            if phrase in user_lower:
                return "positive"
        
        for phrase in negative_feedback:
            if phrase in user_lower:
                return "negative"
                
        return "none"

    def get_conversation_insights(self) -> Dict[str, Any]:
        """Get insights from conversation history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Average response time
        cursor.execute('SELECT AVG(response_time_ms) FROM conversations')
        avg_response_time = cursor.fetchone()[0] or 0
        
        # Most common topics (simplified)
        cursor.execute('SELECT user_input FROM conversations ORDER BY timestamp DESC LIMIT 50')
        recent_inputs = [row[0] for row in cursor.fetchall()]
        
        # Response quality (based on feedback)
        cursor.execute('''
            SELECT user_feedback, COUNT(*) 
            FROM conversations 
            WHERE user_feedback IS NOT NULL 
            GROUP BY user_feedback
        ''')
        feedback_stats = dict(cursor.fetchall())
        
        conn.close()
        
        return {
            "avg_response_time_ms": avg_response_time,
            "total_conversations": len(recent_inputs),
            "feedback_stats": feedback_stats,
            "recent_topics": recent_inputs[:10]
        }

class RAGKnowledgeBase:
    """RAG (Retrieval-Augmented Generation) Knowledge Base"""
    
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        
        # Try Azure OpenAI embeddings first, with fallback options
        self.embeddings = self._initialize_embeddings()
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vectorstore = None
        self.retriever = None
        self.init_vectorstore()
    
    def _initialize_embeddings(self):
        """Initialize embeddings with fallback options"""
        
        # Try Azure OpenAI embeddings first
        try:
            from langchain_openai import AzureOpenAIEmbeddings
            
            # Try with a dedicated embedding deployment first
            embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
            if embedding_deployment:
                print(f"🔧 Trying Azure embedding deployment: {embedding_deployment}")
                embeddings = AzureOpenAIEmbeddings(
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                    deployment=embedding_deployment,
                    model=os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
                )
                # Test with a simple query
                test_embedding = embeddings.embed_query("test")
                print("✅ Azure OpenAI embeddings initialized successfully")
                return embeddings
            else:
                print("⚠️ No AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME specified")
        except Exception as e:
            print(f"⚠️ Azure OpenAI embeddings failed: {e}")
        
        # Fallback 1: Try OpenAI directly (if you have an OpenAI API key)
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from langchain_community.embeddings import OpenAIEmbeddings
                print("🔄 Falling back to direct OpenAI embeddings...")
                embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
                # Test with a simple query
                test_embedding = embeddings.embed_query("test")
                print("✅ OpenAI embeddings initialized successfully")
                return embeddings
            else:
                print("⚠️ No OPENAI_API_KEY found for fallback")
        except Exception as e:
            print(f"⚠️ OpenAI embeddings fallback failed: {e}")
        
        # Fallback 2: Use sentence transformers (local)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            print("🔄 Falling back to local HuggingFace embeddings...")
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",  # Small, fast model
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            # Test with a simple query
            test_embedding = embeddings.embed_query("test")
            print("✅ Local HuggingFace embeddings initialized successfully")
            return embeddings
        except Exception as e:
            print(f"⚠️ HuggingFace embeddings fallback failed: {e}")
            # Try the older import as fallback
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                print("🔄 Trying legacy HuggingFace embeddings...")
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'}
                )
                test_embedding = embeddings.embed_query("test")
                print("✅ Legacy HuggingFace embeddings initialized successfully")
                return embeddings
            except Exception as legacy_error:
                print(f"⚠️ Legacy HuggingFace embeddings also failed: {legacy_error}")
                
        # Fallback 3: Create a simple local embedding using TF-IDF
        try:
            print("🔄 Creating simple local embeddings as final fallback...")
            from langchain_community.embeddings import FakeEmbeddings
            embeddings = FakeEmbeddings(size=384)  # 384-dimensional embeddings
            test_embedding = embeddings.embed_query("test")
            print("✅ Simple local embeddings initialized (limited functionality)")
            return embeddings
        except Exception as fake_error:
            print(f"⚠️ Simple local embeddings also failed: {fake_error}")
        
        # Final fallback: Disable RAG features
        print("❌ All embedding options failed - RAG features will be disabled")
        return None
    
    def init_vectorstore(self):
        """Initialize or load existing vector store"""
        if not self.embeddings:
            print("⚠️ No embeddings available - RAG features disabled")
            return
            
        try:
            # Try to load existing vectorstore
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            print("✅ Loaded existing RAG knowledge base")
        except Exception as e:
            print(f"⚠️ No existing knowledge base found, creating new one: {e}")
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
                print("✅ Created new RAG knowledge base")
            except Exception as create_error:
                print(f"❌ Failed to create knowledge base: {create_error}")
                self.vectorstore = None
                self.retriever = None
    
    def add_documents(self, documents: List[str], metadatas: List[Dict] = None):
        """Add documents to the knowledge base"""
        if self.embeddings is None or self.vectorstore is None:
            print("⚠️ RAG not available - skipping document addition")
            return
            
        if not documents:
            return
            
        # Split documents into chunks
        docs = []
        for i, doc in enumerate(documents):
            chunks = self.text_splitter.split_text(doc)
            for chunk in chunks:
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                docs.append(Document(page_content=chunk, metadata=metadata))
        
        # Add to vectorstore
        try:
            self.vectorstore.add_documents(docs)
            print(f"✅ Added {len(docs)} document chunks to knowledge base")
        except Exception as e:
            print(f"❌ Failed to add documents: {e}")
    
    def search_knowledge(self, query: str, k: int = 3) -> List[Document]:
        """Search knowledge base for relevant information"""
        if not self.retriever:
            return []
        
        try:
            docs = self.retriever.get_relevant_documents(query)
            return docs[:k]
        except Exception as e:
            print(f"❌ Error searching knowledge base: {e}")
            return []
    
    def get_context_for_query(self, query: str) -> str:
        """Get relevant context for a query"""
        if not self.embeddings:
            return ""  # No RAG available
            
        docs = self.search_knowledge(query)
        if not docs:
            return ""
        
        context_parts = []
        for doc in docs:
            context_parts.append(doc.page_content)
        
        return "\n\n".join(context_parts)

# Enhanced timing instrumentation
class LatencyTracker:
    def __init__(self, verbose=False):
        self.timings = {}
        self.conversation_start = None
        self.verbose = verbose  # Control detailed logging
        
    def start_timing(self, event_name):
        """Start timing an event"""
        self.timings[event_name] = time.time()
        
    def end_timing(self, event_name, description=""):
        """End timing an event and log the duration"""
        if event_name in self.timings:
            duration = (time.time() - self.timings[event_name]) * 1000  # Convert to ms
            if self.verbose:
                print(f"⏱️  {event_name}: {duration:.1f}ms {description}")
            return duration
        return None
    
    def log_milestone(self, milestone):
        """Log a milestone with timestamp"""
        if not self.conversation_start:
            self.conversation_start = time.time()
        elapsed = (time.time() - self.conversation_start) * 1000
        if self.verbose:
            print(f"📍 {milestone} (T+{elapsed:.1f}ms)")
    
    def log_integration_point(self, stage, message):
        """Log key integration points always (simplified logging)"""
        if not self.conversation_start:
            self.conversation_start = time.time()
        elapsed = (time.time() - self.conversation_start) * 1000
        print(f"{stage} {message} (T+{elapsed:.0f}ms)")

class EnhancedRAGVoiceAgent:
    def __init__(self, verbose_logging=False):
        # Initialize latency tracker with logging control
        self.latency_tracker = LatencyTracker(verbose=verbose_logging)
        self.verbose_logging = verbose_logging
        
        # Initialize RAG and Learning components
        self.knowledge_base = RAGKnowledgeBase()
        self.conversation_learner = ConversationLearner()
        
        # AssemblyAI Universal-Streaming API v3 configuration (official format)
        self.assemblyai_api_key = os.getenv("ASSEMBLYAI_API_KEY")
        self.CONNECTION_PARAMS = {
            "sample_rate": 16000,
            "format_turns": True,
        }
        self.API_ENDPOINT_BASE_URL = "wss://streaming.assemblyai.com/v3/ws"
        self.streaming_endpoint = f"{self.API_ENDPOINT_BASE_URL}?{urlencode(self.CONNECTION_PARAMS)}"
        
        # Cartesia TTS
        self.cartesia = Cartesia(api_key=os.getenv("CARTESIA_API_KEY"))
        
        # Audio settings
        self.sample_rate = 16000
        self.chunk_duration = 0.05
        self.chunk_size = int(self.sample_rate * self.chunk_duration)
        
        # Voice Activity Detection
        self.vad = webrtcvad.Vad(1)
        
        # Audio buffers and state
        self.audio_queue = queue.Queue()
        self.transcription_queue = queue.Queue()
        self.is_recording = False
        self.is_speaking = False
        self.silence_threshold = 15
        self.silence_count = 0
        
        # WebSocket streaming state
        self.ws_app = None
        self.ws_connected = False
        self.stream_active = False
        self.stop_streaming = threading.Event()
        
        # Enhanced thread management for better resource control
        # Global variables for better resource management (following AssemblyAI example)
        self.stop_event = threading.Event()  # Global stop event for coordinated shutdown
        self.audio_thread = None  # Main audio streaming thread reference
        
        # AssemblyAI pattern: global stream, audio resource references
        self.stream = None       # Audio stream reference for robust cleanup
        self.audio = None        # Audio device reference for robust cleanup
        
        # Thread management (existing threads for compatibility)
        self.stream_thread = None
        self.ws_thread = None
        self.current_audio_stream = None
        
        # TTS Engine (fallback)
        self.tts_engine = pyttsx3.init()
        self.tts_engine.setProperty('rate', 180)
        self.tts_engine.setProperty('volume', 0.9)
        
        # Enhanced LangChain setup with RAG
        from langchain_openai import AzureChatOpenAI
        
        self.llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            # temperature removed - not supported by this Azure model
            model_kwargs={"stop": None}
        )
        
        # Enhanced memory with summarization
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            memory_key="chat_history",
            return_messages=True,
            max_token_limit=2000,  # Prevent memory from growing too large
        )
        
        # Initialize agent with enhanced tools
        self.agent = initialize_agent(
            tools=self.get_enhanced_tools(),
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            memory=self.memory,
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="generate"
        )
        
        # Conversation state
        self.conversation_active = False
        self.current_conversation_id = None
    
    def get_enhanced_tools(self):
        """Define enhanced tools with RAG capabilities"""
        
        class RAGSearchTool(BaseTool):
            name: str = "knowledge_search"
            description: str = "Search the knowledge base for relevant information about any topic"
            knowledge_base: RAGKnowledgeBase = None
            
            def __init__(self, knowledge_base):
                super().__init__()
                self.knowledge_base = knowledge_base
            
            def _run(self, query: str) -> str:
                context = self.knowledge_base.get_context_for_query(query)
                if context:
                    return f"Knowledge base context: {context}"
                return "No relevant information found in knowledge base."
        
        class ConversationInsightsTool(BaseTool):
            name: str = "conversation_insights"
            description: str = "Get insights from previous conversations and user patterns"
            learner: ConversationLearner = None
            
            def __init__(self, learner):
                super().__init__()
                self.learner = learner
            
            def _run(self, query: str) -> str:
                insights = self.learner.get_conversation_insights()
                return f"Conversation insights: {json.dumps(insights, indent=2)}"
        
        class TimeTool(BaseTool):
            name: str = "get_current_time"
            description: str = "Get the current time and date"
            
            def _run(self, query: str) -> str:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                return f"The current time is {current_time}"
        
        class WeatherTool(BaseTool):
            name: str = "get_weather"
            description: str = "Get weather information (placeholder - implement with real API)"
            
            def _run(self, query: str) -> str:
                return "I don't have access to real-time weather data, but you can check your local weather app or website."
        
        # Initialize tools with dependencies
        rag_tool = RAGSearchTool(self.knowledge_base)
        insights_tool = ConversationInsightsTool(self.conversation_learner)
        
        return [rag_tool, insights_tool, TimeTool(), WeatherTool()]
    
    def add_knowledge_to_rag(self, documents: List[str], metadatas: List[Dict] = None):
        """Add documents to RAG knowledge base"""
        self.knowledge_base.add_documents(documents, metadatas)
    
    # --- WebSocket Event Handlers for AssemblyAI Streaming ---
    
    def on_ws_open(self, ws):
        """Enhanced WebSocket open handler with improved thread management."""
        print("🔗 AssemblyAI WebSocket connection opened")
        print(f"📡 Connected to: {self.streaming_endpoint}")
        self.ws_connected = True
        self.stream_active = True
        
        # Enhanced audio streaming function following AssemblyAI example pattern
        def stream_audio():
            """Enhanced audio streaming with better error handling using global stop_event"""
            print("🎤 Starting real-time audio streaming with enhanced thread management...")
            
            # FRAMES_PER_BUFFER = 800  # 50ms of audio (0.05s * 16000Hz)
            # SAMPLE_RATE = CONNECTION_PARAMS["sample_rate"] = 16000
            # CHANNELS = 1
            # FORMAT = pyaudio.paInt16 (equivalent to int16)
            
            while not self.stop_event.is_set():  # Use global stop_event for coordination
                try:
                    # Get audio from queue with timeout to allow clean shutdown
                    if not self.audio_queue.empty():
                        audio_chunk = self.audio_queue.get_nowait()
                        if audio_chunk is not None:
                            # Convert numpy to int16 bytes (FORMAT = pyaudio.paInt16)
                            audio_int16 = (audio_chunk * 32767).astype(np.int16)
                            audio_bytes = audio_int16.tobytes()
                            # Send as binary message (50ms chunks for optimal streaming)
                            ws.send(audio_bytes, websocket.ABNF.OPCODE_BINARY)
                    else:
                        time.sleep(0.01)  # Small delay to prevent busy waiting
                except websocket.WebSocketConnectionClosedException:
                    print("🔌 WebSocket connection closed during streaming")
                    break
                except Exception as e:
                    print(f"❌ Error streaming audio: {e}")
                    break
            print("🔇 Enhanced audio streaming stopped")
        
        # Enhanced thread creation with global reference management
        # global audio_thread (following AssemblyAI example pattern)
        self.audio_thread = threading.Thread(target=stream_audio, name="EnhancedAudioStreamThread")
        self.audio_thread.daemon = True
        self.audio_thread.start()
        print("✅ Enhanced audio thread started with global stop_event coordination")
    
    def on_ws_message(self, ws, message):
        """Enhanced message handling following AssemblyAI Universal-Streaming API v3 example pattern"""
        try:
            data = json.loads(message)
            msg_type = data.get('type')

            if msg_type == "Begin":
                session_id = data.get('id')
                expires_at = data.get('expires_at')
                # Enhanced session begin handling with timestamp conversion (AssemblyAI example pattern)
                if expires_at:
                    expiry_time = datetime.fromtimestamp(expires_at)
                    print(f"\n🚀 Session began: ID={session_id}, ExpiresAt={expiry_time}")
                else:
                    print(f"\n🚀 Session began: ID={session_id}")
                    
            elif msg_type == "Turn":
                transcript = data.get('transcript', '')
                formatted = data.get('turn_is_formatted', False)
                
                # Enhanced transcript handling following AssemblyAI example pattern
                if formatted:
                    # Clear partial transcript line and show final (exact AssemblyAI pattern)
                    print('\r' + ' ' * 80 + '\r', end='')
                    print(transcript)
                    
                    # Enhanced final transcript processing with latency tracking
                    if transcript.strip():
                        self.latency_tracker.end_timing("speech_end_to_final", "- Speech end to final transcript")
                        self.latency_tracker.start_timing("transcript_to_ai_response")
                        self.latency_tracker.log_integration_point("📝", "Text received")
                        self.transcription_queue.put(transcript)
                else:
                    # Show partial transcript in real-time (simplified)
                    if self.verbose_logging:
                        print(f"\r{transcript}", end='')
                    
                    # Track latency for first partial transcript
                    if "speech_to_partial" in self.latency_tracker.timings:
                        self.latency_tracker.end_timing("speech_to_partial", "- Speech start to first partial")
                        if self.verbose_logging:
                            self.latency_tracker.log_milestone("🎧 First partial transcript received")
                        
            elif msg_type == "Termination":
                audio_duration = data.get('audio_duration_seconds', 0)
                session_duration = data.get('session_duration_seconds', 0)
                # Enhanced termination handling (exact AssemblyAI example pattern)
                print(f"\n🔚 Session Terminated: Audio Duration={audio_duration}s, Session Duration={session_duration}s")
                
        except json.JSONDecodeError as e:
            # Enhanced error handling following AssemblyAI example pattern
            print(f"❌ Error decoding message: {e}")
        except Exception as e:
            # Enhanced error handling following AssemblyAI example pattern
            print(f"❌ Error handling message: {e}")
    
    def on_ws_error(self, ws, error):
        """Robust error handling following AssemblyAI example pattern"""
        print(f"\nWebSocket Error: {error}")
        
        # Enhanced error coordination following AssemblyAI pattern
        # stop_event.set()  # Signal threads to stop on error
        self.stop_event.set()      # Global stop_event for coordinated shutdown
        self.stop_streaming.set()  # Legacy compatibility
        
        # Enhanced state management on error
        self.ws_connected = False
        self.stream_active = False
        
        # Trigger immediate cleanup on error
        try:
            self.conversation_active = False
        except Exception as cleanup_error:
            print(f"⚠️ Error during emergency cleanup: {cleanup_error}")

    def on_ws_close(self, ws, close_status_code, close_msg):
        """Robust resource management following AssemblyAI example pattern"""
        print(f"\nWebSocket Disconnected: Status={close_status_code}, Msg={close_msg}")
        
        # AssemblyAI pattern: global stream, audio
        # stop_event.set()  # Signal audio thread to stop
        self.stop_event.set()      # Signal audio thread to stop (global stop_event pattern)
        self.stop_streaming.set()  # Legacy compatibility
        
        # Enhanced state management
        self.ws_connected = False
        self.stream_active = False
        
        # Robust audio resource cleanup following exact AssemblyAI pattern
        # if stream: if stream.is_active(): stream.stop_stream(); stream.close(); stream = None
        if self.stream:
            try:
                if hasattr(self.stream, 'is_active') and self.stream.is_active():
                    self.stream.stop_stream()
                if hasattr(self.stream, 'close'):
                    self.stream.close()
                self.stream = None
                print("✅ Global stream cleaned up successfully")
            except Exception as stream_error:
                print(f"⚠️ Global stream cleanup warning: {stream_error}")
        
        # if audio: audio.terminate(); audio = None
        if self.audio:
            try:
                if hasattr(self.audio, 'terminate'):
                    self.audio.terminate()
                self.audio = None
                print("✅ Global audio cleaned up successfully")
            except Exception as audio_error:
                print(f"⚠️ Global audio cleanup warning: {audio_error}")
        
        # Additional current stream cleanup (existing pattern)
        if hasattr(self, 'current_audio_stream') and self.current_audio_stream:
            try:
                if hasattr(self.current_audio_stream, 'is_active') and self.current_audio_stream.is_active():
                    self.current_audio_stream.stop_stream()
                if hasattr(self.current_audio_stream, 'close'):
                    self.current_audio_stream.close()
                self.current_audio_stream = None
                print("✅ Current audio stream cleaned up successfully")
            except Exception as stream_error:
                print(f"⚠️ Current audio stream cleanup warning: {stream_error}")
        
        # Ensure audio thread cleanup following exact AssemblyAI pattern
        # if audio_thread and audio_thread.is_alive(): audio_thread.join(timeout=1.0)
        cleanup_threads = [
            ("Stream Thread", self.stream_thread),
            ("Audio Thread", self.audio_thread)  # AssemblyAI pattern: audio_thread reference
        ]
        
        for thread_name, thread in cleanup_threads:
            if thread and thread.is_alive():
                print(f"🔄 Cleaning up {thread_name}...")
                thread.join(timeout=1.0)  # AssemblyAI timeout pattern
                if thread.is_alive():
                    print(f"⚠️ {thread_name} cleanup timeout")
                else:
                    print(f"✅ {thread_name} cleaned up successfully")
        
        # Enhanced resource cleanup
        try:
            self.conversation_active = False
            self._clear_audio_resources()
        except Exception as e:
            print(f"❌ Error during WebSocket close cleanup: {e}")
    
    def _clear_audio_resources(self):
        """Enhanced audio resource cleanup helper"""
        try:
            # Clear audio queue
            cleared_audio = 0
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                    cleared_audio += 1
                except:
                    break
            
            # Clear transcription queue  
            cleared_transcripts = 0
            while not self.transcription_queue.empty():
                try:
                    self.transcription_queue.get_nowait()
                    cleared_transcripts += 1
                except:
                    break
            
            if cleared_audio > 0 or cleared_transcripts > 0:
                print(f"🗑️ Cleared {cleared_audio} audio buffers, {cleared_transcripts} transcripts")
                
        except Exception as e:
            print(f"⚠️ Audio resource cleanup warning: {e}")
    
    def audio_callback(self, indata, frames, time, status):
        """Callback for audio input - optimized for streaming with latency tracking"""
        if status:
            print(f"⚠️ Audio input error: {status}")
        
        try:
            # Convert to bytes for VAD (single channel)
            audio_mono = indata[:, 0]
            audio_data = (audio_mono * 32767).astype(np.int16).tobytes()
            
            # Voice Activity Detection - ensure frame size is valid for VAD
            # WebRTC VAD expects specific frame sizes (10ms, 20ms, 30ms at 16kHz)
            # Our chunk_size (800 frames) = 50ms, so we'll use it as-is
            try:
                is_speech = self.vad.is_speech(audio_data, self.sample_rate)
            except Exception as vad_error:
                # If VAD fails, assume it's speech to be safe
                is_speech = True
                if frames != self.chunk_size:
                    # Only log VAD errors occasionally to avoid spam
                    pass
            
            if is_speech:
                # Always send audio when speech is detected (for streaming)
                if self.ws_connected and not self.is_speaking:
                    self.audio_queue.put(audio_mono)
                
                self.silence_count = 0
                if not self.is_recording:
                    self.is_recording = True
                    self.latency_tracker.start_timing("speech_to_partial")
                    self.latency_tracker.log_integration_point("🎤", "Speech detected")
                    print("\n🎤 Speech detected - streaming...")
            else:
                if self.is_recording:
                    self.silence_count += 1
                    # Still send some audio during short silences
                    if self.silence_count < self.silence_threshold:
                        if self.ws_connected and not self.is_speaking:
                            self.audio_queue.put(audio_mono)
                    else:
                        self.is_recording = False
                        self.latency_tracker.start_timing("speech_end_to_final")
                        self.latency_tracker.log_milestone("🔇 Speech detection ended")
                        print("\n🔇 Speech ended")
                        # Could send a pause signal here if needed
                        
        except Exception as e:
            print(f"❌ Audio callback error: {e}")
    
    async def close_streaming_transcription(self):
        """Close the streaming transcription connection"""
        try:
            print("🔌 Closing AssemblyAI streaming connection...")
            self.stream_active = False
            self.stop_streaming.set()
            
            # Send termination message
            if self.ws_app and self.ws_connected:
                try:
                    terminate_message = {"type": "Terminate"}
                    self.ws_app.send(json.dumps(terminate_message))
                    await asyncio.sleep(1)  # Give time to process
                except Exception as e:
                    print(f"⚠️ Error sending termination: {e}")
            
            # Close WebSocket
            if self.ws_app:
                self.ws_app.close()
            
            # Wait for thread to finish
            if hasattr(self, 'ws_thread') and self.ws_thread.is_alive():
                self.ws_thread.join(timeout=2.0)
                
            print("✅ Streaming connection closed")
            
        except Exception as e:
            print(f"❌ Error closing streaming connection: {e}")
    
    def start_audio_stream(self):
        """Start the audio input stream with enhanced thread management and resource tracking"""
        try:
            print("🎤 Starting audio input stream...")
            
            # Enhanced audio stream with better resource management and tracking
            with sd.InputStream(
                callback=self.audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
            ) as stream:
                # Track the stream for cleanup (enhanced resource management)
                self.current_audio_stream = stream
                print("✅ Audio stream started")
                
                # Keep the stream alive while conversation is active
                # Enhanced monitoring with stop event integration
                while self.conversation_active and not self.stop_streaming.is_set():
                    time.sleep(0.1)
                
                # Clear stream reference when done
                self.current_audio_stream = None
                    
        except Exception as e:
            print(f"❌ Audio stream error: {e}")
            self.conversation_active = False
            self.stop_streaming.set()  # Signal all threads to stop
            self.current_audio_stream = None  # Ensure cleanup
    
    def cleanup(self):
        """Enhanced cleanup with robust resource management following blog's best practices"""
        try:
            print("🧹 Starting comprehensive resource cleanup...")
            
            # Enhanced state signaling (blog's stop_event.set() pattern)
            self.conversation_active = False
            self.stream_active = False
            self.stop_streaming.set()
            
            # Robust WebSocket cleanup
            if self.ws_app:
                try:
                    if self.ws_connected:
                        print("🔌 Closing WebSocket connection...")
                        self.ws_app.close()
                except Exception as ws_error:
                    print(f"⚠️ WebSocket close warning: {ws_error}")
            
            # Enhanced audio stream cleanup (blog's stream management pattern)
            if self.current_audio_stream:
                try:
                    print("🎤 Cleaning up audio stream...")
                    # sounddevice context manager handles this, but ensure state is clear
                    self.current_audio_stream = None
                except Exception as stream_error:
                    print(f"⚠️ Audio stream cleanup warning: {stream_error}")
            
            # Enhanced thread cleanup with proper joining (blog's thread.join pattern)
            threads_to_cleanup = [
                ("Audio Thread", self.audio_thread),
                ("Stream Thread", self.stream_thread), 
                ("WebSocket Thread", self.ws_thread)
            ]
            
            for thread_name, thread in threads_to_cleanup:
                if thread and thread.is_alive():
                    print(f"🔄 Waiting for {thread_name} to finish...")
                    thread.join(timeout=2.0)  # Enhanced timeout for robust cleanup
                    if thread.is_alive():
                        print(f"⚠️ {thread_name} did not finish cleanly (timeout)")
                    else:
                        print(f"✅ {thread_name} finished cleanly")
            
            # Enhanced queue cleanup with detailed reporting
            self._clear_audio_resources()
            
            print("✅ Comprehensive cleanup completed")
            
        except Exception as e:
            print(f"❌ Critical cleanup error: {e}")
            # Emergency cleanup - ensure basic state is reset
            try:
                self.conversation_active = False
                self.stream_active = False
                self.stop_streaming.set()
            except:
                pass
    
    async def setup_streaming_transcription(self):
        """Setup AssemblyAI Universal-Streaming WebSocket connection with enhanced thread management"""
        try:
            print("🔗 Setting up AssemblyAI Universal-Streaming API v3...")
            
            # Reset stop events for new session (enhanced thread management)
            self.stop_event.clear()      # Clear global stop_event for new session
            self.stop_streaming.clear()  # Clear legacy stop event for compatibility
            print("✅ Thread stop events reset for new streaming session")
            
            # Create WebSocketApp with v3 endpoint
            self.ws_app = websocket.WebSocketApp(
                self.streaming_endpoint,
                header={"Authorization": self.assemblyai_api_key},
                on_open=self.on_ws_open,
                on_message=self.on_ws_message,
                on_error=self.on_ws_error,
                on_close=self.on_ws_close,
            )
            
            # Enhanced WebSocket thread management with graceful shutdown support
            self.ws_thread = threading.Thread(
                target=self.ws_app.run_forever, 
                name="WebSocketThread"
            )
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            # Wait for connection with enhanced monitoring
            for i in range(50):  # Wait up to 5 seconds
                if self.ws_connected:
                    print("✅ AssemblyAI streaming connection established")
                    return True
                if self.stop_streaming.is_set():
                    print("🛑 Setup cancelled by stop event")
                    return False
                await asyncio.sleep(0.1)
            
            print("❌ Failed to establish AssemblyAI streaming connection")
            return False
            
        except Exception as e:
            print(f"❌ Error setting up streaming transcription: {e}")
            self.stop_streaming.set()  # Signal cleanup on error
            return False

    async def synthesize_speech(self, text):
        """Convert text to speech using Cartesia with QUALITY-FIRST optimization"""
        import base64  # Import base64 at the beginning of the method
        
        try:
            self.latency_tracker.end_timing("ai_response_to_tts_start", "- AI response to TTS start")
            self.latency_tracker.start_timing("tts_generation")
            self.latency_tracker.log_integration_point("🔊", "Voice synthesis")
            
            print(f"🔊 TTS: Starting speech synthesis...")
            print(f"🏥 Healthcare Expert: {text}")
            
            # Try Cartesia with optimized playback (full audio for quality)
            try:
                print("🎭 Using Cartesia AI TTS...")
                
                # Track Cartesia API call time
                cartesia_start = time.time()
                
                # Generate speech using Cartesia (fixed API)
                response = self.cartesia.tts.sse(
                    model_id="sonic-english",
                    transcript=text,
                    voice={
                        "mode": "id",
                        "id": "a0e99841-438c-4a64-b679-ae501e7d6091"  # Default voice
                    },
                    output_format={
                        "container": "raw",
                        "encoding": "pcm_f32le",
                        "sample_rate": 22050,
                    }
                )
                
                # CONSERVATIVE STREAMING: Collect more audio before playing to avoid cutting words
                audio_chunks = []
                first_chunk_time = None
                
                # Track first chunk for timing but don't start playing immediately
                self.latency_tracker.start_timing("first_audio_chunk")
                
                for chunk in response:
                    # Track time to first audio chunk
                    if not first_chunk_time:
                        first_chunk_time = time.time()
                        first_chunk_latency = (first_chunk_time - cartesia_start) * 1000
                        print(f"⏱️  First TTS chunk: {first_chunk_latency:.1f}ms")
                        self.latency_tracker.end_timing("first_audio_chunk", "- Time to first audio chunk")
                    
                    # Cartesia SSE returns audio data directly in chunk.data as base64 string
                    if hasattr(chunk, 'data') and chunk.data and chunk.type == 'chunk':
                        try:
                            # Decode base64 audio data
                            audio_data = base64.b64decode(chunk.data)
                            audio_chunks.append(audio_data)
                        except Exception as e:
                            print(f"⚠️ Error decoding audio chunk: {e}")
                            continue
                
                if audio_chunks:
                    # Combine all audio chunks
                    audio_data = b"".join(audio_chunks)
                    
                    # Track total TTS generation time
                    tts_total_time = (time.time() - cartesia_start) * 1000
                    print(f"⏱️  Total TTS Generation: {tts_total_time:.1f}ms")
                    
                    # Play all audio at once for better quality (no streaming cutting)
                    self.latency_tracker.start_timing("audio_playback")
                    await self.play_audio(audio_data, sample_rate=22050)
                    
                    # Track total TTS pipeline time
                    self.latency_tracker.end_timing("tts_generation", "- Complete TTS pipeline")
                    self.latency_tracker.log_integration_point("🔊", "Voice output complete")
                    
                    print("✅ TTS: Cartesia synthesis successful (full audio)")
                    return
                else:
                    print("⚠️ No audio data received from Cartesia")
                    
            except Exception as cartesia_error:
                print(f"⚠️ Cartesia TTS failed: {cartesia_error}")
                print("🔄 Falling back to macOS 'say' command...")
            
            # Fallback to macOS say command
            try:
                import subprocess
                import shlex
                
                fallback_start = time.time()
                escaped_text = shlex.quote(text)
                result = subprocess.run(
                    ["say", "-v", "Samantha", escaped_text], 
                    capture_output=True, 
                    text=True, 
                    timeout=30
                )
                
                fallback_time = (time.time() - fallback_start) * 1000
                print(f"⏱️  macOS 'say' command: {fallback_time:.1f}ms")
                
                if result.returncode == 0:
                    self.latency_tracker.end_timing("tts_generation", "- macOS say fallback")
                    print("✅ TTS: macOS 'say' command successful")
                    return
                    
            except Exception as say_error:
                print(f"⚠️ macOS 'say' failed: {say_error}")
            
            # Final fallback to pyttsx3
            print("🔄 Using pyttsx3 as final fallback...")
            fallback_start = time.time()
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            fallback_time = (time.time() - fallback_start) * 1000
            print(f"⏱️  pyttsx3 fallback: {fallback_time:.1f}ms")
            self.latency_tracker.end_timing("tts_generation", "- pyttsx3 fallback")
            print("✅ TTS: pyttsx3 fallback successful")
                
        except Exception as e:
            print(f"❌ TTS Error: {e}")
            # Emergency fallback
            try:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            except:
                print("❌ All TTS methods failed")
    
    async def generate_speech_data(self, text):
        """Generate speech data for web browser playback (returns audio data instead of playing locally)"""
        import base64  # Import base64 at the beginning of the method
        
        try:
            print(f"🔊 TTS: Generating speech data for browser...")
            print(f"🏥 Healthcare Expert: {text}")
            
            # Try Cartesia with data return (not local playback)
            try:
                print("🎭 Using Cartesia AI TTS for browser...")
                
                # Generate speech using Cartesia
                response = self.cartesia.tts.sse(
                    model_id="sonic-english",
                    transcript=text,
                    voice={
                        "mode": "id",
                        "id": "a0e99841-438c-4a64-b679-ae501e7d6091"  # Default voice
                    },
                    output_format={
                        "container": "raw",
                        "encoding": "pcm_f32le",
                        "sample_rate": 22050,
                    }
                )
                
                # Collect audio chunks
                audio_chunks = []
                
                for chunk in response:
                    if hasattr(chunk, 'data') and chunk.data and chunk.type == 'chunk':
                        try:
                            # Decode base64 audio data
                            audio_data = base64.b64decode(chunk.data)
                            audio_chunks.append(audio_data)
                        except Exception as e:
                            print(f"⚠️ Error decoding audio chunk: {e}")
                            continue
                
                if audio_chunks:
                    # Combine all audio chunks
                    raw_audio_data = b"".join(audio_chunks)
                    
                    # Convert to WAV format for browser compatibility
                    wav_data = self.convert_to_wav(raw_audio_data, sample_rate=22050)
                    
                    # Encode as base64 for JSON transmission
                    audio_base64 = base64.b64encode(wav_data).decode('utf-8')
                    
                    print("✅ TTS: Speech data generated for browser")
                    return audio_base64
                else:
                    print("⚠️ No audio data received from Cartesia")
                    return None
                    
            except Exception as cartesia_error:
                print(f"⚠️ Cartesia TTS failed for browser: {cartesia_error}")
                return None
                
        except Exception as e:
            print(f"❌ TTS Error for browser: {e}")
            return None

    def convert_to_wav(self, pcm_data, sample_rate=22050):
        """Convert raw PCM data to WAV format for browser playback"""
        try:
            import wave
            import io
            import struct
            
            # Convert PCM f32le to 16-bit PCM for WAV
            import numpy as np
            
            # PCM data is float32 little-endian
            audio_array = np.frombuffer(pcm_data, dtype=np.float32)
            
            # Convert to 16-bit PCM (WAV standard)
            audio_16bit = (audio_array * 32767).astype(np.int16)
            
            # Create WAV file in memory
            wav_buffer = io.BytesIO()
            
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 2 bytes (16-bit)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_16bit.tobytes())
            
            wav_buffer.seek(0)
            return wav_buffer.getvalue()
            
        except Exception as e:
            print(f"❌ Error converting to WAV: {e}")
            return pcm_data  # Return original data as fallback
    
    async def play_audio(self, audio_data, sample_rate=22050):
        """Play audio data with latency tracking"""
        try:
            playback_start = time.time()
            
            # Convert bytes to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            # Play audio using sounddevice
            sd.play(audio_array, samplerate=sample_rate)
            sd.wait()  # Wait until audio is finished
            
            # Track audio playback time
            playback_time = (time.time() - playback_start) * 1000
            print(f"⏱️  Audio Playback: {playback_time:.1f}ms")
            self.latency_tracker.end_timing("audio_playback", "- Audio playback duration")
            
            # Small delay to ensure audio completion
            await asyncio.sleep(0.5)
            
        except Exception as e:
            print(f"❌ Audio playback error: {e}")
    
    async def graceful_shutdown(self):
        """Graceful shutdown handling following blog's best practices"""
        try:
            print("🛑 Initiating graceful shutdown...")
            
            # Signal all components to stop
            self.conversation_active = False
            self.stream_active = False
            self.stop_streaming.set()
            
            # Send termination message to AssemblyAI if WebSocket is connected
            if self.ws_app and self.ws_connected:
                try:
                    print("📤 Sending termination message to AssemblyAI...")
                    terminate_message = {"type": "Terminate"}
                    self.ws_app.send(json.dumps(terminate_message))
                    
                    # Allow time for message processing (blog's pattern)
                    print("⏳ Waiting for termination acknowledgment...")
                    await asyncio.sleep(5)  # Blog's 5-second wait pattern
                    
                except Exception as e:
                    print(f"⚠️ Error sending termination message: {e}")
            
            # Close WebSocket connection gracefully
            if self.ws_app:
                try:
                    print("🔌 Closing WebSocket connection...")
                    self.ws_app.close()
                except Exception as e:
                    print(f"⚠️ WebSocket close warning: {e}")
            
            # Wait for WebSocket thread to finish (blog's join pattern)
            if self.ws_thread and self.ws_thread.is_alive():
                print("⏳ Waiting for WebSocket thread to finish...")
                self.ws_thread.join(timeout=2.0)  # Blog's timeout pattern
                if self.ws_thread.is_alive():
                    print("⚠️ WebSocket thread did not finish cleanly")
                else:
                    print("✅ WebSocket thread finished gracefully")
            
            print("✅ Graceful shutdown completed")
            
        except Exception as e:
            print(f"❌ Error during graceful shutdown: {e}")
            # Fallback to basic cleanup
            self.conversation_active = False
            self.stop_streaming.set()

    async def process_streaming_conversation_step(self):
        """Process a single step of the streaming conversation"""
        try:
            # Check for completed transcriptions
            if not self.transcription_queue.empty():
                transcript = self.transcription_queue.get_nowait()
                
                if transcript and transcript.strip():
                    print(f"\n👤 User: {transcript}")
                    
                    # Check for exit commands
                    if any(word in transcript.lower() for word in ['goodbye', 'exit', 'quit', 'stop']):
                        response = await self.generate_enhanced_response(transcript)
                        await self.synthesize_speech(response)
                        self.conversation_active = False
                        return False  # Signal to stop conversation
                    
                    # Generate and speak response
                    print("🤖 Thinking...")
                    self.is_speaking = True  # Pause audio streaming during TTS
                    
                    response = await self.generate_enhanced_response(transcript)
                    await self.synthesize_speech(response)
                    
                    self.is_speaking = False  # Resume audio streaming
                    print("🎧 Ready for next input...\n")
            
            return True  # Continue conversation
            
        except Exception as e:
            print(f"❌ Conversation step error: {e}")
            return True  # Continue despite errors
    
    async def generate_enhanced_response(self, user_input: str):
        """Generate response using RAG-enhanced LangChain agent"""
        try:
            self.latency_tracker.log_integration_point("�", "AI processing")
            if self.verbose_logging:
                print(f"🔄 Sending to Enhanced AI with RAG: '{user_input}'")
            
            # Start conversation timing
            conversation_start = time.time()
            
            # Check for exit commands
            if any(word in user_input.lower() for word in ['goodbye', 'exit', 'quit', 'stop']):
                return "Goodbye! Have a great day!"
            
            # Check for feedback from previous response
            feedback = self.conversation_learner.detect_feedback(user_input)
            if feedback != "none" and hasattr(self, 'current_conversation_id') and self.current_conversation_id:
                # Store feedback for the last conversation
                self.conversation_learner.learn_from_feedback(self.current_conversation_id, feedback)
                if feedback == "positive":
                    return "Thank you for the feedback! I'm glad I could help."
                else:
                    return "Thanks for the feedback. I'll try to do better next time."
            
            # Get conversation history for context
            conversation_history = ""
            history_context_type = "none"
            
            # Check if user is asking for summary or conversation-related questions
            summary_keywords = ['summarize', 'summary', 'what have we', 'our conversation', 'we discussed', 'we talked about', 'history']
            if any(keyword in user_input.lower() for keyword in summary_keywords):
                conversation_history = self.conversation_learner.get_conversation_summary(last_n=10)
                history_context_type = "full_history"
            else:
                # For regular questions, include limited recent context
                conversation_history = self.conversation_learner.get_conversation_summary(last_n=3)
                history_context_type = "recent_context"
            
            # Enhanced query classification - detect question types first
            is_math_question = self._is_math_or_general_question(user_input)
            is_summary_question = any(keyword in user_input.lower() for keyword in summary_keywords)
            
            # Get RAG context only if it's likely to be relevant
            self.latency_tracker.start_timing("rag_retrieval")
            if is_math_question:
                # Skip RAG for obvious math/general questions
                rag_context = ""
                if self.verbose_logging:
                    print("🔢 Math/general question detected - skipping RAG retrieval")
            else:
                rag_context = self.knowledge_base.get_context_for_query(user_input)
            self.latency_tracker.end_timing("rag_retrieval", "- RAG context retrieval")
            
            # Enhanced context evaluation with relevance threshold
            has_meaningful_rag_context = self._has_meaningful_rag_context(rag_context, user_input)
            has_conversation_context = bool(conversation_history and conversation_history.strip())
            
            # Enhanced response type classification
            if is_summary_question and has_conversation_context:
                response_type = "Memory"
                context_indicator = "💭"
            elif has_meaningful_rag_context and has_conversation_context:
                response_type = "RAG+Memory"
                context_indicator = "📚💭"
            elif has_meaningful_rag_context:
                response_type = "RAG"
                context_indicator = "📚"
            elif has_conversation_context and not is_math_question:
                response_type = "Memory"
                context_indicator = "💭"
            else:
                response_type = "LLM"
                context_indicator = "🤖"
            
            # Log the processing type
            self.latency_tracker.log_integration_point("🧠", f"AI processing ({response_type})")
            
            # Enhanced verbose logging for debugging
            if self.verbose_logging:
                if has_meaningful_rag_context:
                    print(f"📚 Meaningful RAG context found: {len(rag_context)} chars")
                else:
                    print("📚 No meaningful RAG context available")
                if has_conversation_context:
                    print(f"💭 Conversation context: {len(conversation_history)} chars")
                else:
                    print("💭 No conversation context")
                print(f"🔍 Question type - Math/General: {is_math_question}, Summary: {is_summary_question}")
                print(f"🎯 Response type determined: {response_type}")
            
            # Enhanced prompt with both RAG context and conversation history
            from langchain.schema import HumanMessage, SystemMessage
            
            system_prompt = """You are a healthcare expert AI assistant specializing in medical information and health advice, with access to a comprehensive medical knowledge base and conversation history. 
            Keep responses VERY concise and conversational - maximum 2 sentences or 150 characters for voice output.
            
            Conversation History:
            {conversation_history}
            
            Knowledge Base Context:
            {rag_context}
            
            Instructions:
            - If asked to summarize or about conversation history, focus on the conversation history above
            - For other questions, use both conversation context and knowledge base
            - Keep responses brief and natural for voice interaction""".format(
                conversation_history=conversation_history if conversation_history else "No previous conversation context.",
                rag_context=rag_context if rag_context else "No specific knowledge base context available."
            )
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input)
            ]
            
            # Track AI response time
            ai_start = time.time()
            response = self.llm.invoke(messages)
            ai_duration = (time.time() - ai_start) * 1000
            print(f"⏱️  Enhanced AI Response Generation: {ai_duration:.1f}ms")
            
            response_text = response.content.strip()
            
            # Clean up response and limit length for voice
            response_text = response_text.replace("*", "").replace("#", "").strip()
            if len(response_text) > 150:
                response_text = response_text[:150] + "..."
            
            # Log conversation for learning
            total_response_time = (time.time() - conversation_start) * 1000
            context_summary = f"Type: {response_type}, RAG: {bool(rag_context)}, History: {history_context_type}, Context_chars: {len(rag_context) if rag_context else 0}"
            
            self.current_conversation_id = self.conversation_learner.log_conversation(
                user_input, response_text, total_response_time, context_summary
            )
            
            # Track timing
            self.latency_tracker.end_timing("transcript_to_ai_response", "- Enhanced transcript to AI response")
            self.latency_tracker.start_timing("ai_response_to_tts_start")
            self.latency_tracker.log_integration_point("🧠", f"AI response ready ({response_type})")
            
            return response_text
            
        except Exception as e:
            print(f"❌ Enhanced response generation error: {e}")
            return "I'm sorry, I encountered an error processing your request."
    
    def _is_math_or_general_question(self, user_input: str) -> bool:
        """Detect if the question is mathematical or general knowledge (non-healthcare)"""
        user_lower = user_input.lower()
        
        # Math keywords and patterns
        math_keywords = [
            'calculate', 'multiply', 'times', 'plus', 'minus', 'divide', 'divided',
            'square', 'root', 'power', 'equation', 'solve', 'derivative', 'integral',
            'sum', 'difference', 'product', 'quotient', 'percent', 'percentage',
            'area', 'volume', 'circumference', 'diameter', 'radius'
        ]
        
        # Math operators and number patterns
        math_patterns = [
            r'\d+\s*[\+\-\*\/×÷]\s*\d+',  # Basic math operations
            r'\d+\s*(times|multiplied\s+by|plus|minus|divided\s+by)\s*\d+',
            r'what\s+is\s+\d+',  # "what is 15 times 8"
            r'\d+\s*[\+\-\*\/×÷]',  # Numbers with operators
        ]
        
        # General knowledge keywords (non-healthcare)
        general_keywords = [
            'capital', 'country', 'president', 'year', 'date', 'time', 'weather',
            'geography', 'history', 'literature', 'author', 'book', 'movie',
            'planet', 'solar system', 'continent', 'ocean', 'mountain',
            'who wrote', 'when did', 'where is', 'which country', 'what year'
        ]
        
        # Summary/conversation keywords should NOT be classified as general
        summary_keywords = ['summarize', 'summary', 'what have we', 'our conversation', 'we discussed', 'we talked about', 'history']
        is_summary_question = any(keyword in user_lower for keyword in summary_keywords)
        
        # Don't classify summary questions as general knowledge
        if is_summary_question:
            return False
        
        # Check for math keywords
        if any(keyword in user_lower for keyword in math_keywords):
            return True
            
        # Check for math patterns using regex
        for pattern in math_patterns:
            if re.search(pattern, user_lower):
                return True
        
        # Check for general knowledge keywords
        if any(keyword in user_lower for keyword in general_keywords):
            return True
            
        # Check if it's a simple arithmetic question
        if re.search(r'what\s+is\s+\d+', user_lower) and any(op in user_lower for op in ['times', 'plus', 'minus', 'divided']):
            return True
            
        return False
    
    def _has_meaningful_rag_context(self, rag_context: str, user_input: str) -> bool:
        """Determine if RAG context is meaningful and relevant"""
        if not rag_context or not rag_context.strip():
            return False
            
        # Minimum length threshold - too short context is likely irrelevant
        if len(rag_context.strip()) < 50:
            return False
            
        # Check if the context contains healthcare-related terms
        healthcare_terms = [
            'patient', 'treatment', 'medication', 'symptom', 'diagnosis', 'therapy',
            'disease', 'condition', 'medical', 'clinical', 'hospital', 'doctor',
            'nurse', 'health', 'care', 'drug', 'dosage', 'side effect', 'procedure'
        ]
        
        context_lower = rag_context.lower()
        user_lower = user_input.lower()
        
        # If user asks healthcare question, require healthcare context
        user_has_healthcare_terms = any(term in user_lower for term in healthcare_terms)
        context_has_healthcare_terms = any(term in context_lower for term in healthcare_terms)
        
        if user_has_healthcare_terms and not context_has_healthcare_terms:
            return False
            
        # For non-healthcare questions, be more restrictive about RAG context
        if not user_has_healthcare_terms and context_has_healthcare_terms:
            return False  # Healthcare context for non-healthcare question
            
        return True
    
    # [Include all other methods from original agent - audio processing, TTS, etc.]
    # For brevity, I'm showing the key enhanced methods
    
    def initialize_with_sample_knowledge(self):
        """Initialize with sample knowledge for demonstration"""
        sample_docs = [
            """
            Voice AI Technology Overview:
            Voice AI systems combine speech recognition, natural language processing, and text-to-speech synthesis.
            Key components include:
            1. Speech-to-Text (STT) - Converting audio to text
            2. Natural Language Understanding (NLU) - Understanding intent and context
            3. Dialog Management - Managing conversation flow
            4. Text-to-Speech (TTS) - Converting text back to audio
            
            Modern voice AI systems use deep learning models and can achieve human-like performance
            in many scenarios. They're used in virtual assistants, customer service, and accessibility tools.
            """,
            
            """
            LangChain and RAG Systems:
            LangChain is a framework for developing applications with language models.
            RAG (Retrieval-Augmented Generation) combines the power of retrieval systems with generative models.
            
            Benefits of RAG:
            - Access to up-to-date information
            - Reduced hallucinations
            - Domain-specific knowledge integration
            - Better factual accuracy
            
            RAG systems typically use vector databases like Chroma, Pinecone, or Weaviate
            to store and retrieve relevant document chunks.
            """,
            
            """
            Conversation Learning and Memory:
            Advanced AI agents can learn from conversations through:
            1. Session memory - Remembering context within a conversation
            2. Long-term memory - Storing information across sessions
            3. User preference learning - Adapting to user preferences
            4. Feedback integration - Learning from user corrections
            
            Memory types in LangChain:
            - ConversationBufferMemory: Stores raw conversation
            - ConversationSummaryMemory: Stores summarized conversation
            - ConversationSummaryBufferMemory: Hybrid approach
            - VectorStoreRetrieverMemory: Uses vector similarity for memory retrieval
            """
        ]
        
        metadatas = [
            {"source": "voice_ai_docs", "category": "technology"},
            {"source": "langchain_docs", "category": "rag"},
            {"source": "ai_memory_docs", "category": "learning"}
        ]
        
        self.add_knowledge_to_rag(sample_docs, metadatas)
        print("🧠 Initialized RAG knowledge base with sample documents")

    async def initialize_for_api(self):
        """Initialize the voice agent for API access without starting the conversation loop"""
        print("🔄 Initializing Healthcare Expert AI Agent for API access...")
        
        # Initialize with sample knowledge (without the full conversation setup)
        self.initialize_with_sample_knowledge()
        
        print("✅ Healthcare Expert AI Agent ready for API access")
        print(f"🏥 Medical knowledge base initialized")
        print(f"💭 Conversation memory system ready")
        
        return True

    async def start_enhanced_conversation(self):
        """Start the enhanced conversation with RAG and learning"""
        print("🎙️ Enhanced RAG Voice AI Agent started!")
        print("🚀 Features: RAG Knowledge Base, Conversation Learning, Real-time STT, Cartesia TTS")
        print("🧠 Enhanced Capabilities:")
        print("   • RAG-powered responses with knowledge base search")
        print("   • Conversation learning and pattern recognition")
        print("   • Enhanced memory with summarization")
        print("   • User feedback integration")
        print("📝 Instructions:")
        print("   • Speak naturally - streaming VAD will detect your voice")
        print("   • Ask about topics in the knowledge base for enhanced responses")
        print("   • Say 'goodbye' or 'exit' to stop")
        print("   • Press Ctrl+C for graceful shutdown")
        print()
        
        # Initialize with sample knowledge
        self.initialize_with_sample_knowledge()
        
        # Show conversation insights
        insights = self.conversation_learner.get_conversation_insights()
        print(f"📊 Previous conversations: {insights.get('total_conversations', 0)}")
        print(f"⚡ Avg response time: {insights.get('avg_response_time_ms', 0):.1f}ms")
        print()
        
        # Test audio devices
        print("🔧 Testing audio setup...")
        try:
            # Test audio input
            print(f"🎤 Audio input: {sd.default.device[0]} (sample rate: {self.sample_rate}Hz)")
            print(f"🔊 Audio output: {sd.default.device[1]}")
            print("✅ Audio devices ready")
        except Exception as e:
            print(f"⚠️ Audio setup warning: {e}")
        
        # Setup streaming transcription
        print("🔗 Setting up streaming transcription...")
        setup_success = await self.setup_streaming_transcription()
        if not setup_success:
            print("❌ Failed to setup streaming - aborting")
            return
        
        # Test TTS initialization with OPTIMIZED short message
        print("🧪 Testing TTS engines...")
        test_text = "Healthcare Expert AI ready! How can I help with your health questions?"  # Healthcare-focused startup message
        await self.synthesize_speech(test_text)
        
        self.conversation_active = True
        
        print("\n🎧 Listening for speech... (speak now)")
        
        try:
            # Enhanced audio thread management with proper reference tracking
            self.audio_thread = threading.Thread(
                target=self.start_audio_stream, 
                name="MainAudioThread"
            )
            self.audio_thread.daemon = True
            self.audio_thread.start()
            
            # Graceful shutdown handling - monitor threads while conversation is active
            try:
                while (self.conversation_active and 
                       not self.stop_streaming.is_set() and
                       (self.ws_thread and self.ws_thread.is_alive())):
                    
                    # Process conversation with streaming
                    await self.process_streaming_conversation_step()
                    await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                    
            except KeyboardInterrupt:
                print("\n🛑 Ctrl+C received. Initiating graceful shutdown...")
                await self.graceful_shutdown()
                
        except Exception as e:
            print(f"❌ Audio stream error: {e}")
            self.conversation_active = False
            self.stop_streaming.set()
        finally:
            # Final cleanup - ensure all resources are released
            print("🧹 Performing final cleanup...")
            self.cleanup()
            await self.close_streaming_transcription()
            print("✅ Enhanced Voice AI Agent shutdown complete")

    def run_with_graceful_shutdown(self):
        """Run with graceful shutdown handling following AssemblyAI example pattern"""
        try:
            print("🚀 Starting graceful shutdown monitoring...")
            
            # Don't start the WebSocket thread here - it's already started in setup_streaming_transcription
            # Just monitor the existing thread
            if not self.ws_thread:
                print("❌ WebSocket thread not initialized")
                return
            
            if not self.ws_thread.is_alive():
                print("❌ WebSocket thread is not running")
                return
                
            print("✅ WebSocket thread is running, monitoring for shutdown...")
            
            # Enhanced main loop with graceful shutdown (AssemblyAI pattern)
            try:
                while self.ws_thread.is_alive() and not self.stop_event.is_set():
                    time.sleep(0.1)  # AssemblyAI pattern: small sleep for responsiveness
            except KeyboardInterrupt:
                print("\nCtrl+C received. Stopping...")
                # stop_event.set() - AssemblyAI pattern
                self.stop_event.set()

                # Send termination message following AssemblyAI pattern
                if self.ws_app and hasattr(self.ws_app, 'sock') and self.ws_app.sock and hasattr(self.ws_app.sock, 'connected') and self.ws_app.sock.connected:
                    try:
                        # terminate_message = {"type": "Terminate"} - AssemblyAI pattern
                        terminate_message = {"type": "Terminate"}
                        self.ws_app.send(json.dumps(terminate_message))
                        time.sleep(5)  # Allow message processing (AssemblyAI pattern)
                        print("✅ Termination message sent")
                    except Exception as e:
                        print(f"⚠️ Error sending termination message: {e}")

                # ws_app.close() - AssemblyAI pattern
                if self.ws_app:
                    self.ws_app.close()
                    print("✅ WebSocket connection closed")
                
                # ws_thread.join(timeout=2.0) - AssemblyAI pattern
                self.ws_thread.join(timeout=2.0)
                if self.ws_thread.is_alive():
                    print("⚠️ WebSocket thread cleanup timeout")
                else:
                    print("✅ WebSocket thread cleaned up successfully")
                    
        finally:
            # Final cleanup following AssemblyAI pattern
            print("🧹 Performing final graceful cleanup...")
            try:
                # Enhanced resource cleanup coordination
                self.stop_event.set()
                self.stop_streaming.set()
                self.conversation_active = False
                self.stream_active = False
                self.ws_connected = False
                
                # Cleanup global resources (AssemblyAI pattern: global stream, audio)
                if self.stream:
                    try:
                        if hasattr(self.stream, 'is_active') and self.stream.is_active():
                            self.stream.stop_stream()
                        if hasattr(self.stream, 'close'):
                            self.stream.close()
                        self.stream = None
                        print("✅ Global stream cleaned up in shutdown")
                    except Exception as e:
                        print(f"⚠️ Global stream cleanup error: {e}")
                
                if self.audio:
                    try:
                        if hasattr(self.audio, 'terminate'):
                            self.audio.terminate()
                        self.audio = None
                        print("✅ Global audio cleaned up in shutdown")
                    except Exception as e:
                        print(f"⚠️ Global audio cleanup error: {e}")
                
                # Final thread cleanup
                cleanup_threads = [
                    ("Audio Thread", self.audio_thread),
                    ("Stream Thread", self.stream_thread)
                ]
                
                for thread_name, thread in cleanup_threads:
                    if thread and thread.is_alive():
                        print(f"🔄 Final cleanup of {thread_name}...")
                        thread.join(timeout=1.0)
                        if thread.is_alive():
                            print(f"⚠️ {thread_name} final cleanup timeout")
                        else:
                            print(f"✅ {thread_name} final cleanup successful")
                
                print("✅ Graceful shutdown completed successfully")
                
            except Exception as cleanup_error:
                print(f"❌ Error during final cleanup: {cleanup_error}")

# Example usage and integration
async def main():
    """Enhanced main function with graceful shutdown handling following AssemblyAI pattern"""
    # Check for required environment variables
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT", 
        "ASSEMBLYAI_API_KEY",
        "CARTESIA_API_KEY"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Error: Missing environment variables: {', '.join(missing_vars)}")
        print("Please add them to your .env file")
        return
    
    print("🚀 Initializing Healthcare Expert AI Voice Agent...")
    # Set verbose_logging=True for detailed logs, False for simplified integration point logs
    agent = EnhancedRAGVoiceAgent(verbose_logging=False)
    
    try:
        # Start the enhanced conversation (includes setup, TTS testing, and audio streaming)
        await agent.start_enhanced_conversation()
        
    except KeyboardInterrupt:
        print("\n🛑 Graceful shutdown initiated...")
        print("👋 Thank you for using Healthcare Expert AI!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        # Ensure cleanup on unexpected errors
        try:
            agent.stop_event.set()
            agent.stop_streaming.set()
        except:
            pass
    finally:
        print("🔚 Healthcare Expert AI Agent terminated with graceful shutdown")

if __name__ == "__main__":
    asyncio.run(main())
