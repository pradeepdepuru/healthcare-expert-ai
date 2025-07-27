#!/usr/bin/env python3
"""
Inspect the current knowledge base contents
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env')

def inspect_knowledge_base():
    """Inspect what's currently in the knowledge base"""
    
    print("🔍 Inspecting current knowledge base contents...")
    
    try:
        from enhanced_rag_voice_agent import RAGKnowledgeBase
        
        # Initialize the knowledge base (will load existing data)
        print("\n🔧 Loading existing knowledge base...")
        kb = RAGKnowledgeBase()
        
        if not kb.embeddings:
            print("❌ No embeddings available")
            return
        
        if not kb.vectorstore:
            print("❌ No vector store available")
            return
        
        print("✅ Knowledge base loaded successfully")
        
        # Try to get information about the database
        print(f"\n📊 Vector Store Info:")
        print(f"   Type: {type(kb.vectorstore).__name__}")
        print(f"   Persist Directory: {kb.persist_directory}")
        
        # Check if we can get collection info
        try:
            collection = kb.vectorstore._collection
            count = collection.count()
            print(f"   Document Count: {count}")
            
            if count > 0:
                print(f"\n📚 Sample Documents in Knowledge Base:")
                
                # Get a sample of documents
                results = collection.get(limit=10)
                
                if results and 'documents' in results:
                    documents = results['documents']
                    metadatas = results.get('metadatas', [])
                    ids = results.get('ids', [])
                    
                    for i, (doc_id, doc, metadata) in enumerate(zip(ids, documents, metadatas), 1):
                        print(f"\n   Document {i}:")
                        print(f"     ID: {doc_id}")
                        print(f"     Content: {doc[:200]}{'...' if len(doc) > 200 else ''}")
                        if metadata:
                            print(f"     Metadata: {metadata}")
                else:
                    print("   No document content available")
            else:
                print("   📝 Knowledge base is empty")
                
        except Exception as e:
            print(f"   ⚠️ Could not get detailed collection info: {e}")
        
        # Test some sample queries to see what's retrievable
        print(f"\n🔍 Testing Sample Queries:")
        test_queries = [
            "voice AI technology",
            "machine learning",
            "RAG systems",
            "LangChain",
            "embeddings",
            "conversation memory",
            "speech recognition"
        ]
        
        for query in test_queries:
            try:
                docs = kb.search_knowledge(query, k=2)
                if docs:
                    print(f"\n   Query: '{query}'")
                    print(f"   Found: {len(docs)} relevant documents")
                    for j, doc in enumerate(docs, 1):
                        preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
                        print(f"     Doc {j}: {preview}")
                        if hasattr(doc, 'metadata') and doc.metadata:
                            print(f"     Metadata: {doc.metadata}")
            except Exception as e:
                print(f"   ❌ Query '{query}' failed: {e}")
        
        print(f"\n✅ Knowledge base inspection complete!")
        
    except Exception as e:
        print(f"❌ Failed to inspect knowledge base: {e}")
        import traceback
        traceback.print_exc()

def show_sample_documents():
    """Show what sample documents are typically added to the knowledge base"""
    print(f"\n📖 Sample Documents Available in Code:")
    
    try:
        from enhanced_rag_voice_agent import EnhancedRAGVoiceAgent
        
        # Create a temporary agent to access the sample documents
        print("   Loading sample documents from code...")
        
        # The sample documents are defined in the initialize_with_sample_knowledge method
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
        
        print(f"\n   🎯 Sample Knowledge Topics:")
        for i, doc in enumerate(sample_docs, 1):
            lines = doc.strip().split('\n')
            title = lines[0].strip() if lines else f"Document {i}"
            print(f"     {i}. {title}")
            
            # Show first few key points
            for line in lines[1:6]:
                if line.strip() and not line.strip().startswith('Key components') and not line.strip().startswith('Benefits'):
                    print(f"        • {line.strip()}")
                elif line.strip().startswith(('1.', '2.', '3.', '4.', '-')):
                    print(f"        • {line.strip()}")
        
    except Exception as e:
        print(f"❌ Could not load sample documents: {e}")

if __name__ == "__main__":
    inspect_knowledge_base()
    show_sample_documents()
