#!/usr/bin/env python3
"""Inspect the conversation_memory.db database contents"""

import sqlite3
import json
from datetime import datetime

def inspect_conversation_db(db_path="conversation_memory.db"):
    """Inspect and display contents of the conversation memory database"""
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("🗄️  CONVERSATION MEMORY DATABASE INSPECTION")
        print("=" * 60)
        
        # Check if database exists and has tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        if not tables:
            print("❌ No tables found in database")
            return
        
        print(f"📊 Found {len(tables)} table(s): {[t[0] for t in tables]}")
        print("-" * 60)
        
        # Inspect conversations table
        print("\n📝 CONVERSATIONS TABLE:")
        print("-" * 30)
        
        # Get table schema
        cursor.execute("PRAGMA table_info(conversations);")
        columns = cursor.fetchall()
        print("🏗️  Table Schema:")
        for col in columns:
            print(f"   - {col[1]} ({col[2]})")
        
        # Get total count
        cursor.execute("SELECT COUNT(*) FROM conversations")
        total_conversations = cursor.fetchone()[0]
        print(f"\n📈 Total conversations: {total_conversations}")
        
        if total_conversations > 0:
            # Get recent conversations
            cursor.execute("""
                SELECT id, timestamp, user_input, agent_response, user_feedback, 
                       response_time_ms, context_used 
                FROM conversations 
                ORDER BY timestamp DESC 
                LIMIT 10
            """)
            
            recent_conversations = cursor.fetchall()
            
            print(f"\n🔍 Recent {len(recent_conversations)} conversations:")
            print("-" * 50)
            
            for i, conv in enumerate(recent_conversations, 1):
                conv_id, timestamp, user_input, agent_response, feedback, response_time, context = conv
                
                print(f"\n{i}. Conversation ID: {conv_id[:8]}...")
                print(f"   📅 Time: {timestamp}")
                print(f"   👤 User: {user_input}")
                print(f"   🤖 Agent: {agent_response}")
                print(f"   ⏱️  Response Time: {response_time:.1f}ms" if response_time else "   ⏱️  Response Time: N/A")
                print(f"   👍 Feedback: {feedback or 'None'}")
                print(f"   🏷️  Context: {context or 'None'}")
                print("   " + "-" * 40)
            
            # Get conversation statistics
            print(f"\n📊 CONVERSATION STATISTICS:")
            print("-" * 30)
            
            # Average response time
            cursor.execute("SELECT AVG(response_time_ms) FROM conversations WHERE response_time_ms IS NOT NULL")
            avg_response_time = cursor.fetchone()[0]
            if avg_response_time:
                print(f"⏱️  Average response time: {avg_response_time:.1f}ms")
            
            # Feedback statistics
            cursor.execute("""
                SELECT user_feedback, COUNT(*) 
                FROM conversations 
                WHERE user_feedback IS NOT NULL 
                GROUP BY user_feedback
            """)
            feedback_stats = cursor.fetchall()
            if feedback_stats:
                print("👍 Feedback distribution:")
                for feedback, count in feedback_stats:
                    print(f"   - {feedback}: {count}")
            
            # Response time distribution
            cursor.execute("""
                SELECT 
                    COUNT(CASE WHEN response_time_ms < 1000 THEN 1 END) as under_1s,
                    COUNT(CASE WHEN response_time_ms BETWEEN 1000 AND 3000 THEN 1 END) as one_to_3s,
                    COUNT(CASE WHEN response_time_ms > 3000 THEN 1 END) as over_3s
                FROM conversations 
                WHERE response_time_ms IS NOT NULL
            """)
            timing_stats = cursor.fetchone()
            if timing_stats and any(timing_stats):
                print("⚡ Response time distribution:")
                print(f"   - Under 1s: {timing_stats[0]}")
                print(f"   - 1-3s: {timing_stats[1]}")
                print(f"   - Over 3s: {timing_stats[2]}")
            
            # Context type analysis
            cursor.execute("""
                SELECT context_used, COUNT(*) 
                FROM conversations 
                WHERE context_used IS NOT NULL AND context_used != ''
                GROUP BY context_used
                ORDER BY COUNT(*) DESC
                LIMIT 5
            """)
            context_stats = cursor.fetchall()
            if context_stats:
                print("🧠 Most common context types:")
                for context, count in context_stats:
                    # Try to extract the type from context
                    if "Type:" in context:
                        try:
                            type_part = context.split("Type:")[1].split(",")[0].strip()
                            print(f"   - {type_part}: {count}")
                        except:
                            print(f"   - {context[:50]}...: {count}")
                    else:
                        print(f"   - {context[:50]}...: {count}")
        
        # Inspect user_preferences table
        print(f"\n👤 USER PREFERENCES TABLE:")
        print("-" * 30)
        
        cursor.execute("SELECT COUNT(*) FROM user_preferences")
        pref_count = cursor.fetchone()[0]
        print(f"📈 Total preferences: {pref_count}")
        
        if pref_count > 0:
            cursor.execute("""
                SELECT preference_type, preference_value, confidence_score, last_updated
                FROM user_preferences 
                ORDER BY last_updated DESC
                LIMIT 5
            """)
            preferences = cursor.fetchall()
            
            print("🔍 Recent preferences:")
            for pref in preferences:
                pref_type, value, confidence, updated = pref
                print(f"   - {pref_type}: {value} (confidence: {confidence}, updated: {updated})")
        
        # Inspect conversation_patterns table
        print(f"\n🔄 CONVERSATION PATTERNS TABLE:")
        print("-" * 30)
        
        cursor.execute("SELECT COUNT(*) FROM conversation_patterns")
        pattern_count = cursor.fetchone()[0]
        print(f"📈 Total patterns: {pattern_count}")
        
        if pattern_count > 0:
            cursor.execute("""
                SELECT pattern_type, frequency, last_seen
                FROM conversation_patterns 
                ORDER BY frequency DESC
                LIMIT 5
            """)
            patterns = cursor.fetchall()
            
            print("🔍 Top patterns:")
            for pattern in patterns:
                pattern_type, frequency, last_seen = pattern
                print(f"   - {pattern_type}: {frequency} times (last: {last_seen})")
        
        conn.close()
        print(f"\n✅ Database inspection completed!")
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    except FileNotFoundError:
        print(f"❌ Database file not found: {db_path}")
        print("💡 The database will be created when the voice agent runs for the first time.")
    except Exception as e:
        print(f"❌ Error inspecting database: {e}")

if __name__ == "__main__":
    inspect_conversation_db()
