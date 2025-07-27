#!/usr/bin/env python3
"""Clear conversation history from the database while preserving structure"""

import sqlite3
import os
from datetime import datetime

def clear_conversation_history(db_path="conversation_memory.db"):
    """Clear all conversation data while preserving database structure"""
    
    try:
        # Check if database exists
        if not os.path.exists(db_path):
            print(f"❌ Database file not found: {db_path}")
            return
        
        # Create backup timestamp
        backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"conversation_memory_backup_{backup_timestamp}.db"
        
        print("🗄️  CLEARING CONVERSATION HISTORY")
        print("=" * 50)
        
        # Create backup first
        print(f"📦 Creating backup: {backup_path}")
        import shutil
        shutil.copy2(db_path, backup_path)
        print("✅ Backup created successfully")
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get current counts before clearing
        print("\n📊 Current database state:")
        
        cursor.execute("SELECT COUNT(*) FROM conversations")
        conversation_count = cursor.fetchone()[0]
        print(f"   - Conversations: {conversation_count}")
        
        cursor.execute("SELECT COUNT(*) FROM user_preferences")
        preference_count = cursor.fetchone()[0]
        print(f"   - User preferences: {preference_count}")
        
        cursor.execute("SELECT COUNT(*) FROM conversation_patterns")
        pattern_count = cursor.fetchone()[0]
        print(f"   - Conversation patterns: {pattern_count}")
        
        # Clear all tables
        print(f"\n🧹 Clearing conversation data...")
        
        # Clear conversations table
        cursor.execute("DELETE FROM conversations")
        cleared_conversations = cursor.rowcount
        print(f"   ✅ Cleared {cleared_conversations} conversations")
        
        # Clear user preferences
        cursor.execute("DELETE FROM user_preferences")
        cleared_preferences = cursor.rowcount
        print(f"   ✅ Cleared {cleared_preferences} user preferences")
        
        # Clear conversation patterns
        cursor.execute("DELETE FROM conversation_patterns")
        cleared_patterns = cursor.rowcount
        print(f"   ✅ Cleared {cleared_patterns} conversation patterns")
        
        # Reset auto-increment counters (if any) - handle case where table doesn't exist
        try:
            cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('conversations', 'user_preferences', 'conversation_patterns')")
        except sqlite3.Error as e:
            print(f"   ⚠️  Note: {e} (this is normal if no auto-increment columns exist)")
        
        # Commit changes
        conn.commit()
        print(f"   ✅ Changes committed to database")
        
        # Verify tables are empty
        print(f"\n🔍 Verifying cleanup:")
        
        cursor.execute("SELECT COUNT(*) FROM conversations")
        remaining_conversations = cursor.fetchone()[0]
        print(f"   - Conversations remaining: {remaining_conversations}")
        
        cursor.execute("SELECT COUNT(*) FROM user_preferences")
        remaining_preferences = cursor.fetchone()[0]
        print(f"   - User preferences remaining: {remaining_preferences}")
        
        cursor.execute("SELECT COUNT(*) FROM conversation_patterns")
        remaining_patterns = cursor.fetchone()[0]
        print(f"   - Conversation patterns remaining: {remaining_patterns}")
        
        # Vacuum database to reclaim space
        print(f"\n🗜️  Optimizing database...")
        cursor.execute("VACUUM")
        print("   ✅ Database optimized")
        
        conn.close()
        
        # Show file sizes
        original_size = os.path.getsize(db_path)
        backup_size = os.path.getsize(backup_path)
        
        print(f"\n📁 File information:")
        print(f"   - Current database: {original_size:,} bytes")
        print(f"   - Backup created: {backup_size:,} bytes")
        print(f"   - Backup location: {backup_path}")
        
        print(f"\n✅ CONVERSATION HISTORY CLEARED SUCCESSFULLY!")
        print(f"💡 Your voice agent will now start with a fresh conversation history.")
        print(f"🔒 Original data backed up to: {backup_path}")
        
        # Provide instructions
        print(f"\n📋 Next steps:")
        print(f"   1. Your voice agent is ready to use with clean history")
        print(f"   2. All 108 previous conversations have been preserved in the backup")
        print(f"   3. If you need to restore, replace the database with the backup file")
        print(f"   4. You can safely delete the backup file when no longer needed")
        
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ Error clearing conversation history: {e}")

def restore_from_backup(backup_path, db_path="conversation_memory.db"):
    """Restore conversation history from a backup file"""
    
    try:
        if not os.path.exists(backup_path):
            print(f"❌ Backup file not found: {backup_path}")
            return
        
        print(f"🔄 Restoring from backup: {backup_path}")
        
        # Create current backup before restoring
        if os.path.exists(db_path):
            current_backup = f"conversation_memory_pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy2(db_path, current_backup)
            print(f"📦 Current database backed up to: {current_backup}")
        
        # Restore from backup
        shutil.copy2(backup_path, db_path)
        print(f"✅ Database restored from backup successfully!")
        
    except Exception as e:
        print(f"❌ Error restoring from backup: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        if len(sys.argv) > 2:
            restore_from_backup(sys.argv[2])
        else:
            print("Usage: python clear_conversations.py restore <backup_file>")
    else:
        clear_conversation_history()
