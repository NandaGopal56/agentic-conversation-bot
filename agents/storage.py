import aiosqlite
import asyncio
from typing import List, Dict, Any
from .logger import logger
import json

db_path = "./bot_data.db"

async def init_db():
    async with aiosqlite.connect(db_path) as db:
        await db.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id INTEGER NOT NULL,
            user_message TEXT,
            ai_message TEXT,
            metadata JSON
        )
    ''')


async def get_messages(thread_id: int) -> List[Dict[str, Any]]:
    '''Get messages for a specific thread ID'''

    logger.debug(f"Getting messages for thread ID: {thread_id}")
    # Set the row factory to return dictionaries
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM messages WHERE thread_id = ?", (thread_id,))
        data = await cursor.fetchall()
    
    # Convert each row to a dictionary
    messages = [dict(row) for row in data]
    return messages

async def save_message(thread_id: int, user_message: str, ai_message: str, metadata: Dict[str, Any]):
    logger.debug(f"Saving message for thread ID: {thread_id}, User Message: {user_message}, AI Message: {ai_message}, Metadata: {metadata}")
    async with aiosqlite.connect(db_path) as db:
        await db.execute("INSERT INTO messages (thread_id, user_message, ai_message, metadata) VALUES (?, ?, ?, ?)", (thread_id, user_message, ai_message, json.dumps(metadata)))
        await db.commit()

async def delete_messages(thread_id: int):
    logger.debug(f"Deleting messages for thread ID: {thread_id}")
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
        await db.commit()

if __name__ == "__main__":
    asyncio.run(init_db())
    asyncio.run(delete_messages(1))
    asyncio.run(save_message(1, "Hello", "Hello", {"key": "value"}))
    asyncio.run(get_messages(1))
