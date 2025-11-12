import aiosqlite
import asyncio
from typing import List, Dict, Any
from .logger import logger
import json

db_path = "./bot_data.db"


# ===================== INIT DB =====================

async def init_db():
    async with aiosqlite.connect(db_path) as db:
        # Create tables
        await db.executescript('''
        CREATE TABLE IF NOT EXISTS threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS user_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS ai_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_msg_id INTEGER NOT NULL,
            content TEXT,
            sequence INTEGER DEFAULT 1,
            metadata JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_msg_id) REFERENCES user_messages(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS tool_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ai_msg_id INTEGER NOT NULL,
            tool_name TEXT,
            input JSON,
            output JSON,
            sequence INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(ai_msg_id) REFERENCES ai_messages(id) ON DELETE CASCADE
        );
        ''')
        await db.commit()
    logger.info("Database initialized successfully.")


# ===================== THREAD FUNCTIONS =====================

async def create_thread(title: str = None) -> int:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("INSERT INTO threads (title) VALUES (?)", (title,))
        await db.commit()
        return cursor.lastrowid


# ===================== USER MESSAGE FUNCTIONS =====================

async def save_user_message(thread_id: int, content: str) -> int:
    logger.debug(f"Saving user message: thread_id={thread_id}, content={content}")
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "INSERT INTO user_messages (thread_id, content) VALUES (?, ?)",
            (thread_id, content)
        )
        await db.commit()
        return cursor.lastrowid


async def get_user_messages(thread_id: int) -> List[Dict[str, Any]]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM user_messages WHERE thread_id = ? ORDER BY created_at", (thread_id,))
        data = await cursor.fetchall()
    return [dict(row) for row in data]


# ===================== AI MESSAGE FUNCTIONS =====================

async def save_ai_message(user_msg_id: int, content: str, sequence: int = 1, metadata: Dict[str, Any] = None) -> int:
    logger.debug(f"Saving AI message: user_msg_id={user_msg_id}, content={content}")
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "INSERT INTO ai_messages (user_msg_id, content, sequence, metadata) VALUES (?, ?, ?, ?)",
            (user_msg_id, content, sequence, json.dumps(metadata or {}))
        )
        await db.commit()
        return cursor.lastrowid


async def get_ai_messages(user_msg_id: int) -> List[Dict[str, Any]]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM ai_messages WHERE user_msg_id = ? ORDER BY sequence", (user_msg_id,))
        data = await cursor.fetchall()
    return [dict(row) for row in data]


# ===================== TOOL MESSAGE FUNCTIONS =====================

async def save_tool_message(ai_msg_id: int, tool_name: str, input_data: Dict[str, Any], output_data: Dict[str, Any], sequence: int = 1) -> int:
    logger.debug(f"Saving tool message: ai_msg_id={ai_msg_id}, tool={tool_name}")
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            "INSERT INTO tool_messages (ai_msg_id, tool_name, input, output, sequence) VALUES (?, ?, ?, ?, ?)",
            (ai_msg_id, tool_name, json.dumps(input_data), json.dumps(output_data), sequence)
        )
        await db.commit()
        return cursor.lastrowid


async def get_tool_messages(ai_msg_id: int) -> List[Dict[str, Any]]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute("SELECT * FROM tool_messages WHERE ai_msg_id = ? ORDER BY sequence", (ai_msg_id,))
        data = await cursor.fetchall()
    return [dict(row) for row in data]


# ===================== CLEANUP / DELETE =====================

async def delete_thread(thread_id: int):
    logger.debug(f"Deleting thread and related messages for thread ID: {thread_id}")
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        await db.commit()


# ===================== TEST / DEMO =====================

if __name__ == "__main__":
    async def demo():
        await init_db()

        # Create a new thread
        thread_id = await create_thread("Weather Info")
        print(f"Created Thread ID: {thread_id}")

        # Add user message
        user_msg_id = await save_user_message(thread_id, "What's the weather in Paris?")
        print(f"User Message ID: {user_msg_id}")

        # Add AI message
        ai_msg_id = await save_ai_message(user_msg_id, "Let me check that for you...")
        print(f"AI Message ID: {ai_msg_id}")

        # Add tool message
        tool_msg_id = await save_tool_message(
            ai_msg_id,
            "weather_api",
            {"city": "Paris"},
            {"temperature": "22°C", "condition": "Sunny"}
        )
        print(f"Tool Message ID: {tool_msg_id}")

        # Fetch conversation
        user_msgs = await get_user_messages(thread_id)
        ai_msgs = await get_ai_messages(user_msg_id)
        tool_msgs = await get_tool_messages(ai_msg_id)

        print("User Messages:", user_msgs)
        print("AI Messages:", ai_msgs)
        print("Tool Messages:", tool_msgs)

    asyncio.run(demo())
