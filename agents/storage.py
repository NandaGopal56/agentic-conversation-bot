import aiosqlite
import json

DB_PATH = "bot_data.db"


# -------------------------------------------------------------
#  DATABASE INIT
# -------------------------------------------------------------
async def init_db():
    """
    Initialize the SQLite database with minimal required tables.
    Keeps schema simple and framework-independent.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS threads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id INTEGER NOT NULL,
            role TEXT NOT NULL,                 -- user / assistant / tool
            message_type TEXT NOT NULL,         -- text / tool_call / tool_result
            content TEXT,
            metadata JSON,
            sequence INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(thread_id) REFERENCES threads(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS tool_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message_id INTEGER NOT NULL,
            ai_message_id INTEGER NOT NULL,
            input JSON NOT NULL,
            FOREIGN KEY(user_message_id) REFERENCES messages(id) ON DELETE CASCADE,
            FOREIGN KEY(ai_message_id) REFERENCES messages(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS tool_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_message_id INTEGER NOT NULL,
            ai_message_id INTEGER NOT NULL,
            output JSON NOT NULL,
            FOREIGN KEY(user_message_id) REFERENCES messages(id) ON DELETE CASCADE,
            FOREIGN KEY(ai_message_id) REFERENCES messages(id) ON DELETE CASCADE
        );
        """)
        await db.commit()


# -------------------------------------------------------------
#  INTERNAL UTILITY
# -------------------------------------------------------------
async def _get_next_sequence(db, thread_id: int) -> int:
    """
    Get next sequence number for messages in a thread.
    Ensures strict ordering across all message types.
    """
    row = await db.execute(
        "SELECT COALESCE(MAX(sequence), 0) + 1 FROM messages WHERE thread_id = ?",
        (thread_id,)
    )
    return (await row.fetchone())[0]


# -------------------------------------------------------------
#  THREAD CRUD
# -------------------------------------------------------------
async def create_thread(title: str = None) -> int:
    """
    Create a new thread. Returns thread_id.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "INSERT INTO threads (title) VALUES (?)",
            (title,)
        )
        await db.commit()
        return cur.lastrowid


async def get_all_threads():
    """
    Fetch list of all threads.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        rows = await db.execute_fetchall(
            "SELECT id, title, created_at FROM threads ORDER BY created_at DESC"
        )
        return rows


# -------------------------------------------------------------
#  MESSAGE CRUD
# -------------------------------------------------------------
async def add_message(thread_id: int, role: str, message_type: str,
                      content: str = None, metadata: dict = None) -> int:
    """
    Insert a message into a thread.
    Returns message_id.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        seq = await _get_next_sequence(db, thread_id)
        print(thread_id, role, message_type, content, metadata, seq)
        cur = await db.execute("""
            INSERT INTO messages (thread_id, role, message_type, content, metadata, sequence)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            thread_id,
            role,
            message_type,
            content,
            json.dumps(metadata) if metadata else None,
            seq
        ))

        await db.commit()
        return cur.lastrowid


async def get_messages(thread_id: int):
    """
    Fetch all messages in a thread in correct order.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        rows = await db.execute_fetchall("""
            SELECT id, role, message_type, content, metadata, sequence, created_at
            FROM messages
            WHERE thread_id = ?
            ORDER BY sequence ASC
        """, (thread_id,))
        return rows


# -------------------------------------------------------------
#  TOOL CALL CRUD
# -------------------------------------------------------------
async def add_tool_call(user_message_id: int, ai_message_id: int, input_data: dict) -> int:
    """
    Store tool invocation details for a message.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("""
            INSERT INTO tool_calls (user_message_id, ai_message_id, input)
            VALUES (?, ?, ?)
        """, (user_message_id, ai_message_id, json.dumps(input_data)))
        await db.commit()
        return cur.lastrowid


async def get_tool_call(user_message_id: int, ai_message_id: int):
    """
    Get tool call information for a message.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        return await db.execute_fetchone(
            "SELECT user_message_id, ai_message_id, input FROM tool_calls WHERE user_message_id = ? AND ai_message_id = ?",
            (user_message_id, ai_message_id)
        )


# -------------------------------------------------------------
#  TOOL RESULT CRUD
# -------------------------------------------------------------
async def add_tool_result(user_message_id: int, ai_message_id: int, output_data: dict) -> int:
    """
    Store tool result output for a message.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("""
            INSERT INTO tool_results (user_message_id, ai_message_id, output)
            VALUES (?, ?, ?)
        """, (user_message_id, ai_message_id, json.dumps(output_data)))
        await db.commit()
        return cur.lastrowid


async def get_tool_result(user_message_id: int, ai_message_id: int):
    """
    Get tool execution result for a message.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        return await db.execute_fetchone(
            "SELECT output FROM tool_results WHERE user_message_id = ? AND ai_message_id = ?",
            (user_message_id, ai_message_id)
        )


# -------------------------------------------------------------
#  FULL CHAT RECONSTRUCTION
# -------------------------------------------------------------
async def get_full_thread(thread_id: int):
    """
    Reconstruct complete conversation for a thread in correct order.
    Includes text, tool calls, and tool results.
    """
    messages = await get_messages(thread_id)
    final = []

    async with aiosqlite.connect(DB_PATH) as db:
        for msg in messages:
            msg_id, role, msg_type, content, metadata, seq, created = msg

            entry = {
                "id": msg_id,
                "role": role,
                "type": msg_type,
                "content": content,
                "metadata": json.loads(metadata) if metadata else None,
                "sequence": seq,
                "created_at": created
            }

            if msg_type == "tool_call":
                tool_call = await db.execute_fetchone(
                    "SELECT user_message_id, ai_message_id, input FROM tool_calls WHERE user_message_id = ? AND ai_message_id = ?",
                    (msg_id,)
                )
                entry["tool_call"] = {
                    "user_message_id": tool_call[0],
                    "ai_message_id": tool_call[1],
                    "input": json.loads(tool_call[2])
                }

            if msg_type == "tool_result":
                result = await db.execute_fetchone(
                    "SELECT user_message_id, ai_message_id, output FROM tool_results WHERE user_message_id = ? AND ai_message_id = ?",
                    (msg_id,)
                )
                entry["tool_result"] = {
                    "user_message_id": result[0],
                    "ai_message_id": result[1],
                    "output": json.loads(result[2])
                }

            final.append(entry)

    return final

import asyncio

def main():
    asyncio.run(init_db())  
    asyncio.run(create_thread('test1'))

if __name__ == "__main__":
    main()