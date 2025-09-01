import sqlite3
from typing import List, Dict, Any

db_path = "./bot_data.db"

connection = sqlite3.connect(db_path)

# connection.execute('''
#     CREATE TABLE IF NOT EXISTS messages (
#         id INTEGER PRIMARY KEY AUTOINCREMENT,
#         thread_id INTEGER NOT NULL,
#         user_message TEXT,
#         ai_message TEXT,
#         summary TEXT
#     )
# ''')


def get_messages(thread_id: int) -> List[Dict[str, Any]]:
    '''Get messages for a specific thread ID
    
    Args:
        thread_id: The ID of the thread to retrieve messages for
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries where each dictionary represents a message
        with column names as keys and row values as values
    '''
    # Set the row factory to return dictionaries
    connection.row_factory = sqlite3.Row
    cursor = connection.cursor()
    
    cursor.execute("SELECT * FROM messages WHERE thread_id = ?", (thread_id,))
    
    # Convert each row to a dictionary
    messages = [dict(row) for row in cursor.fetchall()]
    
    return messages

def save_message(thread_id: int, user_message: str, ai_message: str, summary: str):
    cursor = connection.cursor()
    # print(thread_id, user_message, ai_message, summary)
    cursor.execute("INSERT INTO messages (thread_id, user_message, ai_message, summary) VALUES (?, ?, ?, ?)", (thread_id, user_message, ai_message, summary))
    connection.commit()

def delete_messages(thread_id: int):
    cursor = connection.cursor()
    cursor.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
    connection.commit()

if __name__ == "__main__":
    thread_history = get_messages(1)
    
    for msg_pair in thread_history:
        if msg_pair['user_message']:
            print(msg_pair['user_message'])
        if msg_pair['ai_message']:
            print(msg_pair['ai_message'])
        if msg_pair['summary']:
            print(msg_pair['summary'])
