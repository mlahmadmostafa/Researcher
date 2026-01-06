import sqlite3
import os
from src.models import PaperRelevance

def init_db(db_path: str):
    """Initialize the SQLite database."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            relevance_score INTEGER,
            is_relevant BOOLEAN,
            summary TEXT,
            md_file_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_paper_evaluation(db_path: str, evaluation: PaperRelevance, md_file_path: str):
    """Save paper evaluation to the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO papers (id, relevance_score, is_relevant, summary, md_file_path)
        VALUES (?, ?, ?, ?, ?)
    ''', (evaluation.id, evaluation.relevance_score, evaluation.is_relevant, evaluation.summary, md_file_path))
    conn.commit()
    conn.close()

def get_relevant_summaries(db_path: str, min_score: int = 5) -> str:
    """Get all relevant summaries from the database."""
    if not os.path.exists(db_path):
        return ""
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, summary, relevance_score FROM papers 
        WHERE is_relevant = 1 AND relevance_score >= ?
        ORDER BY relevance_score DESC
    ''', (min_score,))
    rows = cursor.fetchall()
    conn.close()
    
    summaries = []
    for row in rows:
        summaries.append(f"Paper ID: {row[0]}\nScore: {row[2]}\nSummary: {row[1]}\n---")
    
    return "\n".join(summaries)
