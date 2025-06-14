import json
import os
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import sqlite3
import pickle

from .schema import ApiElement, ApiElementType


class KnowledgeStorage:
    """Manages storage and retrieval of D3 API knowledge."""
    
    def __init__(self, storage_path: Union[str, Path] = None):
        """Initialize the knowledge storage.
        
        Args:
            storage_path: Path to the storage directory. If None, a default location is used.
        """
        if storage_path is None:
            storage_path = Path.home() / ".d3-neuro" / "knowledge"
        
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_path / "knowledge.db"
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_elements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            description TEXT,
            serialized BLOB NOT NULL,
            source_url TEXT,
            version_introduced TEXT,
            version_deprecated TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS element_tags (
            element_id INTEGER,
            tag_id INTEGER,
            PRIMARY KEY (element_id, tag_id),
            FOREIGN KEY (element_id) REFERENCES api_elements(id),
            FOREIGN KEY (tag_id) REFERENCES tags(id)
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS related_elements (
            source_id INTEGER,
            target_id INTEGER,
            relationship_type TEXT,
            PRIMARY KEY (source_id, target_id),
            FOREIGN KEY (source_id) REFERENCES api_elements(id),
            FOREIGN KEY (target_id) REFERENCES api_elements(id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_element(self, element: ApiElement) -> int:
        """Add a new API element to the knowledge base.
        
        Args:
            element: The API element to add
            
        Returns:
            The ID of the newly added element
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize the element
        serialized = pickle.dumps(element)
        
        # Insert the element
        cursor.execute(
            '''
            INSERT INTO api_elements (name, type, description, serialized, source_url, 
                                     version_introduced, version_deprecated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''',
            (element.name, element.type.name, element.description, serialized,
             element.source_url, element.version_introduced, element.version_deprecated)
        )
        
        element_id = cursor.lastrowid
        
        # Add tags
        if element.tags:
            for tag in element.tags:
                # Get or create tag
                cursor.execute("SELECT id FROM tags WHERE name = ?", (tag,))
                result = cursor.fetchone()
                
                if result:
                    tag_id = result[0]
                else:
                    cursor.execute("INSERT INTO tags (name) VALUES (?)", (tag,))
                    tag_id = cursor.lastrowid
                
                # Link tag to element
                cursor.execute(
                    "INSERT INTO element_tags (element_id, tag_id) VALUES (?, ?)",
                    (element_id, tag_id)
                )
        
        conn.commit()
        conn.close()
        
        return element_id
    
    def get_element(self, element_id: int) -> Optional[ApiElement]:
        """Get an API element by ID.
        
        Args:
            element_id: The ID of the element to retrieve
            
        Returns:
            The API element, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT serialized FROM api_elements WHERE id = ?", (element_id,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return pickle.loads(result[0])
        return None
    
    def get_element_by_name(self, name: str) -> Optional[ApiElement]:
        """Get an API element by its name.
        
        Args:
            name: The name of the API element
            
        Returns:
            The API element, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT serialized FROM api_elements WHERE name = ?", (name,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return pickle.loads(result[0])
        return None
    
    def get_all_elements(self) -> List[ApiElement]:
        """Get all API elements in the knowledge base.
        
        Returns:
            A list of all API elements
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT serialized FROM api_elements")
        results = cursor.fetchall()
        
        conn.close()
        
        return [pickle.loads(result[0]) for result in results]
    
    def get_elements_by_type(self, element_type: ApiElementType) -> List[ApiElement]:
        """Get all API elements of a specific type.
        
        Args:
            element_type: The type of API elements to retrieve
            
        Returns:
            A list of API elements of the specified type
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT serialized FROM api_elements WHERE type = ?", (element_type.name,))
        results = cursor.fetchall()
        
        conn.close()
        
        return [pickle.loads(result[0]) for result in results]
    
    def get_elements_by_tag(self, tag: str) -> List[ApiElement]:
        """Get all API elements with a specific tag.
        
        Args:
            tag: The tag to filter by
            
        Returns:
            A list of API elements with the specified tag
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT a.serialized 
            FROM api_elements a
            JOIN element_tags et ON a.id = et.element_id
            JOIN tags t ON et.tag_id = t.id
            WHERE t.name = ?
        """, (tag,))
        results = cursor.fetchall()
        
        conn.close()
        
        return [pickle.loads(result[0]) for result in results]
    
    def update_element(self, element_id: int, element: ApiElement) -> bool:
        """Update an existing API element.
        
        Args:
            element_id: The ID of the element to update
            element: The updated API element
            
        Returns:
            True if the update was successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Serialize the element
        serialized = pickle.dumps(element)
        
        # Update the element
        cursor.execute(
            """
            UPDATE api_elements 
            SET name = ?, type = ?, description = ?, serialized = ?, 
                source_url = ?, version_introduced = ?, version_deprecated = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (element.name, element.type.name, element.description, serialized,
             element.source_url, element.version_introduced, element.version_deprecated,
             element_id)
        )
        
        if cursor.rowcount == 0:
            conn.close()
            return False
        
        # Clear existing tags
        cursor.execute("DELETE FROM element_tags WHERE element_id = ?", (element_id,))
        
        # Add new tags
        if element.tags:
            for tag in element.tags:
                # Get or create tag
                cursor.execute("SELECT id FROM tags WHERE name = ?", (tag,))
                result = cursor.fetchone()
                
                if result:
                    tag_id = result[0]
                else:
                    cursor.execute("INSERT INTO tags (name) VALUES (?)", (tag,))
                    tag_id = cursor.lastrowid
                
                # Link tag to element
                cursor.execute(
                    "INSERT INTO element_tags (element_id, tag_id) VALUES (?, ?)",
                    (element_id, tag_id)
                )
        
        conn.commit()
        conn.close()
        
        return True
    
    def delete_element(self, element_id: int) -> bool:
        """Delete an API element.
        
        Args:
            element_id: The ID of the element to delete
            
        Returns:
            True if the deletion was successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete related tags
        cursor.execute("DELETE FROM element_tags WHERE element_id = ?", (element_id,))
        
        # Delete the element
        cursor.execute("DELETE FROM api_elements WHERE id = ?", (element_id,))
        
        success = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return success 