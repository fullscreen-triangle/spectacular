from typing import List, Dict, Optional, Any, Union, Callable
import sqlite3
import pickle
import re
from pathlib import Path

from .schema import ApiElement, ApiElementType
from .storage import KnowledgeStorage


class KnowledgeQuery:
    """Query interface for retrieving D3 API knowledge."""
    
    def __init__(self, storage: Optional[KnowledgeStorage] = None):
        """Initialize the knowledge query.
        
        Args:
            storage: The knowledge storage to query. If None, a new one is created.
        """
        self.storage = storage or KnowledgeStorage()
    
    def search(self, query: str, limit: int = 10) -> List[ApiElement]:
        """Search for API elements matching the query.
        
        This performs a simple text search on element names and descriptions.
        
        Args:
            query: The search query
            limit: Maximum number of results to return
            
        Returns:
            A list of API elements matching the query
        """
        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()
        
        # Simple search on name and description
        cursor.execute(
            """
            SELECT serialized FROM api_elements
            WHERE name LIKE ? OR description LIKE ?
            LIMIT ?
            """,
            (f"%{query}%", f"%{query}%", limit)
        )
        results = cursor.fetchall()
        
        conn.close()
        
        return [pickle.loads(result[0]) for result in results]
    
    def advanced_search(self, 
                       name_pattern: Optional[str] = None,
                       element_type: Optional[ApiElementType] = None,
                       tags: Optional[List[str]] = None,
                       version: Optional[str] = None,
                       limit: int = 10) -> List[ApiElement]:
        """Search for API elements with more specific criteria.
        
        Args:
            name_pattern: Pattern to match against element names
            element_type: Type of elements to search for
            tags: List of tags that elements must have
            version: Version in which elements are available
            limit: Maximum number of results to return
            
        Returns:
            A list of API elements matching the criteria
        """
        conn = sqlite3.connect(self.storage.db_path)
        cursor = conn.cursor()
        
        query_parts = ["SELECT a.serialized FROM api_elements a"]
        conditions = []
        params = []
        
        # Add conditions based on provided parameters
        if name_pattern:
            conditions.append("a.name LIKE ?")
            params.append(f"%{name_pattern}%")
        
        if element_type:
            conditions.append("a.type = ?")
            params.append(element_type.name)
        
        if version:
            conditions.append("(a.version_introduced <= ? OR a.version_introduced IS NULL)")
            params.append(version)
            conditions.append("(a.version_deprecated IS NULL OR a.version_deprecated > ?)")
            params.append(version)
        
        # Handle tags with a JOIN if needed
        if tags and len(tags) > 0:
            tag_placeholders = ", ".join(["?"] * len(tags))
            query_parts.append("""
                JOIN element_tags et ON a.id = et.element_id
                JOIN tags t ON et.tag_id = t.id
                WHERE t.name IN ({})
            """.format(tag_placeholders))
            params.extend(tags)
            
            # Count the number of matching tags to ensure all required tags are present
            query_parts.append("""
                GROUP BY a.id
                HAVING COUNT(DISTINCT t.id) = ?
            """)
            params.append(len(tags))
        elif conditions:
            query_parts.append("WHERE " + " AND ".join(conditions))
        
        # Add the limit
        query_parts.append("LIMIT ?")
        params.append(limit)
        
        # Execute the query
        cursor.execute(" ".join(query_parts), params)
        results = cursor.fetchall()
        
        conn.close()
        
        return [pickle.loads(result[0]) for result in results]
    
    def get_similar_elements(self, element: ApiElement, limit: int = 5) -> List[ApiElement]:
        """Find API elements similar to the given element.
        
        Args:
            element: The API element to find similar elements for
            limit: Maximum number of results to return
            
        Returns:
            A list of similar API elements
        """
        # Get elements of the same type
        same_type = self.storage.get_elements_by_type(element.type)
        
        # Filter out the input element
        same_type = [e for e in same_type if e.name != element.name]
        
        # Score elements based on shared tags and related elements
        scored_elements = []
        for e in same_type:
            score = 0
            
            # Score based on shared tags
            if element.tags and e.tags:
                shared_tags = set(element.tags).intersection(set(e.tags))
                score += len(shared_tags) * 2
            
            # Score based on related elements
            if element.related_elements and e.related_elements:
                shared_related = set(element.related_elements).intersection(set(e.related_elements))
                score += len(shared_related) * 3
            
            # Score based on name similarity (simple word overlap)
            element_words = set(re.findall(r'\w+', element.name.lower()))
            e_words = set(re.findall(r'\w+', e.name.lower()))
            shared_words = element_words.intersection(e_words)
            score += len(shared_words)
            
            scored_elements.append((e, score))
        
        # Sort by score (descending) and return top elements
        scored_elements.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored_elements[:limit]]
    
    def get_related_elements(self, element_name: str) -> List[ApiElement]:
        """Get elements related to the specified element.
        
        Args:
            element_name: Name of the element to find relations for
            
        Returns:
            A list of related API elements
        """
        # Get the element
        element = self.storage.get_element_by_name(element_name)
        if not element or not element.related_elements:
            return []
        
        # Get all related elements
        related = []
        for related_name in element.related_elements:
            related_element = self.storage.get_element_by_name(related_name)
            if related_element:
                related.append(related_element)
        
        return related
    
    def find_usage_patterns(self, visualization_type: Optional[str] = None) -> List[ApiElement]:
        """Find API elements with specific usage patterns.
        
        Args:
            visualization_type: Optional filter for visualization type
            
        Returns:
            A list of API elements with matching usage patterns
        """
        elements = self.storage.get_all_elements()
        
        # Filter elements with usage patterns
        with_patterns = [e for e in elements if e.usage_patterns and len(e.usage_patterns) > 0]
        
        # Further filter by visualization type if specified
        if visualization_type:
            with_patterns = [
                e for e in with_patterns 
                if any(pattern.visualization_type and pattern.visualization_type.name == visualization_type
                      for pattern in e.usage_patterns)
            ]
        
        return with_patterns 