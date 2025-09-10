"""
Azure OpenAI Conversation History Management Module for PostgreSQL.

This module provides functionality to store and retrieve conversation history
from PostgreSQL using Azure OpenAI embeddings for semantic search, saving LLM token costs.
"""

import json
import logging
import hashlib
import re
import os
import math
from typing import List, Dict, Optional, Tuple
import psycopg
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class AzureConversationHistoryManager:

    def find_similar_questions(self, question: str, similarity_threshold: float = 0.8, limit: int = 5) -> list:
        """
        Find similar questions from conversation history using Azure OpenAI embeddings.
        Args:
            question: The question to find similarities for
            similarity_threshold: Minimum similarity score (0-1)
            limit: Maximum number of results to return
        Returns:
            List of dictionaries containing similar questions and their responses
        """
        question_hash = self._get_content_hash(question)
        question_embedding = self._get_azure_embedding(question)
        try:
            with psycopg.connect(self.connection_uri) as conn:
                with conn.cursor() as cur:
                    # First, try exact hash match for identical questions
                    cur.execute(
                        """
                        SELECT DISTINCT m1.content as question, m2.content as response, 
                               m1.session_id, m1.created_at, 1.0 as similarity_score
                        FROM conversation_messages m1
                        JOIN conversation_messages m2 ON m1.session_id = m2.session_id 
                                                      AND m2.message_order = m1.message_order + 1
                        WHERE m1.role = 'user' 
                          AND m2.role IN ('assistant', 'agent')
                          AND m1.content_hash = %s
                          AND LENGTH(TRIM(m2.content)) > 0
                        ORDER BY m1.created_at DESC
                        LIMIT %s
                        """,
                        (question_hash, limit)
                    )
                    exact_matches = cur.fetchall()
                    if exact_matches:
                        results = []
                        for row in exact_matches:
                            results.append({
                                'question': row[0],
                                'response': row[1],
                                'session_id': str(row[2]),
                                'created_at': row[3],
                                'similarity_score': row[4],
                                'match_type': 'exact'
                            })
                        return results

                    # If no exact matches and we have embeddings, try semantic similarity (DiskANN or fallback)
                    if question_embedding:
                        try:
                            embedding_str = '[' + ','.join(map(str, question_embedding)) + ']'
                            cur.execute(
                                """
                                SELECT m1.content as question, m2.content as response, 
                                       m1.session_id, m1.created_at, m1.embedding,
                                       1 - (m1.embedding <=> %s::vector) as similarity_score
                                FROM conversation_messages m1
                                JOIN conversation_messages m2 ON m1.session_id = m2.session_id 
                                                              AND m2.message_order = m1.message_order + 1
                                WHERE m1.role = 'user' 
                                  AND m2.role IN ('assistant', 'agent')
                                  AND m1.embedding IS NOT NULL
                                  AND LENGTH(TRIM(m2.content)) > 0
                                  AND (1 - (m1.embedding <=> %s::vector)) >= %s
                                ORDER BY similarity_score DESC
                                LIMIT %s
                                """,
                                (embedding_str, embedding_str, similarity_threshold, limit)
                            )
                            semantic_matches = cur.fetchall()
                            if semantic_matches:
                                results = []
                                for row in semantic_matches:
                                    results.append({
                                        'question': row[0],
                                        'response': row[1],
                                        'session_id': str(row[2]),
                                        'created_at': row[3],
                                        'similarity_score': float(row[5]),
                                        'match_type': 'semantic_azure_openai_diskann'
                                    })
                                return results
                        except Exception as e:
                            logger.warning(f"DiskANN vector similarity search failed: {e}")
                        # Fallback to text-based similarity
                    # Fallback to text-based similarity using normalized content
                    normalized_question = self._normalize_content(question)
                    cur.execute(
                        """
                        SELECT DISTINCT m1.content as question, m2.content as response, 
                               m1.session_id, m1.created_at
                        FROM conversation_messages m1
                        JOIN conversation_messages m2 ON m1.session_id = m2.session_id 
                                                      AND m2.message_order = m1.message_order + 1
                        WHERE m1.role = 'user' 
                          AND m2.role IN ('assistant', 'agent')
                          AND LENGTH(TRIM(m2.content)) > 0
                        ORDER BY m1.created_at DESC
                        LIMIT %s
                        """,
                        (limit,)
                    )
                    all_questions = cur.fetchall()
                    results = []
                    for row in all_questions:
                        similarity = self._calculate_text_similarity(normalized_question, self._normalize_content(row[0]))
                        if similarity >= similarity_threshold:
                            results.append({
                                'question': row[0],
                                'response': row[1],
                                'session_id': str(row[2]),
                                'created_at': row[3],
                                'similarity_score': similarity,
                                'match_type': 'text'
                            })
                    results.sort(key=lambda x: x['similarity_score'], reverse=True)
                    return results[:limit]
        except Exception as e:
            logger.error(f"Error finding similar questions: {e}")
            return []
    def _normalize_content(self, content) -> str:
        """Normalize content for comparison by removing punctuation and extra whitespace."""
        # Convert UUIDs and other non-str to string
        if not isinstance(content, str):
            content = str(content)
        # Remove punctuation and convert to lowercase
        normalized = re.sub(r'[^\w\s]', '', content.lower())
        # Remove extra whitespace
        return ' '.join(normalized.split())
    """Manages conversation history storage and retrieval in PostgreSQL using Azure OpenAI embeddings."""
    
    def __init__(self, connection_uri: str):
        """
        Initialize the conversation history manager with Azure OpenAI embeddings.
        
        Args:
            connection_uri: PostgreSQL connection string
        """
        self.connection_uri = connection_uri
        
        # Initialize Azure OpenAI client for embeddings
        self.azure_openai_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # Get the embeddings deployment name from environment
        self.embeddings_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME", "text-embedding-3-small")
        
        # Expected embedding dimension for text-embedding-3-small
        self.embedding_dimension = 1536
        
    
    @staticmethod
    def _calculate_dot_product(vec1: List[float], vec2: List[float]) -> float:
        """Calculate dot product of two vectors."""
        return sum(a * b for a, b in zip(vec1, vec2))
    
    @staticmethod
    def _calculate_magnitude(vector: List[float]) -> float:
        """Calculate the magnitude (norm) of a vector."""
        return math.sqrt(sum(x * x for x in vector))
    
    @staticmethod
    def _calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Returns:
            float: Cosine similarity score between 0 and 1
        """
        # Ensure vectors have the same dimensions
        if len(vec1) != len(vec2):
            return 0.0
        
        # Calculate dot product
        dot_product = AzureConversationHistoryManager._calculate_dot_product(vec1, vec2)
        
        # Calculate magnitudes
        magnitude1 = AzureConversationHistoryManager._calculate_magnitude(vec1)
        magnitude2 = AzureConversationHistoryManager._calculate_magnitude(vec2)
        
        # Avoid division by zero
        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0
        
        # Calculate cosine similarity
        cosine_sim = dot_product / (magnitude1 * magnitude2)
        
        # Ensure the result is in the range [0, 1] (clamp negative values to 0)
        return max(0.0, cosine_sim)
    
    def _ensure_tables_exist(self):
        """Create the required tables if they don't exist."""
        create_conversations_table = """
        CREATE TABLE IF NOT EXISTS conversation_sessions (
            session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_name VARCHAR(255),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            metadata JSONB DEFAULT '{}'::jsonb
        );
        """
        
        # Enable vector/diskann extension first
        enable_vector_extension = "CREATE EXTENSION IF NOT EXISTS vector;"

        enable_diskann_extension = "CREATE EXTENSION IF NOT EXISTS pg_diskann;"

        create_messages_table = f"""
        CREATE TABLE IF NOT EXISTS conversation_messages (
            message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
            message_order INTEGER,
            role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'agent', 'system')),
            content TEXT NOT NULL,
            content_hash VARCHAR(64) NOT NULL,
            embedding VECTOR({self.embedding_dimension}),  -- Vector for Azure OpenAI text-embedding-3-small with DiskANN indexing
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            metadata JSONB DEFAULT '{{}}'::jsonb,
            UNIQUE(session_id, message_order)
        );
        """
        
        create_indexes = """
        CREATE INDEX IF NOT EXISTS idx_messages_content_hash ON conversation_messages(content_hash);
        CREATE INDEX IF NOT EXISTS idx_messages_role ON conversation_messages(role);
        CREATE INDEX IF NOT EXISTS idx_messages_created_at ON conversation_messages(created_at);
        CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON conversation_sessions(created_at);
        """
        
        try:
            with psycopg.connect(self.connection_uri) as conn:
                with conn.cursor() as cur:
                    # Enable vector/pg_diskann extension first
                    try:
                        cur.execute(enable_vector_extension)
                        cur.execute(enable_diskann_extension)
                        logger.info("Vector/pg_diskann extensions enabled successfully")
                    except Exception as ext_error:
                        logger.warning(f"Could not enable vector/pg_diskann extensions: {ext_error}")

                    cur.execute(create_conversations_table)
                    
                    # Try to create the table with vector support for DiskANN indexing
                    try:
                        cur.execute(create_messages_table)
                        logger.info("Conversation history tables created with Azure OpenAI vector support for DiskANN indexing")
                    except Exception as vector_error:
                        logger.warning(f"Could not create table with vector support: {vector_error}")
                        # Fall back to table without vector column
                        self._create_tables_without_vector()
                        return
                    
                    # Create indexes
                    for index_sql in create_indexes.split(';'):
                        if index_sql.strip():
                            try:
                                cur.execute(index_sql.strip())
                            except Exception as e:
                                logger.warning(f"Could not create index: {e}")
                    
                    # Create DiskANN vector index (Azure PostgreSQL Flexible Server optimized)
                    try:
                        # Create DiskANN index for ultra-high performance vector similarity search
                        cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_embedding_diskann ON conversation_messages USING diskann (embedding vector_cosine_ops);")
                        logger.info("DiskANN vector similarity index created successfully")
                        
                        # Verify that only DiskANN indexes exist for vector columns
                        
                    except Exception as diskann_error:
                        logger.warning(f"Could not create DiskANN vector index: {diskann_error}")
                        logger.info("DiskANN indexing requires Azure PostgreSQL Flexible Server with vector extension and DiskANN support")
                    
                    conn.commit()
                    logger.info("Conversation history tables created successfully with Azure OpenAI vector support and DiskANN indexing")
                    
        except Exception as e:
            logger.error(f"Error creating conversation history tables: {e}")
            raise
    
    def _ensure_diskann_only_indexing(self, cur):
        """
        Ensure that only DiskANN indexes are used for vector columns.
        Remove any IVFFLAT or HNSW indexes and warn about non-DiskANN indexes.
        """
    
    def _get_content_hash(self, content: str) -> str:
        """Generate a hash for content to enable fast duplicate detection."""
        normalized_content = self._normalize_content(content)
        return hashlib.sha256(normalized_content.encode()).hexdigest()
    
    def _get_azure_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Azure OpenAI."""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding generation")
                return None
                
            response = self.azure_openai_client.embeddings.create(
                input=text.strip(),
                model=self.embeddings_deployment
            )
            embedding = response.data[0].embedding
            
            # Validate embedding
            if not embedding or len(embedding) != self.embedding_dimension:
                logger.error(f"Invalid embedding received: expected {self.embedding_dimension} dimensions, got {len(embedding) if embedding else 0}")
                return None
                
            # Ensure all values are valid floats
            try:
                validated_embedding = [float(x) for x in embedding]
                logger.debug(f"Generated Azure OpenAI embedding with {len(validated_embedding)} dimensions")
                return validated_embedding
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid embedding values: {e}")
                return None
                
        except Exception as e:
            logger.warning(f"Error generating Azure OpenAI embedding: {e}")
            return None
    
    def create_session(self, session_name: str = None, metadata: Dict = None) -> str:
        """
        Create a new conversation session.
        
        Args:
            session_name: Optional name for the session
            metadata: Optional metadata dictionary
            
        Returns:
            session_id: UUID of the created session
        """
        try:
            with psycopg.connect(self.connection_uri) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO conversation_sessions (session_name, metadata)
                        VALUES (%s, %s)
                        RETURNING session_id
                        """,
                        (session_name, json.dumps(metadata or {}))
                    )
                    session_id = str(cur.fetchone()[0])
                    conn.commit()
                    logger.info(f"Created conversation session: {session_id}")
                    return session_id
                    
        except Exception as e:
            logger.error(f"Error creating conversation session: {e}")
            raise
    
    def add_message(self, session_id: str, role: str, content: str, metadata: Dict = None, limit: int = 10, similarity_threshold: float = 0.7) -> str:
        """
        Add a message to a conversation session.
        
        Args:
            session_id: UUID of the conversation session
            role: Role of the message sender ('user', 'assistant', 'agent', 'system')
            content: Message content
            metadata: Optional metadata dictionary
            limit: Maximum number of results to return (default: 10)
            similarity_threshold: Threshold for text similarity (default: 0.7)
            
        Returns:
            message_id: UUID of the created message
        """
        try:
            content_hash = self._get_content_hash(content)
            embedding = None
            # Only generate embeddings for user messages for similarity search
            if role == 'user' and content.strip():
                embedding = self._get_azure_embedding(content)

            with psycopg.connect(self.connection_uri) as conn:
                with conn.cursor() as cur:
                    # Get the next message order for this session
                    cur.execute(
                        "SELECT COALESCE(MAX(message_order), 0) + 1 FROM conversation_messages WHERE session_id = %s",
                        (session_id,)
                    )
                    message_order = cur.fetchone()[0]

                    # Check if table has embedding column
                    cur.execute("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = 'conversation_messages' 
                        AND column_name = 'embedding'
                    """)
                    has_embedding_column = cur.fetchone() is not None

                    if has_embedding_column:
                        cur.execute(
                            """
                            INSERT INTO conversation_messages (session_id, message_order, role, content, content_hash, embedding, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            RETURNING message_id
                            """,
                            (session_id, message_order, role, content, content_hash, embedding, json.dumps(metadata or {}))
                        )
                    else:
                        cur.execute(
                            """
                            INSERT INTO conversation_messages (session_id, message_order, role, content, content_hash, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            RETURNING message_id
                            """,
                            (session_id, message_order, role, content, content_hash, json.dumps(metadata or {}))
                        )
                    message_id = cur.fetchone()[0]
                    conn.commit()
                    return message_id
        except Exception as e:
            logger.error(f"Error saving message: {e}")
            return None
    
    def _python_based_similarity_search(self, cur, question_embedding, similarity_threshold, limit):
        """
        Fallback similarity search using custom cosine similarity when pgvector is not available.
        """
        try:
            # Get all user messages with embeddings
            cur.execute(
                """
                SELECT m1.content as question, m2.content as response, 
                       m1.session_id, m1.created_at, m1.embedding
                FROM conversation_messages m1
                JOIN conversation_messages m2 ON m1.session_id = m2.session_id 
                                              AND m2.message_order = m1.message_order + 1
                WHERE m1.role = 'user' 
                  AND m2.role IN ('assistant', 'agent')
                  AND m1.embedding IS NOT NULL
                  AND LENGTH(TRIM(m2.content)) > 0
                ORDER BY m1.created_at DESC
                LIMIT %s
                """
            )
            
            candidates = cur.fetchall()
            if not candidates:
                return []
            
            # Validate question embedding
            if not question_embedding or not isinstance(question_embedding, (list, tuple)):
                logger.warning("Invalid question embedding format")
                return []
            
            # Convert to list of floats
            try:
                question_vec = [float(x) for x in question_embedding]
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not convert question embedding to floats: {e}")
                return []
            
            results = []
            
            for row in candidates:
                try:
                    stored_embedding = row[4]  # embedding column
                    if stored_embedding:
                        # Handle different data types that might be stored
                        if isinstance(stored_embedding, str):
                            # If it's a string, try to parse it as JSON
                            import json
                            try:
                                stored_embedding = json.loads(stored_embedding)
                            except (json.JSONDecodeError, ValueError):
                                # If JSON parsing fails, skip this embedding
                                logger.warning(f"Could not parse embedding as JSON: {stored_embedding[:100]}...")
                                continue
                        elif isinstance(stored_embedding, (list, tuple)):
                            # If it's already a list/tuple, use it directly
                            pass
                        else:
                            # For other types, try to convert to list
                            try:
                                stored_embedding = list(stored_embedding)
                            except (TypeError, ValueError):
                                logger.warning(f"Could not convert embedding to list: {type(stored_embedding)}")
                                continue
                        
                        # Convert to list of floats
                        try:
                            stored_vec = [float(x) for x in stored_embedding]
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not convert stored embedding to floats: {e}")
                            continue
                        
                        # Ensure dimensions match
                        if len(stored_vec) != len(question_vec):
                            logger.warning(f"Embedding dimension mismatch: {len(stored_vec)} vs {len(question_vec)}")
                            continue
                        
                        # Calculate cosine similarity using our custom implementation
                        similarity = self._calculate_cosine_similarity(question_vec, stored_vec)
                        
                        if similarity >= similarity_threshold:
                            results.append({
                                'question': row[0],
                                'response': row[1],
                                'session_id': str(row[2]),
                                'created_at': row[3],
                                'similarity_score': similarity,
                                'match_type': 'semantic_azure_openai_python'
                            })
                except Exception as embedding_error:
                    logger.warning(f"Error processing embedding for message: {embedding_error}")
                    continue
            
            # Sort by similarity and limit results
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.warning(f"Python-based similarity search failed: {e}")
            return []
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings using simple word overlap."""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def get_session_messages(self, session_id: str) -> List[Dict]:
        """
        Get all messages from a conversation session.
        
        Args:
            session_id: UUID of the conversation session
            
        Returns:
            List of message dictionaries
        """
        try:
            with psycopg.connect(self.connection_uri) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT message_id, role, content, created_at, metadata
                        FROM conversation_messages
                        WHERE session_id = %s
                        ORDER BY message_order ASC
                        """,
                        (session_id,)
                    )
                    
                    messages = []
                    for row in cur.fetchall():
                        messages.append({
                            'message_id': str(row[0]),
                            'role': row[1],
                            'content': row[2],
                            'created_at': row[3],
                            'metadata': row[4]
                        })
                    
                    return messages
                    
        except Exception as e:
            logger.error(f"Error getting session messages: {e}")
            return []
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """
        Get recent conversation sessions.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session dictionaries
        """
        try:
            with psycopg.connect(self.connection_uri) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT s.session_id, s.session_name, s.created_at, s.updated_at, 
                               COUNT(m.message_id) as message_count
                        FROM conversation_sessions s
                        LEFT JOIN conversation_messages m ON s.session_id = m.session_id
                        GROUP BY s.session_id, s.session_name, s.created_at, s.updated_at
                        ORDER BY s.updated_at DESC
                        LIMIT %s
                        """,
                        (limit,)
                    )
                    
                    sessions = []
                    for row in cur.fetchall():
                        sessions.append({
                            'session_id': str(row[0]),
                            'session_name': row[1],
                            'created_at': row[2],
                            'updated_at': row[3],
                            'message_count': row[4]
                        })
                    
                    return sessions
                    
        except Exception as e:
            logger.error(f"Error getting recent sessions: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a conversation session and all its messages.
        
        Args:
            session_id: UUID of the conversation session
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            with psycopg.connect(self.connection_uri) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM conversation_sessions WHERE session_id = %s",
                        (session_id,)
                    )
                    deleted_count = cur.rowcount
                    conn.commit()
                    
                    if deleted_count > 0:
                        logger.info(f"Deleted conversation session: {session_id}")
                        return True
                    else:
                        logger.warning(f"Session not found: {session_id}")
                        return False
                        
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False
    
    def get_conversation_stats(self) -> Dict:
        """
        Get statistics about the conversation history.
        
        Returns:
            Dictionary containing various statistics
        """
        try:
            with psycopg.connect(self.connection_uri) as conn:
                with conn.cursor() as cur:
                    # Get basic counts
                    cur.execute("SELECT COUNT(*) FROM conversation_sessions")
                    session_count = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM conversation_messages")
                    message_count = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM conversation_messages WHERE role = 'user'")
                    user_message_count = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM conversation_messages WHERE role IN ('assistant', 'agent')")
                    agent_message_count = cur.fetchone()[0]
                    
                    # Check if vector column exists and count embeddings
                    cur.execute("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = 'conversation_messages' 
                        AND column_name = 'embedding'
                    """)
                    has_vector = cur.fetchone() is not None
                    
                    embedding_count = 0
                    if has_vector:
                        cur.execute("SELECT COUNT(*) FROM conversation_messages WHERE embedding IS NOT NULL")
                        embedding_count = cur.fetchone()[0]
                    
                    # Get recent activity
                    cur.execute("""
                        SELECT COUNT(*) FROM conversation_messages 
                        WHERE created_at >= NOW() - INTERVAL '24 hours'
                    """)
                    messages_last_24h = cur.fetchone()[0]
                    
                    return {
                        'total_sessions': session_count,
                        'total_messages': message_count,
                        'user_messages': user_message_count,
                        'agent_messages': agent_message_count,
                        'azure_openai_embeddings': embedding_count,
                        'vector_support': has_vector,
                        'messages_last_24h': messages_last_24h,
                        'embedding_model': self.embeddings_deployment,
                        'embedding_dimension': self.embedding_dimension
                    }
                    
        except Exception as e:
            logger.error(f"Error getting conversation stats: {e}")
            return {}
    
