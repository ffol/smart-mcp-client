import os
import warnings
import logging
import asyncio
import uuid
import re

# Set environment variable to suppress google-crc32c warnings
os.environ['GOOGLE_CRC32C_IGNORE_IMPORT_ERROR'] = '1'

# Suppress warnings before any imports
warnings.filterwarnings("ignore", category=ResourceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.mcp import MCPStreamableHttpPlugin

# Import the Azure OpenAI conversation history manager
from azure_conversation_history import AzureConversationHistoryManager as ConversationHistoryManager
HISTORY_TYPE = "azure_openai"

from tokengen import TokenValidation
from dotenv import load_dotenv
load_dotenv()

# Remove INFO:httpx from output
logging.getLogger("httpx").setLevel(logging.WARNING)

def normalize_text(text):
    """Normalize text by removing punctuation and converting to lowercase for comparison."""
    import re
    # Remove punctuation and convert to lowercase
    normalized = re.sub(r'[^\w\s]', '', str(text).lower())
    # Remove extra whitespace
    return ' '.join(normalized.split())

class SmartMCPClientImproved:
    """Enhanced MCP Client with improved PostgreSQL conversation history for better token savings."""
    
    def __init__(self, mcp_url=None, similarity_threshold=0.80):
        # Use environment variable if mcp_url is not provided
        if mcp_url is None:
            self.mcp_url = os.getenv('MCP_URL', 'http://localhost:8003/mcp')
            print(f"🔗 Using MCP URL from environment: {self.mcp_url}")
        else:
            self.mcp_url = mcp_url
            print(f"🔗 Using provided MCP URL: {self.mcp_url}")
        self.similarity_threshold = similarity_threshold  # Improved default threshold
        self.agent_instance = None
        self.thread_instance = None
        self.local_plugin = None
        self.session_id = None
        self.conversation_history_enabled = True
        self.conversation_manager = None
        self._init_messages_saved = False
        self._last_saved_messages = set()
        self._currently_saving = set()
        
    @staticmethod
    def normalize_text(text):
        """Normalize text by removing punctuation and converting to lowercase for comparison."""
        import re
        # Remove punctuation and convert to lowercase
        normalized = re.sub(r'[^\w\s]', '', str(text).lower())
        # Remove extra whitespace
        return ' '.join(normalized.split())
        
    async def _init_conversation_manager(self):
        """Initialize the conversation history manager with Azure AD authentication."""
        try:
            # Get database connection using environment variables and AAD token
            token_validator = TokenValidation()
            token_validator.check_and_refresh_token()
            token = token_validator.get_password()
            
            # Get connection details from environment variables
            pg_user = os.getenv('PGUSER', 'pgflexsp17')
            pg_host = os.getenv('PGHOST', 'pgflexscs17.postgres.database.azure.com')
            pg_database = os.getenv('PGDATABASE', 'advworks')
            
            # For Azure PostgreSQL Flexible Server with AAD, use just the username
            username = pg_user
            
            # Build the connection URI
            import urllib.parse
            encoded_user = urllib.parse.quote(username)
            connection_uri = f"postgresql://{encoded_user}:{token}@{pg_host}:5432/{pg_database}?sslmode=require"
            
            self.conversation_manager = ConversationHistoryManager(connection_uri)
            print(f"✅ Improved conversation history manager initialized ({HISTORY_TYPE} version)")
        except Exception as e:
            print(f"⚠️ Could not initialize conversation manager: {e}")
            self.conversation_history_enabled = False
        
    async def init_chat(self):
        """Initialize the chat agent with improved conversation history support."""
        try:
            # Initialize conversation history manager first
            await self._init_conversation_manager()
            
            # Connect to MCP server via HTTP
            self.local_plugin = MCPStreamableHttpPlugin(
                url=self.mcp_url,
                name="LocalPlugin",
                description="Local Resources Plugin with Improved Conversation History",
                load_tools=True,
            )
            
            # Connect to the plugin
            await self.local_plugin.connect()
            
            # Create the agent
            self.agent_instance = ChatCompletionAgent(
                service=AzureChatCompletion(),
                name="Agent",
                instructions=(
                    "You are a helpful agent that can answer questions about Azure postgres databases. "
                    "You have access to conversation history tools that can save LLM tokens by finding similar previously asked questions. "
                    f"Always check for similar questions first using find_similar_questions with similarity_threshold={self.similarity_threshold} before generating new responses. "
                    "If similar questions are found with good responses that match the current query's intent and requirements, use those responses and mention that you're using cached results to save tokens. "
                    "Save all conversation messages using save_conversation_message for future reference. "
                    "When connected to postgres resource, just say 'connected to postgres resource'. Do not share any additional information. "
                    "You can use provided tools to answer questions about Azure postgres databases. "
                    "Do not include resource group name, username and subscription id when giving answers."
                ),
                plugins=[self.local_plugin],
            )
            
            # Create a ChatHistoryAgentThread explicitly
            self.thread_instance = ChatHistoryAgentThread()
            
            # Create a new conversation session in PostgreSQL
            if self.conversation_history_enabled:
                await self._create_conversation_session()
            
            # Set context for postgres resource
            init_input = (
                "connect to postgres database {advworks} using tools connection details. "
                "Do not include resource group name, username and subscription id when giving answers."
            )
            
            # Store the database context for later extraction
            import re
            db_match = re.search(r'\{([^}]+)\}', init_input)
            if db_match:
                self._init_database_context = db_match.group(1)
            else:
                self._init_database_context = os.getenv('PGDATABASE', 'advworks')

            response = await self.agent_instance.get_response(messages=init_input, thread=self.thread_instance)
            print(f"# {response.name}: {response}")
            print(f"\n✅ Improved Smart MCP Agent initialized (similarity threshold: {self.similarity_threshold:.0%})!")
            
            # Save the initialization messages
            if self.conversation_history_enabled and self.session_id and not self._init_messages_saved:
                await self._save_message("user", init_input)
                await self._save_message("agent", str(response))
                self._init_messages_saved = True
            
            response = None
            
        except Exception as e:
            print(f"❌ Error initializing improved smart agent: {e}")
            if self.local_plugin:
                try:
                    if hasattr(self.local_plugin, 'disconnect'):
                        await self.local_plugin.disconnect()
                    elif hasattr(self.local_plugin, 'close'):
                        await self.local_plugin.close()
                except Exception as cleanup_error:
                    print(f"⚠️ Error during cleanup: {cleanup_error}")

    async def _create_conversation_session(self):
        """Create a new conversation session in PostgreSQL."""
        try:
            from datetime import datetime
            session_name = f"Improved Smart Chat Session - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            if self.conversation_manager:
                self.session_id = self.conversation_manager.create_session(session_name)
                print(f"🆔 Generated conversation session ID: {self.session_id}")
                return
            
            self.session_id = str(uuid.uuid4())
            print(f"🆔 Generated local session ID: {self.session_id}")
            
        except Exception as e:
            print(f"⚠️ Warning: Could not create conversation session: {e}")
            self.conversation_history_enabled = False

    async def _save_message(self, role: str, content: str):
        """Save a message to the conversation history."""
        if not self.conversation_history_enabled or not self.session_id:
            return
            
        message_hash = f"{self.session_id}:{role}:{content[:200]}"
        
        if message_hash in self._currently_saving or message_hash in self._last_saved_messages:
            return
            
        try:
            self._currently_saving.add(message_hash)
            
            if self.conversation_manager and self.session_id:
                message_id = self.conversation_manager.add_message(self.session_id, role, content)
                print(f"💾 Saved: [{role}] {content[:50]}...")
                
                self._currently_saving.discard(message_hash)
                self._last_saved_messages.add(message_hash)
                
                if len(self._last_saved_messages) > 15:
                    temp_list = list(self._last_saved_messages)
                    temp_list = temp_list[-10:]
                    self._last_saved_messages = set(temp_list)
                    
                return message_id
            else:
                self._currently_saving.discard(message_hash)
                print(f"💾 Local: [{role}] {content[:50]}...")
                
        except Exception as e:
            self._currently_saving.discard(message_hash)
            print(f"❌ Error saving message: {e}")

    def _normalize_database_context(self, question: str) -> str:
        """Normalize database references to handle 'my database', 'my db', etc. = extracted database context."""
        
        
        # Extract database name from agent instructions context
        database_name = self._extract_database_from_context()
        
        # Replace common database references with the actual database name
        replacements = [
            (r'\bmy database\b', database_name),
            (r'\bmy db\b', f'{database_name} database'),  # Expand abbreviation for better matching
            (r'\bthe database\b', database_name), 
            (r'\bthe db\b', f'{database_name} database'),  # Expand abbreviation for better matching
            (r'\bour database\b', database_name),
            (r'\bour db\b', f'{database_name} database'),  # Expand abbreviation for better matching
            (r'\bthis database\b', database_name),
            (r'\bthis db\b', f'{database_name} database'),  # Expand abbreviation for better matching
        ]
        
        normalized = question.lower()
        for pattern, replacement in replacements:
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        
        return normalized

    def _extract_database_from_context(self) -> str:
        """Extract database name from the agent's instructions context."""
        try:
            if self.agent_instance and hasattr(self.agent_instance, 'instructions'):
                instructions = self.agent_instance.instructions
                
                # Look for database name in format {database_name} in instructions
                import re
                match = re.search(r'\{([^}]+)\}', instructions)
                if match:
                    return match.group(1)
                
                # Also check initialization input pattern
                # Look for "postgres database {name}" pattern
                db_match = re.search(r'postgres database\s+\{([^}]+)\}', instructions, re.IGNORECASE)
                if db_match:
                    return db_match.group(1)
            
            # Check if we have stored initialization context
            if hasattr(self, '_init_database_context'):
                return self._init_database_context
                
            # Fallback to environment variable if available
            return os.getenv('PGDATABASE', 'contosodb')
            
        except Exception as e:
            print(f"⚠️ Could not extract database from context: {e}")
            # Fallback to default
            return 'contosodb'

    def _analyze_question_intent(self, question: str) -> dict:
        """Analyze the intent and requirements of a question."""
        question_lower = question.lower()
        
        # Define intent categories - improved for better detection
        schema_keywords = ['schema', 'column', 'type', 'structure', 'describe', 'definition', 'column types', 'data types']
        simple_keywords = ['tables', 'list', 'show tables', 'what tables']
        count_keywords = ['count', 'how many', 'number of']
        data_keywords = ['data', 'rows', 'records', 'content', 'values']
        
        # Analyze intent
        intent = {
            'wants_schema': any(keyword in question_lower for keyword in schema_keywords),
            'wants_simple_list': any(keyword in question_lower for keyword in simple_keywords) and not any(keyword in question_lower for keyword in schema_keywords),
            'wants_count': any(keyword in question_lower for keyword in count_keywords),
            'wants_data': any(keyword in question_lower for keyword in data_keywords),
            'complexity': 'high' if any(keyword in question_lower for keyword in schema_keywords) else 'low'
        }
        
        return intent

    def _validate_response_completeness(self, question: str, cached_response: str) -> bool:
        """Check if the cached response adequately answers the current question."""
        question_intent = self._analyze_question_intent(question)
        response_lower = cached_response.lower()
        
        # If question needs detailed schema info, check if response has it
        if question_intent['wants_schema']:
            # Check for schema indicators in response
            schema_indicators = [':', 'character varying', 'integer', 'uuid', 'text', 'numeric', 'boolean', 'date', 'timestamp']
            has_schema_info = any(indicator in response_lower for indicator in schema_indicators)
            
            if not has_schema_info:
                return False
        
        # If question needs count, check if response has numbers
        if question_intent['wants_count']:
            import re
            has_numbers = bool(re.search(r'\d+', cached_response))
            if not has_numbers:
                return False
        
        # Additional validation can be added here
        return True

    async def ask_question(self, question: str, force_fresh: bool = False, save_to_history: bool = True):
        """
        Ask a question to the agent with improved smart conversation history.
        
        Args:
            question (str): The question to ask
            force_fresh (bool): If True, bypass similarity search and get a fresh response
            save_to_history (bool): If True, save the conversation to history
        """
        if self.agent_instance is None:
            print("❌ Agent not initialized. Please run init_chat() first.")
            return
        
        print(f"# User: {question}")
        
        # Check for similar questions with improved logic
        if not force_fresh and self.conversation_history_enabled:
            similar_found = await self._check_similar_questions_improved(question)
            if similar_found:
                return
        
        try:
            # Get fresh response from the agent
            response = await self.agent_instance.get_response(messages=question, thread=self.thread_instance)
            print(f"# {response.name}: {response}")
            
            # Save to conversation history if enabled
            if save_to_history and self.conversation_history_enabled:
                await self._save_message("user", question)
                await self._save_message("agent", str(response))
            
            if not force_fresh:
                print("💡 Response saved for future similar questions")
            
            response = None
        except Exception as e:
            print(f"❌ Error: {e}")

    async def _check_similar_questions_improved(self, question: str) -> bool:
        """
        Improved method to check for similar questions with smart context-aware matching.
        
        Returns:
            bool: True if similar question found and response provided, False otherwise
        """
        try:
            print(f"🔍 Checking for similar questions (smart context-aware mode)...")
            
            if self.conversation_manager:
                try:
                    # Step 1: Normalize the input question for database context
                    normalized_question = self._normalize_database_context(question)
                    current_intent = self._analyze_question_intent(question)
                    
                    # Use the normalized question for similarity search
                    search_question = normalized_question if normalized_question != question.lower() else question
                    
                    similar_questions = self.conversation_manager.find_similar_questions(
                        search_question, similarity_threshold=0.50, limit=5  # Use lower threshold for initial search
                    )
                    
                    if similar_questions:
                        # Apply smart filtering to results
                        for match in similar_questions:
                            cached_response = match.get("response", "")
                            similarity = match.get("similarity_score", 0)
                            cached_question = match.get("question", "")
                            
                            if not cached_response:
                                continue
                            
                            # Analyze intent compatibility
                            cached_intent = self._analyze_question_intent(cached_question)
                            
                            # Check intent compatibility
                            intent_compatible = True
                            if current_intent['wants_schema'] != cached_intent['wants_schema']:
                                intent_compatible = False
                            
                            # Apply dynamic threshold based on context and intent
                            if intent_compatible:
                                # If intents match, use lower threshold for context-normalized questions
                                if normalized_question != question.lower():
                                    effective_threshold = 0.60  # Context-aware threshold
                                else:
                                    effective_threshold = self.similarity_threshold  # Regular threshold
                            else:
                                # If intents don't match, use higher threshold to prevent false matches
                                effective_threshold = self.similarity_threshold + 0.10
                            
                            # Check if similarity meets effective threshold and intents are compatible
                            if similarity >= effective_threshold and intent_compatible:
                                # Validate response completeness
                                if self._validate_response_completeness(question, cached_response):
                                    print(f"# Agent (smart cached - {similarity:.1%} similar): {cached_response}")
                                    print(f"💡 ✨ Using smart cached response (ZERO tokens used!)")
                                    print(f"🧠 Match reason: Intent compatible, effective threshold: {effective_threshold:.0%}")
                                    return True
                                else:
                                    print(f"🚫 Found similar question ({similarity:.1%}) but response lacks required detail")
                                    continue
                            else:
                                if not intent_compatible:
                                    print(f"🚫 Found similar question ({similarity:.1%}) but intent incompatible")
                                    print(f"   Current: {'schema' if current_intent['wants_schema'] else 'simple'}")
                                    print(f"   Cached: {'schema' if cached_intent['wants_schema'] else 'simple'}")
                                continue
                        
                except Exception as e:
                    print(f"⚠️ Could not search PostgreSQL history: {e}")
            
            # Fallback to thread history
            cached_response = await self._check_thread_history(question)
            
            if cached_response:
                print(f"# Agent (from thread history): {cached_response}")
                print("💡 ✨ Using cached response (ZERO tokens used!)")
                return True
            else:
                print("🆕 No suitable matches found - generating fresh response")
                return False
                
        except Exception as e:
            print(f"⚠️ Error in smart similarity check: {e}")
            return False

    async def _check_thread_history(self, question: str) -> str:
        """Check the current thread history for similar questions (fallback method)."""
        if self.thread_instance is None:
            return None
        
        try:
            messages = []
            async for message in self.thread_instance.get_messages():
                messages.append(message)
            
            normalized_question = self.normalize_text(question)
            
            for i in range(len(messages)):
                current_msg = messages[i]
                current_role = current_msg.role.value if hasattr(current_msg.role, 'value') else str(current_msg.role)
                
                if current_role.lower() == 'user':
                    normalized_content = self.normalize_text(current_msg.content)
                    
                    if normalized_content == normalized_question:
                        for j in range(i + 1, len(messages)):
                            next_msg = messages[j]
                            next_role = next_msg.role.value if hasattr(next_msg.role, 'value') else str(next_msg.role)
                            
                            if next_role.lower() in ['assistant', 'agent']:
                                content = str(next_msg.content).strip()
                                if content:
                                    return content
                            elif next_role.lower() == 'user':
                                break
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error checking thread history: {e}")
            return None

    # Include all other methods from the original class...
    async def get_conversation_stats(self):
        """Get statistics about stored conversations."""
        if not self.conversation_history_enabled:
            print("❌ Conversation history not enabled")
            return
            
        try:
            print("📊 Getting conversation statistics...")
            
            if self.conversation_manager:
                try:
                    stats = self.conversation_manager.get_conversation_stats()
                    print(f"📈 Sessions: {stats.get('total_sessions', 0)}")
                    print(f"📈 Messages: {stats.get('total_messages', 0)}")
                    print(f"📈 Users: {stats.get('unique_users', 0)}")
                    print(f"📈 Similarity Threshold: {self.similarity_threshold:.0%}")
                    return stats
                except Exception as e:
                    print(f"⚠️ Could not get stats: {e}")
            
            print("💡 No statistics available")
        except Exception as e:
            print(f"❌ Error getting conversation stats: {e}")

    async def get_recent_sessions(self, limit: int = 10):
        """Get recent conversation sessions."""
        if not self.conversation_history_enabled:
            print("❌ Conversation history not enabled")
            return
            
        try:
            print(f"📝 Getting {limit} recent sessions...")
            
            if self.conversation_manager:
                try:
                    sessions = self.conversation_manager.get_recent_sessions(limit)
                    for i, session in enumerate(sessions, 1):
                        print(f"{i}. {session['topic']} - {session['created_at']}")
                    return sessions
                except Exception as e:
                    print(f"⚠️ Could not get sessions: {e}")
            
            print("💡 No recent sessions available")
        except Exception as e:
            print(f"❌ Error getting recent sessions: {e}")

    async def get_chat_history(self):
        """Get the current chat history from the thread."""
        if self.thread_instance is None:
            print("❌ Thread not initialized. Please run init_chat() first.")
            return
        
        print("📝 Current Chat History:")
        
        try:
            messages = []
            async for message in self.thread_instance.get_messages():
                messages.append(message)
            
            if not messages:
                print("No messages in chat history yet.")
                return []
            
            for i, message in enumerate(messages, 1):
                role = message.role.value if hasattr(message.role, 'value') else str(message.role)
                content = str(message.content)
                print(f"{i}. [{role.upper()}]: {content}")
            
            return messages
            
        except Exception as e:
            print(f"❌ Error accessing chat history: {e}")
            return []

    async def clear_chat_history(self):
        """Clear the chat history and create a new thread."""
        self.thread_instance = ChatHistoryAgentThread()
        
        if self.conversation_history_enabled:
            self.session_id = None
            self._init_messages_saved = False
            self._last_saved_messages.clear()
            await self._create_conversation_session()
        
        print("🧹 Chat history cleared. New thread and conversation session created.")

    async def cleanup(self):
        """Call this function to properly disconnect from the MCP server when done."""
        try:
            if self.local_plugin:
                if hasattr(self.local_plugin, 'disconnect'):
                    await self.local_plugin.disconnect()
                    print("✅ Disconnected from MCP server.")
                elif hasattr(self.local_plugin, 'close'):
                    await self.local_plugin.close()
                    print("✅ Closed MCP plugin connection.")
                else:
                    print("ℹ️ No disconnect/close method available for MCP plugin.")
        except Exception as e:
            print(f"⚠️ Warning during cleanup: {e}")

if __name__ == "__main__":
    # Interactive mode
    async def main():
        client = SmartMCPClientImproved()
        await client.init_chat()
        
        while True:
            user_input = input("\nEnter your question (or type 'bye' to exit, 'stats' for statistics): ").strip()
            
            if user_input.lower() == "bye":
                print("Exiting the conversation.")
                break
            elif user_input.lower() == "stats":
                await client.get_conversation_stats()
                continue
            elif user_input.lower() == "history":
                await client.get_chat_history()
                continue
            elif user_input.lower().startswith("fresh:"):
                question = user_input[6:].strip()
                await client.ask_question(question, force_fresh=True)
                continue
            elif user_input:
                await client.ask_question(user_input)
        
        await client.cleanup()
    
    asyncio.run(main())
