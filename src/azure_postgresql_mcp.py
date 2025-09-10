"""
MCP server for Azure Database for PostgreSQL - Flexible Server.

This server exposes the following capabilities:

Tools:
- create_table: Creates a table in a database.
- drop_table: Drops a table in a database.
- get_databases: Gets the list of all the databases in a server instance.
- get_schemas: Gets schemas of all the tables.
- get_server_config: Gets the configuration of a server instance. [Available with Microsoft EntraID]
- get_server_parameter: Gets the value of a server parameter. [Available with Microsoft EntraID]
- query_data: Runs read queries on a database.
- update_values: Updates or inserts values into a table.
- create_conversation_session: Creates a new conversation session for storing chat history.
- find_similar_questions: Finds similar questions from conversation history to save LLM tokens.
- save_conversation_message: Saves a message to conversation history.
- get_conversation_stats: Gets statistics about stored conversations.
- get_recent_sessions: Gets recent conversation sessions.

Resources:
- databases: Gets the list of all databases in a server instance.

Run the MCP Server using the following command:

```
python azure_postgresql_mcp.py
```

For detailed usage instructions, please refer to the README.md file.

"""

import json
import logging
import os
import sys
import urllib.parse
import psycopg
import uvicorn
from azure.identity import DefaultAzureCredential
from azure.mgmt.postgresqlflexibleservers import PostgreSQLManagementClient
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.resources import FunctionResource

from tokengen import TokenValidation
from azure_conversation_history import AzureConversationHistoryManager as ConversationHistoryManager
ADVANCED_HISTORY = True
HISTORY_TYPE = "azure_openai"

logger = logging.getLogger("azure")
logger.setLevel(logging.ERROR)


class AzurePostgreSQLMCP:
    def init(self):
        self.aad_in_use = os.environ.get("AZURE_USE_AAD")
        self.dbhost = self.get_environ_variable("PGHOST")
        self.dbuser = urllib.parse.quote(self.get_environ_variable("PGUSER"))
        self.dbname = self.get_environ_variable("PGDATABASE")

        if self.aad_in_use == "True":
            self.subscription_id = self.get_environ_variable("AZURE_SUBSCRIPTION_ID")
            self.resource_group_name = self.get_environ_variable("AZURE_RESOURCE_GROUP")
            self.server_name = (
                self.dbhost.split(".", 1)[0] if "." in self.dbhost else self.dbhost
            )
            self.credential = DefaultAzureCredential()
            self.postgresql_client = PostgreSQLManagementClient(
                self.credential, self.subscription_id
            )
        
        # Initialize conversation history manager
        try:
            connection_uri = self.get_connection_uri(self.dbname)
            self.conversation_manager = ConversationHistoryManager(connection_uri)
            logger.info(f"Conversation history manager initialized successfully ({HISTORY_TYPE} version)")
        except Exception as e:
            logger.warning(f"Failed to initialize conversation history manager: {e}")
            self.conversation_manager = None

    @staticmethod
    def get_environ_variable(name: str):
        """Helper function to get environment variable or raise an error."""
        value = os.environ.get(name)
        if value is None:
            raise EnvironmentError(f"Environment variable {name} not found.")
        return value

    def get_password(self) -> str:
        """Get password based on the auth mode set"""
        if self.aad_in_use == "True":
            # Use tokengen.py script to get the token
            token_validator = TokenValidation()
            return token_validator.get_password()
        else:
            return self.get_environ_variable("PGPASSWORD")

    def get_dbs_resource_uri(self):
        """Gets the resource URI exposed as MCP resource for getting list of dbs."""
        dbhost_normalized = (
            self.dbhost.split(".", 1)[0] if "." in self.dbhost else self.dbhost
        )
        return f"flexpg://{dbhost_normalized}/databases"

    def get_databases_internal(self) -> str:
        """Internal function which gets the list of all databases in a server instance."""
        try:
            with psycopg.connect(
                f"host={self.dbhost} user={self.dbuser} dbname='postgres' password={self.get_password()}"
            ) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT datname FROM pg_database WHERE datistemplate = false;"
                    )
                    colnames = [desc[0] for desc in cur.description]
                    dbs = cur.fetchall()
                    return json.dumps(
                        {
                            "columns": str(colnames),
                            "rows": "".join(str(row) for row in dbs),
                        }
                    )
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return ""

    def get_databases_resource(self):
        """Gets list of databases as a resource"""
        return self.get_databases_internal()

    def get_databases(self):
        """Gets the list of all the databases in a server instance."""
        return self.get_databases_internal()

    def get_connection_uri(self, dbname: str) -> str:
        """Construct URI for connection."""
        return f"host={self.dbhost} dbname={dbname} user={self.dbuser} password={self.get_password()}"

    def get_schemas(self, database: str):
        """Gets schemas of all the tables in the database (all schemas, not just public), with a row and LLM token limit."""
        # Set a sensible default row limit, can be overridden by env var
        row_limit = int(os.environ.get("SCHEMA_ROW_LIMIT", 5000))
        llm_token_limit = int(os.environ.get("SCHEMA_LLM_TOKEN_LIMIT", 1500))
        try:
            with psycopg.connect(self.get_connection_uri(database)) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"""
                        SELECT table_schema, table_name, column_name, data_type
                        FROM information_schema.columns
                        ORDER BY table_schema, table_name, ordinal_position
                        LIMIT {row_limit};
                        """
                    )
                    colnames = [desc[0] for desc in cur.description]
                    tables = cur.fetchall()
                    # Estimate LLM tokens: assume ~8 tokens per value (conservative, varies by model)
                    def estimate_tokens(row):
                        return sum(len(str(val)) // 4 + 1 for val in row)  # ~4 chars/token
                    total_tokens = 0
                    limited_tables = []
                    for row in tables:
                        row_tokens = estimate_tokens(row)
                        if total_tokens + row_tokens > llm_token_limit:
                            break
                        limited_tables.append(row)
                        total_tokens += row_tokens
                    return json.dumps(
                        {
                            "columns": colnames,
                            "rows": limited_tables,
                            "row_limit": row_limit,
                            "row_count": len(limited_tables),
                            "llm_token_limit": llm_token_limit,
                            "llm_token_estimate": total_tokens
                        }
                    )
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return ""

    def query_data(self, dbname: str, s: str) -> str:
        """Runs read queries on a database."""
        try:
            with psycopg.connect(self.get_connection_uri(dbname)) as conn:
                with conn.cursor() as cur:
                    cur.execute(s)
                    rows = cur.fetchall()
                    colnames = [desc[0] for desc in cur.description]
                    return json.dumps(
                        {
                            "columns": str(colnames),
                            "rows": ",".join(str(row) for row in rows),
                        }
                    )
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            return ""

    def exec_and_commit(self, dbname: str, s: str) -> None:
        """Internal function to execute and commit transaction."""
        try:
            with psycopg.connect(self.get_connection_uri(dbname)) as conn:
                with conn.cursor() as cur:
                    cur.execute(s)
                    conn.commit()
        except Exception as e:
            logger.error(f"Error: {str(e)}")

    def update_values(self, dbname: str, s: str):
        """Updates or inserts values into a table."""
        self.exec_and_commit(dbname, s)

    def create_table(self, dbname: str, s: str):
        """Creates a table in a database."""
        self.exec_and_commit(dbname, s)

    def drop_table(self, dbname: str, s: str):
        """Drops a table in a database."""
        self.exec_and_commit(dbname, s)

    def get_server_config(self) -> str:
        """Gets the configuration of a server instance. [Available with Microsoft EntraID]"""
        if self.aad_in_use:
            try:
                server = self.postgresql_client.servers.get(
                    self.resource_group_name, self.server_name
                )
                return json.dumps(
                    {
                        "server": {
                            "name": server.name,
                            "location": server.location,
                            "version": server.version,
                            "sku": server.sku.name,
                            "storage_profile": {
                                "storage_size_gb": server.storage.storage_size_gb,
                                "backup_retention_days": server.backup.backup_retention_days,
                                "geo_redundant_backup": server.backup.geo_redundant_backup,
                            },
                        },
                    }
                )
            except Exception as e:
                logger.error(f"Failed to get PostgreSQL server configuration: {e}")
                raise e

        else:
            raise NotImplementedError(
                "This tool is available only with Microsoft EntraID"
            )

    def get_server_parameter(self, parameter_name: str) -> str:
        """Gets the value of a server parameter. [Available with Microsoft EntraID]"""
        if self.aad_in_use:
            try:
                configuration = self.postgresql_client.configurations.get(
                    self.resource_group_name, self.server_name, parameter_name
                )
                return json.dumps(
                    {"param": configuration.name, "value": configuration.value}
                )
            except Exception as e:
                logger.error(
                    f"Failed to get PostgreSQL server parameter '{parameter_name}': {e}"
                )
                raise e
        else:
            raise NotImplementedError(
                "This tool is available only with Microsoft EntraID"
            )

    # Conversation History Management Methods
    def create_conversation_session(self, session_name: str = None) -> str:
        """
        Creates a new conversation session for storing chat history.
        
        Args:
            session_name: Optional name for the conversation session
            
        Returns:
            JSON string containing the session_id
        """
        if not self.conversation_manager:
            return json.dumps({"error": "Conversation history not available"})
        
        try:
            session_id = self.conversation_manager.create_session(
                session_name=session_name,
                metadata={"created_via": "mcp_server"}
            )
            return json.dumps({
                "session_id": session_id,
                "session_name": session_name,
                "status": "created"
            })
        except Exception as e:
            logger.error(f"Error creating conversation session: {e}")
            return json.dumps({"error": str(e)})

    def find_similar_questions(self, question: str, similarity_threshold: float = 0.8, max_results: int = 3) -> str:
        """
        Finds similar questions from conversation history to save LLM tokens.
        
        Args:
            question: The question to find similarities for
            similarity_threshold: Minimum similarity score (0.0 to 1.0, default 0.8)
            max_results: Maximum number of similar questions to return (default 3)
            
        Returns:
            JSON string containing similar questions and their responses
        """
        if not self.conversation_manager:
            return json.dumps({"error": "Conversation history not available"})
        
        try:
            similar_questions = self.conversation_manager.find_similar_questions(
                question=question,
                similarity_threshold=similarity_threshold,
                limit=max_results
            )
            
            if similar_questions:
                return json.dumps({
                    "found_similar": True,
                    "count": len(similar_questions),
                    "similar_questions": similar_questions,
                    "message": f"Found {len(similar_questions)} similar question(s) - you can reuse these responses to save LLM tokens!"
                })
            else:
                return json.dumps({
                    "found_similar": False,
                    "count": 0,
                    "similar_questions": [],
                    "message": "No similar questions found in conversation history"
                })
        except Exception as e:
            logger.error(f"Error finding similar questions: {e}")
            return json.dumps({"error": str(e)})

    def save_conversation_message(self, session_id: str, role: str, content: str) -> str:
        """
        Saves a message to conversation history.
        
        Args:
            session_id: UUID of the conversation session
            role: Role of the message sender (user, assistant, agent, system)
            content: Message content
            
        Returns:
            JSON string containing the message_id and status
        """
        if not self.conversation_manager:
            return json.dumps({"error": "Conversation history not available"})
        
        if role not in ['user', 'assistant', 'agent', 'system']:
            return json.dumps({"error": f"Invalid role: {role}. Must be one of: user, assistant, agent, system"})
        
        try:
            message_id = self.conversation_manager.add_message(
                session_id=session_id,
                role=role,
                content=content,
                metadata={"saved_via": "mcp_server"}
            )
            return json.dumps({
                "message_id": message_id,
                "session_id": session_id,
                "role": role,
                "status": "saved"
            })
        except Exception as e:
            logger.error(f"Error saving conversation message: {e}")
            return json.dumps({"error": str(e)})

    def get_conversation_stats(self) -> str:
        """
        Gets statistics about stored conversations.
        
        Returns:
            JSON string containing conversation statistics
        """
        if not self.conversation_manager:
            return json.dumps({"error": "Conversation history not available"})
        
        try:
            stats = self.conversation_manager.get_conversation_stats()
            return json.dumps({
                "statistics": stats,
                "status": "success"
            })
        except Exception as e:
            logger.error(f"Error getting conversation stats: {e}")
            return json.dumps({"error": str(e)})

    def get_recent_sessions(self, limit: int = 10) -> str:
        """
        Gets recent conversation sessions.
        
        Args:
            limit: Maximum number of sessions to return (default 10)
            
        Returns:
            JSON string containing recent conversation sessions
        """
        if not self.conversation_manager:
            return json.dumps({"error": "Conversation history not available"})
        
        try:
            sessions = self.conversation_manager.get_recent_sessions(limit=limit)
            return json.dumps({
                "sessions": sessions,
                "count": len(sessions),
                "status": "success"
            })
        except Exception as e:
            logger.error(f"Error getting recent sessions: {e}")
            return json.dumps({"error": str(e)})


if __name__ == "__main__":
    mcp = FastMCP("pgflex-mcp-server", stateless_http=True)
    azure_pg_mcp = AzurePostgreSQLMCP()
    azure_pg_mcp.init()
    
    # Database management tools
    mcp.add_tool(azure_pg_mcp.get_databases)
    mcp.add_tool(azure_pg_mcp.get_schemas)
    mcp.add_tool(azure_pg_mcp.query_data)
    mcp.add_tool(azure_pg_mcp.update_values)
    mcp.add_tool(azure_pg_mcp.create_table)
    mcp.add_tool(azure_pg_mcp.drop_table)
    mcp.add_tool(azure_pg_mcp.get_server_config)
    mcp.add_tool(azure_pg_mcp.get_server_parameter)
    
    # Conversation history management tools
    mcp.add_tool(azure_pg_mcp.create_conversation_session)
    mcp.add_tool(azure_pg_mcp.find_similar_questions)
    mcp.add_tool(azure_pg_mcp.save_conversation_message)
    mcp.add_tool(azure_pg_mcp.get_conversation_stats)
    mcp.add_tool(azure_pg_mcp.get_recent_sessions)
    
    databases_resource = FunctionResource(
        name=azure_pg_mcp.get_dbs_resource_uri(),
        uri=azure_pg_mcp.get_dbs_resource_uri(),
        description="List of databases in the server",
        mime_type="application/json",
        fn=azure_pg_mcp.get_databases_resource,
    )
    mcp.add_resource(databases_resource)

    # Now convert to Starlette app
    mcp = mcp.streamable_http_app()

    import argparse

    parser = argparse.ArgumentParser(description="Run MCP HTTP-streamable-based server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8003, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run(mcp, host=args.host, port=args.port)




