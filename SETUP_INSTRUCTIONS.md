# Smart MCP Client Capstone project Setup Instructions

## 1. Prerequisites
- **An active Azure Subscription**
- **Docker Desktop**
- **Visual Studio Code**
- **Azure PostgreSQL Flexible Server**

## 2. AAD Service Principal Creation

### a. Create a New Service Principal
If a Service Principal is not yet created, use the command below (requires Entra ID access). Make sure to capture the `subscription_id`, `client id`, `secret`, etc.

```
az ad sp create-for-rbac --name pgflexsp17 --role "Reader" --scopes /subscriptions/{subscription_id}/resourceGroups/{rg_name}/providers/Microsoft.DBforPostgreSQL/flexibleServers/{postgresflexservername}
```
> **Note:** If you want to use a different service principal, update the login in `advworks.sql` accordingly.

### b. Assign Permissions to an Existing Service Principal
If you already have a Service Principal or an Entra Administrator has created one for you, use the following commands to add the required permissions:

```
az role assignment create --assignee {service_principal_client_id} --role "Reader" --scope /subscriptions/{subscription_id}/resourceGroups/{rg_name}/providers/Microsoft.DBforPostgreSQL/flexibleServer {postgresflexservername}
az role assignment list --all --assignee {service_principal_client_id}
```

### c. Add Service Principal to Entra Administrators
Add the Service Principal to Microsoft Entra administrators via the Portal:
- Go to the respective Azure Database for PostgreSQL Flexible Server and Under the **Security** blade, select **Authentication**

## 3. Enable Required PostgreSQL Extensions
In **Azure PostgreSQL Flexible Server** > **Server Parameters** > `azure.extensions`, select the following extensions:
- `pg_diskann`
- `tablefunc`
- `uuid-ossp`
- `vector`

## 4. Install Visual Studio Code Extensions
- **Dev Containers**
- **Python**

## 5. Project Setup
- Browse to the folder where the Dockerfile and source code are located (e.g., `mcp-server-py`)
- Populate the `.env` file with all required details (see provided `env.sample` file)

## 6. Build and Start Docker Containers
```
docker compose build
docker compose up
```

## 7. Initialize Database Inside Docker Container
In the Docker container terminal, navigate to the `advworks` folder and run:

```
sh init-db.sh   # Creates the advworks database and populates it with adventureworks data
python create_pg_extensions_tables.py   # Creates diskann and pgvector extensions, conversation tables, and indexes
```
