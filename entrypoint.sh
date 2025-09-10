#!/bin/sh
crond -f -l 8 & python azure_postgresql_mcp.py --host localhost --port 8003
