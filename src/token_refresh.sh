#!/bin/sh
# /home/ffol/token_refresh.sh - Script to check and refresh Azure AD token if expired

# Set environment variables (if needed)
export PATH="/usr/local/bin:/usr/bin:/bin"
# Log file for debugging
LOG_FILE="/mcp/token_refresh.log"

# Load environment variables from .env file
if [ -f /mcp/.env ]; then
    export $(cat /mcp/.env | grep -v '^#' | xargs)
fi

# Function to get timestamp (fallback if date command not available)
get_timestamp() {
    if command -v date >/dev/null 2>&1; then
        date
    else
        # Fallback using stat command or simple timestamp
        echo "$(stat -c %Y /proc/self 2>/dev/null || echo 'Unknown time')"
    fi
}

# Add timestamp to log
echo "$(get_timestamp): Starting token check..." >> "$LOG_FILE"

# Change to the working directory
cd /mcp

# Debug environment variables
echo "AZURE_CLIENT_ID: $AZURE_CLIENT_ID" >> "$LOG_FILE"
echo "AZURE_TENANT_ID: $AZURE_TENANT_ID" >> "$LOG_FILE"
echo "AZURE_CLIENT_SECRET: $([ -n "$AZURE_CLIENT_SECRET" ] && echo "[PRESENT]" || echo "[MISSING]")" >> "$LOG_FILE"

# Run the token check and refresh script
python /mcp/tokengen.py >> "$LOG_FILE" 2>&1
# Log completion
echo "$(get_timestamp): Token check completed" >> "$LOG_FILE"
echo "-----------------------------------------------------------------------------" >> "$LOG_FILE"

