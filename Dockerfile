FROM ghcr.io/astral-sh/uv:python3.12-alpine


# Install tzdata and set timezone
RUN apk add --no-cache tzdata && \
cp /usr/share/zoneinfo/Europe/Bucharest /etc/localtime && \
echo "Europe/Bucharest" > /etc/timezone


RUN apk update
RUN apk add gcc musl-dev linux-headers build-base libffi-dev openssl-dev
RUN apk add postgresql17-client

# Set working directory
WORKDIR /mcp

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt


COPY ./src/azure_postgresql_mcp.py .
COPY ./src/test_pg_connection.py .
COPY ./src/azure_conversation_history.py .
COPY ./src/smart_mcp_client_improved.py .
COPY ./src/smart-mcp-client_improved.ipynb .
COPY env.sample .env
COPY advworks.tar.gz .
RUN tar -zxvf advworks.tar.gz

COPY entrypoint.sh .
COPY ./src/tokengen.py .
COPY ./src/token_refresh.sh /etc/periodic/5min/token_refresh.sh

# Remove trailing whitespace from all lines
RUN sed -i 's/[[:space:]]*$//' /etc/periodic/5min/token_refresh.sh

RUN chmod +x ./entrypoint.sh
RUN chmod +x /etc/periodic/5min/token_refresh.sh

RUN echo "*/5 * * * * run-parts /etc/periodic/5min" >> /etc/crontabs/root



ENTRYPOINT ["./entrypoint.sh"]
