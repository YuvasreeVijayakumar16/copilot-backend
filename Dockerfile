FROM debian:bookworm-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN set -eux; \
    # Remove conflicting ODBC packages forcibly (ignore errors)
    dpkg -r --force-all libodbc2 libodbcinst2 unixodbc-common libodbc1 odbcinst1debian2 odbcinst unixodbc unixodbc-dev || true; \
    apt-get clean; rm -rf /var/lib/apt/lists/*; \
    \
    # Update package lists and install base dependencies + python3 + venv
    apt-get update; \
    apt-get install -y --no-install-recommends \
        curl \
        gnupg \
        apt-transport-https \
        gcc \
        g++ \
        ca-certificates \
        python3 \
        python3-venv \
        python3-pip; \
    \
    # Setup Microsoft package repository key and list
    mkdir -p /etc/apt/keyrings; \
    curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > /etc/apt/keyrings/microsoft.gpg; \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/11/prod bullseye main" > /etc/apt/sources.list.d/microsoft-prod.list; \
    \
    # Update and install unixodbc and msodbcsql17 freshly
    apt-get update; \
    apt-get install -y --no-install-recommends unixodbc unixodbc-dev; \
    ACCEPT_EULA=Y apt-get install -y msodbcsql17; \
    \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*; \
    \
    # Create Python virtual environment
    python3 -m venv $VIRTUAL_ENV; \
    $VIRTUAL_ENV/bin/pip install --upgrade pip setuptools wheel

# Copy requirements and install Python dependencies inside venv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app source code
COPY . .

EXPOSE 8000

# Run the app using virtual environment Python
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]