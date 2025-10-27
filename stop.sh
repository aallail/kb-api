#!/bin/bash

# Knowledge Base API - Stop Script
# This script stops the entire application

echo "🛑 Stopping Knowledge Base API..."

docker compose down

echo ""
echo "✅ Application stopped successfully!"
echo ""
echo "To start again, run: ./start.sh"
