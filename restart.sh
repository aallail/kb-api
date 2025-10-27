#!/bin/bash

# Knowledge Base API - Restart Script
# This script restarts the entire application

echo "🔄 Restarting Knowledge Base API..."

./stop.sh
echo ""
./start.sh
