#!/bin/bash

# Knowledge Base API - Start Script
# This script starts the entire application with one command

echo "🚀 Starting Knowledge Base API..."
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running!"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

# Start the containers
echo "📦 Starting containers..."
docker compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 5

# Check if API is responding
if curl -s http://localhost:8000/healthz > /dev/null 2>&1; then
    echo ""
    echo "✅ Knowledge Base API is running!"
    echo ""
    echo "📍 Access the application:"
    echo "   • API:        http://localhost:8000"
    echo "   • Swagger UI: http://localhost:8000/docs"
    echo "   • Health:     http://localhost:8000/healthz"
    echo ""
    echo "🔑 Your API Key: dev-key"
    echo ""
    echo "💡 Tip: Use the Swagger UI to upload documents and ask questions!"
    echo ""
    echo "To stop the application, run: ./stop.sh"
else
    echo ""
    echo "⚠️  Services started but API is not responding yet."
    echo "Run 'docker compose logs -f' to see what's happening."
fi
