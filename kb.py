#!/usr/bin/env python3
"""
Knowledge Base CLI - Command-line interface for the Knowledge Base API.

Usage:
    python kb.py upload <file> [--title TITLE]
    python kb.py ask "<question>" [--hybrid] [--reranker] [--mmr]
    python kb.py list
    python kb.py delete <doc_id>
    python kb.py analytics
    python kb.py health

Examples:
    python kb.py upload document.pdf --title "Research Paper"
    python kb.py ask "What is machine learning?" --hybrid --reranker
    python kb.py analytics
"""

import argparse
import requests
import sys
import json
from pathlib import Path
from typing import Optional

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = "dev-key"  # Change this to match your API key


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_success(message: str):
    """Print success message in green."""
    print(f"{Colors.GREEN}✓ {message}{Colors.ENDC}")


def print_error(message: str):
    """Print error message in red."""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}", file=sys.stderr)


def print_info(message: str):
    """Print info message in cyan."""
    print(f"{Colors.CYAN}ℹ {message}{Colors.ENDC}")


def print_header(message: str):
    """Print header message in bold."""
    print(f"\n{Colors.BOLD}{Colors.HEADER}{message}{Colors.ENDC}\n")


def upload_document(file_path: str, title: Optional[str] = None):
    """Upload a document to the knowledge base."""
    file_path = Path(file_path)

    if not file_path.exists():
        print_error(f"File not found: {file_path}")
        return False

    if not file_path.is_file():
        print_error(f"Not a file: {file_path}")
        return False

    print_info(f"Uploading {file_path.name}...")

    with open(file_path, 'rb') as f:
        files = {'file': (file_path.name, f, 'application/octet-stream')}
        data = {}
        if title:
            data['title'] = title

        headers = {'password': API_KEY}

        try:
            response = requests.post(
                f"{API_BASE_URL}/documents",
                files=files,
                data=data,
                headers=headers
            )

            if response.status_code == 201:
                result = response.json()
                print_success("Document uploaded successfully!")
                print(f"  Document ID: {Colors.BLUE}{result['doc_id']}{Colors.ENDC}")
                print(f"  Filename: {result['filename']}")
                print(f"  Chunks: {result['chunks']}")
                return True
            else:
                error = response.json()
                print_error(f"Upload failed: {error.get('detail', 'Unknown error')}")
                return False

        except requests.exceptions.RequestException as e:
            print_error(f"Connection error: {e}")
            return False


def ask_question(query: str, hybrid: bool = False, reranker: bool = False, mmr: bool = False):
    """Ask a question to the knowledge base."""
    print_info(f"Searching for: '{query}'")

    request_data = {
        "query": query,
        "use_hybrid": hybrid,
        "use_reranker": reranker,
        "use_mmr": mmr,
        "top_k": 6
    }

    headers = {
        'Content-Type': 'application/json',
        'password': API_KEY
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/ask",
            json=request_data,
            headers=headers
        )

        if response.status_code == 200:
            result = response.json()

            print_header("Answer")
            print(f"{Colors.GREEN}{result['answer']}{Colors.ENDC}\n")

            print_header(f"Sources ({len(result['sources'])})")
            for idx, source in enumerate(result['sources'], 1):
                print(f"{Colors.BOLD}[{idx}]{Colors.ENDC} Page {source.get('page', 'N/A')} "
                      f"(Score: {Colors.CYAN}{source['score']:.3f}{Colors.ENDC})")
                preview = source['text_preview']
                if len(preview) > 150:
                    preview = preview[:150] + "..."
                print(f"    {preview}\n")

            return True
        else:
            error = response.json()
            detail = error.get('detail', {})
            if isinstance(detail, dict):
                message = detail.get('message', str(detail))
                suggestions = detail.get('suggestions', [])
                print_error(f"Query failed: {message}")
                if suggestions:
                    print("\nSuggestions:")
                    for suggestion in suggestions:
                        print(f"  • {suggestion}")
            else:
                print_error(f"Query failed: {detail}")
            return False

    except requests.exceptions.RequestException as e:
        print_error(f"Connection error: {e}")
        return False


def list_documents():
    """List all documents in the knowledge base."""
    print_error("List functionality not yet implemented in the API")
    print_info("Use the web UI at http://localhost:8000 or API docs at /docs")
    return False


def delete_document(doc_id: str):
    """Delete a document from the knowledge base."""
    print_info(f"Deleting document {doc_id}...")

    headers = {'password': API_KEY}

    try:
        response = requests.delete(
            f"{API_BASE_URL}/documents/{doc_id}",
            headers=headers
        )

        if response.status_code == 204:
            print_success(f"Document {doc_id} deleted successfully!")
            return True
        elif response.status_code == 404:
            print_error(f"Document {doc_id} not found")
            return False
        else:
            print_error(f"Delete failed: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print_error(f"Connection error: {e}")
        return False


def show_analytics():
    """Show analytics and usage statistics."""
    print_info("Fetching analytics...")

    try:
        response = requests.get(f"{API_BASE_URL}/analytics")

        if response.status_code == 200:
            data = response.json()

            print_header("Analytics Overview")
            overview = data['overview']
            print(f"Total Queries: {Colors.BOLD}{overview['total_queries']}{Colors.ENDC}")
            print(f"Total Uploads: {Colors.BOLD}{overview['total_uploads']}{Colors.ENDC}")
            print(f"Recent Queries: {overview['recent_queries']}")
            print(f"Recent Uploads: {overview['recent_uploads']}")

            print_header("Query Performance")
            perf = data['query_performance']
            print(f"Avg Response Time: {Colors.CYAN}{perf['avg_response_time_ms']:.0f}ms{Colors.ENDC}")
            print(f"Avg Results: {perf['avg_results_per_query']:.1f}")
            print(f"Cache Hit Rate: {Colors.GREEN}{perf['cache_hit_rate_percent']:.1f}%{Colors.ENDC}")
            print(f"Cache Hits: {perf['cache_hits']}")
            print(f"Cache Misses: {perf['cache_misses']}")

            if data.get('popular_queries'):
                print_header("Popular Queries")
                for query_data in data['popular_queries'][:5]:
                    print(f"  {query_data['count']}x: {query_data['query']}")

            return True
        else:
            print_error("Failed to fetch analytics")
            return False

    except requests.exceptions.RequestException as e:
        print_error(f"Connection error: {e}")
        return False


def check_health():
    """Check API health status."""
    try:
        response = requests.get(f"{API_BASE_URL}/healthz")

        if response.status_code == 200:
            data = response.json()
            print_success("API is healthy")
            print(f"  Status: {data['status']}")
            print(f"  Database: {data['database']}")
            return True
        else:
            print_error("API is unhealthy")
            return False

    except requests.exceptions.RequestException as e:
        print_error(f"Connection error: {e}")
        print_info(f"Make sure the API is running at {API_BASE_URL}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Knowledge Base CLI - Interact with your knowledge base from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s upload document.pdf --title "Research Paper"
  %(prog)s ask "What is machine learning?" --hybrid --reranker
  %(prog)s analytics
  %(prog)s health
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload a document')
    upload_parser.add_argument('file', help='Path to the document file')
    upload_parser.add_argument('--title', help='Optional document title')

    # Ask command
    ask_parser = subparsers.add_parser('ask', help='Ask a question')
    ask_parser.add_argument('query', help='The question to ask')
    ask_parser.add_argument('--hybrid', action='store_true', help='Use hybrid search (BM25 + Vector)')
    ask_parser.add_argument('--reranker', action='store_true', help='Use cross-encoder reranking')
    ask_parser.add_argument('--mmr', action='store_true', help='Use MMR for diverse results')

    # List command
    subparsers.add_parser('list', help='List all documents')

    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a document')
    delete_parser.add_argument('doc_id', help='Document ID to delete')

    # Analytics command
    subparsers.add_parser('analytics', help='Show usage analytics')

    # Health command
    subparsers.add_parser('health', help='Check API health')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    success = False
    if args.command == 'upload':
        success = upload_document(args.file, args.title)
    elif args.command == 'ask':
        success = ask_question(args.query, args.hybrid, args.reranker, args.mmr)
    elif args.command == 'list':
        success = list_documents()
    elif args.command == 'delete':
        success = delete_document(args.doc_id)
    elif args.command == 'analytics':
        success = show_analytics()
    elif args.command == 'health':
        success = check_health()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
