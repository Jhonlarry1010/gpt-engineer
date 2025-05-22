"""
Run script for the GPT Engineer Modern IDE.

This script provides a simple CLI entry point to launch the Modern IDE
without needing to import and call the main function directly.
"""

import os
import sys
import argparse
import importlib.util
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed and provide install instructions if not."""
    missing_deps = []
    
    # Check for FastAPI
    if importlib.util.find_spec("fastapi") is None:
        missing_deps.append("fastapi")
    
    # Check for Uvicorn
    if importlib.util.find_spec("uvicorn") is None:
        missing_deps.append("uvicorn")
    
    if missing_deps:
        print("\nError: Missing required dependencies for GPT Engineer Modern IDE:")
        for dep in missing_deps:
            print(f"  - {dep}")
        
        print("\nTo install all required dependencies, run:")
        print("  gpte-modern-ide-install")
        return False
    
    return True

def create_frontend_if_missing():
    """Create the frontend directory and placeholder if it doesn't exist."""
    script_dir = Path(__file__).parent
    frontend_dir = script_dir / "frontend" / "build"
    index_path = frontend_dir / "index.html"
    
    if not frontend_dir.exists() or not index_path.exists():
        print("Frontend files not found. Creating placeholder...")
        try:
            # Import and run the create_frontend_directory function from install.py
            from gpt_engineer.applications.modern_ide.install import create_frontend_directory
            create_frontend_directory()
        except ImportError:
            # Fallback if import fails
            frontend_dir.mkdir(parents=True, exist_ok=True)
            with open(index_path, "w") as f:
                f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT Engineer Modern IDE</title>
    <style>
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            line-height: 1.6;
            color: #333;
        }
        h1 {
            color: #2563eb;
        }
        .note {
            background-color: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 1rem;
            margin: 1rem 0;
        }
    </style>
</head>
<body>
    <h1>GPT Engineer Modern IDE</h1>
    <p>Placeholder frontend page. API endpoints are available.</p>
    <div class="note">
        <p>The full React frontend is not yet built.</p>
    </div>
</body>
</html>
""")
            print("Created placeholder frontend")

def main():
    """Main entry point for the Modern IDE."""
    parser = argparse.ArgumentParser(description="GPT Engineer Modern IDE")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode with auto-reload"
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open a browser window after starting the server"
    )
    
    args = parser.parse_args()
    
    if not check_dependencies():
        return 1
    
    # Create frontend placeholder if needed
    create_frontend_if_missing()
    
    try:
        # Import the server module and run it
        from gpt_engineer.applications.modern_ide.server import run_server
        
        print(f"Starting GPT Engineer Modern IDE on http://{args.host}:{args.port}")
        
        # Open browser if requested
        if args.browser:
            import webbrowser
            webbrowser.open(f"http://{args.host}:{args.port}")
        
        # Run the server
        run_server(host=args.host, port=args.port, dev_mode=args.dev)
        return 0
    except ImportError as e:
        print(f"Error importing required modules: {str(e)}")
        print("\nThis may be because GPT Engineer is not properly installed")
        return 1
    except Exception as e:
        print(f"\nError starting the GPT Engineer Modern IDE: {str(e)}")
        print("\nIf the error persists, please check:")
        print("1. Your OpenAI API key is correctly set")
        print("2. You have the latest version of all dependencies")
        return 1

if __name__ == "__main__":
    sys.exit(main())
