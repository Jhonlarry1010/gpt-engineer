"""
Command-line interface for the GPT Engineer Modern IDE.

This module provides command-line commands for installing
and running the Modern IDE.
"""

import os
import sys
import importlib.util
import argparse
from pathlib import Path

def install_command():
    """Run the Modern IDE installer."""
    try:
        # Import and run the install module
        from gpt_engineer.applications.modern_ide.install import install_dependencies, test_installation, create_frontend_directory
        
        print("Installing GPT Engineer Modern IDE dependencies...")
        success = install_dependencies()
        
        if success:
            test_installation()
            create_frontend_directory()
            
            print("\n✅ Installation complete!")
            print("Run 'gpte-modern-ide' to start the Modern IDE.")
        
        return 0 if success else 1
    except ImportError as e:
        print(f"Error: {str(e)}")
        print("Make sure GPT Engineer is properly installed.")
        return 1
    except Exception as e:
        print(f"Error during installation: {str(e)}")
        return 1

def run_command(args=None):
    """Run the Modern IDE."""
    if args is None:
        # Parse command-line arguments
        parser = argparse.ArgumentParser(description="GPT Engineer Modern IDE")
        parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to (default: 127.0.0.1)")
        parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to (default: 8000)")
        parser.add_argument("--dev", action="store_true", help="Run in development mode with auto-reload")
        parser.add_argument("--browser", action="store_true", help="Open a browser window after starting the server")
        args = parser.parse_args()
    
    try:
        # Import and run the run module
        from gpt_engineer.applications.modern_ide.run import main
        return main()
    except ImportError as e:
        print(f"Error: {str(e)}")
        print("Make sure GPT Engineer Modern IDE dependencies are installed.")
        print("Run 'gpte-modern-ide-install' to install them.")
        return 1
    except Exception as e:
        print(f"Error starting the Modern IDE: {str(e)}")
        return 1

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "run"
    
    if command == "install":
        sys.exit(install_command())
    else:
        sys.exit(run_command())
