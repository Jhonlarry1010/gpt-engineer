"""
Run script for the GPT Engineer Web IDE.

This script provides a simple CLI entry point to launch the web IDE
without needing to import and call the main function directly.
"""

import sys
import os
import traceback
import importlib.util

def check_dependencies():
    """Check if required dependencies are installed and provide install instructions if not."""
    missing_deps = []
    
    # Check for Gradio
    if importlib.util.find_spec("gradio") is None:
        missing_deps.append("gradio")
    
    if missing_deps:
        print("\nError: Missing required dependencies for GPT Engineer Web IDE:")
        for dep in missing_deps:
            print(f"  - {dep}")
        
        print("\nTo install all required dependencies, run:")
        print("  pip install " + " ".join(missing_deps))
        print("\nOr use the built-in installer:")
        print("  gpte-web-install")
        return False
    
    return True

def main():
    """Main entry point for the Web IDE."""
    if not check_dependencies():
        return 1
    
    try:
        # Only import main if dependencies are available
        from gpt_engineer.applications.web.main import main as run_main
        run_main()
        return 0
    except ImportError as e:
        print(f"Error importing required modules: {str(e)}")
        print("\nThis may be because:")
        print("1. GPT Engineer is not properly installed")
        print("2. You're missing required dependencies")
        print("\nTry running: gpte-web-install")
        return 1
    except Exception as e:
        print(f"\nError starting the GPT Engineer Web IDE: {str(e)}")
        print("\nDetailed error information:")
        traceback.print_exc()
        print("\nIf the error persists, please check:")
        print("1. Your OpenAI API key is correctly set")
        print("2. You have the latest version of GPT Engineer and Gradio")
        print("3. Your Python environment is properly configured")
        return 1

if __name__ == "__main__":
    sys.exit(main())
