"""
Run script for the GPT Engineer Web IDE.

This script provides a simple CLI entry point to launch the web IDE
without needing to import and call the main function directly.
"""

import sys
import os
import traceback

def check_gradio_installed():
    """Check if Gradio is installed and provide install instructions if not."""
    try:
        import gradio
        return True
    except ImportError:
        print("Error: Gradio is not installed.")
        print("To use the GPT Engineer Web IDE, you need to install Gradio:")
        print("\npip install gradio\n")
        print("After installing Gradio, try running the Web IDE again.")
        return False

if __name__ == "__main__":
    try:
        # First check if Gradio is installed
        if check_gradio_installed():
            # Only import main if Gradio is available
            from gpt_engineer.applications.web.main import main
            main()
    except Exception as e:
        print(f"Error starting the GPT Engineer Web IDE: {str(e)}")
        print("\nDetailed error information:")
        traceback.print_exc()
        print("\nIf the error persists, please check:")
        print("1. Your OpenAI API key is correctly set")
        print("2. You have the latest version of GPT Engineer and Gradio")
        print("3. Your Python environment is properly configured")

        sys.exit(1)
