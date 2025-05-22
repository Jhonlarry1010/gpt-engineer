"""
Installer script for GPT Engineer Web IDE dependencies.

This script helps users easily install the required dependencies
for the Web IDE without modifying the core package requirements.
"""

import subprocess
import sys
import os
import tempfile
from pathlib import Path

def create_requirements_file():
    """Create a temporary requirements file with the necessary dependencies."""
    requirements = [
        "gradio>=4.0.0",
        "python-dotenv>=0.21.0",
        "openai>=1.0.0",
        "google-generativeai>=0.3.0"
    ]

    temp_fd, temp_path = tempfile.mkstemp(suffix=".txt", prefix="gpte_web_requirements_")
    with os.fdopen(temp_fd, 'w') as f:
        f.write("\n".join(requirements))

    return temp_path

def install_dependencies():
    """Install the required dependencies for the Web IDE."""
    print("Installing dependencies for GPT Engineer Web IDE...")

    # Create a temporary requirements file
    requirements_path = create_requirements_file()

    try:
        # Install the dependencies
        subprocess.check_call([
            sys.executable,
            "-m", "pip", "install",
            "-r", requirements_path
        ])

        print("\nDependencies installed successfully!")
        print("You can now run the Web IDE with the command: gpte-web")

        # Clean up
        os.unlink(requirements_path)
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print("\nPlease try installing them manually with:")
        print("pip install gradio>=4.0.0 python-dotenv>=0.21.0 openai>=1.0.0")

        # Clean up
        os.unlink(requirements_path)
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")

        # Clean up
        if os.path.exists(requirements_path):
            os.unlink(requirements_path)
        return False

def test_installation():
    """Test if the installation was successful."""
    print("\nTesting installation...")

    try:
        # Test importing gradio
        import gradio
        print(f"✅ Gradio {gradio.__version__} successfully installed")

        # Test importing other dependencies
        import dotenv
        print(f"✅ python-dotenv {dotenv.__version__} successfully installed")

        import openai
        print(f"✅ openai {openai.__version__} successfully installed")

        # Test importing Gemini (optional)
        try:
            import google.generativeai
            print(f"✅ google-generativeai successfully installed")
        except ImportError:
            print("⚠️ google-generativeai not installed (optional for Gemini support)")

        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Some dependencies may not have been installed correctly.")
        return False

if __name__ == "__main__":
    success = install_dependencies()
    if success:
        test_installation()
