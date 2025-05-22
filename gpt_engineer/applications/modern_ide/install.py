"""
Installer script for GPT Engineer Modern IDE dependencies.

This script helps users easily install the required dependencies
for the Modern IDE without modifying the core package requirements.
"""

import subprocess
import sys
import os
import tempfile
from pathlib import Path

def create_requirements_file():
    """Create a temporary requirements file with the necessary dependencies."""
    requirements = [
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "python-dotenv>=0.21.0",
        "openai>=1.0.0",
        "httpx>=0.24.0",
        "pydantic>=2.3.0"
    ]

    temp_fd, temp_path = tempfile.mkstemp(suffix=".txt", prefix="gpte_modern_ide_requirements_")
    with os.fdopen(temp_fd, 'w') as f:
        f.write("\n".join(requirements))

    return temp_path

def install_dependencies():
    """Install the required dependencies for the Modern IDE."""
    print("Installing dependencies for GPT Engineer Modern IDE...")

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
        print("You can now run the Modern IDE with the command: gpte-modern-ide")

        # Clean up
        os.unlink(requirements_path)
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print("\nPlease try installing them manually with:")
        print("pip install fastapi uvicorn python-dotenv openai httpx pydantic")

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
        # Test importing fastapi
        import fastapi
        print(f"✅ FastAPI {fastapi.__version__} successfully installed")

        # Test importing uvicorn
        import uvicorn
        print(f"✅ Uvicorn {uvicorn.__version__} successfully installed")

        # Test importing other dependencies
        import dotenv
        print(f"✅ python-dotenv {dotenv.__version__} successfully installed")

        import openai
        print(f"✅ openai {openai.__version__} successfully installed")

        import httpx
        print(f"✅ httpx {httpx.__version__} successfully installed")

        import pydantic
        print(f"✅ pydantic {pydantic.__version__} successfully installed")

        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Some dependencies may not have been installed correctly.")
        return False

def create_frontend_directory():
    """Create the frontend directory structure."""
    script_dir = Path(__file__).parent
    frontend_dir = script_dir / "frontend" / "build"
    frontend_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a placeholder index.html
    index_path = frontend_dir / "index.html"
    if not index_path.exists():
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
        .card {
            padding: 1.5rem;
            border-radius: 0.5rem;
            background-color: #f3f4f6;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        }
        code {
            background-color: #e5e7eb;
            padding: 0.2rem 0.4rem;
            border-radius: 0.25rem;
            font-family: monospace;
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
    <p>The React-based frontend for the Modern IDE is not yet built. Please follow these instructions to build it:</p>
    
    <div class="card">
        <h2>API Endpoints Available</h2>
        <p>The backend API is running and can be accessed at: <code>/api/*</code></p>
        <p>For example, try: <a href="/api/health">/api/health</a> to check the server status.</p>
    </div>
    
    <div class="note">
        <p>Note: This is a placeholder page. The full React frontend will be available in future releases.</p>
    </div>
</body>
</html>
""")
        print(f"✅ Created placeholder frontend at {index_path}")
    else:
        print(f"✅ Frontend already exists at {frontend_dir}")

if __name__ == "__main__":
    success = install_dependencies()
    if success:
        test_installation()
        create_frontend_directory()
