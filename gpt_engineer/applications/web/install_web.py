"""
Installer script for GPT Engineer Web IDE dependencies.

This script helps users easily install the required dependencies
for the Web IDE without modifying the core package requirements.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install the required dependencies for the Web IDE."""
    print("Installing dependencies for GPT Engineer Web IDE...")
    
    # Get the path to the requirements file
    current_dir = Path(__file__).parent
    requirements_path = current_dir / "requirements.txt"
    
    if not requirements_path.exists():
        print(f"Error: Could not find requirements file at {requirements_path}")
        return False
    
    try:
        # Install the dependencies
        subprocess.check_call([
            sys.executable, 
            "-m", "pip", "install", 
            "-r", str(requirements_path)
        ])
        
        print("\nDependencies installed successfully!")
        print("You can now run the Web IDE with the command: gpte-web")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        print("\nPlease try installing them manually with:")
        print(f"pip install -r {requirements_path}")
        return False

if __name__ == "__main__":
    install_dependencies()
