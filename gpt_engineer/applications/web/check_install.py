"""
Diagnostic tool for GPT Engineer Web IDE.

This script checks the installation status of required dependencies
and provides detailed information about the system configuration.
"""

import sys
import os
import platform
import importlib.util
import subprocess
import traceback


def check_python_version():
    """Check if the Python version is compatible."""
    print(f"Python Version: {sys.version}")
    version_info = sys.version_info
    if version_info.major < 3 or (version_info.major == 3 and version_info.minor < 10):
        print("WARNING: GPT Engineer works best with Python 3.10 or newer")
        return False
    return True


def check_dependency(package_name, min_version=None):
    """Check if a dependency is installed and meets the minimum version requirement."""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            print(f"❌ {package_name}: Not installed")
            return False
        
        module = importlib.import_module(package_name)
        version = getattr(module, "__version__", "unknown")
        
        if min_version and version != "unknown":
            # Very simple version comparison - not handling complex semver
            if version < min_version:
                print(f"⚠️ {package_name}: Installed (version {version}, but {min_version} or newer recommended)")
                return False
        
        print(f"✅ {package_name}: Installed (version {version})")
        return True
    
    except Exception as e:
        print(f"❌ {package_name}: Error checking ({str(e)})")
        return False


def check_openai_api_key():
    """Check if the OpenAI API key is set."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
        print(f"✅ OPENAI_API_KEY: Set ({masked_key})")
        return True
    else:
        print("❌ OPENAI_API_KEY: Not set")
        return False


def check_pip_installation():
    """Check pip installation status."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ pip: Installed ({result.stdout.strip()})")
            return True
        else:
            print(f"❌ pip: Error ({result.stderr.strip()})")
            return False
    except Exception as e:
        print(f"❌ pip: Error checking ({str(e)})")
        return False


def check_system_info():
    """Check system information."""
    print(f"Operating System: {platform.system()} {platform.release()} ({platform.architecture()[0]})")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")


def check_internet_connection():
    """Check internet connection."""
    try:
        result = subprocess.run(
            ["ping", "api.openai.com", "-c", "1"] if platform.system() != "Windows" else ["ping", "api.openai.com", "-n", "1"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ Internet Connection: OK")
            return True
        else:
            print("❌ Internet Connection: Failed to reach api.openai.com")
            return False
    except Exception as e:
        print(f"❌ Internet Connection: Error checking ({str(e)})")
        return False


def suggest_fixes(issues):
    """Suggest fixes for common issues."""
    if "gradio" in issues:
        print("\nTo install Gradio:")
        print("  pip install gradio>=4.0.0")
        print("  OR")
        print("  gpte-web-install")
    
    if "openai" in issues:
        print("\nTo install OpenAI Python client:")
        print("  pip install openai>=1.0.0")
    
    if "python-dotenv" in issues:
        print("\nTo install python-dotenv:")
        print("  pip install python-dotenv>=0.21.0")
    
    if "openai_api_key" in issues:
        print("\nTo set up your OpenAI API key:")
        print("  1. Get your API key from https://platform.openai.com/api-keys")
        print("  2. Set it as an environment variable:")
        if platform.system() == "Windows":
            print("     set OPENAI_API_KEY=your-key-here")
        else:
            print("     export OPENAI_API_KEY=your-key-here")
        print("  3. Or create a .env file in your project directory with:")
        print("     OPENAI_API_KEY=your-key-here")


def main():
    """Run the diagnostic checks."""
    print("===== GPT Engineer Web IDE Diagnostic Tool =====\n")
    
    print("System Information:")
    check_system_info()
    print()
    
    print("Python Environment:")
    python_ok = check_python_version()
    pip_ok = check_pip_installation()
    print()
    
    print("Internet Connectivity:")
    internet_ok = check_internet_connection()
    print()
    
    print("API Keys:")
    api_key_ok = check_openai_api_key()
    print()
    
    print("Required Dependencies:")
    gradio_ok = check_dependency("gradio", "4.0.0")
    openai_ok = check_dependency("openai", "1.0.0")
    dotenv_ok = check_dependency("dotenv", "0.21.0")
    print()
    
    # Collect issues
    issues = []
    if not gradio_ok:
        issues.append("gradio")
    if not openai_ok:
        issues.append("openai")
    if not dotenv_ok:
        issues.append("python-dotenv")
    if not api_key_ok:
        issues.append("openai_api_key")
    
    # Print summary
    print("===== Diagnostic Summary =====")
    if issues:
        print(f"❌ Found {len(issues)} issue(s) that need to be resolved:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRecommended fixes:")
        suggest_fixes(issues)
    else:
        print("✅ All checks passed! Your system should be ready to run GPT Engineer Web IDE.")
        print("   Start the Web IDE with: gpte-web")
    
    return len(issues)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Diagnostic tool encountered an error: {str(e)}")
        print("\nDetailed error information:")
        traceback.print_exc()
        sys.exit(1)
