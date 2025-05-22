"""
Example script for using the Gemini API with GPT Engineer.

This script demonstrates how to use the Gemini API for code generation
and provides a simple command-line interface for testing.
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path to allow importing from gpt_engineer
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from gpt_engineer.applications.web.gemini_ai import GeminiAI
except ImportError:
    print("Error: Could not import GeminiAI. Make sure google-generativeai is installed.")
    print("Install it with: pip install google-generativeai")
    sys.exit(1)

def main():
    """Run the Gemini example script."""
    parser = argparse.ArgumentParser(description="Test Gemini API with GPT Engineer")
    parser.add_argument("--api-key", help="Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model to use")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for generation")
    parser.add_argument("--prompt", help="Prompt for generation")
    
    args = parser.parse_args()
    
    # Get API key from args or environment
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: No Gemini API key provided. Use --api-key or set GEMINI_API_KEY environment variable.")
        sys.exit(1)
    
    # Initialize the Gemini AI
    try:
        ai = GeminiAI(
            model_name=args.model,
            temperature=args.temperature,
            api_key=api_key
        )
        print(f"Initialized Gemini model: {args.model}")
    except Exception as e:
        print(f"Error initializing Gemini model: {e}")
        sys.exit(1)
    
    # Get prompt from args or interactive input
    prompt = args.prompt
    if not prompt:
        print("\nEnter your prompt (type 'exit' to quit):")
        print("----------------------------------------")
        prompt = input("> ")
    
    if prompt.lower() == "exit":
        sys.exit(0)
    
    # Generate response
    print("\nGenerating response...\n")
    response = ai.complete(prompt)
    
    print("Response:")
    print("----------------------------------------")
    print(response)
    print("----------------------------------------")
    
    # Print token usage info
    print(f"\nTokens used (approximate): {ai.token_usage_log.total_tokens()}")
    print(f"Estimated cost: ${ai.token_usage_log.usage_cost():.6f}")

if __name__ == "__main__":
    main()
