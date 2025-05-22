# Using Gemini with GPT Engineer Web IDE

GPT Engineer Web IDE now supports Google's Gemini models as an alternative to OpenAI's models. This guide explains how to set up and use Gemini with GPT Engineer.

## Getting a Gemini API Key

To use Gemini, you'll need a Google AI Studio API key:

1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create or sign in to your Google account
3. Navigate to "Get API key" in the menu
4. Create a new API key or use an existing one
5. Copy the API key for use with GPT Engineer

## Installation

Make sure you have the Google Generative AI package installed:

```bash
pip install google-generativeai
```

Or use the built-in installer which now includes Gemini support:

```bash
gpte-web-install
```

## Using Gemini in the Web IDE

1. Start the Web IDE with `gpte-web`
2. In the "AI Setup" tab, select the "Gemini" sub-tab
3. Enter your Gemini API key
4. Select a Gemini model (gemini-2.0-flash, gemini-2.0-pro, etc.)
5. Set the temperature (0.1 for more deterministic results)
6. Click "Set Up Gemini"
7. Continue with project creation and code generation as usual

## Available Gemini Models

The Web IDE supports several Gemini models:

- **gemini-2.0-flash**: Fast and efficient, good for most code generation tasks
- **gemini-2.0-pro**: Higher quality results but slower and more expensive
- **gemini-1.5-pro**: Previous generation model with good performance

## Environment Variables

You can set your Gemini API key as an environment variable instead of entering it in the UI:

```bash
export GEMINI_API_KEY=your_api_key_here
```

Or in a .env file in your project directory:

```
GEMINI_API_KEY=your_api_key_here
```

## Testing Gemini Separately

You can test Gemini without the Web IDE using the included example script:

```bash
python -m gpt_engineer.applications.web.gemini_example --api-key your_api_key_here --prompt "Create a Python function that sorts a list of dictionaries by a specific key"
```

## Limitations

There are some differences between OpenAI and Gemini models:

- Gemini models may have different strengths and weaknesses compared to OpenAI models
- Token counting is approximate for Gemini as they use a different tokenization method
- Some advanced features may not be available with all Gemini models
- Error handling may differ between the two APIs

## Switching Between Providers

You can easily switch between OpenAI and Gemini by selecting the appropriate tab in the AI Setup section. The Web IDE will use the most recently configured provider for code generation.

## Troubleshooting

If you encounter issues with Gemini:

1. Verify your API key is correct and has adequate quota
2. Ensure you have the latest version of the google-generativeai package
3. Try a different model if you're experiencing specific issues
4. Check the console output for detailed error messages
