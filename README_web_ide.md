# GPT Engineer Web IDE

The GPT Engineer Web IDE provides a user-friendly interface for interacting with GPT Engineer through your web browser instead of the command line.

## Features

- **AI-Powered Code Generation**: Describe what you want to build in natural language, and let GPT Engineer generate the code for you.
- **Code Improvement**: Have the AI improve existing code based on your specifications.
- **File Explorer**: View and edit all files in your project.
- **Integrated Terminal**: Run and test your code directly from the web interface.
- **AI Chat Assistant**: Interact with the AI to discuss your project, ask questions, and get programming help.

## Installation

### Using the Built-in Installer (Recommended)

The simplest way to install the Web IDE dependencies is with the built-in installer:

```bash
gpte-web-install
```

This will automatically install Gradio and other required dependencies.

### Manual Installation

If the built-in installer doesn't work, you can manually install the dependencies:

```bash
pip install gradio>=4.0.0 python-dotenv>=0.21.0 openai>=1.0.0
```

### Troubleshooting Installation

If you encounter installation issues:

1. Make sure your pip is up to date:

   ```bash
   pip install --upgrade pip
   ```

2. Try installing with the `--user` flag:

   ```bash
   pip install --user gradio>=4.0.0
   ```

3. If you're using a virtual environment, make sure it's activated:

   ```bash
   # For virtualenv
   source venv/bin/activate  # On Linux/Mac
   venv\Scripts\activate     # On Windows

   # For conda
   conda activate your-env-name
   ```

## Usage

To start the Web IDE, run:

```bash
gpte-web
```

Or, if you installed from source:

```bash
python -m gpt_engineer.applications.web.run
```

This will launch the web interface in your default browser. If it doesn't open automatically, you can access it at http://localhost:7860 by default.

## Quick Start Guide

1. **Set Up AI**:

   - Enter your OpenAI API key
   - Select a model (GPT-4o recommended)
   - Click "Set Up AI"

2. **Create a Project**:

   - Enter a project name
   - Click "Create Project"
   - Click "Initialize Agent for Selected Project"

3. **Generate Code**:

   - In the "Generate Code" tab, describe what you want to build
   - Click "Generate Code"
   - Wait for the AI to write your code

4. **Explore and Edit Files**:

   - Use the file explorer to navigate through generated files
   - Edit files as needed in the code editor
   - Click "Save File" to save changes

5. **Run Your Code**:
   - Click "Run Code" to execute your project
   - View the output in the terminal section

## Tips for Better Results

- **Be Specific**: The more detailed your description, the better the generated code will match your expectations.
- **Iterative Development**: Use the "Improve Code" feature to refine your project in steps.
- **Use the Chat**: The AI assistant can help explain concepts or suggest improvements.

## Requirements

- Python 3.10 or newer
- An OpenAI API key with access to GPT-4 models
- Gradio (for the web interface)

## Limitations

- The Web IDE requires an internet connection to communicate with the OpenAI API.
- Complex projects may require multiple iterations or manual adjustments.

## Troubleshooting

If you encounter errors when running the Web IDE:

1. **Startup Issues**:

   - Run `gpte-web-install` to ensure all dependencies are installed
   - Check the console output for specific error messages
   - Make sure you're using Python 3.10 or newer

2. **API Key Problems**:

   - Ensure your OpenAI API key is valid and has sufficient quota
   - Check that the API key is correctly entered in the Web IDE interface

3. **Web Interface Problems**:

   - If the interface doesn't load properly, try a different web browser
   - Clear your browser cache and cookies
   - Check if your firewall or antivirus is blocking the connection

4. **Error Messages**:

   - Note down any error messages shown in the terminal
   - For "Error: X is not defined" messages, make sure all dependencies are installed

5. **Code Generation Issues**:

   - If code generation takes too long, try using a simpler prompt
   - Check your internet connection stability
   - Verify that your OpenAI API key has adequate rate limits

6. **UI Freezes or Crashes**:
   - Restart the application with `gpte-web`
   - Clear your browser cache and try again
   - Check system resources (memory usage, CPU)

If problems persist, you can always fall back to the command-line interface with `gpte` while troubleshooting the web interface.
