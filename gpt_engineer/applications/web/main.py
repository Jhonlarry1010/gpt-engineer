"""
Main entry point for the GPT Engineer Web IDE.

This module provides a web-based interface for GPT Engineer using Gradio,
allowing users to:
- Write project specifications in natural language
- Generate code from these specifications
- View and edit generated files
- Improve existing code
- Run and test the generated code
"""

import os
import sys
import logging
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable

# Check if gradio is installed
GRADIO_AVAILABLE = importlib.util.find_spec("gradio") is not None

# Only import gradio if it's available
if GRADIO_AVAILABLE:
    import gradio as gr

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("gpt_engineer.web")

# Import the rest of our dependencies
# We'll do this in a try/except block to provide helpful error messages
try:
    from gpt_engineer.core.ai import AI
    from gpt_engineer.core.default.disk_execution_env import DiskExecutionEnv
    from gpt_engineer.core.default.disk_memory import DiskMemory
    from gpt_engineer.core.default.file_store import FileStore
    from gpt_engineer.core.default.paths import PREPROMPTS_PATH, memory_path
    from gpt_engineer.core.default.steps import gen_code, improve_fn, execute_entrypoint
    from gpt_engineer.core.preprompts_holder import PrepromptsHolder
    from gpt_engineer.core.prompt import Prompt
    from gpt_engineer.core.files_dict import FilesDict

    # We'll import ide_agent in a function to avoid circular imports
    WEB_CORE_DEPS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing core dependencies: {e}")
    WEB_CORE_DEPS_AVAILABLE = False


class WebIde:
    """Main web IDE application class."""

    def __init__(self):
        """Initialize the Web IDE."""
        self.projects_dir = Path("projects")
        self.projects_dir.mkdir(exist_ok=True)
        self.current_project = None
        self.ai = None
        self.agent = None
        self.files_dict = None
        self.execution_env = DiskExecutionEnv() if WEB_CORE_DEPS_AVAILABLE else None
        self.logger = logger

    def list_projects(self) -> List[str]:
        """List all available projects."""
        try:
            return [dir.name for dir in self.projects_dir.iterdir() if dir.is_dir()]
        except Exception as e:
            self.logger.error(f"Error listing projects: {e}")
            return ["Error listing projects"]

    def init_ai(self, model_name: str, temperature: float, api_key: Optional[str] = None, provider: str = "openai") -> str:
        """Initialize the AI with given parameters."""
        if not WEB_CORE_DEPS_AVAILABLE:
            return "Error: Core dependencies not available"

        try:
            if provider == "openai":
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key

                from gpt_engineer.core.ai import AI
                self.ai = AI(model_name=model_name, temperature=temperature)
                return f"OpenAI initialized with model {model_name}"
            elif provider == "gemini":
                if api_key:
                    os.environ["GEMINI_API_KEY"] = api_key

                try:
                    # Import here to avoid dependency issues
                    from gpt_engineer.applications.web.gemini_ai import GeminiAI
                    self.ai = GeminiAI(model_name=model_name, temperature=temperature, api_key=api_key)
                    return f"Gemini initialized with model {model_name}"
                except ImportError:
                    return "Error: Gemini support not available. Please install the google-generativeai package with: pip install google-generativeai"
            else:
                return f"Error: Unknown provider {provider}"
        except Exception as e:
            self.logger.error(f"Error initializing AI: {e}")
            return f"Error: {str(e)}"

    def create_project(self, project_name: str) -> str:
        """Create a new project with the given name."""
        if not project_name or not project_name.strip():
            return "Error: Project name cannot be empty"

        try:
            project_path = self.projects_dir / project_name
            if project_path.exists():
                return f"Error: Project '{project_name}' already exists"

            project_path.mkdir(parents=True)
            self.current_project = project_path
            return f"Project '{project_name}' created successfully"
        except Exception as e:
            self.logger.error(f"Error creating project: {e}")
            return f"Error: {str(e)}"

    def select_project(self, project_name: str) -> str:
        """Select an existing project to work with."""
        if not project_name:
            return "No project selected"

        try:
            project_path = self.projects_dir / project_name
            if not project_path.exists() or not project_path.is_dir():
                return f"Error: Project '{project_name}' does not exist"

            self.current_project = project_path
            return f"Project '{project_name}' selected"
        except Exception as e:
            self.logger.error(f"Error selecting project: {e}")
            return f"Error: {str(e)}"

    def init_agent(self) -> str:
        """Initialize the agent for the current project."""
        if not WEB_CORE_DEPS_AVAILABLE:
            return "Error: Core dependencies not available"

        if not self.current_project:
            return "Error: No project selected"

        if not self.ai:
            return "Error: AI not initialized. Please set up the AI first."

        try:
            # Import here to avoid circular imports
            from gpt_engineer.applications.web.ide_agent import WebIdeAgent

            memory = DiskMemory(memory_path(self.current_project))
            preprompts_holder = PrepromptsHolder(PREPROMPTS_PATH)

            def status_callback(msg):
                self.logger.info(f"Agent status: {msg}")

            self.agent = WebIdeAgent.with_default_config(
                memory,
                self.execution_env,
                ai=self.ai,
                code_gen_fn=gen_code,
                improve_fn=improve_fn,
                process_code_fn=execute_entrypoint,
                preprompts_holder=preprompts_holder,
                status_callback=status_callback
            )

            return f"Agent initialized for project '{self.current_project.name}'"
        except Exception as e:
            self.logger.error(f"Error initializing agent: {e}")
            return f"Error: {str(e)}"

    def generate_code(self, prompt_text: str) -> Tuple[str, List[str]]:
        """Generate code from the given prompt."""
        if not WEB_CORE_DEPS_AVAILABLE:
            return "Error: Core dependencies not available", []

        if not self.current_project:
            return "Error: No project selected", []

        if not self.agent:
            return "Error: Agent not initialized", []

        if not prompt_text or not prompt_text.strip():
            return "Error: Prompt cannot be empty", []

        try:
            # Save the prompt to a file
            prompt_file = self.current_project / "prompt"
            prompt_file.write_text(prompt_text)

            # Create prompt object
            prompt = Prompt(prompt_text)

            # Generate code
            self.logger.info("Generating code...")
            self.files_dict = self.agent.init(prompt)

            # Save generated files
            file_store = FileStore(self.current_project)
            file_store.push(self.files_dict)

            # Return success message and list of generated files
            return "Code generated successfully", list(self.files_dict.keys())
        except Exception as e:
            self.logger.error(f"Error generating code: {e}")
            return f"Error: {str(e)}", []

    def improve_code(self, prompt_text: str) -> Tuple[str, List[str]]:
        """Improve existing code with the given prompt."""
        if not WEB_CORE_DEPS_AVAILABLE:
            return "Error: Core dependencies not available", []

        if not self.current_project:
            return "Error: No project selected", []

        if not self.agent:
            return "Error: Agent not initialized", []

        if not prompt_text or not prompt_text.strip():
            return "Error: Improvement prompt cannot be empty", []

        try:
            # Load existing files
            file_store = FileStore(self.current_project)
            files_dict_before = file_store.get_all_files()

            if not files_dict_before:
                return "Error: No existing files found to improve", []

            # Save the prompt to a file
            improve_prompt_file = self.current_project / "improve_prompt"
            improve_prompt_file.write_text(prompt_text)

            # Create prompt object
            prompt = Prompt(prompt_text)

            # Improve code
            self.logger.info("Improving code...")
            self.files_dict = self.agent.improve(prompt, files_dict_before)

            # Save improved files
            file_store.push(self.files_dict)

            # Return success message and list of improved files
            return "Code improved successfully", list(self.files_dict.keys())
        except Exception as e:
            self.logger.error(f"Error improving code: {e}")
            return f"Error: {str(e)}", []

    def get_file_content(self, filename: str) -> str:
        """Get the content of a file in the current project."""
        if not self.current_project:
            return "Error: No project selected"

        try:
            file_path = self.current_project / filename
            if not file_path.exists():
                return f"Error: File '{filename}' does not exist"

            return file_path.read_text()
        except Exception as e:
            self.logger.error(f"Error reading file: {e}")
            return f"Error: {str(e)}"

    def save_file_content(self, filename: str, content: str) -> str:
        """Save content to a file in the current project."""
        if not self.current_project:
            return "Error: No project selected"

        if not filename or not filename.strip():
            return "Error: Filename cannot be empty"

        try:
            file_path = self.current_project / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            return f"File '{filename}' saved successfully"
        except Exception as e:
            self.logger.error(f"Error saving file: {e}")
            return f"Error: {str(e)}"

    def run_code(self) -> str:
        """Run the code in the current project."""
        if not WEB_CORE_DEPS_AVAILABLE:
            return "Error: Core dependencies not available"

        if not self.current_project:
            return "Error: No project selected"

        try:
            # Find entrypoint based on common patterns
            entrypoint_candidates = ["main.py", "app.py", "index.py", "run.py"]
            entrypoint = None

            for candidate in entrypoint_candidates:
                if (self.current_project / candidate).exists():
                    entrypoint = candidate
                    break

            if not entrypoint:
                return "Error: No entrypoint found (main.py, app.py, index.py, or run.py)"

            # Execute code using the execution environment
            self.logger.info(f"Running {entrypoint}...")
            result = self.execution_env.execute_command(
                self.current_project, f"python {entrypoint}"
            )

            return result.output
        except Exception as e:
            self.logger.error(f"Error running code: {e}")
            return f"Error: {str(e)}"


def create_web_ui(ide: WebIde):
    """Create the Gradio web interface for the IDE."""
    if not GRADIO_AVAILABLE:
        raise ImportError("Gradio is not installed. Please install it with: pip install gradio")

    try:
        with gr.Blocks(title="GPT Engineer - AI IDE", theme=gr.themes.Soft()) as app:
            gr.Markdown("# GPT Engineer - AI IDE")
            gr.Markdown(
                "Specify what you want to build, and let AI write and execute the code."
            )

            with gr.Tabs() as tabs:
            # AI Setup Tab
            with gr.TabItem("AI Setup"):
                with gr.Tabs() as api_tabs:
                    with gr.TabItem("OpenAI"):
                        openai_api_key = gr.Textbox(
                            label="OpenAI API Key",
                            placeholder="Enter your OpenAI API key...",
                            type="password"
                        )

                        with gr.Row():
                            openai_model_name = gr.Dropdown(
                                label="AI Model",
                                choices=["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
                                value="gpt-4o"
                            )
                            openai_temperature = gr.Slider(
                                label="Temperature",
                                minimum=0,
                                maximum=1,
                                value=0.1,
                                step=0.1
                            )

                        openai_setup_button = gr.Button("Set Up OpenAI", variant="primary")

                    with gr.TabItem("Gemini"):
                        gemini_api_key = gr.Textbox(
                            label="Gemini API Key",
                            placeholder="Enter your Gemini API key...",
                            type="password"
                        )

                        with gr.Row():
                            gemini_model_name = gr.Dropdown(
                                label="AI Model",
                                choices=["gemini-2.0-flash", "gemini-2.0-pro", "gemini-1.5-pro"],
                                value="gemini-2.0-flash"
                            )
                            gemini_temperature = gr.Slider(
                                label="Temperature",
                                minimum=0,
                                maximum=1,
                                value=0.1,
                                step=0.1
                            )

                        gemini_setup_button = gr.Button("Set Up Gemini", variant="primary")

                setup_output = gr.Markdown()

                openai_setup_button.click(
                    fn=lambda model, temp, key: ide.init_ai(model, temp, key, provider="openai"),
                    inputs=[openai_model_name, openai_temperature, openai_api_key],
                    outputs=[setup_output]
                )

                gemini_setup_button.click(
                    fn=lambda model, temp, key: ide.init_ai(model, temp, key, provider="gemini"),
                    inputs=[gemini_model_name, gemini_temperature, gemini_api_key],
                    outputs=[setup_output]
                )

                # Project Management Tab
                with gr.TabItem("Project Management"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Create New Project")
                            new_project_name = gr.Textbox(
                                label="Project Name",
                                placeholder="Enter project name..."
                            )
                            create_button = gr.Button("Create Project", variant="primary")
                            create_output = gr.Markdown()

                        with gr.Column():
                            gr.Markdown("### Select Existing Project")
                            refresh_button = gr.Button("Refresh Project List")
                            project_dropdown = gr.Dropdown(label="Projects", choices=ide.list_projects())
                            select_button = gr.Button("Select Project")
                            select_output = gr.Markdown()

                    # Initialize agent button
                    init_agent_button = gr.Button("Initialize Agent for Selected Project")
                    init_agent_output = gr.Markdown()

                    # Button click functions
                    refresh_button.click(
                        fn=ide.list_projects,
                        inputs=[],
                        outputs=[project_dropdown]
                    )

                    create_button.click(
                        fn=ide.create_project,
                        inputs=[new_project_name],
                        outputs=[create_output]
                    ).then(
                        fn=ide.list_projects,
                        inputs=[],
                        outputs=[project_dropdown]
                    )

                    select_button.click(
                        fn=ide.select_project,
                        inputs=[project_dropdown],
                        outputs=[select_output]
                    )

                    init_agent_button.click(
                        fn=ide.init_agent,
                        inputs=[],
                        outputs=[init_agent_output]
                    )

                # Code Generation Tab
                with gr.TabItem("Generate Code"):
                    generation_prompt = gr.Textbox(
                        label="Project Specification",
                        placeholder="Describe the application you want to build...",
                        lines=10
                    )
                    generate_button = gr.Button("Generate Code", variant="primary")
                    generation_status = gr.Markdown()

                    with gr.Row():
                        generation_result = gr.Markdown(label="Result")
                        generated_files = gr.Dropdown(label="Generated Files")

                    generate_button.click(
                        fn=ide.generate_code,
                        inputs=[generation_prompt],
                        outputs=[generation_result, generated_files]
                    )

                # Code Improvement Tab
                with gr.TabItem("Improve Code"):
                    improvement_prompt = gr.Textbox(
                        label="Improvement Specification",
                        placeholder="Describe how you want to improve the existing code...",
                        lines=10
                    )
                    improve_button = gr.Button("Improve Code", variant="primary")
                    improvement_status = gr.Markdown()

                    with gr.Row():
                        improvement_result = gr.Markdown(label="Result")
                        improved_files = gr.Dropdown(label="Improved Files")

                    improve_button.click(
                        fn=ide.improve_code,
                        inputs=[improvement_prompt],
                        outputs=[improvement_result, improved_files]
                    )

                # File Explorer Tab
                with gr.TabItem("File Explorer"):
                    with gr.Row():
                        file_dropdown = gr.Dropdown(label="Select File")
                        refresh_files_button = gr.Button("Refresh Files")

                    file_editor = gr.Code(language="python", label="File Editor", interactive=True)
                    save_button = gr.Button("Save File")
                    file_status = gr.Markdown()

                    # Button click functions
                    def refresh_files():
                        if not ide.current_project:
                            return ["No project selected"]
                        try:
                            return [f.name for f in ide.current_project.glob("**/*") if f.is_file()]
                        except Exception as e:
                            ide.logger.error(f"Error refreshing files: {e}")
                            return ["Error refreshing files"]

                    refresh_files_button.click(
                        fn=refresh_files,
                        inputs=[],
                        outputs=[file_dropdown]
                    )

                    file_dropdown.change(
                        fn=ide.get_file_content,
                        inputs=[file_dropdown],
                        outputs=[file_editor]
                    )

                    save_button.click(
                        fn=ide.save_file_content,
                        inputs=[file_dropdown, file_editor],
                        outputs=[file_status]
                    )

                # Run & Test Tab
                with gr.TabItem("Run & Test"):
                    run_button = gr.Button("Run Code", variant="primary")
                    run_status = gr.Markdown()
                    output_terminal = gr.Textbox(label="Output", lines=10, max_lines=30)

                    run_button.click(
                        fn=ide.run_code,
                        inputs=[],
                        outputs=[output_terminal]
                    )

        return app

    except Exception as e:
        logger.error(f"Error creating web UI: {e}")
        # Create a minimal fallback UI if there's an error with the main UI
        with gr.Blocks(title="GPT Engineer - AI IDE (Fallback Mode)") as fallback_app:
            gr.Markdown("# GPT Engineer - AI IDE (Fallback Mode)")
            gr.Markdown(f"There was an error creating the full UI: {str(e)}")
            gr.Markdown("This is a minimal fallback interface.")

            with gr.Row():
                api_key = gr.Textbox(label="OpenAI API Key", type="password")
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"],
                    value="gpt-4o"
                )

            prompt = gr.Textbox(label="Project Specification", lines=10)
            generate_btn = gr.Button("Generate Code")
            output = gr.Textbox(label="Output", lines=10)

            def simple_generate(api, model, prompt):
                try:
                    if api:
                        os.environ["OPENAI_API_KEY"] = api
                    return f"Would generate code with {model} based on: {prompt}"
                except Exception as e:
                    return f"Error: {str(e)}"

            generate_btn.click(
                fn=simple_generate,
                inputs=[api_key, model_dropdown, prompt],
                outputs=[output]
            )

        return fallback_app


def main():
    """Main entry point for the web IDE."""
    logger.info("Starting GPT Engineer Web IDE...")

    if not GRADIO_AVAILABLE:
        print("Error: Gradio is not installed. Please install it with:")
        print("pip install gradio")
        print("Then run the IDE again.")
        return

    if not WEB_CORE_DEPS_AVAILABLE:
        print("Error: Some core dependencies are missing.")
        print("Please ensure you have the required packages installed.")
        return

    try:
        ide = WebIde()
        app = create_web_ui(ide)

        # Launch the Gradio app with more explicit parameters
        app.launch(
            server_name="127.0.0.1",  # Local only by default for security
            server_port=7860,         # Standard Gradio port
            share=False,              # No public URL by default
            debug=False,              # Disable debug mode in production
            show_error=True,          # Show detailed errors
            favicon_path=None         # Default favicon
        )

    except Exception as e:
        logger.error(f"Error starting the Web IDE: {str(e)}")
        print(f"Error starting the Web IDE: {str(e)}")
        print("If this is related to Gradio, try reinstalling it with:")
        print("pip install --upgrade gradio")


if __name__ == "__main__":
    main()
