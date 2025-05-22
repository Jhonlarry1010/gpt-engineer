"""
FastAPI server for the Modern GPT Engineer IDE.

This module provides a FastAPI-based server that serves both the API endpoints
and the frontend static files for the Modern GPT Engineer IDE.
"""

import os
import sys
import logging
import uvicorn
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from fastapi import FastAPI, HTTPException, Request, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("gpt_engineer.modern_ide")

# Import GPT Engineer core components
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
    from gpt_engineer.applications.web.ide_agent import WebIdeAgent

    CORE_DEPS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Error importing core dependencies: {e}")
    CORE_DEPS_AVAILABLE = False

# Get the current directory
CURRENT_DIR = Path(__file__).parent

# Define path to frontend static files
FRONTEND_DIR = CURRENT_DIR / "frontend" / "build"

# Check if frontend files exist
FRONTEND_AVAILABLE = FRONTEND_DIR.exists() and (FRONTEND_DIR / "index.html").exists()

# Define Pydantic models for API requests and responses
class AISetupRequest(BaseModel):
    provider: str = Field("openai", description="AI provider (openai or gemini)")
    model: str = Field("gpt-4o", description="AI model name")
    temperature: float = Field(0.1, description="Temperature setting for AI responses", ge=0, le=1)
    api_key: Optional[str] = Field(None, description="API key for the AI provider")

class ProjectRequest(BaseModel):
    name: str = Field(..., description="Project name")

class PromptRequest(BaseModel):
    prompt: str = Field(..., description="Project specification or improvement prompt")

class FileRequest(BaseModel):
    filename: str = Field(..., description="Filename")
    content: str = Field(..., description="File content")

class StatusResponse(BaseModel):
    status: str = Field(..., description="Status message")
    details: Optional[str] = Field(None, description="Additional details")

class AISetupResponse(StatusResponse):
    provider: str = Field(..., description="AI provider used")
    model: str = Field(..., description="AI model used")

class ProjectResponse(StatusResponse):
    project_name: Optional[str] = Field(None, description="Project name")

class CodeResponse(StatusResponse):
    files: List[str] = Field(default_factory=list, description="Generated or improved files")

class FileResponse(StatusResponse):
    filename: str = Field(..., description="Filename")
    content: str = Field(..., description="File content")

class FilesListResponse(BaseModel):
    files: List[str] = Field(..., description="List of files")

class TaskUpdateModel(BaseModel):
    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Current status")
    message: str = Field(..., description="Status message")
    progress: float = Field(0, description="Progress percentage", ge=0, le=100)
    result: Optional[Dict[str, Any]] = Field(None, description="Task result if complete")

# Background tasks dictionary to track long-running operations
background_tasks = {}

# Create the FastAPI app
app = FastAPI(
    title="GPT Engineer Modern IDE",
    description="A modern web IDE for GPT Engineer",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a global IDE state
class ModernIdeState:
    def __init__(self):
        self.projects_dir = Path("projects")
        self.projects_dir.mkdir(exist_ok=True)
        self.current_project = None
        self.ai = None
        self.agent = None
        self.files_dict = None
        self.execution_env = DiskExecutionEnv() if CORE_DEPS_AVAILABLE else None
        self.logger = logger

    def list_projects(self) -> List[str]:
        """List all available projects."""
        try:
            return [dir.name for dir in self.projects_dir.iterdir() if dir.is_dir()]
        except Exception as e:
            self.logger.error(f"Error listing projects: {e}")
            return []

    def init_ai(self, provider: str, model: str, temperature: float, api_key: Optional[str] = None) -> Dict[str, Any]:
        """Initialize the AI with given parameters."""
        if not CORE_DEPS_AVAILABLE:
            return {"status": "error", "details": "Core dependencies not available"}

        try:
            if provider == "openai":
                if api_key:
                    os.environ["OPENAI_API_KEY"] = api_key

                self.ai = AI(model_name=model, temperature=temperature)
                return {
                    "status": "success",
                    "details": f"OpenAI initialized with model {model}",
                    "provider": "openai",
                    "model": model
                }
            elif provider == "gemini":
                if api_key:
                    os.environ["GEMINI_API_KEY"] = api_key

                try:
                    # Import here to avoid dependency issues
                    from gpt_engineer.applications.web.gemini_ai import GeminiAI
                    self.ai = GeminiAI(model_name=model, temperature=temperature, api_key=api_key)
                    return {
                        "status": "success",
                        "details": f"Gemini initialized with model {model}",
                        "provider": "gemini",
                        "model": model
                    }
                except ImportError:
                    return {
                        "status": "error",
                        "details": "Gemini support not available. Please install the google-generativeai package"
                    }
            else:
                return {"status": "error", "details": f"Unknown provider {provider}"}
        except Exception as e:
            self.logger.error(f"Error initializing AI: {e}")
            return {"status": "error", "details": str(e)}

    def create_project(self, project_name: str) -> Dict[str, Any]:
        """Create a new project with the given name."""
        if not project_name or not project_name.strip():
            return {"status": "error", "details": "Project name cannot be empty"}

        try:
            project_path = self.projects_dir / project_name
            if project_path.exists():
                return {"status": "error", "details": f"Project '{project_name}' already exists"}

            project_path.mkdir(parents=True)
            self.current_project = project_path
            return {
                "status": "success",
                "details": f"Project '{project_name}' created successfully",
                "project_name": project_name
            }
        except Exception as e:
            self.logger.error(f"Error creating project: {e}")
            return {"status": "error", "details": str(e)}

    def select_project(self, project_name: str) -> Dict[str, Any]:
        """Select an existing project to work with."""
        if not project_name:
            return {"status": "error", "details": "No project selected"}

        try:
            project_path = self.projects_dir / project_name
            if not project_path.exists() or not project_path.is_dir():
                return {"status": "error", "details": f"Project '{project_name}' does not exist"}

            self.current_project = project_path
            return {
                "status": "success",
                "details": f"Project '{project_name}' selected",
                "project_name": project_name
            }
        except Exception as e:
            self.logger.error(f"Error selecting project: {e}")
            return {"status": "error", "details": str(e)}

    def init_agent(self) -> Dict[str, Any]:
        """Initialize the agent for the current project."""
        if not CORE_DEPS_AVAILABLE:
            return {"status": "error", "details": "Core dependencies not available"}

        if not self.current_project:
            return {"status": "error", "details": "No project selected"}

        if not self.ai:
            return {"status": "error", "details": "AI not initialized. Please set up the AI first."}

        try:
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

            return {
                "status": "success",
                "details": f"Agent initialized for project '{self.current_project.name}'"
            }
        except Exception as e:
            self.logger.error(f"Error initializing agent: {e}")
            return {"status": "error", "details": str(e)}

    def generate_code(self, prompt_text: str) -> Dict[str, Any]:
        """Generate code from the given prompt."""
        if not CORE_DEPS_AVAILABLE:
            return {"status": "error", "details": "Core dependencies not available", "files": []}

        if not self.current_project:
            return {"status": "error", "details": "No project selected", "files": []}

        if not self.agent:
            return {"status": "error", "details": "Agent not initialized", "files": []}

        if not prompt_text or not prompt_text.strip():
            return {"status": "error", "details": "Prompt cannot be empty", "files": []}

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
            return {
                "status": "success",
                "details": "Code generated successfully",
                "files": list(self.files_dict.keys())
            }
        except Exception as e:
            self.logger.error(f"Error generating code: {e}")
            return {"status": "error", "details": str(e), "files": []}

    def improve_code(self, prompt_text: str) -> Dict[str, Any]:
        """Improve existing code with the given prompt."""
        if not CORE_DEPS_AVAILABLE:
            return {"status": "error", "details": "Core dependencies not available", "files": []}

        if not self.current_project:
            return {"status": "error", "details": "No project selected", "files": []}

        if not self.agent:
            return {"status": "error", "details": "Agent not initialized", "files": []}

        if not prompt_text or not prompt_text.strip():
            return {"status": "error", "details": "Improvement prompt cannot be empty", "files": []}

        try:
            # Load existing files
            file_store = FileStore(self.current_project)
            files_dict_before = file_store.get_all_files()

            if not files_dict_before:
                return {"status": "error", "details": "No existing files found to improve", "files": []}

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
            return {
                "status": "success",
                "details": "Code improved successfully",
                "files": list(self.files_dict.keys())
            }
        except Exception as e:
            self.logger.error(f"Error improving code: {e}")
            return {"status": "error", "details": str(e), "files": []}

    def get_file_content(self, filename: str) -> Dict[str, Any]:
        """Get the content of a file in the current project."""
        if not self.current_project:
            return {"status": "error", "details": "No project selected", "filename": filename, "content": ""}

        try:
            file_path = self.current_project / filename
            if not file_path.exists():
                return {"status": "error", "details": f"File '{filename}' does not exist", "filename": filename, "content": ""}

            content = file_path.read_text()
            return {
                "status": "success",
                "details": f"File '{filename}' read successfully",
                "filename": filename,
                "content": content
            }
        except Exception as e:
            self.logger.error(f"Error reading file: {e}")
            return {"status": "error", "details": str(e), "filename": filename, "content": ""}

    def save_file_content(self, filename: str, content: str) -> Dict[str, Any]:
        """Save content to a file in the current project."""
        if not self.current_project:
            return {"status": "error", "details": "No project selected", "filename": filename}

        if not filename or not filename.strip():
            return {"status": "error", "details": "Filename cannot be empty", "filename": filename}

        try:
            file_path = self.current_project / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)
            return {
                "status": "success",
                "details": f"File '{filename}' saved successfully",
                "filename": filename
            }
        except Exception as e:
            self.logger.error(f"Error saving file: {e}")
            return {"status": "error", "details": str(e), "filename": filename}

    def list_files(self) -> Dict[str, Any]:
        """List all files in the current project."""
        if not self.current_project:
            return {"files": []}

        try:
            files = []
            for file_path in self.current_project.glob("**/*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(self.current_project)
                    files.append(str(relative_path))
            return {"files": files}
        except Exception as e:
            self.logger.error(f"Error listing files: {e}")
            return {"files": []}

    def run_code(self) -> Dict[str, Any]:
        """Run the code in the current project."""
        if not CORE_DEPS_AVAILABLE:
            return {"status": "error", "details": "Core dependencies not available"}

        if not self.current_project:
            return {"status": "error", "details": "No project selected"}

        try:
            # Find entrypoint based on common patterns
            entrypoint_candidates = ["main.py", "app.py", "index.py", "run.py"]
            entrypoint = None

            for candidate in entrypoint_candidates:
                if (self.current_project / candidate).exists():
                    entrypoint = candidate
                    break

            if not entrypoint:
                return {"status": "error", "details": "No entrypoint found (main.py, app.py, index.py, or run.py)"}

            # Execute code using the execution environment
            self.logger.info(f"Running {entrypoint}...")
            result = self.execution_env.execute_command(
                self.current_project, f"python {entrypoint}"
            )

            return {
                "status": "success",
                "details": "Code execution completed",
                "output": result.output
            }
        except Exception as e:
            self.logger.error(f"Error running code: {e}")
            return {"status": "error", "details": str(e)}

# Create the global state instance
ide_state = ModernIdeState()

# Define API endpoints
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}

@app.get("/api/status")
async def get_status():
    """Get the current IDE status."""
    return {
        "core_available": CORE_DEPS_AVAILABLE,
        "frontend_available": FRONTEND_AVAILABLE,
        "ai_initialized": ide_state.ai is not None,
        "agent_initialized": ide_state.agent is not None,
        "current_project": ide_state.current_project.name if ide_state.current_project else None,
    }

@app.post("/api/ai/setup", response_model=AISetupResponse)
async def setup_ai(request: AISetupRequest):
    """Set up the AI with the given parameters."""
    result = ide_state.init_ai(
        provider=request.provider,
        model=request.model,
        temperature=request.temperature,
        api_key=request.api_key
    )
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["details"])
    return result

@app.get("/api/projects", response_model=FilesListResponse)
async def list_projects():
    """List all available projects."""
    projects = ide_state.list_projects()
    return {"files": projects}

@app.post("/api/projects", response_model=ProjectResponse)
async def create_project(request: ProjectRequest):
    """Create a new project."""
    result = ide_state.create_project(request.name)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["details"])
    return result

@app.post("/api/projects/select", response_model=ProjectResponse)
async def select_project(request: ProjectRequest):
    """Select an existing project."""
    result = ide_state.select_project(request.name)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["details"])
    return result

@app.post("/api/agent/init", response_model=StatusResponse)
async def init_agent():
    """Initialize the agent for the current project."""
    result = ide_state.init_agent()
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["details"])
    return result

@app.post("/api/code/generate")
async def generate_code(request: PromptRequest, background_tasks: BackgroundTasks):
    """Generate code from the given prompt."""
    # Check prerequisites
    if not ide_state.current_project:
        raise HTTPException(status_code=400, detail="No project selected")
    if not ide_state.ai:
        raise HTTPException(status_code=400, detail="AI not initialized")
    if not ide_state.agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")

    # Start code generation in background
    task_id = f"generate_{id(request)}"

    async def run_generation():
        try:
            # Update task status
            background_tasks[task_id] = {
                "task_id": task_id,
                "status": "running",
                "message": "Generating code...",
                "progress": 10
            }

            # Generate code
            result = ide_state.generate_code(request.prompt)

            # Update task status based on result
            if result["status"] == "success":
                background_tasks[task_id] = {
                    "task_id": task_id,
                    "status": "completed",
                    "message": "Code generation completed",
                    "progress": 100,
                    "result": result
                }
            else:
                background_tasks[task_id] = {
                    "task_id": task_id,
                    "status": "failed",
                    "message": result["details"],
                    "progress": 100,
                    "result": result
                }
        except Exception as e:
            # Handle any exceptions
            background_tasks[task_id] = {
                "task_id": task_id,
                "status": "failed",
                "message": str(e),
                "progress": 100,
                "result": {"status": "error", "details": str(e), "files": []}
            }

    # Start the background task
    background_tasks.add_task(run_generation)

    # Initialize task status
    background_tasks[task_id] = {
        "task_id": task_id,
        "status": "started",
        "message": "Starting code generation...",
        "progress": 0
    }

    # Return the task ID for polling
    return {"status": "accepted", "details": "Code generation started", "task_id": task_id}

@app.post("/api/code/improve")
async def improve_code(request: PromptRequest, background_tasks: BackgroundTasks):
    """Improve code with the given prompt."""
    # Check prerequisites
    if not ide_state.current_project:
        raise HTTPException(status_code=400, detail="No project selected")
    if not ide_state.ai:
        raise HTTPException(status_code=400, detail="AI not initialized")
    if not ide_state.agent:
        raise HTTPException(status_code=400, detail="Agent not initialized")

    # Start code improvement in background
    task_id = f"improve_{id(request)}"

    async def run_improvement():
        try:
            # Update task status
            background_tasks[task_id] = {
                "task_id": task_id,
                "status": "running",
                "message": "Improving code...",
                "progress": 10
            }

            # Improve code
            result = ide_state.improve_code(request.prompt)

            # Update task status based on result
            if result["status"] == "success":
                background_tasks[task_id] = {
                    "task_id": task_id,
                    "status": "completed",
                    "message": "Code improvement completed",
                    "progress": 100,
                    "result": result
                }
            else:
                background_tasks[task_id] = {
                    "task_id": task_id,
                    "status": "failed",
                    "message": result["details"],
                    "progress": 100,
                    "result": result
                }
        except Exception as e:
            # Handle any exceptions
            background_tasks[task_id] = {
                "task_id": task_id,
                "status": "failed",
                "message": str(e),
                "progress": 100,
                "result": {"status": "error", "details": str(e), "files": []}
            }

    # Start the background task
    background_tasks.add_task(run_improvement)

    # Initialize task status
    background_tasks[task_id] = {
        "task_id": task_id,
        "status": "started",
        "message": "Starting code improvement...",
        "progress": 0
    }

    # Return the task ID for polling
    return {"status": "accepted", "details": "Code improvement started", "task_id": task_id}

@app.get("/api/tasks/{task_id}", response_model=TaskUpdateModel)
async def get_task_status(task_id: str):
    """Get the status of a long-running task."""
    if task_id not in background_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    return background_tasks[task_id]

@app.get("/api/files", response_model=FilesListResponse)
async def list_files():
    """List all files in the current project."""
    if not ide_state.current_project:
        raise HTTPException(status_code=400, detail="No project selected")

    return ide_state.list_files()

@app.get("/api/files/{filename:path}")
async def get_file(filename: str):
    """Get the content of a file."""
    if not ide_state.current_project:
        raise HTTPException(status_code=400, detail="No project selected")

    result = ide_state.get_file_content(filename)
    if result["status"] == "error":
        raise HTTPException(status_code=404, detail=result["details"])

    return result

@app.post("/api/files/{filename:path}")
async def save_file(filename: str, request: FileRequest):
    """Save content to a file."""
    if not ide_state.current_project:
        raise HTTPException(status_code=400, detail="No project selected")

    result = ide_state.save_file_content(filename, request.content)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["details"])

    return result

@app.post("/api/code/run", response_model=StatusResponse)
async def run_code():
    """Run the code in the current project."""
    if not ide_state.current_project:
        raise HTTPException(status_code=400, detail="No project selected")

    result = ide_state.run_code()
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["details"])

    return result

# Mount static files if available
if FRONTEND_AVAILABLE:
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

    @app.get("/{rest_of_path:path}")
    async def serve_frontend(rest_of_path: str):
        """Serve the frontend for any path that doesn't match API routes."""
        index_path = FRONTEND_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        else:
            raise HTTPException(status_code=404, detail="Frontend files not found")

def run_server(host: str = "127.0.0.1", port: int = 8000, dev_mode: bool = False):
    """Run the FastAPI server."""
    logger.info(f"Starting GPT Engineer Modern IDE server on http://{host}:{port}")

    if not CORE_DEPS_AVAILABLE:
        logger.error("Core dependencies not available. Some features may not work.")

    if not FRONTEND_AVAILABLE:
        logger.warning("Frontend files not found. Only API endpoints will be available.")

    # Run the server
    uvicorn.run("gpt_engineer.applications.modern_ide.server:app", host=host, port=port, reload=dev_mode)

if __name__ == "__main__":
    run_server()
