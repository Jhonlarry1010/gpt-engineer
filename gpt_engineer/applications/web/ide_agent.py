"""
Web IDE Agent for GPT Engineer.

This module defines a specialized agent for the web IDE interface,
extending the base agent functionality with web-specific capabilities.
"""

import logging
import traceback
from typing import Callable, Dict, List, Optional, Any

# Core imports - wrap in try/except to provide better error handling
try:
    from gpt_engineer.core.base_agent import BaseAgent
    from gpt_engineer.core.base_execution_env import BaseExecutionEnv
    from gpt_engineer.core.base_memory import BaseMemory
    from gpt_engineer.core.ai import AI
    from gpt_engineer.core.files_dict import FilesDict
    from gpt_engineer.core.preprompts_holder import PrepromptsHolder
    from gpt_engineer.core.prompt import Prompt
    
    AGENT_DEPS_AVAILABLE = True
except ImportError as e:
    logging.error(f"Error importing agent dependencies: {e}")
    AGENT_DEPS_AVAILABLE = False


class WebIdeAgent:
    """Agent for running GPT Engineer processes in a web interface context."""

    def __init__(
        self,
        memory: Any,
        execution_env: Any,
        ai: Any,
        preprompts_holder: Any,
        code_gen_fn: Callable,
        improve_fn: Callable,
        process_code_fn: Callable,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the Web IDE Agent.

        Parameters
        ----------
        memory : BaseMemory
            The memory interface for storing and retrieving data.
        execution_env : BaseExecutionEnv
            The execution environment for running code.
        ai : AI
            The AI interface.
        preprompts_holder : PrepromptsHolder
            The holder for preprompts.
        code_gen_fn : Callable
            The function to generate code.
        improve_fn : Callable
            The function to improve code.
        process_code_fn : Callable
            The function to process code.
        status_callback : Optional[Callable[[str], None]]
            A callback function to report status updates to the web UI.
        """
        if not AGENT_DEPS_AVAILABLE:
            raise ImportError("Agent dependencies not available")
            
        # Create a BaseAgent instance if available
        if isinstance(memory, BaseMemory) and isinstance(execution_env, BaseExecutionEnv):
            self.agent = BaseAgent(memory, execution_env)
        else:
            self.agent = None
            
        self.memory = memory
        self.execution_env = execution_env
        self.ai = ai
        self.preprompts_holder = preprompts_holder
        self.code_gen_fn = code_gen_fn
        self.improve_fn = improve_fn
        self.process_code_fn = process_code_fn
        self.status_callback = status_callback
        self.logger = logging.getLogger(__name__)

    @classmethod
    def with_default_config(
        cls,
        memory: Any,
        execution_env: Any,
        ai: Any,
        code_gen_fn: Callable,
        improve_fn: Callable,
        process_code_fn: Callable,
        preprompts_holder: Any,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> "WebIdeAgent":
        """
        Create a new WebIdeAgent with the default configuration.

        Parameters
        ----------
        memory : BaseMemory
            The memory interface for storing and retrieving data.
        execution_env : BaseExecutionEnv
            The execution environment for running code.
        ai : AI
            The AI interface.
        code_gen_fn : Callable
            The function to generate code.
        improve_fn : Callable
            The function to improve code.
        process_code_fn : Callable
            The function to process code.
        preprompts_holder : PrepromptsHolder
            The holder for preprompts.
        status_callback : Optional[Callable[[str], None]]
            A callback function to report status updates to the web UI.

        Returns
        -------
        WebIdeAgent
            The configured WebIdeAgent.
        """
        try:
            return cls(
                memory,
                execution_env,
                ai,
                preprompts_holder,
                code_gen_fn,
                improve_fn,
                process_code_fn,
                status_callback,
            )
        except Exception as e:
            logging.error(f"Error creating WebIdeAgent: {e}")
            logging.error(traceback.format_exc())
            raise

    def update_status(self, message: str):
        """
        Update the status via the callback if available.

        Parameters
        ----------
        message : str
            The status message to report.
        """
        if self.status_callback:
            try:
                self.status_callback(message)
            except Exception as e:
                self.logger.error(f"Error in status callback: {e}")
                
        self.logger.info(message)

    def init(self, prompt: Any) -> Dict[str, str]:
        """
        Initialize a new project with the given prompt.

        Parameters
        ----------
        prompt : Prompt
            The prompt to use for generating code.

        Returns
        -------
        FilesDict
            The generated files.
        """
        self.update_status("Starting code generation...")
        
        try:
            files_dict = self.code_gen_fn(
                self.ai, prompt, self.memory, self.preprompts_holder
            )
            
            self.update_status("Code generation complete")
            self.update_status("Processing code...")
            
            try:
                self.process_code_fn(files_dict, self.execution_env)
                self.update_status("Code processing complete")
            except Exception as e:
                self.update_status(f"Error processing code: {str(e)}")
                self.logger.error(f"Error processing code: {e}")
                self.logger.error(traceback.format_exc())
            
            return files_dict
        except Exception as e:
            self.update_status(f"Error generating code: {str(e)}")
            self.logger.error(f"Error generating code: {e}")
            self.logger.error(traceback.format_exc())
            return {"error.txt": f"Error generating code: {str(e)}\n\n{traceback.format_exc()}"}

    def improve(self, prompt: Any, files_dict: Dict[str, str]) -> Dict[str, str]:
        """
        Improve an existing project with the given prompt.

        Parameters
        ----------
        prompt : Prompt
            The improvement prompt.
        files_dict : FilesDict
            The existing files to improve.

        Returns
        -------
        FilesDict
            The improved files.
        """
        self.update_status("Starting code improvement...")
        
        try:
            improved_files_dict = self.improve_fn(
                self.ai, prompt, self.memory, files_dict, self.preprompts_holder
            )
            
            self.update_status("Code improvement complete")
            self.update_status("Processing improved code...")
            
            try:
                self.process_code_fn(improved_files_dict, self.execution_env)
                self.update_status("Improved code processing complete")
            except Exception as e:
                self.update_status(f"Error processing improved code: {str(e)}")
                self.logger.error(f"Error processing improved code: {e}")
                self.logger.error(traceback.format_exc())
            
            return improved_files_dict
        except Exception as e:
            self.update_status(f"Error improving code: {str(e)}")
            self.logger.error(f"Error improving code: {e}")
            self.logger.error(traceback.format_exc())
            # Return the original files with an error note
            files_dict["error.txt"] = f"Error improving code: {str(e)}\n\n{traceback.format_exc()}"
            return files_dict

    def chat(self, message: str) -> str:
        """
        Chat with the AI about the current project.

        Parameters
        ----------
        message : str
            The message to send to the AI.

        Returns
        -------
        str
            The AI's response.
        """
        self.update_status("Sending message to AI...")
        
        try:
            # Use a simple chat context for now
            response = self.ai.chat(
                system_prompt="You are an AI assistant helping with code. Answer questions about programming, debugging, and software development.",
                message=message,
            )
            
            self.update_status("Received AI response")
            return response
        except Exception as e:
            error_msg = f"Error communicating with AI: {str(e)}"
            self.update_status(error_msg)
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            return f"I'm sorry, I encountered an error: {error_msg}. Please check your API key and connection."
