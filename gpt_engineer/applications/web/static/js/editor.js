/**
 * GPT Engineer Web IDE JavaScript
 * Enhances the web IDE functionality with client-side features
 */

// Wait for the DOM to be fully loaded
document.addEventListener("DOMContentLoaded", function () {
  // Initialize CodeMirror instances if the library is available
  initCodeEditors();

  // Setup file tree interactions
  setupFileTree();

  // Setup chat interface
  setupChatInterface();

  // Setup keyboard shortcuts
  setupKeyboardShortcuts();

  // Setup terminal output handling
  setupTerminal();
});

/**
 * Initialize code editors using CodeMirror if available
 */
function initCodeEditors() {
  if (typeof CodeMirror !== "undefined") {
    // Find all code editor textareas
    const editorElements = document.querySelectorAll(".code-editor-textarea");

    editorElements.forEach((element) => {
      const editor = CodeMirror.fromTextArea(element, {
        lineNumbers: true,
        mode: element.getAttribute("data-language") || "javascript",
        theme: "default",
        indentUnit: 4,
        smartIndent: true,
        lineWrapping: false,
        matchBrackets: true,
        autoCloseBrackets: true,
        styleActiveLine: true,
        extraKeys: {
          "Ctrl-Space": "autocomplete",
          Tab: function (cm) {
            if (cm.somethingSelected()) {
              cm.indentSelection("add");
            } else {
              cm.replaceSelection("    ", "end");
            }
          },
        },
      });

      // Store editor instance in data attribute for later access
      element.editor = editor;

      // Trigger resize to ensure proper rendering
      setTimeout(() => {
        editor.refresh();
      }, 10);
    });
  } else {
    console.warn(
      "CodeMirror library not found. Falling back to regular textareas.",
    );
  }
}

/**
 * Set up file tree interactions
 */
function setupFileTree() {
  const fileItems = document.querySelectorAll(".file-item");
  const fileContent = document.querySelector(".code-editor-content");
  const fileNameDisplay = document.querySelector(".code-editor-filename");

  fileItems.forEach((item) => {
    item.addEventListener("click", function () {
      // Remove selected class from all items
      fileItems.forEach((fi) => fi.classList.remove("selected"));

      // Add selected class to clicked item
      this.classList.add("selected");

      // Update file name display if it exists
      if (fileNameDisplay) {
        fileNameDisplay.textContent =
          this.querySelector(".file-item-name").textContent;
      }

      // Load file content (in a real app, this would call an API)
      const filePath = this.getAttribute("data-path");
      if (filePath) {
        fetchFileContent(filePath);
      }
    });
  });

  // Handle new file button if it exists
  const newFileButton = document.querySelector(".new-file-btn");
  if (newFileButton) {
    newFileButton.addEventListener("click", function () {
      const fileName = prompt("Enter new file name:");
      if (fileName) {
        createNewFile(fileName);
      }
    });
  }
}

/**
 * Fetch file content from the server
 * @param {string} filePath - Path to the file
 */
function fetchFileContent(filePath) {
  // In a real implementation, this would make an API call
  // For demonstration, we're using a placeholder

  const editorContent = document.querySelector(".code-editor-content");
  const loadingIndicator = document.createElement("div");
  loadingIndicator.className = "loading-indicator";
  loadingIndicator.textContent = "Loading...";

  editorContent.innerHTML = "";
  editorContent.appendChild(loadingIndicator);

  // Simulate API call
  setTimeout(() => {
    // Get the editor instance if using CodeMirror
    const editorTextarea = document.querySelector(".code-editor-textarea");
    if (editorTextarea && editorTextarea.editor) {
      editorTextarea.editor.setValue(
        `// Content of ${filePath}\n\n// This is a placeholder for demonstration purposes\n// In a real implementation, this would show the actual file content`,
      );
    } else {
      editorContent.innerHTML = `<pre>// Content of ${filePath}\n\n// This is a placeholder for demonstration purposes\n// In a real implementation, this would show the actual file content</pre>`;
    }
  }, 500);
}

/**
 * Create a new file in the project
 * @param {string} fileName - Name of the new file
 */
function createNewFile(fileName) {
  // In a real implementation, this would make an API call
  // For demonstration, we're adding a placeholder item

  const fileTree = document.querySelector(".file-tree-content");
  if (!fileTree) return;

  const newItem = document.createElement("div");
  newItem.className = "file-item";
  newItem.setAttribute("data-path", fileName);

  newItem.innerHTML = `
    <span class="file-item-icon">📄</span>
    <span class="file-item-name">${fileName}</span>
  `;

  fileTree.appendChild(newItem);

  // Setup click handler for the new item
  newItem.addEventListener("click", function () {
    const fileItems = document.querySelectorAll(".file-item");
    fileItems.forEach((fi) => fi.classList.remove("selected"));
    this.classList.add("selected");

    const fileNameDisplay = document.querySelector(".code-editor-filename");
    if (fileNameDisplay) {
      fileNameDisplay.textContent = fileName;
    }

    // Load empty file content
    const editorTextarea = document.querySelector(".code-editor-textarea");
    if (editorTextarea && editorTextarea.editor) {
      editorTextarea.editor.setValue("");
    } else {
      const editorContent = document.querySelector(".code-editor-content");
      if (editorContent) {
        editorContent.innerHTML = "<pre></pre>";
      }
    }
  });
}

/**
 * Set up chat interface
 */
function setupChatInterface() {
  const chatForm = document.querySelector(".chat-form");
  const chatInput = document.querySelector(".chat-input");
  const chatMessages = document.querySelector(".chat-messages");

  if (!chatForm || !chatInput || !chatMessages) return;

  chatForm.addEventListener("submit", function (e) {
    e.preventDefault();

    const message = chatInput.value.trim();
    if (!message) return;

    // Add user message to chat
    addChatMessage(message, "user");

    // Clear input
    chatInput.value = "";

    // In a real implementation, send message to server and get AI response
    // For demonstration, we're simulating a response
    simulateAiResponse(message);
  });
}

/**
 * Add a message to the chat
 * @param {string} message - Message text
 * @param {string} sender - 'user' or 'ai'
 */
function addChatMessage(message, sender) {
  const chatMessages = document.querySelector(".chat-messages");
  if (!chatMessages) return;

  const messageElement = document.createElement("div");
  messageElement.className = `message ${sender}`;
  messageElement.textContent = message;

  chatMessages.appendChild(messageElement);

  // Scroll to bottom
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

/**
 * Simulate AI response (in a real app, this would call an API)
 * @param {string} userMessage - User's message
 */
function simulateAiResponse(userMessage) {
  // Create typing indicator
  const chatMessages = document.querySelector(".chat-messages");
  const typingIndicator = document.createElement("div");
  typingIndicator.className = "typing-indicator";
  typingIndicator.textContent = "AI is typing...";
  chatMessages.appendChild(typingIndicator);

  // Simulate response delay
  setTimeout(() => {
    // Remove typing indicator
    chatMessages.removeChild(typingIndicator);

    // Add AI response
    let response = "I'm sorry, I couldn't process that request.";

    if (
      userMessage.toLowerCase().includes("hello") ||
      userMessage.toLowerCase().includes("hi")
    ) {
      response = "Hello! How can I help you with your project today?";
    } else if (userMessage.toLowerCase().includes("help")) {
      response =
        "I can help you generate code, explain concepts, or improve your existing code. What would you like assistance with?";
    } else if (
      userMessage.toLowerCase().includes("code") ||
      userMessage.toLowerCase().includes("generate")
    ) {
      response =
        "To generate code, please provide a detailed description of what you want to build. The more specific you are, the better results you'll get.";
    } else if (
      userMessage.toLowerCase().includes("error") ||
      userMessage.toLowerCase().includes("bug")
    ) {
      response =
        "To help debug your code, please share the error message and the relevant code snippet. I'll do my best to identify the issue.";
    } else {
      response =
        "I understand you're asking about: \"" +
        userMessage +
        '". Could you provide more details so I can assist you better?';
    }

    addChatMessage(response, "ai");
  }, 1000);
}

/**
 * Set up keyboard shortcuts
 */
function setupKeyboardShortcuts() {
  document.addEventListener("keydown", function (e) {
    // Ctrl+S or Cmd+S to save
    if ((e.ctrlKey || e.metaKey) && e.key === "s") {
      e.preventDefault();
      saveCurrentFile();
    }

    // Ctrl+R or Cmd+R to run
    if ((e.ctrlKey || e.metaKey) && e.key === "r") {
      e.preventDefault();
      runCode();
    }
  });
}

/**
 * Save the current file
 */
function saveCurrentFile() {
  const saveButton = document.querySelector(".save-btn");
  if (saveButton) {
    saveButton.click();
  } else {
    // Flash a save notification
    const notification = document.createElement("div");
    notification.className = "save-notification";
    notification.textContent = "File saved";
    document.body.appendChild(notification);

    setTimeout(() => {
      document.body.removeChild(notification);
    }, 2000);
  }
}

/**
 * Run the code
 */
function runCode() {
  const runButton = document.querySelector(".run-btn");
  if (runButton) {
    runButton.click();
  }
}

/**
 * Set up terminal output handling
 */
function setupTerminal() {
  const terminal = document.querySelector(".terminal");
  if (!terminal) return;

  // Method to add output to terminal
  window.addTerminalOutput = function (text, type = "info") {
    const line = document.createElement("div");
    line.className = `terminal-line ${type}`;
    line.textContent = text;
    terminal.appendChild(line);

    // Scroll to bottom
    terminal.scrollTop = terminal.scrollHeight;
  };

  // Example usage
  window.addTerminalOutput("Terminal initialized", "info");
  window.addTerminalOutput("Ready to execute commands", "success");
}
