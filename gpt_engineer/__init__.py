# Make web applications accessible
try:
    from gpt_engineer.applications import web
except ImportError:
    # This handles cases where the web module dependencies aren't installed
    pass

try:
    from gpt_engineer.applications import modern_ide
except ImportError:
    # This handles cases where the modern_ide module dependencies aren't installed
    pass
