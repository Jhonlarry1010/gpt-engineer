import logging


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Make web application accessible
try:
    from gpt_engineer.applications import web
except ImportError:
    # This handles cases where the web module dependencies aren't installed
    pass
