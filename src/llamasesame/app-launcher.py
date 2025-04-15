#!/usr/bin/env python3
"""
LlamaSesame Web Application Launcher

This script provides a convenient way to start the LlamaSesame web interface.
"""

import logging
import os
import sys
import threading
import time
import webbrowser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("llamasesame.launcher")


def open_browser():
    """Open the web browser after a short delay."""
    time.sleep(2)
    url = "http://localhost:8501"
    logger.info(f"Opening browser at {url}")
    webbrowser.open(url)


def main():
    """Main entry point for the launcher."""
    logger.info("Starting LlamaSesame Voice Cloning Studio Pro")

    # Print welcome message
    print("\n" + "=" * 80)
    print("  LlamaSesame Voice Cloning Studio Pro")
    print("  Version 1.0.0")
    print("=" * 80 + "\n")

    try:
        # Import the app module
        from app import home, initialize, web

        # Initialize the application
        logger.info("Initializing application")
        initialize()

        # Start browser in a separate thread
        thread = threading.Thread(target=open_browser)
        thread.daemon = True
        thread.start()

        # Start the web app
        logger.info("Starting web interface")
        web.start(home)

    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        print(
            "\nError: Failed to start LlamaSesame. Please make sure all dependencies are installed."
        )
        print("Try running: pip install -r requirements.txt\n")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
