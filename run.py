#!/usr/bin/env python3
"""
NSE Stock Analyzer - Local Runner Script

This script runs the Streamlit app locally on your machine.
Make sure you have installed the required dependencies:
    pip install -r requirements.txt

Usage:
    python run.py
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app locally."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to Home.py
    home_py = os.path.join(script_dir, 'Home.py')

    # Check if Home.py exists
    if not os.path.exists(home_py):
        print(f"Error: {home_py} not found!")
        sys.exit(1)

    # Run streamlit
    try:
        cmd = [
            sys.executable, '-m', 'streamlit', 'run', home_py,
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--server.headless', 'true'
        ]

        print("Starting NSE Stock Analyzer...")
        print("App will be available at: http://localhost:8501")
        print("Press Ctrl+C to stop the server")

        subprocess.run(cmd, cwd=script_dir)

    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error running the app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()