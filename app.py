"""Streamlit Cloud entrypoint.

The real dashboard lives in webapp/app.py. This wrapper exists because some
Streamlit deployments are configured with app.py as the main module.
"""
from pathlib import Path
import runpy


runpy.run_path(Path(__file__).parent / "webapp" / "app.py", run_name="__main__")
