import os
import sys

# Ensure parent directory is in sys.path for Vercel serverless environment
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from app import app

# Export WSGI application for Vercel
app = app
