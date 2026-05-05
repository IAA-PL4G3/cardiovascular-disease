"""
adapter for the API - loads missing_data, recommendations, and explainability
in a way that works from the project root without touching the original scripts.
"""
import sys
from pathlib import Path

# missing_data.py uses bare imports and relative paths, both anchored to src/data/
_src_data = Path(__file__).resolve().parent
_project_root = _src_data.parents[1]

sys.path.insert(0, str(_src_data))

# patch the CWD-relative model paths by temporarily changing directory
import os
_orig_cwd = os.getcwd()
os.chdir(_src_data)

from missing_data import models, scaler, process_and_predict
from recommendations import generate_recommendations
from explainability import generate_all_explanations

os.chdir(_orig_cwd)
sys.path.remove(str(_src_data))
