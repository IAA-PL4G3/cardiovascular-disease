#!/bin/bash
set -e
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'duartebranco/cardiovascular-disease-models',
    local_dir='/app',
    repo_type='model',
)
snapshot_download(
    'duartebranco/cardiovascular-disease',
    local_dir='/app',
    repo_type='space',
    allow_patterns='output/plots/*',
)
print('Models and plots ready')
"
exec uvicorn api.main:app --host 0.0.0.0 --port 7860
