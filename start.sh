#!/bin/bash
set -e
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'duartebranco/cardiovascular-disease-models',
    local_dir='/app/models',
    repo_type='model',
    local_dir_use_symlinks=False,
)
print('Models ready')
"
exec uvicorn api.main:app --host 0.0.0.0 --port 7860
