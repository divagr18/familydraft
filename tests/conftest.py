"""Shared pytest configuration.

Sets a short Hugging Face Hub download timeout BEFORE any test module imports
transformers/huggingface_hub, so the one network-touching test (bogus repo
load) cannot hang on a blackholed network. Offline runs fail fast too.
"""

import os

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "5")
