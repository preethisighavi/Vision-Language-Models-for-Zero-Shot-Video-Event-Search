"""
export_results.py
─────────────────
Task 9: Sync all final outputs from /data/outputs to the Google Drive
export directory (/data/export) using rsync for efficiency and FUSE safety.
"""

import logging
import os
import subprocess
import yaml

log = logging.getLogger(__name__)

def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def run(config_path: str, **kwargs) -> str:
    cfg = _load_config(config_path)
    output_dir = cfg["paths"]["output_dir"]
    
    # This path must be mounted in docker-compose as a Google Drive folder
    export_dir = "/data/export"
    
    if not os.path.exists(export_dir):
        log.warning("Export directory %s not found. Skipping Drive auto-sync.", export_dir)
        return "Skipped"

    log.info("Exporting results to Google Drive: %s", export_dir)
    
    # Use rsync to avoid OSError 35 deadlocks and handle partial transfers
    # -a: archive mode, -v: verbose, -z: compress (optional), -u: update
    try:
        # Create target subfolder to avoid clutter
        target_path = os.path.join(export_dir, "latest_processed")
        os.makedirs(target_path, exist_ok=True)
        
        cmd = [
            "rsync", "-av", "--ignore-errors",
            output_dir + "/",  # Trailing slash to copy contents
            target_path + "/"
        ]
        
        log.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            log.info("Sync complete!")
        else:
            log.error("Sync failed with code %d: %s", result.returncode, result.stderr)
            
    except Exception as e:
        log.error("Failed to export results: %s", e)
        return "Failed"

    return "Success"
