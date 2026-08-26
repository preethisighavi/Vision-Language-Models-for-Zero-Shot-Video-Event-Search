import os
import time
import json
import shutil
import subprocess
from datetime import datetime

# --- Configuration ---
BATCH_SIZE = 10
BASE_DIR = "/Users/spartan/Library/CloudStorage/GoogleDrive-vummidichettyabhinav@gmail.com/Shared drives/DATA 298A/DATA"

MSRVTT_SRC = os.path.join(BASE_DIR, "MSRVTT", "raw_videos")
MSVD_SRC = os.path.join(BASE_DIR, "MSVD", "raw_videos")

# Local mounting directories
VLM_LOCAL = os.path.join(BASE_DIR, "vlm_pipeline", "data_local")
MSRVTT_LOCAL = os.path.join(VLM_LOCAL, "MSRVTT", "raw_videos")
MSVD_LOCAL = os.path.join(VLM_LOCAL, "MSVD", "raw_videos")

PROCESSED_TRACKER = os.path.join(BASE_DIR, "vlm_pipeline", "processed_videos.json")
OUTPUTS_DIR = os.path.join(BASE_DIR, "vlm_pipeline", "processed_outputs")
LATEST_OUTPUT = os.path.join(OUTPUTS_DIR, "latest_processed")
ARCHIVE_DIR = os.path.join(OUTPUTS_DIR, "batch_archives")

# Ensure required directories exist
os.makedirs(MSRVTT_LOCAL, exist_ok=True)
os.makedirs(MSVD_LOCAL, exist_ok=True)
os.makedirs(ARCHIVE_DIR, exist_ok=True)


def get_all_videos():
    msrvtt = {f: os.path.join(MSRVTT_SRC, f) for f in os.listdir(MSRVTT_SRC) if f.endswith((".mp4", ".avi"))} if os.path.exists(MSRVTT_SRC) else {}
    msvd = {f: os.path.join(MSVD_SRC, f) for f in os.listdir(MSVD_SRC) if f.endswith((".mp4", ".avi"))} if os.path.exists(MSVD_SRC) else {}
    return msrvtt, msvd


def empty_dir(directory):
    for f in os.listdir(directory):
        os.remove(os.path.join(directory, f))


def trigger_airflow_dag():
    print("🚀 Triggering Airflow DAG 'vlm_data_pipeline'...")
    # Trigger using docker compose
    cmd = [
        "/usr/local/bin/docker", "compose", "exec", "airflow-webserver", 
        "airflow", "dags", "trigger", "vlm_data_pipeline"
    ]
    cwd = os.path.join(BASE_DIR, "vlm_pipeline")
    subprocess.run(cmd, cwd=cwd, check=True)
    
    # Give Airflow a few seconds to register the run
    time.sleep(10)
    
    # We must find the latest run_id
    run_cmd = ["/usr/local/bin/docker", "compose", "exec", "airflow-webserver", "airflow", "runs", "list", "-d", "vlm_data_pipeline", "-n", "1"]
    res = subprocess.run(run_cmd, cwd=cwd, capture_output=True, text=True)
    output_lines = [l for l in res.stdout.split('\n') if 'manual__' in l]
    
    if not output_lines:
        print("⚠️ Could not locate the run. Waiting 30 seconds to fallback...")
        time.sleep(30)
        return "unknown"
        
    # The run_id is usually the second column in the CLI table
    cols = output_lines[0].split('|')
    run_id = cols[2].strip()
    print(f"📊 Tracking Run ID: {run_id}")
    return run_id


def wait_for_dag(run_id):
    if run_id == "unknown":
        print("⏳ Polling gracefully for 2 minutes as fallback...")
        time.sleep(120)
        return True
        
    cwd = os.path.join(BASE_DIR, "vlm_pipeline")
    cmd = ["/usr/local/bin/docker", "compose", "exec", "airflow-webserver", "airflow", "dags", "state", "vlm_data_pipeline", run_id]
    
    while True:
        res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        state = res.stdout.strip().split('\n')[-1]
        
        if 'success' in state.lower():
            print("✅ DAG completed successfully!")
            return True
        elif 'failed' in state.lower():
            print("❌ DAG failed. Halting orchestrator.")
            return False
            
        print(f"⏳ DAG state: {state}. Waiting 15s...")
        time.sleep(15)


def main():
    if os.path.exists(PROCESSED_TRACKER):
        with open(PROCESSED_TRACKER) as f:
            processed = set(json.load(f))
    else:
        processed = set()

    msrvtt_all, msvd_all = get_all_videos()
    all_videos = {**msrvtt_all, **msvd_all}
    
    # Find unprocessed
    unprocessed = [v for v in all_videos.keys() if v not in processed]
    print(f"Total videos: {len(all_videos)} | Processed: {len(processed)} | Remaining: {len(unprocessed)}")
    
    batch_num = 1
    while unprocessed:
        print(f"\n{'='*50}\n🚀 Starting Batch {batch_num}\n{'='*50}")
        current_batch = unprocessed[:BATCH_SIZE]
        
        print("🧹 Clearing local staging folders...")
        empty_dir(MSRVTT_LOCAL)
        empty_dir(MSVD_LOCAL)
        
        # Flush intermediate outputs locally so we don't carry over old files
        outputs_local = os.path.join(VLM_LOCAL, "outputs")
        if os.path.exists(outputs_local):
            shutil.rmtree(outputs_local)
        os.makedirs(outputs_local)
        
        print(f"📦 Staging {len(current_batch)} videos locally...")
        for vid in current_batch:
            src_path = all_videos[vid]
            if "MSRVTT" in src_path:
                shutil.copy2(src_path, os.path.join(MSRVTT_LOCAL, vid))
            else:
                shutil.copy2(src_path, os.path.join(MSVD_LOCAL, vid))
                
        # Run Airflow
        run_id = trigger_airflow_dag()
        success = wait_for_dag(run_id)
        
        if not success:
            break
            
        # Archive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_archive_path = os.path.join(ARCHIVE_DIR, f"batch_{timestamp}")
        
        if os.path.exists(LATEST_OUTPUT):
            print(f"💾 Archiving outputs to {batch_archive_path}")
            shutil.copytree(LATEST_OUTPUT, batch_archive_path)
        else:
            print("⚠️ No output found in latest_processed to archive!")
            
        # Mark as processed
        processed.update(current_batch)
        with open(PROCESSED_TRACKER, "w") as f:
            json.dump(list(processed), f)
            
        unprocessed = unprocessed[BATCH_SIZE:]
        batch_num += 1
        print(f"🎉 Batch {batch_num-1} complete! Waiting 10s cooldown.")
        time.sleep(10)
        
    print("✅ All data processed!")

if __name__ == "__main__":
    main()
