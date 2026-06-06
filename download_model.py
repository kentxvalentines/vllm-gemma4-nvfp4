import os
import sys
from huggingface_hub import snapshot_download

def download():
    # Model ID passed via environment variable
    repo_id = os.getenv("MODEL_NAME")
    if not repo_id:
        print("!!! ERROR: MODEL_NAME environment variable not set.")
        sys.exit(1)

    # Token handling
    secret_path = "/run/secrets/HF_TOKEN"
    token = None
    
    if os.path.exists(secret_path):
        print(f"--- INFO: Found secret at {secret_path}")
        with open(secret_path, "r") as f:
            token = f.read().strip()
    else:
        token = os.getenv("HF_TOKEN")
        if token:
            print("--- INFO: Using token from HF_TOKEN environment variable.")
        else:
            print("--- WARNING: No token found. Attempting public download.")

    print(f"--- STARTING DOWNLOAD: {repo_id}")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir="/model",
            token=token,
            max_workers=8,
            # Disable hf_transfer for the build step to ensure reliability
            # (It's re-enabled for runtime in the Dockerfile)
        )
        print("--- DOWNLOAD COMPLETE!")
    except Exception as e:
        print(f"\n!!! DOWNLOAD ERROR: {e}\n", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    download()
