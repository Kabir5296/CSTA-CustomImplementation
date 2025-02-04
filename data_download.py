from huggingface_hub import snapshot_download

snapshot_download(repo_id="flwrlabs/ucf101", repo_type="dataset", local_dir="DATA/UCF101")