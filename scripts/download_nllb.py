from huggingface_hub import snapshot_download

MODEL_NAME = "facebook/nllb-200-distilled-600M"

print("Downloading NLLB model...")

path = snapshot_download(
    repo_id=MODEL_NAME,
)

print("\nDownload completed!")
print(f"Model location:\n{path}")