import modal
from ctm_shakespeare import main

# Define the image with all necessary dependencies
image = (
    modal.Image.debian_slim()
    .uv_pip_install(
        "torch",
        "torchvision",
        "datasets",
        "scikit-learn",
        "numpy",
        "matplotlib",
        "torch_geometric",
        "wandb",
        "tokenizers"
    ).env({
        "HF_TOKEN": "",
        "WANDB_API_KEY": "5c0d2d6b1fcad21af4e0cc3894c119285c4ddae5"
    }).add_local_python_source("ctm_shakespeare", "ctm", "save_utils")
)

app = modal.App("ctm", image=image)

# Define a volume for persistent storage of datasets and models
volume = modal.Volume.from_name("ctm-data", create_if_missing=True)

@app.function(
    gpu="any",
    volumes={"/data": volume},
    timeout=3600 * 8,
)
def run_training(checkpoint_dir: str = "/data/checkpoints"):
    main(data_path="/data", checkpoint_dir=checkpoint_dir)
    volume.commit()

@app.local_entrypoint()
def local_main():
    run_training.remote()

