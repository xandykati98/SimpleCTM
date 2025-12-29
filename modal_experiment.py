import modal
from ctm_opengenome import main

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
    ).env({
        "HF_TOKEN": ""
    }).add_local_python_source("ctm_opengenome_conv1d", "ctm_opengenome", "ctm")
)

app = modal.App("ctm-opengenome-conv1d", image=image)

# Define a volume for persistent storage of datasets and models
volume = modal.Volume.from_name("ctm-data", create_if_missing=True)

@app.function(
    gpu="any",
    volumes={"/data": volume},
    timeout=3600,
)
def run_training():
    main(data_path="/data")
    volume.commit()

@app.local_entrypoint()
def local_main():
    run_training.remote()

