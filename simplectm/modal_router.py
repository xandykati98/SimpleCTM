import modal
from ctm_shakespeare import main as ctm_shakespeare_main
from transformer_shakespeare import main as transformer_shakespeare_main
from ctm_modelnet import main as ctm_modelnet_main
from ctm_imagenette import main as ctm_imagenette_main
from ctm_opengenome import main as ctm_opengenome_main
from ctm_opengenome_conv1d import main as ctm_opengenome_conv1d_main
from ctm import main as ctm_main
# Define the image with all necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
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
    }).add_local_python_source(
        "ctm_shakespeare", 
        "transformer_shakespeare",
        "ctm_modelnet",
        "ctm_imagenette",
        "ctm_opengenome",
        "ctm_opengenome_conv1d",
        "ctm", 
        "save_utils"
    )
)

app = modal.App("ctm", image=image)

# Define a volume for persistent storage of datasets and models
volume = modal.Volume.from_name("ctm-data", create_if_missing=True)

@app.function(
    gpu="any",
    volumes={"/data": volume},
    timeout=3600 * 8,
)
def run_training_shakespeare(checkpoint_dir: str = "/data/checkpoints"):
    ctm_shakespeare_main(data_path="/data", checkpoint_dir=checkpoint_dir)
    volume.commit()

@app.function(
    gpu="any",
    volumes={"/data": volume},
    timeout=3600 * 8,
)
def run_training_transformer(checkpoint_dir: str = "/data/checkpoints"):
    transformer_shakespeare_main(data_path="/data", checkpoint_dir=checkpoint_dir)
    volume.commit()

@app.function(
    gpu="any",
    volumes={"/data": volume},
    timeout=3600 * 8,
)
def run_training_modelnet(checkpoint_dir: str = "/data/checkpoints", resume_from: str | None = None):
    ctm_modelnet_main(data_path="/data", checkpoint_dir=checkpoint_dir, resume_from=resume_from)
    volume.commit()

@app.function(
    gpu="any",
    volumes={"/data": volume},
    timeout=3600 * 8,
)
def run_training_imagenette(checkpoint_dir: str = "/data/checkpoints"):
    ctm_imagenette_main(data_path="/data", checkpoint_dir=checkpoint_dir)
    volume.commit()

@app.function(
    gpu="any",
    volumes={"/data": volume},
    timeout=3600 * 8,
)
def run_training_opengenome():
    ctm_opengenome_main(data_path="/data")
    volume.commit()

@app.function(
    gpu="any",
    volumes={"/data": volume},
    timeout=3600 * 8,
)
def run_training_opengenome_conv1d():
    ctm_opengenome_conv1d_main(data_path="/data")
    volume.commit()

@app.function(
    gpu="any",
    volumes={"/data": volume},
    timeout=3600 * 8,
)
def run_training_ctm():
    ctm_main()
    volume.commit()

@app.local_entrypoint()
def local_main():
    """Local entrypoint to run training functions remotely."""
    import sys
    if len(sys.argv) < 2:
        print("Usage: modal run simplectm/modal_router.py <function_name>")
        print("Available functions:")
        print("  - shakespeare")
        print("  - transformer")
        print("  - modelnet")
        print("  - imagenette")
        print("  - opengenome")
        print("  - opengenome_conv1d")
        print("  - ctm")
        return
    
    function_name = sys.argv[1]
    function_map = {
        "shakespeare": run_training_shakespeare,
        "transformer": run_training_transformer,
        "modelnet": run_training_modelnet,
        "imagenette": run_training_imagenette,
        "opengenome": run_training_opengenome,
        "opengenome_conv1d": run_training_opengenome_conv1d,
        "ctm": run_training_ctm,
    }
    
    if function_name not in function_map:
        print(f"Unknown function: {function_name}")
        return
    
    function_map[function_name].remote()