"""Cross-platform cache directory for OpenGenome / HuggingFace datasets."""
import os


def default_opengenome_data_path() -> str:
    """
    Root directory passed as load_dataset(..., cache_dir=...) and for filtered
    on-disk caches. Uses the same env vars as Hugging Face when unset.
    """
    explicit = os.environ.get("OPENGENOME_DATA_PATH")
    if explicit is not None and explicit != "":
        return explicit
    hf_datasets = os.environ.get("HF_DATASETS_CACHE")
    if hf_datasets is not None and hf_datasets != "":
        return hf_datasets
    hf_home = os.environ.get("HF_HOME")
    if hf_home is not None and hf_home != "":
        return os.path.join(hf_home, "datasets")
    return os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")
