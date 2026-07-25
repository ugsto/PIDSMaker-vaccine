import os
import subprocess
import sys


def run_command(command):
    print(f"Executing: {command}")
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in process.stdout:
        print(line, end="")
    process.wait()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        sys.exit(process.returncode)


def main():
    print("=== Google Colab Auto-Setup for PIDSMaker ===")

    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Installing standard PyTorch...")
        run_command("pip install -q torch")
        import torch

    torch_ver = torch.__version__.split("+")[0]
    has_cuda = torch.cuda.is_available()
    device_suffix = "+cu121" if has_cuda else "+cpu"
    print(f"Detected PyTorch: {torch.__version__} (CUDA available: {has_cuda})")

    if "+" in torch.__version__:
        pyg_torch_ver = torch.__version__
    else:
        pyg_torch_ver = f"{torch_ver}{device_suffix}"

    print(f"Using PyG index for torch version: {pyg_torch_ver}")

    print("\nInstalling PyTorch Geometric packages...")
    run_command("pip install -q torch_geometric")

    pyg_wheels = ["pyg_lib", "torch_scatter", "torch_sparse", "torch_cluster", "torch_spline_conv"]
    pyg_cmd = f"pip install -q {' '.join(pyg_wheels)} -f https://data.pyg.org/whl/torch-{pyg_torch_ver}.html"
    try:
        run_command(pyg_cmd)
    except SystemExit:
        print("Failed to install matching PyG wheels. Trying standard installation...")
        run_command("pip install -q pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv")

    print(f"\nDetecting Python version: {sys.version}")
    if sys.version_info >= (3, 10):
        print("Python 3.10+ detected. Installing unpinned packages to use pre-built wheels...")
        deps = [
            "scikit-learn",
            "networkx",
            "xxhash",
            "scipy",
            "numpy",
            "yacs",
            "gensim",
            "duckdb",
            "wandb",
            "psycopg2-binary",
            "igraph",
            "tqdm",
            "pandas",
            "pytz",
            "matplotlib",
            "gdown",
            "psutil",
            "graphviz",
            "chardet",
            "nltk",
            "cairocffi",
            "wget",
        ]
    else:
        print("Python 3.9 or older detected. Installing pinned packages...")
        deps = [
            "scikit-learn==1.2.0",
            "networkx==2.8.7",
            "xxhash==3.2.0",
            "scipy==1.10.1",
            "numpy==1.26.4",
            "yacs==0.1.8",
            "gensim==4.3.1",
            "duckdb",
            "wandb",
            "psycopg2-binary",
            "igraph==0.11.5",
            "tqdm",
            "pandas==2.2.2",
            "pytz==2024.1",
            "matplotlib==3.8.4",
            "gdown==5.2.0",
            "psutil",
            "graphviz==0.20.1",
            "chardet==5.2.0",
            "nltk==3.8.1",
            "cairocffi==1.7.0",
            "wget==3.2",
        ]

    run_command(f"pip install -q {' '.join(deps)}")

    print("\nInstalling PIDSMaker in editable mode...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    run_command(f"pip install -e {current_dir}")

    print("\nDownloading NLTK datasets (punkt, punkt_tab)...")
    try:
        import nltk

        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
    except Exception as e:
        print(f"Warning: Failed to download NLTK datasets: {e}")

    print("\n=== Setup completed successfully! ===")
    print("You can now import pidsmaker and run your experiments.")
    print("Remember to set PIDS_DUCKDB=1 and PIDS_PARQUET_DIR='/path/to/parquet' (or R2 credentials).")


if __name__ == "__main__":
    main()
