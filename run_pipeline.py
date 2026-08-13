from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"


PIPELINE_STAGES = [
    ("Process raw dataset", "data_processing.py"),
    ("Clean data", "clean_data.py"),
    ("Split dataset", "split_data.py"),
    ("Fine-tune model", "train_data.py"),
    ("Evaluate model", "evaluate_model.py"),
    ("Generate sample prediction", "sample_prediction.py"),
]


def run_stage(description, script_name):
    script_path = SRC_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Required script not found: {script_path}")

    print("\n" + "=" * 60)
    print(description)
    print(f"Running: {script_path}")
    print("=" * 60)

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main():
    print("Starting the complete ML pipeline...")

    for description, script_name in PIPELINE_STAGES:
        run_stage(description, script_name)

    print("\n" + "=" * 60)
    print("Complete ML pipeline finished successfully.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as error:
        print(f"\nPipeline configuration error: {error}")
        sys.exit(1)
    except subprocess.CalledProcessError as error:
        failed_script = error.cmd[-1]
        print(
            f"\nPipeline stopped because {failed_script} "
            f"failed with exit code {error.returncode}."
        )
        sys.exit(error.returncode)