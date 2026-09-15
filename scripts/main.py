from pathlib import Path
import torch

from scripts.preprocess.preprocess import create_dataloaders
from scripts.model.model import DummyClassifier
from scripts.model.train import train_and_evaluate
from scripts.evaluate.eval import display_results

def run_pipeline(
    data_dir: Path,
    results_path: Path,
    model_factory,
    batch_size: int = 64
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running pipeline on device: {device}")

    # 1. Dependency Injection: Data Processing
    _, test_loader = create_dataloaders(data_dir=data_dir, batch_size=batch_size)

    # 2. Dependency Injection: Instantiating Model Architecture
    model = model_factory(num_classes=43)

    # 3. Train / Predict & Log Metrics
    train_and_evaluate(model, test_loader, results_path, device=device)

    # 4. Display Evaluation Metrics
    display_results(results_path)

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "GTSRB"
    RESULTS_PATH = PROJECT_ROOT / "results" / "res.csv"

    run_pipeline(
        data_dir=DATA_DIR,
        results_path=RESULTS_PATH,
        model_factory=DummyClassifier,
        batch_size=64
    )