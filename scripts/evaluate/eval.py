from pathlib import Path
import pandas as pd

def display_results(results_path: Path):
    if not results_path.exists():
        print(f"Results file not found at {results_path}")
        return

    df = pd.read_csv(results_path)
    print("\n================ EVALUATION RESULTS ================")
    print(df.to_string(index=False))
    print("====================================================\n")