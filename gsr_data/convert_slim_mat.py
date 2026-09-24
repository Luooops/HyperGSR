"""Convert SLIM MAT graphs to the existing HyperGSR CSV layout.

Rows retain their original MAT order. Generated IDs are one-based row numbers,
not subject IDs. Cross-modal subject alignment and homologous LH/RH ROI order
must be established from the dataset documentation.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from src.matrix_vectorizer import MatrixVectorizer


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = (ROOT / "gsr_data" / "SLIM-multi-modal dataset-SG-Net-2020"
                 / "SLIM-multi-modal dataset-SG-Net-2020")


def load_vectors(folder, filename, key, n_nodes):
    values = np.asarray(loadmat(folder / filename, variable_names=[key])[key])
    n_edges = n_nodes * (n_nodes - 1) // 2
    if (values.ndim != 2 or values.shape[0] == 0
            or values.shape[1] != n_edges or not np.isrealobj(values)
            or not np.isfinite(values).all()):
        raise ValueError(f"{filename}: expected finite real rows of {n_edges} edges")
    return values.astype(np.float64, copy=False)


def average_hemispheres(left, right, input_order):
    """Average homologous edges and emit column-wise upper-triangle vectors."""
    if left.shape != right.shape:
        raise ValueError("LH and RH must have identical shapes")
    averaged = (left + right) / 2.0
    result = []
    rows, cols = np.triu_indices(35, k=1)
    for vector in averaged:
        if input_order == "row":
            matrix = np.zeros((35, 35))
            matrix[rows, cols] = vector
            matrix[cols, rows] = vector
        else:
            matrix = MatrixVectorizer.anti_vectorize(vector, 35)
        converted = MatrixVectorizer.vectorize(matrix)
        # Guard against changing the project's vectorization convention.
        if not np.array_equal(MatrixVectorizer.anti_vectorize(converted, 35), matrix):
            raise ValueError("Morphology matrix/vector round-trip failed")
        result.append(converted)
    return np.asarray(result)


def write_csv(path, vectors):
    """Match legacy CSVs: E header fields, then ID + E values per graph."""
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(range(vectors.shape[1]))
        for graph_id, vector in enumerate(vectors, start=1):
            writer.writerow([graph_id, *vector])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "gsr_data" / "slim_csv")
    parser.add_argument(
        "--morph-input-order", choices=("row", "column"), default="row",
        help="MAT morphology upper-triangle order (default: row, inferred from data)",
    )
    args = parser.parse_args()
    functional = [
        load_vectors(args.input_dir, "func_data_160.mat", "LR", 160),
        load_vectors(args.input_dir, "func_data_268.mat", "HR", 268),
    ]
    hemispheres = [
        load_vectors(args.input_dir, f"morph_thickness_data_35_{side}.mat",
                     f"morph_thickness_data_35_{side}", 35)
        for side in ("lh", "rh")
    ]
    if len({len(v) for v in functional + hemispheres}) != 1:
        raise ValueError("All four MAT files must have the same number of graphs")
    # Functional column order already matches the existing project CSVs.
    outputs = {
        "func_data_160.csv": np.maximum(functional[0], 0),
        "func_data_268.csv": np.maximum(functional[1], 0),
        "morph_thickness_data_35_mean.csv": average_hemispheres(
            *hemispheres, args.morph_input_order),
    }
    for name in outputs:
        if (args.output_dir / name).exists():
            raise FileExistsError(f"Refusing to overwrite {args.output_dir / name}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, vectors in outputs.items():
        path = args.output_dir / name
        write_csv(path, vectors)
        print(f"{path}: {len(vectors)} graphs, {vectors.shape[1]} edges per graph")


if __name__ == "__main__":
    main()
