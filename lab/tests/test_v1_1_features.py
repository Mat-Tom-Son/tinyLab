"""Validation tests for v1.1 features."""
import pandas as pd
from pathlib import Path


def validate_head_impact_schema(df):
    """Validate head impact table schema.

    Required columns:
    - run_id: str
    - seed: int
    - layer: int
    - head: int
    - scale: float
    - metric: str
    - value: float
    """
    required_cols = ["run_id", "seed", "layer", "head", "scale", "metric", "value"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # Check dtypes
    assert df["run_id"].dtype == object, "run_id should be str"
    assert pd.api.types.is_integer_dtype(df["seed"]), "seed should be int"
    assert pd.api.types.is_integer_dtype(df["layer"]), "layer should be int"
    assert pd.api.types.is_integer_dtype(df["head"]), "head should be int"
    assert pd.api.types.is_float_dtype(df["scale"]), "scale should be float"
    assert df["metric"].dtype == object, "metric should be str"
    assert pd.api.types.is_float_dtype(df["value"]), "value should be float"

    print("✓ Head impact schema valid")
    return True


def validate_layer_impact_schema(df):
    """Validate layer impact table schema.

    Required columns:
    - run_id: str
    - seed: int
    - layer: int
    - granularity: str
    - metric: str
    - value: float
    """
    required_cols = ["run_id", "seed", "layer", "granularity", "metric", "value"]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # Check dtypes
    assert df["run_id"].dtype == object, "run_id should be str"
    assert pd.api.types.is_integer_dtype(df["seed"]), "seed should be int"
    assert pd.api.types.is_integer_dtype(df["layer"]), "layer should be int"
    assert df["granularity"].dtype == object, "granularity should be str"
    assert df["metric"].dtype == object, "metric should be str"
    assert pd.api.types.is_float_dtype(df["value"]), "value should be float"

    print("✓ Layer impact schema valid")
    return True


def validate_verify_json(verify_data):
    """Validate verify.json structure.

    Required keys:
    - device_main: str
    - device_verify: str
    - n_examples: int
    - n_seeds: int
    - metrics: dict
    """
    required_keys = ["device_main", "device_verify", "n_examples", "n_seeds", "metrics"]
    for key in required_keys:
        assert key in verify_data, f"Missing key: {key}"

    # Check metrics structure
    for metric_name, metric_data in verify_data["metrics"].items():
        assert "main" in metric_data, f"Missing 'main' in {metric_name}"
        assert "verify" in metric_data, f"Missing 'verify' in {metric_name}"
        assert "abs_diff" in metric_data, f"Missing 'abs_diff' in {metric_name}"

    print("✓ Verify JSON structure valid")
    return True


def validate_cross_condition_matrix(df, expected_conditions):
    """Validate cross-condition matrix has condition column.

    Args:
        df: DataFrame (head_matrix or layer_matrix)
        expected_conditions: List of expected condition tags
    """
    assert "condition" in df.columns, "Missing 'condition' column"

    conditions = df["condition"].unique()
    assert len(conditions) == len(expected_conditions), \
        f"Expected {len(expected_conditions)} conditions, found {len(conditions)}"

    for cond in expected_conditions:
        assert cond in conditions, f"Missing condition: {cond}"

    print(f"✓ Cross-condition matrix valid ({len(conditions)} conditions)")
    return True


def validate_invariants_json(invariants_data):
    """Validate invariants.json structure.

    Required keys:
    - k: int
    - metrics: list
    - heads: dict
    - layers: dict
    """
    required_keys = ["k", "metrics", "heads", "layers"]
    for key in required_keys:
        assert key in invariants_data, f"Missing key: {key}"

    # Check heads structure
    for metric in invariants_data["metrics"]:
        if metric in invariants_data["heads"]:
            heads_list = invariants_data["heads"][metric]
            if heads_list:
                # Check first entry has layer and head
                assert "layer" in heads_list[0], "Missing 'layer' in head entry"
                assert "head" in heads_list[0], "Missing 'head' in head entry"

        # Check layers structure
        if metric in invariants_data["layers"]:
            layers_list = invariants_data["layers"][metric]
            if layers_list:
                assert isinstance(layers_list[0], int), "Layer should be int"

    print("✓ Invariants JSON structure valid")
    return True


def test_schemas_on_files(run_dir):
    """Test all schemas on a completed run directory.

    Args:
        run_dir: Path to run directory

    Usage:
        python3 -c "from lab.tests.test_v1_1_features import test_schemas_on_files; test_schemas_on_files('lab/runs/h1_heads_zero_abc123')"
    """
    run_path = Path(run_dir)

    # Test head impact
    head_impact_path = run_path / "metrics" / "head_impact.parquet"
    if head_impact_path.exists():
        df = pd.read_parquet(head_impact_path)
        validate_head_impact_schema(df)

    # Test layer impact
    layer_impact_path = run_path / "metrics" / "layer_impact.parquet"
    if layer_impact_path.exists():
        df = pd.read_parquet(layer_impact_path)
        validate_layer_impact_schema(df)

    # Test verify
    verify_path = run_path / "metrics" / "verify.json"
    if verify_path.exists():
        import json
        with open(verify_path) as f:
            verify_data = json.load(f)
        validate_verify_json(verify_data)

    print(f"✓ All schemas valid for: {run_dir}")


if __name__ == "__main__":
    print("Run test_schemas_on_files() on a completed run directory.")
    print("Example:")
    print("  from lab.tests.test_v1_1_features import test_schemas_on_files")
    print("  test_schemas_on_files('lab/runs/h1_heads_zero_abc123')")
