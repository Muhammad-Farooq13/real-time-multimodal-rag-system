from pathlib import Path

from train_demo import load_or_rebuild_bundle, train_and_build_bundle


def test_train_and_build_bundle_creates_required_artifacts(tmp_path: Path) -> None:
    bundle_path = tmp_path / "demo_bundle.pkl"
    bundle = train_and_build_bundle(bundle_path)

    assert bundle_path.exists()
    assert bundle["best_model_name"]
    assert bundle["model_results"]
    assert bundle["full_dataframe"]
    assert bundle["feature_schema"]


def test_load_or_rebuild_bundle_recovers_from_corrupt_file(tmp_path: Path) -> None:
    bundle_path = tmp_path / "demo_bundle.pkl"
    bundle_path.write_bytes(b"not-a-valid-bundle")

    bundle = load_or_rebuild_bundle(bundle_path)

    assert bundle_path.exists()
    assert bundle["best_model"] is not None
    assert bundle["analytics"]["target_rate"] >= 0.0
