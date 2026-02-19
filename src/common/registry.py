from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib


class ModelRegistry:
    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

    def list_versions(self) -> List[str]:
        if not self.registry_path.exists():
            return []
        versions = [
            path.name
            for path in self.registry_path.iterdir()
            if path.is_dir() and (path / "model.joblib").exists()
        ]
        return sorted(versions)

    def get_latest_version(self) -> Optional[str]:
        versions = self.list_versions()
        return versions[-1] if versions else None

    def save_model(
        self,
        model: Any,
        metrics: Dict[str, Any],
        feature_names: List[str],
        version: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        version = version or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        model_dir = self.registry_path / version
        model_dir.mkdir(parents=True, exist_ok=False)

        joblib.dump(model, model_dir / "model.joblib")
        (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        metadata = {
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_type": model.__class__.__name__,
            "feature_names": feature_names,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        return version

    def load_model(
        self, version: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
        if version in (None, "latest"):
            version = self.get_latest_version()
        if not version:
            raise FileNotFoundError("Model registry is empty.")

        model_dir = self.registry_path / version
        model_path = model_dir / "model.joblib"
        metrics_path = model_dir / "metrics.json"
        metadata_path = model_dir / "metadata.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found for version {version}.")

        model = joblib.load(model_path)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        return model, metadata, metrics

    def get_version_info(self, version: str) -> Dict[str, Any]:
        model_dir = self.registry_path / version
        metrics_path = model_dir / "metrics.json"
        metadata_path = model_dir / "metadata.json"

        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

        return {
            "version": version,
            "created_at": metadata.get("created_at"),
            "metrics": metrics,
            "metadata": metadata,
        }

    def get_versions_info(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        versions = sorted(self.list_versions(), reverse=True)
        if limit is not None:
            versions = versions[: max(limit, 0)]
        return [self.get_version_info(version) for version in versions]
