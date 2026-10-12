"""
Model Artifact Integrity Verification
SHA-256 verification before deserialization.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Optional
import logging

try:
    from .config import get_config, get_model_integrity_config, get_resolved_paths
except ImportError:
    from config import get_config, get_model_integrity_config, get_resolved_paths

logger = logging.getLogger(__name__)


class IntegrityError(Exception):
    """Raised when model artifact integrity verification fails."""
    pass


def compute_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        # Read in chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_artifact(
    artifact_name: str,
    file_path: Path,
    expected_hash: Optional[str] = None,
) -> bool:
    """
    Verify a single artifact's SHA-256 hash.
    
    Args:
        artifact_name: Logical name (e.g., "xgboost", "gru_60s")
        file_path: Path to the artifact file
        expected_hash: Optional override; if None, uses config
        
    Returns:
        True if verification passes
        
    Raises:
        IntegrityError: If verification fails (missing, unreadable, hash mismatch)
    """
    if not file_path.exists():
        raise IntegrityError(f"Model artifact '{artifact_name}' not found at {file_path}")
    
    if not file_path.is_file():
        raise IntegrityError(f"Model artifact '{artifact_name}' path is not a file: {file_path}")
    
    try:
        actual_hash = compute_sha256(file_path)
    except Exception as e:
        raise IntegrityError(f"Failed to read artifact '{artifact_name}': {e}")
    
    if expected_hash is None:
        config = get_config()
        integrity_cfg = config.model_integrity
        expected_hash = getattr(integrity_cfg, artifact_name, None)
    
    if expected_hash is None:
        raise IntegrityError(f"No expected hash configured for artifact '{artifact_name}'")
    
    if actual_hash.lower() != expected_hash.lower():
        logger.error(f"Integrity verification FAILED for '{artifact_name}': expected {expected_hash}, got {actual_hash}")
        raise IntegrityError(
            f"Integrity verification failed for '{artifact_name}': "
            f"hash mismatch (expected {expected_hash}, got {actual_hash}). "
            f"Artifact may have been tampered with or corrupted."
        )
    
    logger.info(f"Integrity verification PASSED for '{artifact_name}'")
    return True


def verify_all_artifacts(models_dir: Optional[Path] = None) -> dict[str, bool]:
    """
    Verify all configured model artifacts.
    
    Returns:
        Dict mapping artifact name -> True (all verified)
        
    Raises:
        IntegrityError: If any artifact fails verification
    """
    config = get_config()
    integrity_cfg = config.model_integrity
    models_cfg = config.models
    
    if models_dir is None:
        paths = get_resolved_paths()
        models_dir = paths.models
    
    results = {}
    
    # Map of artifact names to (config_attr, hash_attr)
    artifact_map = {
        "xgboost": ("xgboost", "xgboost"),
        "random_forest": ("random_forest", "random_forest"),
        "logistic_regression": ("logistic_regression", "logistic_regression"),
        "gru_60s": ("gru_60s", "gru_60s"),
    }
    
    for artifact_name, (model_attr, hash_attr) in artifact_map.items():
        model_filename = getattr(config.models, model_attr)
        expected_hash = getattr(config.model_integrity, hash_attr)
        
        file_path = Path(models_dir) / model_filename
        
        try:
            verify_artifact(artifact_name, file_path, expected_hash)
            results[artifact_name] = True
        except IntegrityError:
            # Re-raise with artifact context
            raise
    
    return results


def load_verified_joblib(artifact_name: str, models_dir: Optional[Path] = None):
    """
    Load a joblib artifact after verifying its integrity.
    
    Args:
        artifact_name: One of "xgboost", "random_forest", "logistic_regression"
        models_dir: Optional override for models directory
        
    Returns:
        The loaded joblib object
        
    Raises:
        IntegrityError: If verification fails
    """
    if models_dir is None:
        paths = get_resolved_paths()
        models_dir = paths.models
    
    config = get_config()
    model_filename = getattr(config.models, artifact_name)
    expected_hash = getattr(config.model_integrity, artifact_name)
    file_path = Path(models_dir) / model_filename
    
    # Verify BEFORE loading
    verify_artifact(artifact_name, file_path)
    
    import joblib
    return joblib.load(file_path)


def load_verified_torch(artifact_name: str, models_dir: Optional[Path] = None, device="cpu"):
    """
    Load a PyTorch artifact after verifying its integrity.
    
    Args:
        artifact_name: Currently only "gru_60s" supported
        models_dir: Optional override for models directory
        device: Torch device
        
    Returns:
        The loaded checkpoint dict
        
    Raises:
        IntegrityError: If verification fails
    """
    if models_dir is None:
        paths = get_resolved_paths()
        models_dir = paths.models
    
    config = get_config()
    model_filename = getattr(config.models, artifact_name)
    expected_hash = getattr(config.model_integrity, artifact_name)
    file_path = Path(models_dir) / model_filename
    
    # Verify BEFORE loading
    verify_artifact(artifact_name, file_path, expected_hash)
    
    import torch
    # Use weights_only=False because checkpoint contains config dict and numpy arrays
    # This is safe because we verified the hash first
    return torch.load(file_path, map_location=device)


if __name__ == "__main__":
    # Quick test
    try:
        results = verify_all_artifacts()
        print("All artifacts verified:", results)
    except IntegrityError as e:
        print(f"Verification failed: {e}")