# tests/test_hyperparameter_analysis.py
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import inspect
import hyperparameter_analysis as ha


def test_public_signatures_exist():
    for fn_name in [
        "analyze_learning_rates", "analyze_batch_sizes",
        "analyze_regularization_strengths", "analyze_optimizers",
        "create_comprehensive_hyperparameter_analysis",
    ]:
        assert hasattr(ha, fn_name), f"Missing function: {fn_name}"
        sig = inspect.signature(getattr(ha, fn_name))
        params = list(sig.parameters)
        assert "model_func" in params
        assert "train_dir" in params
        assert "val_dir" in params
