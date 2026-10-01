import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from ..data.preprocessing import get_feature_names_after_preprocessing, prepare_xy
from ..config.settings import ALL_MODEL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES

try:
    import shap
except Exception:
    # Numba (a SHAP dependency) can be blocked by Windows Application Control.
    shap = None

class ModelExplainer:
    """
    SHAP-based and coefficient-based model explainability engine.
    Converts mathematical attributions into engineering decision summaries.
    """
    def __init__(self, pipeline: Any, background_df: pd.DataFrame):
        self.pipeline = pipeline
        self.preprocessor = None
        self.classifier = None
        
        # Extract underlying fitted preprocessor and classifier
        if hasattr(pipeline, "calibrated_classifiers_") and len(pipeline.calibrated_classifiers_) > 0:
            first_base = pipeline.calibrated_classifiers_[0].estimator
            if hasattr(first_base, "named_steps"):
                self.preprocessor = first_base.named_steps.get("preprocessor")
                self.classifier = first_base.named_steps.get("classifier")
        elif hasattr(pipeline, "named_steps"):
            self.preprocessor = pipeline.named_steps.get("preprocessor")
            self.classifier = pipeline.named_steps.get("classifier")
        elif hasattr(pipeline, "estimator") and hasattr(pipeline.estimator, "named_steps"):
            self.preprocessor = pipeline.estimator.named_steps.get("preprocessor")
            self.classifier = pipeline.estimator.named_steps.get("classifier")

        # Transform background data for SHAP
        X_bg_clean, _ = prepare_xy(background_df, is_inference=True)
        if self.preprocessor is not None:
            self.feature_names = get_feature_names_after_preprocessing(self.preprocessor)
            self.X_bg_trans = self.preprocessor.transform(X_bg_clean)
        else:
            self.feature_names = ALL_MODEL_FEATURES
            self.X_bg_trans = X_bg_clean.values

        self.shap_available = shap is not None
        self.bg_mean = np.asarray(self.X_bg_trans).mean(axis=0)

        # Initialize tree or linear explainer
        self.explainer = None
        if self.shap_available:
            try:
                clf_name = type(self.classifier).__name__ if self.classifier is not None else ""
                if "XGB" in clf_name or "GradientBoosting" in clf_name:
                    self.explainer = shap.TreeExplainer(self.classifier)
                elif "LogisticRegression" in clf_name:
                    self.explainer = shap.LinearExplainer(self.classifier, self.X_bg_trans)
                else:
                    sample_bg = self.X_bg_trans[:20] if len(self.X_bg_trans) > 20 else self.X_bg_trans
                    self.explainer = shap.KernelExplainer(self.classifier.predict_proba, sample_bg)
            except Exception:
                try:
                    sample_bg = self.X_bg_trans[:20] if len(self.X_bg_trans) > 20 else self.X_bg_trans
                    self.explainer = shap.KernelExplainer(self.classifier.predict_proba, sample_bg)
                except Exception:
                    self.explainer = None

    def explain_instance(self, instance_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates local feature attributions and engineering explanations for a single failure event.
        """
        X_inst_clean, _ = prepare_xy(instance_df.iloc[0:1], is_inference=True)
        
        if self.preprocessor is not None:
            X_trans = self.preprocessor.transform(X_inst_clean)
        else:
            X_trans = X_inst_clean.values

        # Compute SHAP values, or a local importance fallback when SHAP cannot load.
        try:
            if self.explainer is None:
                raise RuntimeError("SHAP explainer unavailable")
            shap_values = self.explainer.shap_values(X_trans)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                sv = np.array(shap_values[1])[0]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                sv = shap_values[0, :, 1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
                sv = shap_values[0]
            elif hasattr(shap_values, "values"):
                sv_arr = shap_values.values
                sv = sv_arr[0, :, 1] if sv_arr.ndim == 3 else sv_arr[0]
            else:
                sv = np.zeros(len(self.feature_names))
        except Exception:
            sv = self._fallback_contributions(np.asarray(X_trans)[0])

        contrib_list = []
        for feat_name, feat_val, shap_val in zip(self.feature_names, X_trans[0], sv):
            clean_name = feat_name.replace("cat__", "").replace("num__", "").replace("_", " ")
            contrib_list.append({
                "feature": clean_name,
                "raw_feature": feat_name,
                "value": float(feat_val),
                "shap_value": float(shap_val),
                "abs_shap": abs(float(shap_val)),
                "impact": "Increases P(Retest)" if shap_val > 0 else "Decreases P(Retest)"
            })

        contrib_list.sort(key=lambda x: x["abs_shap"], reverse=True)

        bullets = []
        if self.explainer is None:
            bullets.append(
                "SHAP is unavailable on this machine (Numba was blocked by Windows Application Control). "
                "Contributions below are local model-importance estimates, not SHAP values."
            )
        for item in contrib_list[:4]:
            feat = item["feature"]
            s_val = item["shap_value"]
            if s_val > 0:
                bullets.append(
                    f"{feat} contributed toward a higher predicted P(RETEST_BENEFICIAL). "
                    "This is a model contribution, not physical causation."
                )
            else:
                bullets.append(
                    f"{feat} contributed toward a lower predicted P(RETEST_BENEFICIAL). "
                    "This is a model contribution, not physical causation."
                )

        return {
            "feature_contributions": contrib_list,
            "top_features": contrib_list[:8],
            "engineering_explanations": bullets
        }

    def _fallback_contributions(self, x_row: np.ndarray) -> np.ndarray:
        """Instance-level stand-in when SHAP cannot be imported."""
        n = len(self.feature_names)
        weights = np.zeros(n)
        clf = self.classifier
        if clf is not None and hasattr(clf, "feature_importances_"):
            weights = np.asarray(clf.feature_importances_, dtype=float)
        elif clf is not None and hasattr(clf, "coef_"):
            coef = np.asarray(clf.coef_, dtype=float).reshape(-1)
            weights = coef[:n] if len(coef) >= n else np.pad(coef, (0, n - len(coef)))
        if len(weights) != n:
            weights = np.ones(n)
        centered = x_row - self.bg_mean
        if len(centered) != n:
            centered = np.resize(centered, n)
        return weights * centered
