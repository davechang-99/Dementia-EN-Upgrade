"""
Module 6: Model Layer (모델 레이어)

추출된 음성 기반 특징을 활용하여 정상군 대비 인지기능 변화와
연관될 수 있는 위험 경향성을 기술적으로 산출하고, 해당 결과를
연속형 위험 점수와 신뢰도 지표의 형태로 제공하는 모듈.
"""

import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from velora.config.settings import VeloraConfig, get_config
from velora.utils.logging import setup_logger

logger = setup_logger("model_layer")


@dataclass
class ConfidenceScore:
    """결과 신뢰도"""
    overall: float                 # 종합 신뢰도 (0~1)
    audio_quality: float           # 입력 음성 품질 기반
    diarization_confidence: float  # 화자 분리 신뢰도
    model_uncertainty: float       # 모델 불확실성 (Entropy)
    details: Dict[str, float] = field(default_factory=dict)


@dataclass
class RiskAssessment:
    """위험 경향성 평가 결과"""
    risk_score: float              # 연속형 위험 점수 (0~1)
    risk_level: str                # "low", "caution", "high"
    risk_probability: float        # 위험 경향성 확률
    confidence: ConfidenceScore    # 신뢰도 지표
    model_name: str                # 사용된 모델명
    feature_importance: Dict[str, float]  # 주요 특징 기여도
    disclaimer: str                # 비의료적 면책 고지


@dataclass
class TrainingResult:
    """학습 결과"""
    model_name: str
    accuracy: float
    macro_f1: float
    sensitivity: float
    specificity: float
    fold_scores: List[float]
    fold_std: float
    best_params: Optional[Dict] = None


@dataclass
class EnsembleResult:
    """앙상블 결과"""
    risk_score: float
    risk_level: str
    individual_scores: Dict[str, float]
    aggregation_method: str
    confidence: ConfidenceScore


class BaselineModels:
    """
    1단계 기준 모델

    - Logistic Regression
    - SVM (Support Vector Machine)
    - Random Forest
    """

    def __init__(self, config: Optional[VeloraConfig] = None):
        self.config = config or get_config()
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self._is_trained = False

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, TrainingResult]:
        """
        기준 모델 학습 (5-fold Stratified CV)

        Args:
            X: 특징 행렬 (n_samples, n_features)
            y: 레이블 (0: 정상, 1: 위험)
            feature_names: 특징 이름 목록

        Returns:
            Dict[str, TrainingResult]: 모델별 학습 결과
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.metrics import (
            accuracy_score, f1_score, recall_score, confusion_matrix
        )

        model_configs = {
            "logistic_regression": LogisticRegression(
                max_iter=1000, random_state=self.config.model.random_seed, C=1.0
            ),
            "svm": SVC(
                kernel="rbf", probability=True,
                random_state=self.config.model.random_seed, C=1.0
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=100, max_depth=10,
                random_state=self.config.model.random_seed,
            ),
        }

        skf = StratifiedKFold(
            n_splits=self.config.model.n_folds,
            shuffle=True,
            random_state=self.config.model.random_seed,
        )

        results = {}

        for model_name, model in model_configs.items():
            fold_scores = []
            fold_f1s = []
            fold_sensitivities = []
            fold_specificities = []

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # 스케일링
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                # 학습
                model_copy = type(model)(**model.get_params())
                model_copy.fit(X_train_scaled, y_train)

                # 예측
                y_pred = model_copy.predict(X_val_scaled)

                # 메트릭
                acc = accuracy_score(y_val, y_pred)
                f1 = f1_score(y_val, y_pred, average="macro")
                sensitivity = recall_score(y_val, y_pred, pos_label=1, zero_division=0)

                cm = confusion_matrix(y_val, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                else:
                    specificity = 0

                fold_scores.append(acc)
                fold_f1s.append(f1)
                fold_sensitivities.append(sensitivity)
                fold_specificities.append(specificity)

            # 최종 모델 학습 (전체 데이터)
            final_scaler = StandardScaler()
            X_scaled = final_scaler.fit_transform(X)
            model.fit(X_scaled, y)

            self.models[model_name] = model
            self.scalers[model_name] = final_scaler

            results[model_name] = TrainingResult(
                model_name=model_name,
                accuracy=float(np.mean(fold_scores)),
                macro_f1=float(np.mean(fold_f1s)),
                sensitivity=float(np.mean(fold_sensitivities)),
                specificity=float(np.mean(fold_specificities)),
                fold_scores=fold_scores,
                fold_std=float(np.std(fold_scores)),
            )

            logger.info(
                f"Model {model_name}: "
                f"Acc={np.mean(fold_scores):.3f}±{np.std(fold_scores):.3f}, "
                f"F1={np.mean(fold_f1s):.3f}, "
                f"Sens={np.mean(fold_sensitivities):.3f}"
            )

        self._is_trained = True
        return results

    def predict(
        self,
        X: np.ndarray,
        model_name: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        예측 수행

        Args:
            X: 특징 벡터
            model_name: 사용할 모델명 (None이면 best model)

        Returns:
            Tuple[predictions, probabilities]
        """
        if not self._is_trained:
            raise RuntimeError("모델이 학습되지 않았습니다. train()을 먼저 호출하세요.")

        if model_name is None:
            model_name = self._get_best_model_name()

        model = self.models[model_name]
        scaler = self.scalers[model_name]

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)

        return predictions, probabilities

    def _get_best_model_name(self) -> str:
        """가장 성능이 좋은 모델 선택"""
        if not self.models:
            raise RuntimeError("학습된 모델이 없습니다.")
        return list(self.models.keys())[0]  # 기본: 첫 번째 모델

    def get_feature_importance(
        self, model_name: str, feature_names: List[str]
    ) -> Dict[str, float]:
        """특징 중요도 반환"""
        model = self.models.get(model_name)
        if model is None:
            return {}

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return {}

        if len(importances) != len(feature_names):
            return {}

        importance_dict = {
            name: float(imp) for name, imp in zip(feature_names, importances)
        }
        # 상위 10개만 반환
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:10])

    def save(self, model_dir: str) -> None:
        """모델 저장"""
        os.makedirs(model_dir, exist_ok=True)
        for name, model in self.models.items():
            model_path = os.path.join(model_dir, f"{name}_model.pkl")
            scaler_path = os.path.join(model_dir, f"{name}_scaler.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scalers[name], f)
        logger.info(f"Models saved to {model_dir}")

    def load(self, model_dir: str) -> None:
        """모델 로드"""
        for name in ["logistic_regression", "svm", "random_forest"]:
            model_path = os.path.join(model_dir, f"{name}_model.pkl")
            scaler_path = os.path.join(model_dir, f"{name}_scaler.pkl")
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                with open(model_path, "rb") as f:
                    self.models[name] = pickle.load(f)
                with open(scaler_path, "rb") as f:
                    self.scalers[name] = pickle.load(f)
        self._is_trained = bool(self.models)
        logger.info(f"Models loaded from {model_dir}: {list(self.models.keys())}")


class ViTFeatureExtractor:
    """
    2단계 고도화: Vision Transformer (ViT) 기반 특징 추출

    스펙트로그램 이미지를 ViT 모델에 통과시켜
    고차원 특징 벡터를 추출하고 PCA로 차원 축소
    """

    def __init__(self, config: Optional[VeloraConfig] = None):
        self.config = config or get_config()
        self._vit_model = None
        self._vit_processor = None
        self._pca = None

    def _init_vit(self):
        """ViT 모델 lazy loading"""
        if self._vit_model is not None:
            return

        try:
            from transformers import ViTForImageClassification, ViTImageProcessor
            import torch

            model_name = self.config.feature.vit_model_name
            self._vit_processor = ViTImageProcessor.from_pretrained(model_name)
            self._vit_model = ViTForImageClassification.from_pretrained(model_name)
            self._vit_model.eval()
            logger.info(f"ViT model loaded: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load ViT model: {e}")

    def extract_features(
        self, spectrogram_image: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        스펙트로그램 이미지에서 ViT 특징 추출

        Args:
            spectrogram_image: RGB 이미지 (H, W, 3)

        Returns:
            np.ndarray: 특징 벡터
        """
        self._init_vit()

        if self._vit_model is None:
            return None

        try:
            import torch
            from PIL import Image

            img = Image.fromarray(spectrogram_image)
            inputs = self._vit_processor(images=img, return_tensors="pt")

            with torch.no_grad():
                outputs = self._vit_model(**inputs, output_hidden_states=True)
                # CLS 토큰의 hidden state 사용
                features = outputs.hidden_states[-1][:, 0, :].numpy()

            return features.flatten()

        except Exception as e:
            logger.error(f"ViT feature extraction failed: {e}")
            return None

    def setup_pca(
        self, feature_matrix: np.ndarray, n_components: Optional[int] = None
    ):
        """PCA 차원 축소 설정"""
        from sklearn.decomposition import PCA

        n_comp = n_components or self.config.feature.pca_n_components
        self._pca = PCA(n_components=n_comp, random_state=self.config.model.random_seed)
        self._pca.fit(feature_matrix)
        logger.info(
            f"PCA fitted: {feature_matrix.shape[1]} -> {n_comp} dimensions, "
            f"explained variance: {self._pca.explained_variance_ratio_.sum():.3f}"
        )

    def reduce_dimensions(self, features: np.ndarray) -> np.ndarray:
        """PCA 차원 축소"""
        if self._pca is None:
            raise RuntimeError("PCA가 설정되지 않았습니다. setup_pca()를 먼저 호출하세요.")
        if features.ndim == 1:
            features = features.reshape(1, -1)
        return self._pca.transform(features)


class EnsembleModel:
    """
    3단계 운영형 앙상블

    여러 모델의 예측을 조합하여 최종 위험 점수 산출
    """

    def __init__(self, config: Optional[VeloraConfig] = None):
        self.config = config or get_config()
        self.baseline_models = BaselineModels(config)
        self.vit_extractor = ViTFeatureExtractor(config)

    def predict_risk(
        self,
        feature_vector: np.ndarray,
        feature_names: List[str],
        audio_quality_score: float = 1.0,
        diarization_confidence: float = 1.0,
        spectrogram: Optional[np.ndarray] = None,
    ) -> RiskAssessment:
        """
        앙상블 기반 위험 점수 산출

        Args:
            feature_vector: 음향 특징 벡터
            feature_names: 특징 이름
            audio_quality_score: 입력 음성 품질 점수
            diarization_confidence: 화자 분리 신뢰도
            spectrogram: 스펙트로그램 이미지 (ViT용, 선택)

        Returns:
            RiskAssessment: 위험 평가 결과
        """
        model_scores = {}

        # 기준 모델 예측
        if self.baseline_models._is_trained:
            for model_name in self.baseline_models.models:
                try:
                    _, probs = self.baseline_models.predict(
                        feature_vector, model_name
                    )
                    # 위험군(1) 확률
                    risk_prob = probs[0][1] if probs.shape[1] > 1 else probs[0][0]
                    model_scores[model_name] = float(risk_prob)
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_name}: {e}")

        if not model_scores:
            # 모델이 없으면 기본값
            risk_score = 0.5
            model_name_used = "none"
        else:
            # 앙상블 집계 (가중 평균)
            weights = {
                "random_forest": 0.4,
                "svm": 0.35,
                "logistic_regression": 0.25,
            }
            total_weight = 0
            weighted_sum = 0
            for name, score in model_scores.items():
                w = weights.get(name, 0.33)
                weighted_sum += score * w
                total_weight += w

            risk_score = weighted_sum / total_weight if total_weight > 0 else 0.5
            model_name_used = "ensemble"

        # 위험 수준 판정
        risk_level = self._determine_risk_level(risk_score)

        # 신뢰도 계산
        confidence = self._compute_confidence(
            model_scores, audio_quality_score, diarization_confidence
        )

        # 특징 중요도
        feature_importance = {}
        if self.baseline_models._is_trained:
            best_model = max(model_scores, key=model_scores.get) if model_scores else None
            if best_model:
                feature_importance = self.baseline_models.get_feature_importance(
                    best_model, feature_names
                )

        return RiskAssessment(
            risk_score=risk_score,
            risk_level=risk_level,
            risk_probability=risk_score,
            confidence=confidence,
            model_name=model_name_used,
            feature_importance=feature_importance,
            disclaimer=self.config.disclaimer,
        )

    def _determine_risk_level(self, risk_score: float) -> str:
        """위험 수준 판정 (낮음/주의/높음)"""
        thresholds = self.config.model.risk_thresholds
        if risk_score < thresholds["low"]:
            return "low"
        elif risk_score < thresholds["caution"]:
            return "caution"
        else:
            return "high"

    def _compute_confidence(
        self,
        model_scores: Dict[str, float],
        audio_quality: float,
        diarization_conf: float,
    ) -> ConfidenceScore:
        """
        결과 신뢰도 산출

        구성 요소:
        1. 입력 음성 품질
        2. 화자 분리 성공률 및 명확도
        3. AI 모델 자체의 불확실성 (Entropy)
        """
        weights = self.config.model.confidence_weights

        # 모델 불확실성 (예측 분산)
        if len(model_scores) > 1:
            scores = list(model_scores.values())
            # 모델간 일관성이 높을수록 신뢰도 높음
            consistency = 1.0 - np.std(scores)
            model_uncertainty_score = max(0.0, min(1.0, consistency))
        else:
            model_uncertainty_score = 0.5

        # 종합 신뢰도
        overall = (
            audio_quality * weights["audio_quality"]
            + diarization_conf * weights["diarization_confidence"]
            + model_uncertainty_score * weights["model_uncertainty"]
        )

        return ConfidenceScore(
            overall=float(np.clip(overall, 0, 1)),
            audio_quality=float(audio_quality),
            diarization_confidence=float(diarization_conf),
            model_uncertainty=float(model_uncertainty_score),
            details={
                name: float(score) for name, score in model_scores.items()
            },
        )


class FriedmanTest:
    """
    Friedman 통계 검정

    모델 간 성능 유의차 검정
    """

    @staticmethod
    def run_test(
        model_results: Dict[str, TrainingResult],
    ) -> Dict:
        """
        Friedman 검정 수행

        Args:
            model_results: 모델별 학습 결과

        Returns:
            Dict: 검정 결과
        """
        from scipy.stats import friedmanchisquare

        fold_accuracies = {
            name: result.fold_scores
            for name, result in model_results.items()
        }

        model_names = list(fold_accuracies.keys())
        scores_matrix = [fold_accuracies[name] for name in model_names]

        if len(scores_matrix) < 3:
            return {
                "test": "friedman",
                "statistic": None,
                "p_value": None,
                "significant": False,
                "message": "Friedman 검정에는 최소 3개 모델이 필요합니다.",
                "model_rankings": {},
            }

        stat, p_value = friedmanchisquare(*scores_matrix)

        # 모델 순위
        mean_scores = {
            name: np.mean(scores) for name, scores in fold_accuracies.items()
        }
        rankings = sorted(mean_scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "test": "friedman",
            "statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "model_rankings": {name: rank + 1 for rank, (name, _) in enumerate(rankings)},
            "mean_accuracies": mean_scores,
            "fold_accuracies": fold_accuracies,
        }
