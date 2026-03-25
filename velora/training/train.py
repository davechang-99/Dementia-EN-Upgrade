"""
VELORA Training Pipeline

AI Hub 데이터셋을 활용한 모델 학습 파이프라인.
5-fold Stratified Cross-Validation으로 학습하고
Friedman 통계 검정으로 모델 간 유의차를 검증합니다.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from velora.config.settings import VeloraConfig, get_config
from velora.modules.features import FeatureEngine
from velora.modules.model_layer import BaselineModels, FriedmanTest, TrainingResult
from velora.modules.preprocessing import PreprocessingModule
from velora.utils.logging import setup_logger

logger = setup_logger("training")


@dataclass
class DatasetInfo:
    """데이터셋 정보"""
    name: str
    total_samples: int
    positive_samples: int   # 위험군
    negative_samples: int   # 정상군
    audio_dir: str
    metadata_path: Optional[str] = None


@dataclass
class TrainingConfig:
    """학습 설정"""
    dataset_name: str = "aihub"
    audio_dir: str = "./data/audio"
    metadata_path: Optional[str] = None
    output_dir: str = "./models/saved"
    label_column: str = "label"
    audio_column: str = "file_path"
    n_folds: int = 5
    random_seed: int = 42


@dataclass
class TrainingPipelineResult:
    """학습 파이프라인 결과"""
    dataset_info: DatasetInfo
    model_results: Dict[str, TrainingResult]
    friedman_test: Optional[Dict] = None
    best_model: Optional[str] = None
    best_f1: float = 0.0
    output_dir: str = ""


class DataLoader:
    """
    학습 데이터 로더

    지원 형식:
    - AI Hub 데이터셋 (디렉토리 기반: normal/ vs risk/)
    - ADReSS 데이터셋 (csv 메타데이터 + 오디오)
    - 커스텀 데이터셋 (csv/json 메타데이터)
    """

    def __init__(self, config: Optional[VeloraConfig] = None):
        self.config = config or get_config()

    def load_directory_dataset(
        self,
        audio_dir: str,
        normal_subdir: str = "normal",
        risk_subdir: str = "risk",
    ) -> Tuple[List[str], np.ndarray]:
        """
        디렉토리 기반 데이터셋 로드

        구조:
        audio_dir/
            normal/  → 정상군 (label=0)
            risk/    → 위험군 (label=1)

        Returns:
            Tuple[List[str], np.ndarray]: (파일 경로 목록, 레이블 배열)
        """
        audio_extensions = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
        file_paths = []
        labels = []

        # 정상군
        normal_dir = os.path.join(audio_dir, normal_subdir)
        if os.path.isdir(normal_dir):
            for f in sorted(os.listdir(normal_dir)):
                if os.path.splitext(f)[1].lower() in audio_extensions:
                    file_paths.append(os.path.join(normal_dir, f))
                    labels.append(0)

        # 위험군
        risk_dir = os.path.join(audio_dir, risk_subdir)
        if os.path.isdir(risk_dir):
            for f in sorted(os.listdir(risk_dir)):
                if os.path.splitext(f)[1].lower() in audio_extensions:
                    file_paths.append(os.path.join(risk_dir, f))
                    labels.append(1)

        logger.info(
            f"Loaded {len(file_paths)} files: "
            f"{labels.count(0)} normal, {labels.count(1)} risk"
        )

        return file_paths, np.array(labels)

    def load_csv_dataset(
        self,
        csv_path: str,
        audio_dir: str,
        audio_column: str = "file_path",
        label_column: str = "label",
    ) -> Tuple[List[str], np.ndarray]:
        """
        CSV 메타데이터 기반 데이터셋 로드

        Returns:
            Tuple[List[str], np.ndarray]
        """
        import csv

        file_paths = []
        labels = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio_path = os.path.join(audio_dir, row[audio_column])
                if os.path.exists(audio_path):
                    file_paths.append(audio_path)
                    labels.append(int(row[label_column]))

        logger.info(
            f"Loaded {len(file_paths)} files from CSV: "
            f"{labels.count(0)} normal, {labels.count(1)} risk"
        )

        return file_paths, np.array(labels)


class TrainingPipeline:
    """
    VELORA 학습 파이프라인

    처리 흐름:
    1. 데이터 로드
    2. 전처리 + 특징 추출
    3. 모델 학습 (5-fold Stratified CV)
    4. Friedman 통계 검정
    5. 모델 저장
    """

    def __init__(
        self,
        training_config: Optional[TrainingConfig] = None,
        velora_config: Optional[VeloraConfig] = None,
    ):
        self.training_config = training_config or TrainingConfig()
        self.velora_config = velora_config or get_config()

        self.data_loader = DataLoader(self.velora_config)
        self.preprocessing = PreprocessingModule(self.velora_config)
        self.feature_engine = FeatureEngine(self.velora_config)
        self.baseline_models = BaselineModels(self.velora_config)

    def run(
        self,
        audio_dir: Optional[str] = None,
        metadata_path: Optional[str] = None,
    ) -> TrainingPipelineResult:
        """
        전체 학습 파이프라인 실행

        Args:
            audio_dir: 오디오 파일 디렉토리 (None이면 config 사용)
            metadata_path: 메타데이터 CSV 경로 (None이면 디렉토리 모드)

        Returns:
            TrainingPipelineResult
        """
        audio_dir = audio_dir or self.training_config.audio_dir
        metadata_path = metadata_path or self.training_config.metadata_path

        logger.info("=" * 60)
        logger.info("VELORA Training Pipeline Started")
        logger.info("=" * 60)

        # 1. 데이터 로드
        logger.info("Step 1: Loading dataset...")
        if metadata_path and os.path.exists(metadata_path):
            file_paths, labels = self.data_loader.load_csv_dataset(
                metadata_path, audio_dir,
                self.training_config.audio_column,
                self.training_config.label_column,
            )
        else:
            file_paths, labels = self.data_loader.load_directory_dataset(audio_dir)

        if len(file_paths) == 0:
            raise ValueError(f"데이터를 찾을 수 없습니다: {audio_dir}")

        dataset_info = DatasetInfo(
            name=self.training_config.dataset_name,
            total_samples=len(file_paths),
            positive_samples=int(np.sum(labels == 1)),
            negative_samples=int(np.sum(labels == 0)),
            audio_dir=audio_dir,
            metadata_path=metadata_path,
        )

        logger.info(
            f"Dataset: {dataset_info.total_samples} samples "
            f"({dataset_info.negative_samples} normal, "
            f"{dataset_info.positive_samples} risk)"
        )

        # 2. 특징 추출
        logger.info("Step 2: Extracting features...")
        feature_matrix, feature_names, valid_indices = self._extract_all_features(
            file_paths
        )

        # 유효한 샘플만 사용
        valid_labels = labels[valid_indices]

        logger.info(
            f"Feature extraction complete: "
            f"{feature_matrix.shape[0]} samples, "
            f"{feature_matrix.shape[1]} features"
        )

        # 3. 모델 학습
        logger.info("Step 3: Training models (5-fold Stratified CV)...")
        model_results = self.baseline_models.train(
            feature_matrix, valid_labels, feature_names
        )

        # 4. Friedman 검정
        logger.info("Step 4: Friedman statistical test...")
        friedman_result = None
        if len(model_results) >= 3:
            friedman_result = FriedmanTest.run_test(model_results)
            logger.info(
                f"Friedman test: stat={friedman_result.get('statistic', 'N/A')}, "
                f"p-value={friedman_result.get('p_value', 'N/A')}, "
                f"significant={friedman_result.get('significant', False)}"
            )

        # 5. 최적 모델 선택 및 저장
        best_model = max(model_results, key=lambda k: model_results[k].macro_f1)
        best_f1 = model_results[best_model].macro_f1

        output_dir = self.training_config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.baseline_models.save(output_dir)

        # 학습 결과 저장
        self._save_training_report(
            output_dir, dataset_info, model_results,
            friedman_result, feature_names
        )

        logger.info("=" * 60)
        logger.info(f"Training complete! Best model: {best_model} (F1={best_f1:.3f})")
        logger.info(f"Models saved to: {output_dir}")
        logger.info("=" * 60)

        return TrainingPipelineResult(
            dataset_info=dataset_info,
            model_results=model_results,
            friedman_test=friedman_result,
            best_model=best_model,
            best_f1=best_f1,
            output_dir=output_dir,
        )

    def _extract_all_features(
        self,
        file_paths: List[str],
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """모든 파일에서 특징 추출"""
        import tempfile

        all_features = []
        feature_names = None
        valid_indices = []

        for idx, audio_path in enumerate(file_paths):
            try:
                if idx % 10 == 0:
                    logger.info(f"  Processing {idx+1}/{len(file_paths)}...")

                # 전처리
                with tempfile.TemporaryDirectory() as temp_dir:
                    preprocess_result = self.preprocessing.process(
                        audio_path, temp_dir, f"sample_{idx}"
                    )

                    # 특징 추출
                    feature_result = self.feature_engine.extract_features(
                        preprocess_result.output_path,
                        speech_segments=preprocess_result.speech_segments,
                        utterance_stats=preprocess_result.statistics,
                        include_spectrogram=False,
                    )

                    all_features.append(feature_result.feature_vector)
                    if feature_names is None:
                        feature_names = feature_result.feature_names
                    valid_indices.append(idx)

            except Exception as e:
                logger.warning(f"Failed to process {audio_path}: {e}")
                continue

        if not all_features:
            raise ValueError("유효한 특징을 추출할 수 없습니다.")

        # 특징 벡터 길이 맞추기
        max_len = max(len(f) for f in all_features)
        padded_features = []
        for f in all_features:
            if len(f) < max_len:
                f = np.pad(f, (0, max_len - len(f)))
            padded_features.append(f)

        feature_matrix = np.vstack(padded_features)

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(max_len)]

        return feature_matrix, feature_names, np.array(valid_indices)

    def _save_training_report(
        self,
        output_dir: str,
        dataset_info: DatasetInfo,
        model_results: Dict[str, TrainingResult],
        friedman_result: Optional[Dict],
        feature_names: List[str],
    ) -> None:
        """학습 결과 리포트 저장"""
        report = {
            "dataset": {
                "name": dataset_info.name,
                "total_samples": dataset_info.total_samples,
                "positive_samples": dataset_info.positive_samples,
                "negative_samples": dataset_info.negative_samples,
            },
            "models": {},
            "feature_count": len(feature_names),
            "feature_names": feature_names[:50],  # 상위 50개만
        }

        for name, result in model_results.items():
            report["models"][name] = {
                "accuracy": result.accuracy,
                "macro_f1": result.macro_f1,
                "sensitivity": result.sensitivity,
                "specificity": result.specificity,
                "fold_scores": result.fold_scores,
                "fold_std": result.fold_std,
            }

        if friedman_result:
            report["friedman_test"] = {
                k: v for k, v in friedman_result.items()
                if k != "fold_accuracies"
            }

        report_path = os.path.join(output_dir, "training_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Training report saved: {report_path}")


def main():
    """CLI 학습 실행"""
    import argparse

    parser = argparse.ArgumentParser(description="VELORA Training Pipeline")
    parser.add_argument(
        "--audio-dir", type=str, default="./data/audio",
        help="오디오 파일 디렉토리",
    )
    parser.add_argument(
        "--metadata", type=str, default=None,
        help="메타데이터 CSV 경로",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./models/saved",
        help="모델 저장 디렉토리",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Cross-validation fold 수",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    training_config = TrainingConfig(
        audio_dir=args.audio_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir,
        n_folds=args.n_folds,
        random_seed=args.seed,
    )

    pipeline = TrainingPipeline(training_config=training_config)
    result = pipeline.run()

    print(f"\nBest model: {result.best_model} (Macro-F1: {result.best_f1:.3f})")
    print(f"Models saved to: {result.output_dir}")


if __name__ == "__main__":
    main()
