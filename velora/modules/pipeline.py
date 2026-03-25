"""
VELORA End-to-End Pipeline (통합 파이프라인)

음성 업로드부터 결과 산출까지의 전체 분석 흐름을 통합 관리.
각 모듈을 순차적으로 호출하여 완전한 분석 파이프라인을 구성.
"""

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from velora.config.settings import VeloraConfig, get_config
from velora.modules.governance import ConsentType, DataGovernanceModule
from velora.modules.upload import UploadModule, UploadResult, UploadStatus
from velora.modules.speaker import SpeakerProcessingModule, DiarizationResult
from velora.modules.preprocessing import PreprocessingModule, PreprocessingResult
from velora.modules.features import FeatureEngine, FeatureExtractionResult
from velora.modules.model_layer import EnsembleModel, RiskAssessment
from velora.modules.results import AnalysisReport, ResultsModule
from velora.utils.logging import AuditLogger, setup_logger

logger = setup_logger("pipeline")


@dataclass
class PipelineResult:
    """파이프라인 전체 결과"""
    analysis_id: str
    status: str
    upload_result: Optional[UploadResult] = None
    diarization_result: Optional[DiarizationResult] = None
    preprocessing_result: Optional[PreprocessingResult] = None
    feature_result: Optional[FeatureExtractionResult] = None
    risk_assessment: Optional[RiskAssessment] = None
    report: Optional[AnalysisReport] = None
    report_display: Optional[Dict] = None
    error_message: Optional[str] = None
    processing_time_seconds: float = 0.0


class VeloraPipeline:
    """
    VELORA 통합 분석 파이프라인

    처리 흐름:
    1. 동의 검증
    2. 파일 업로드 및 품질 체크
    3. 화자 분리 및 대상자 식별
    4. 전처리 (표준화/VAD/발화 통계)
    5. 특징 추출
    6. 모델 추론 (위험 점수/신뢰도)
    7. 결과 생성 및 안내
    """

    def __init__(self, config: Optional[VeloraConfig] = None):
        self.config = config or get_config()

        # 모듈 초기화
        self.governance = DataGovernanceModule(config)
        self.upload = UploadModule(config)
        self.speaker = SpeakerProcessingModule(config)
        self.preprocessing = PreprocessingModule(config)
        self.features = FeatureEngine(config)
        self.model = EnsembleModel(config)
        self.results = ResultsModule(config)
        self.audit_logger = AuditLogger(
            log_dir=f"{self.config.log_dir}/audit"
        )

    def analyze(
        self,
        audio_path: str,
        user_id: Optional[str] = None,
        consent_token: Optional[str] = None,
        speaker_sample_path: Optional[str] = None,
        transcript: Optional[str] = None,
        age_group: Optional[str] = None,
    ) -> PipelineResult:
        """
        전체 분석 파이프라인 실행

        Args:
            audio_path: 음성 파일 경로
            user_id: 사용자 ID
            consent_token: 동의 토큰
            speaker_sample_path: 본인 음성 샘플 경로 (선택)
            transcript: 전사 텍스트 (선택)
            age_group: 연령대 (선택)

        Returns:
            PipelineResult: 전체 분석 결과
        """
        import time

        analysis_id = str(uuid.uuid4())
        start_time = time.time()

        logger.info(f"Pipeline started: {analysis_id}")
        self.audit_logger.log_analysis(user_id, analysis_id, "started")

        try:
            # 1단계: 동의 검증
            if consent_token:
                if not self.governance.validate_consent_token(consent_token):
                    return PipelineResult(
                        analysis_id=analysis_id,
                        status="error",
                        error_message="유효하지 않은 동의 토큰입니다. 동의 절차를 다시 진행해 주세요.",
                    )
            logger.info(f"[{analysis_id}] Step 1: Consent validated")

            # 2단계: 업로드 및 품질 체크
            upload_result = self.upload.validate_and_upload(
                audio_path, user_id, age_group
            )
            if upload_result.status != UploadStatus.ACCEPTED:
                return PipelineResult(
                    analysis_id=analysis_id,
                    status="rejected",
                    upload_result=upload_result,
                    error_message=upload_result.message,
                )
            logger.info(f"[{analysis_id}] Step 2: Upload accepted")
            self.audit_logger.log_analysis(user_id, analysis_id, "uploaded")

            # 3단계: 화자 분리 및 대상자 식별
            working_audio = upload_result.file_path

            # 화자 샘플 등록 (제공된 경우)
            if speaker_sample_path and user_id:
                try:
                    self.speaker.register_speaker_sample(user_id, speaker_sample_path)
                except Exception as e:
                    logger.warning(f"Speaker sample registration failed: {e}")

            diarization_result = self.speaker.process_audio(
                working_audio, user_id
            )

            # 대상자 음성 추출
            target_audio_dir = os.path.join(
                self.config.temp_dir, analysis_id
            )
            os.makedirs(target_audio_dir, exist_ok=True)
            target_audio_path = os.path.join(target_audio_dir, "target_audio.wav")
            self.speaker.extract_target_audio(
                working_audio, diarization_result, target_audio_path
            )
            logger.info(f"[{analysis_id}] Step 3: Speaker diarization complete")
            self.audit_logger.log_analysis(user_id, analysis_id, "diarized")

            # 4단계: 전처리
            preprocess_dir = os.path.join(target_audio_dir, "preprocessed")
            preprocessing_result = self.preprocessing.process(
                target_audio_path, preprocess_dir, "target"
            )

            if not preprocessing_result.quality_flags.get("is_usable", True):
                warnings = preprocessing_result.quality_flags.get("warnings", [])
                logger.warning(f"[{analysis_id}] Quality issues: {warnings}")

            logger.info(f"[{analysis_id}] Step 4: Preprocessing complete")
            self.audit_logger.log_analysis(user_id, analysis_id, "preprocessed")

            # 5단계: 특징 추출
            feature_result = self.features.extract_features(
                preprocessing_result.output_path,
                speech_segments=preprocessing_result.speech_segments,
                utterance_stats=preprocessing_result.statistics,
                transcript=transcript,
                include_spectrogram=True,
            )
            logger.info(
                f"[{analysis_id}] Step 5: Feature extraction complete "
                f"({feature_result.quality_meta.total_features} features)"
            )
            self.audit_logger.log_analysis(user_id, analysis_id, "features_extracted")

            # 6단계: 모델 추론
            audio_quality = (
                upload_result.quality_report.quality_score
                if upload_result.quality_report
                else 0.5
            )

            risk_assessment = self.model.predict_risk(
                feature_vector=feature_result.feature_vector,
                feature_names=feature_result.feature_names,
                audio_quality_score=audio_quality,
                diarization_confidence=diarization_result.diarization_confidence,
                spectrogram=feature_result.spectrogram_image,
            )
            logger.info(
                f"[{analysis_id}] Step 6: Risk assessment complete "
                f"(score={risk_assessment.risk_score:.3f}, "
                f"level={risk_assessment.risk_level})"
            )
            self.audit_logger.log_analysis(
                user_id, analysis_id, "risk_assessed",
                result={"risk_level": risk_assessment.risk_level}
            )

            # 7단계: 결과 생성
            report = self.results.generate_report(
                risk_assessment, user_id, feature_result.feature_names
            )
            report_display = self.results.format_report_for_display(report)

            processing_time = time.time() - start_time
            logger.info(
                f"[{analysis_id}] Pipeline complete in {processing_time:.1f}s"
            )
            self.audit_logger.log_analysis(
                user_id, analysis_id, "completed",
                result={"processing_time": processing_time}
            )

            # 임시 파일 정리 (원본 음성 삭제 - 개인정보 보호)
            self._cleanup(upload_result.upload_id, analysis_id)

            return PipelineResult(
                analysis_id=analysis_id,
                status="completed",
                upload_result=upload_result,
                diarization_result=diarization_result,
                preprocessing_result=preprocessing_result,
                feature_result=feature_result,
                risk_assessment=risk_assessment,
                report=report,
                report_display=report_display,
                processing_time_seconds=processing_time,
            )

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"[{analysis_id}] Pipeline failed: {e}", exc_info=True)
            self.audit_logger.log_analysis(
                user_id, analysis_id, "failed",
                result={"error": str(e)}
            )
            return PipelineResult(
                analysis_id=analysis_id,
                status="error",
                error_message=f"분석 중 오류가 발생했습니다: {str(e)}",
                processing_time_seconds=processing_time,
            )

    def _cleanup(self, upload_id: str, analysis_id: str) -> None:
        """임시 파일 정리 (원본 음성 삭제)"""
        import shutil

        try:
            self.upload.cleanup_upload(upload_id)
            temp_dir = os.path.join(self.config.temp_dir, analysis_id)
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            logger.info(f"Cleanup complete: upload={upload_id}, analysis={analysis_id}")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
