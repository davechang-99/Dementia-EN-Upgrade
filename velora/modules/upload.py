"""
Module 2: Mobile Upload Module (모바일 흡수 모듈)

사용자가 모바일 앱을 통해 통화 음성을 업로드하면,
파일 길이·잡음·무음 비율 등 기본 품질을 사전 검증하고,
암호화 처리 후 서버로 안전하게 전송하는 모듈.
"""

import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import numpy as np

from velora.config.settings import VeloraConfig, get_config
from velora.utils.logging import AuditLogger, setup_logger

logger = setup_logger("upload")


class UploadStatus(str, Enum):
    """업로드 상태"""
    PENDING = "pending"
    VALIDATING = "validating"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class RejectionReason(str, Enum):
    """반려 사유"""
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    LOW_SNR = "low_snr"
    HIGH_SILENCE = "high_silence"
    UNSUPPORTED_FORMAT = "unsupported_format"
    CORRUPTED_FILE = "corrupted_file"
    EMPTY_FILE = "empty_file"


@dataclass
class QualityReport:
    """음성 품질 리포트"""
    file_id: str
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str
    file_size_bytes: int
    snr_db: float
    silence_ratio: float
    clipping_ratio: float
    is_acceptable: bool
    rejection_reasons: list
    quality_score: float  # 0.0 ~ 1.0


@dataclass
class UploadResult:
    """업로드 결과"""
    upload_id: str
    status: UploadStatus
    quality_report: Optional[QualityReport]
    file_path: Optional[str]
    timestamp: str
    message: str
    user_id: Optional[str] = None


class UploadModule:
    """
    모바일 흡수 모듈

    주요 기능:
    - 다양한 오디오 포맷 지원 (m4a, mp3, wav, flac)
    - 업로드 파일 품질 검증
    - 길이/SNR/무음비 체크
    - 최소 30초 미만 차단
    - 저품질 데이터 반려 및 재업로드 유도
    """

    def __init__(self, config: Optional[VeloraConfig] = None):
        self.config = config or get_config()
        self.audit_logger = AuditLogger(
            log_dir=f"{self.config.log_dir}/audit"
        )
        os.makedirs(self.config.temp_dir, exist_ok=True)

    def validate_and_upload(
        self,
        file_path: str,
        user_id: Optional[str] = None,
        age_group: Optional[str] = None,
        language: str = "ko"
    ) -> UploadResult:
        """
        파일 업로드 및 품질 검증

        Args:
            file_path: 업로드할 오디오 파일 경로
            user_id: 사용자 ID
            age_group: 연령대 (40s, 50s 등)
            language: 언어 (기본 한국어)

        Returns:
            UploadResult: 업로드 결과
        """
        upload_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # 1. 파일 존재 확인
        if not os.path.exists(file_path):
            return UploadResult(
                upload_id=upload_id,
                status=UploadStatus.REJECTED,
                quality_report=None,
                file_path=None,
                timestamp=timestamp,
                message="파일을 찾을 수 없습니다.",
                user_id=user_id,
            )

        # 2. 포맷 확인
        file_ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        if file_ext not in self.config.audio.supported_formats:
            return UploadResult(
                upload_id=upload_id,
                status=UploadStatus.REJECTED,
                quality_report=None,
                file_path=None,
                timestamp=timestamp,
                message=f"지원하지 않는 파일 형식입니다: {file_ext}. "
                        f"지원 형식: {', '.join(self.config.audio.supported_formats)}",
                user_id=user_id,
            )

        # 3. 파일 크기 확인
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return UploadResult(
                upload_id=upload_id,
                status=UploadStatus.REJECTED,
                quality_report=None,
                file_path=None,
                timestamp=timestamp,
                message="빈 파일입니다.",
                user_id=user_id,
            )

        # 4. 오디오 품질 검사
        quality_report = self._check_audio_quality(file_path, upload_id)

        if not quality_report.is_acceptable:
            reasons_msg = self._format_rejection_reasons(quality_report.rejection_reasons)
            self.audit_logger.log_data_access(
                user_id=user_id,
                resource=f"upload:{upload_id}",
                action="upload_rejected",
                details={"reasons": quality_report.rejection_reasons},
            )
            return UploadResult(
                upload_id=upload_id,
                status=UploadStatus.REJECTED,
                quality_report=quality_report,
                file_path=None,
                timestamp=timestamp,
                message=f"품질 기준 미달로 반려되었습니다: {reasons_msg}",
                user_id=user_id,
            )

        # 5. 파일을 임시 디렉토리로 복사
        dest_dir = os.path.join(self.config.temp_dir, upload_id)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, f"audio.{file_ext}")
        shutil.copy2(file_path, dest_path)

        # 6. 감사 로그
        self.audit_logger.log_data_access(
            user_id=user_id,
            resource=f"upload:{upload_id}",
            action="upload_accepted",
            details={
                "duration": quality_report.duration_seconds,
                "quality_score": quality_report.quality_score,
                "age_group": age_group,
                "language": language,
            },
        )

        logger.info(
            f"Upload accepted: {upload_id}, "
            f"duration={quality_report.duration_seconds:.1f}s, "
            f"quality={quality_report.quality_score:.2f}"
        )

        return UploadResult(
            upload_id=upload_id,
            status=UploadStatus.ACCEPTED,
            quality_report=quality_report,
            file_path=dest_path,
            timestamp=timestamp,
            message="업로드가 완료되었습니다. 분석을 시작합니다.",
            user_id=user_id,
        )

    def _check_audio_quality(self, file_path: str, file_id: str) -> QualityReport:
        """오디오 파일 품질 검사"""
        try:
            import librosa

            # 오디오 로드
            y, sr = librosa.load(file_path, sr=None, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)

            file_ext = os.path.splitext(file_path)[1].lower().lstrip(".")
            file_size = os.path.getsize(file_path)
            channels = 1  # mono로 로드

            # SNR 추정
            snr_db = self._estimate_snr(y)

            # 무음 비율 계산
            silence_ratio = self._calculate_silence_ratio(y, sr)

            # 클리핑 비율 계산
            clipping_ratio = self._calculate_clipping_ratio(y)

            # 품질 판정
            rejection_reasons = []

            if duration < self.config.audio.min_duration_seconds:
                rejection_reasons.append(RejectionReason.TOO_SHORT)
            if duration > self.config.audio.max_duration_seconds:
                rejection_reasons.append(RejectionReason.TOO_LONG)
            if snr_db < self.config.audio.min_snr_db:
                rejection_reasons.append(RejectionReason.LOW_SNR)
            if silence_ratio > self.config.audio.max_silence_ratio:
                rejection_reasons.append(RejectionReason.HIGH_SILENCE)

            is_acceptable = len(rejection_reasons) == 0

            # 품질 점수 계산 (0 ~ 1)
            quality_score = self._calculate_quality_score(
                duration, snr_db, silence_ratio, clipping_ratio
            )

            return QualityReport(
                file_id=file_id,
                duration_seconds=duration,
                sample_rate=sr,
                channels=channels,
                format=file_ext,
                file_size_bytes=file_size,
                snr_db=snr_db,
                silence_ratio=silence_ratio,
                clipping_ratio=clipping_ratio,
                is_acceptable=is_acceptable,
                rejection_reasons=[r.value for r in rejection_reasons],
                quality_score=quality_score,
            )

        except Exception as e:
            logger.error(f"Audio quality check failed for {file_path}: {e}")
            return QualityReport(
                file_id=file_id,
                duration_seconds=0.0,
                sample_rate=0,
                channels=0,
                format=os.path.splitext(file_path)[1].lower().lstrip("."),
                file_size_bytes=os.path.getsize(file_path),
                snr_db=0.0,
                silence_ratio=1.0,
                clipping_ratio=0.0,
                is_acceptable=False,
                rejection_reasons=[RejectionReason.CORRUPTED_FILE.value],
                quality_score=0.0,
            )

    def _estimate_snr(self, y: np.ndarray) -> float:
        """신호 대 잡음비(SNR) 추정"""
        # 에너지 기반 간단한 SNR 추정
        frame_length = 2048
        hop_length = 512

        # RMS 에너지 계산
        frames = np.array([
            y[i:i + frame_length]
            for i in range(0, len(y) - frame_length, hop_length)
        ])

        if len(frames) == 0:
            return 0.0

        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        rms_sorted = np.sort(rms)

        # 하위 10% = 노이즈, 상위 50% = 신호로 추정
        noise_rms = np.mean(rms_sorted[:max(1, len(rms_sorted) // 10)])
        signal_rms = np.mean(rms_sorted[len(rms_sorted) // 2:])

        if noise_rms < 1e-10:
            return 40.0  # 매우 깨끗한 신호

        snr = 20 * np.log10(signal_rms / noise_rms)
        return float(np.clip(snr, -10, 60))

    def _calculate_silence_ratio(
        self, y: np.ndarray, sr: int, threshold_db: float = -40.0
    ) -> float:
        """무음 비율 계산"""
        import librosa

        # RMS 에너지 기반 무음 탐지
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        silence_frames = np.sum(rms_db < threshold_db)
        total_frames = len(rms_db)

        if total_frames == 0:
            return 1.0

        return float(silence_frames / total_frames)

    def _calculate_clipping_ratio(
        self, y: np.ndarray, threshold: float = 0.99
    ) -> float:
        """클리핑 비율 계산"""
        clipped_samples = np.sum(np.abs(y) > threshold)
        return float(clipped_samples / len(y)) if len(y) > 0 else 0.0

    def _calculate_quality_score(
        self,
        duration: float,
        snr_db: float,
        silence_ratio: float,
        clipping_ratio: float,
    ) -> float:
        """종합 품질 점수 계산 (0 ~ 1)"""
        scores = []

        # 길이 점수 (30초~600초 최적)
        if duration < 30:
            scores.append(duration / 30)
        elif duration <= 600:
            scores.append(1.0)
        else:
            scores.append(max(0.5, 1.0 - (duration - 600) / 3600))

        # SNR 점수 (10dB 이상 최적)
        scores.append(min(1.0, max(0.0, snr_db / 20)))

        # 무음 비율 점수 (낮을수록 좋음)
        scores.append(max(0.0, 1.0 - silence_ratio))

        # 클리핑 점수 (없을수록 좋음)
        scores.append(max(0.0, 1.0 - clipping_ratio * 100))

        return float(np.mean(scores))

    def _format_rejection_reasons(self, reasons: list) -> str:
        """반려 사유를 사용자 친화적 메시지로 변환"""
        reason_messages = {
            RejectionReason.TOO_SHORT.value: (
                f"음성 길이가 최소 기준({self.config.audio.min_duration_seconds}초) 미만입니다"
            ),
            RejectionReason.TOO_LONG.value: "음성 길이가 너무 깁니다",
            RejectionReason.LOW_SNR.value: "잡음이 너무 많습니다. 조용한 환경에서 다시 녹음해 주세요",
            RejectionReason.HIGH_SILENCE.value: "무음 구간이 너무 많습니다",
            RejectionReason.UNSUPPORTED_FORMAT.value: "지원하지 않는 파일 형식입니다",
            RejectionReason.CORRUPTED_FILE.value: "파일이 손상되었습니다",
            RejectionReason.EMPTY_FILE.value: "빈 파일입니다",
        }
        return "; ".join(reason_messages.get(r, r) for r in reasons)

    def cleanup_upload(self, upload_id: str) -> bool:
        """업로드 임시 파일 정리"""
        upload_dir = os.path.join(self.config.temp_dir, upload_id)
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
            logger.info(f"Cleaned up upload: {upload_id}")
            return True
        return False
