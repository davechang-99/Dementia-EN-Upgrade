"""
VELORA System Configuration
시스템 전역 설정 및 상수 정의
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class AudioConfig:
    """오디오 처리 설정"""
    target_sample_rate: int = 16000
    target_channels: int = 1  # mono
    supported_formats: List[str] = field(
        default_factory=lambda: ["wav", "mp3", "m4a", "flac", "ogg"]
    )
    min_duration_seconds: float = 30.0
    max_duration_seconds: float = 7200.0  # 2 hours
    max_silence_ratio: float = 0.8
    min_snr_db: float = 5.0
    vad_aggressiveness: int = 2  # 0-3, higher = more aggressive


@dataclass
class SpeakerConfig:
    """화자 처리 설정"""
    min_speaker_sample_seconds: float = 5.0
    max_speaker_sample_seconds: float = 10.0
    diarization_min_speakers: int = 1
    diarization_max_speakers: int = 4
    speaker_similarity_threshold: float = 0.7


@dataclass
class FeatureConfig:
    """특징 추출 설정"""
    n_mfcc: int = 13
    n_mels: int = 128
    hop_length: int = 512
    n_fft: int = 2048
    fmax: int = 8000
    use_linguistic_features: bool = False  # 선택적 활성화
    vit_model_name: str = "google/vit-base-patch16-224"
    vit_target_size: tuple = (224, 224)
    pca_n_components: int = 50


@dataclass
class ModelConfig:
    """모델 학습/추론 설정"""
    n_folds: int = 5
    random_seed: int = 42
    test_size: float = 0.2
    risk_thresholds: dict = field(
        default_factory=lambda: {
            "low": 0.3,       # 낮음 (0 ~ 0.3)
            "caution": 0.6,   # 주의 (0.3 ~ 0.6)
            "high": 1.0       # 높음 (0.6 ~ 1.0)
        }
    )
    confidence_weights: dict = field(
        default_factory=lambda: {
            "audio_quality": 0.3,
            "diarization_confidence": 0.3,
            "model_uncertainty": 0.4
        }
    )


@dataclass
class SecurityConfig:
    """보안 설정"""
    encryption_algorithm: str = "AES-256"
    pii_patterns: List[str] = field(
        default_factory=lambda: [
            r"\d{6}[-]\d{7}",           # 주민등록번호
            r"\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4}",  # 전화번호
            r"\d{5}",                    # 우편번호
        ]
    )
    data_retention_days: int = 30
    enable_audit_log: bool = True


@dataclass
class VeloraConfig:
    """VELORA 전체 시스템 설정"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    speaker: SpeakerConfig = field(default_factory=SpeakerConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    # 경로 설정
    data_dir: str = os.environ.get("VELORA_DATA_DIR", "./data")
    model_dir: str = os.environ.get("VELORA_MODEL_DIR", "./models/saved")
    log_dir: str = os.environ.get("VELORA_LOG_DIR", "./logs")
    temp_dir: str = os.environ.get("VELORA_TEMP_DIR", "./temp")

    # 서비스 설정
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # 면책 고지
    disclaimer: str = (
        "본 서비스는 의료적 진단이나 치료 판단을 목적으로 하지 않습니다. "
        "분석 결과는 인지기능 변화와 연관될 수 있는 위험 신호를 참고용으로 "
        "제공하는 비의료적 정보이며, 전문 의료인의 진단을 대체하지 않습니다."
    )


def get_config() -> VeloraConfig:
    """시스템 설정 인스턴스 반환"""
    return VeloraConfig()
