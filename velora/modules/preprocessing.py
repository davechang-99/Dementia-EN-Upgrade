"""
Module 4: Preprocessing Module (전처리 모듈)

서로 다른 기기 및 환경에서 수집된 음성 데이터를 표준 샘플레이트로 통일하고,
무음 구간(VAD)과 발화 통계 정보를 산출하여 모델 입력에 적합한 형태로
정규화하는 모듈.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from velora.config.settings import VeloraConfig, get_config
from velora.utils.logging import setup_logger

logger = setup_logger("preprocessing")


@dataclass
class SpeechSegment:
    """발화 구간"""
    start_time: float
    end_time: float
    energy_db: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class UtteranceStatistics:
    """발화 통계 정보"""
    total_duration: float
    speech_duration: float
    silence_duration: float
    silence_ratio: float
    speech_density: float          # 발화 밀도 (발화 구간 수 / 총 시간)
    avg_utterance_length: float    # 평균 발화 길이 (초)
    max_utterance_length: float    # 최대 발화 길이
    min_utterance_length: float    # 최소 발화 길이
    num_utterances: int            # 발화 구간 수
    num_pauses: int                # 쉼 구간 수
    avg_pause_length: float        # 평균 쉼 길이 (초)
    speech_rate_estimate: float    # 추정 발화 속도 (구간/분)


@dataclass
class PreprocessingResult:
    """전처리 결과"""
    output_path: str
    sample_rate: int
    channels: int
    duration: float
    speech_segments: List[SpeechSegment]
    statistics: UtteranceStatistics
    quality_flags: dict


class PreprocessingModule:
    """
    전처리 모듈

    주요 기능:
    - 오디오 표준화: 16kHz, mono WAV 변환
    - VAD (Voice Activity Detection): 발화/무음 구간 분리
    - 발화 통계 산출: 무음비, 발화 밀도, 평균 발화 길이
    - 품질 필터링: 저품질 구간 제외/가중치 조정
    - 음압 레벨 정규화
    """

    def __init__(self, config: Optional[VeloraConfig] = None):
        self.config = config or get_config()

    def process(
        self,
        audio_path: str,
        output_dir: str,
        file_id: str = "audio",
    ) -> PreprocessingResult:
        """
        오디오 전처리 파이프라인

        Args:
            audio_path: 입력 오디오 파일 경로
            output_dir: 출력 디렉토리
            file_id: 파일 식별자

        Returns:
            PreprocessingResult: 전처리 결과
        """
        import librosa
        import soundfile as sf

        os.makedirs(output_dir, exist_ok=True)

        # 1. 오디오 로드 및 표준화
        y, sr = self._load_and_standardize(audio_path)
        duration = len(y) / sr

        logger.info(f"Audio loaded: duration={duration:.1f}s, sr={sr}")

        # 2. 음압 정규화
        y = self._normalize_amplitude(y)

        # 3. 프리엠퍼시스 필터
        y_filtered = self._apply_preemphasis(y)

        # 4. VAD (발화 구간 탐지)
        speech_segments = self._detect_speech_segments(y_filtered, sr)

        # 5. 발화 통계 계산
        statistics = self._compute_utterance_statistics(
            speech_segments, duration
        )

        # 6. 품질 플래그
        quality_flags = self._assess_quality(statistics, duration)

        # 7. 표준 WAV 저장
        output_path = os.path.join(output_dir, f"{file_id}_preprocessed.wav")
        sf.write(output_path, y_filtered, sr)

        logger.info(
            f"Preprocessing complete: "
            f"utterances={statistics.num_utterances}, "
            f"silence_ratio={statistics.silence_ratio:.2f}, "
            f"speech_density={statistics.speech_density:.2f}"
        )

        return PreprocessingResult(
            output_path=output_path,
            sample_rate=sr,
            channels=1,
            duration=duration,
            speech_segments=speech_segments,
            statistics=statistics,
            quality_flags=quality_flags,
        )

    def _load_and_standardize(
        self, audio_path: str
    ) -> Tuple[np.ndarray, int]:
        """오디오 로드 및 표준화 (16kHz, mono)"""
        import librosa

        target_sr = self.config.audio.target_sample_rate
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

        return y, target_sr

    def _normalize_amplitude(self, y: np.ndarray) -> np.ndarray:
        """음압 레벨 정규화 (peak normalization)"""
        peak = np.max(np.abs(y))
        if peak > 0:
            y = y / peak * 0.95  # 약간의 헤드룸
        return y

    def _apply_preemphasis(
        self, y: np.ndarray, coeff: float = 0.97
    ) -> np.ndarray:
        """프리엠퍼시스 필터 적용"""
        return np.append(y[0], y[1:] - coeff * y[:-1])

    def _detect_speech_segments(
        self,
        y: np.ndarray,
        sr: int,
        frame_length: int = 2048,
        hop_length: int = 512,
        energy_threshold_db: float = -35.0,
        min_speech_duration: float = 0.3,
        min_silence_duration: float = 0.2,
    ) -> List[SpeechSegment]:
        """
        VAD 기반 발화 구간 탐지

        에너지 기반 VAD + 후처리로 안정적인 발화 구간 탐지
        """
        import librosa

        # RMS 에너지 계산
        rms = librosa.feature.rms(
            y=y, frame_length=frame_length, hop_length=hop_length
        )[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # 프레임별 발화 여부
        is_speech = rms_db > energy_threshold_db

        # 후처리: 짧은 무음 제거 (연결)
        frame_duration = hop_length / sr
        min_silence_frames = int(min_silence_duration / frame_duration)
        min_speech_frames = int(min_speech_duration / frame_duration)

        # 짧은 무음 구간 채우기
        is_speech_smoothed = is_speech.copy()
        silence_count = 0
        for i in range(len(is_speech_smoothed)):
            if not is_speech_smoothed[i]:
                silence_count += 1
            else:
                if 0 < silence_count < min_silence_frames:
                    is_speech_smoothed[i - silence_count:i] = True
                silence_count = 0

        # 발화 구간 추출
        segments = []
        in_speech = False
        start_frame = 0

        for i in range(len(is_speech_smoothed)):
            if is_speech_smoothed[i] and not in_speech:
                start_frame = i
                in_speech = True
            elif not is_speech_smoothed[i] and in_speech:
                # 최소 발화 길이 체크
                if (i - start_frame) >= min_speech_frames:
                    start_time = start_frame * frame_duration
                    end_time = i * frame_duration
                    energy = float(np.mean(rms_db[start_frame:i]))

                    segments.append(
                        SpeechSegment(
                            start_time=start_time,
                            end_time=end_time,
                            energy_db=energy,
                        )
                    )
                in_speech = False

        # 마지막 구간
        if in_speech and (len(is_speech_smoothed) - start_frame) >= min_speech_frames:
            segments.append(
                SpeechSegment(
                    start_time=start_frame * frame_duration,
                    end_time=len(is_speech_smoothed) * frame_duration,
                    energy_db=float(np.mean(rms_db[start_frame:])),
                )
            )

        logger.info(f"VAD detected {len(segments)} speech segments")
        return segments

    def _compute_utterance_statistics(
        self,
        segments: List[SpeechSegment],
        total_duration: float,
    ) -> UtteranceStatistics:
        """발화 통계 계산"""
        if not segments:
            return UtteranceStatistics(
                total_duration=total_duration,
                speech_duration=0.0,
                silence_duration=total_duration,
                silence_ratio=1.0,
                speech_density=0.0,
                avg_utterance_length=0.0,
                max_utterance_length=0.0,
                min_utterance_length=0.0,
                num_utterances=0,
                num_pauses=0,
                avg_pause_length=total_duration,
                speech_rate_estimate=0.0,
            )

        # 발화 시간 합계
        utterance_lengths = [s.duration for s in segments]
        speech_duration = sum(utterance_lengths)
        silence_duration = total_duration - speech_duration

        # 쉼 구간 계산
        pauses = []
        for i in range(1, len(segments)):
            pause = segments[i].start_time - segments[i - 1].end_time
            if pause > 0:
                pauses.append(pause)

        # 처음과 끝의 무음도 포함
        if segments[0].start_time > 0:
            pauses.insert(0, segments[0].start_time)
        trailing = total_duration - segments[-1].end_time
        if trailing > 0:
            pauses.append(trailing)

        total_minutes = total_duration / 60.0

        return UtteranceStatistics(
            total_duration=total_duration,
            speech_duration=speech_duration,
            silence_duration=silence_duration,
            silence_ratio=silence_duration / total_duration if total_duration > 0 else 1.0,
            speech_density=len(segments) / total_minutes if total_minutes > 0 else 0.0,
            avg_utterance_length=float(np.mean(utterance_lengths)),
            max_utterance_length=float(np.max(utterance_lengths)),
            min_utterance_length=float(np.min(utterance_lengths)),
            num_utterances=len(segments),
            num_pauses=len(pauses),
            avg_pause_length=float(np.mean(pauses)) if pauses else 0.0,
            speech_rate_estimate=len(segments) / total_minutes if total_minutes > 0 else 0.0,
        )

    def _assess_quality(
        self,
        stats: UtteranceStatistics,
        duration: float,
    ) -> dict:
        """전처리 후 품질 평가 플래그"""
        flags = {
            "is_usable": True,
            "warnings": [],
        }

        # 무음 비율이 너무 높은 경우
        if stats.silence_ratio > 0.8:
            flags["warnings"].append("무음 비율이 80%를 초과합니다")
            flags["is_usable"] = False

        # 발화 구간이 너무 적은 경우
        if stats.num_utterances < 3:
            flags["warnings"].append("발화 구간이 3개 미만입니다")

        # 평균 발화 길이가 너무 짧은 경우
        if stats.avg_utterance_length < 0.5 and stats.num_utterances > 0:
            flags["warnings"].append("평균 발화 길이가 0.5초 미만입니다")

        # 총 발화 시간이 너무 짧은 경우
        if stats.speech_duration < 10.0:
            flags["warnings"].append("총 발화 시간이 10초 미만입니다")
            flags["is_usable"] = False

        flags["quality_level"] = self._determine_quality_level(stats)

        return flags

    def _determine_quality_level(self, stats: UtteranceStatistics) -> str:
        """품질 수준 판정"""
        if stats.silence_ratio > 0.8 or stats.speech_duration < 10:
            return "low"
        elif stats.silence_ratio > 0.6 or stats.num_utterances < 5:
            return "medium"
        else:
            return "high"

    def extract_speech_only(
        self,
        audio_path: str,
        segments: List[SpeechSegment],
        output_path: str,
    ) -> str:
        """발화 구간만 추출하여 새 파일로 저장"""
        import librosa
        import soundfile as sf

        y, sr = librosa.load(
            audio_path,
            sr=self.config.audio.target_sample_rate,
            mono=True,
        )

        speech_parts = []
        silence_gap = np.zeros(int(0.05 * sr))  # 50ms 간격

        for seg in segments:
            start = int(seg.start_time * sr)
            end = int(seg.end_time * sr)
            speech_parts.append(y[start:end])
            speech_parts.append(silence_gap)

        if speech_parts:
            speech_audio = np.concatenate(speech_parts)
        else:
            speech_audio = y

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, speech_audio, sr)

        return output_path
