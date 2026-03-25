"""
Module 3: Speaker Processing Module (화자 처리 모듈)

통화 음성에 혼재된 발화자 중 분석 대상자 중심의 음성 구간을
분리·정제하여, 분석에 적합한 음성만을 선별하는 모듈.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from velora.config.settings import VeloraConfig, get_config
from velora.utils.logging import setup_logger

logger = setup_logger("speaker")


@dataclass
class SpeakerSegment:
    """화자 발화 구간"""
    speaker_id: str
    start_time: float
    end_time: float
    confidence: float

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class DiarizationResult:
    """화자 분리 결과"""
    segments: List[SpeakerSegment]
    num_speakers: int
    total_duration: float
    target_speaker_id: Optional[str]
    target_segments: List[SpeakerSegment]
    target_duration: float
    diarization_confidence: float
    excluded_segments: List[SpeakerSegment]


@dataclass
class SpeakerProfile:
    """화자 프로필 (음성 샘플 기반)"""
    user_id: str
    embedding: Optional[np.ndarray] = None
    sample_duration: float = 0.0
    created_at: str = ""


class SpeakerProcessingModule:
    """
    화자 처리 모듈

    주요 기능:
    - Speaker Diarization (화자 분리): 타임스탬프 단위 화자 구간 분리
    - Target Speaker Identification (대상자 식별): 본인 음성 샘플 매칭
    - Domain Adaptation: 통화 환경 강건성 확보
    """

    def __init__(self, config: Optional[VeloraConfig] = None):
        self.config = config or get_config()
        self._speaker_profiles: Dict[str, SpeakerProfile] = {}
        self._diarization_pipeline = None

    def _init_diarization_pipeline(self):
        """화자 분리 파이프라인 초기화 (lazy loading)"""
        if self._diarization_pipeline is not None:
            return

        try:
            from pyannote.audio import Pipeline
            self._diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1"
            )
            logger.info("Pyannote diarization pipeline loaded")
        except ImportError:
            logger.warning(
                "pyannote.audio not available. Using energy-based fallback diarization."
            )
            self._diarization_pipeline = None
        except Exception as e:
            logger.warning(f"Failed to load pyannote pipeline: {e}. Using fallback.")
            self._diarization_pipeline = None

    def register_speaker_sample(
        self,
        user_id: str,
        sample_path: str,
    ) -> SpeakerProfile:
        """
        본인 음성 샘플 등록 (5~10초)

        Args:
            user_id: 사용자 ID
            sample_path: 음성 샘플 파일 경로

        Returns:
            SpeakerProfile: 등록된 화자 프로필
        """
        import librosa
        from datetime import datetime, timezone

        y, sr = librosa.load(sample_path, sr=16000, mono=True)
        duration = len(y) / sr

        if duration < self.config.speaker.min_speaker_sample_seconds:
            raise ValueError(
                f"음성 샘플이 너무 짧습니다. "
                f"최소 {self.config.speaker.min_speaker_sample_seconds}초 이상 필요합니다."
            )

        # 음성 임베딩 추출
        embedding = self._extract_speaker_embedding(y, sr)

        profile = SpeakerProfile(
            user_id=user_id,
            embedding=embedding,
            sample_duration=duration,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self._speaker_profiles[user_id] = profile
        logger.info(f"Speaker profile registered: {user_id}, duration={duration:.1f}s")

        return profile

    def process_audio(
        self,
        audio_path: str,
        user_id: Optional[str] = None,
    ) -> DiarizationResult:
        """
        통화 음성 화자 분리 및 대상자 식별

        Args:
            audio_path: 통화 음성 파일 경로
            user_id: 대상자 사용자 ID (프로필 등록된 경우)

        Returns:
            DiarizationResult: 화자 분리 결과
        """
        import librosa

        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        total_duration = len(y) / sr

        # 화자 분리 수행
        segments = self._perform_diarization(y, sr, audio_path)

        if not segments:
            # 화자 분리 실패 시 전체를 단일 화자로 처리
            segments = [
                SpeakerSegment(
                    speaker_id="speaker_0",
                    start_time=0.0,
                    end_time=total_duration,
                    confidence=0.5,
                )
            ]

        # 고유 화자 수
        unique_speakers = set(s.speaker_id for s in segments)
        num_speakers = len(unique_speakers)

        # 대상자 식별
        target_speaker_id = self._identify_target_speaker(
            y, sr, segments, user_id
        )

        # 대상자 구간과 제외 구간 분리
        target_segments = [
            s for s in segments if s.speaker_id == target_speaker_id
        ]
        excluded_segments = [
            s for s in segments if s.speaker_id != target_speaker_id
        ]

        target_duration = sum(s.duration for s in target_segments)

        # 전체 신뢰도 계산
        if target_segments:
            diarization_confidence = float(
                np.mean([s.confidence for s in target_segments])
            )
        else:
            diarization_confidence = 0.0

        logger.info(
            f"Diarization complete: {num_speakers} speakers, "
            f"target={target_speaker_id}, "
            f"target_duration={target_duration:.1f}s/{total_duration:.1f}s"
        )

        return DiarizationResult(
            segments=segments,
            num_speakers=num_speakers,
            total_duration=total_duration,
            target_speaker_id=target_speaker_id,
            target_segments=target_segments,
            target_duration=target_duration,
            diarization_confidence=diarization_confidence,
            excluded_segments=excluded_segments,
        )

    def extract_target_audio(
        self,
        audio_path: str,
        diarization_result: DiarizationResult,
        output_path: str,
    ) -> str:
        """
        대상자 음성만 추출하여 새 파일로 저장

        Args:
            audio_path: 원본 오디오 파일 경로
            diarization_result: 화자 분리 결과
            output_path: 출력 파일 경로

        Returns:
            str: 추출된 오디오 파일 경로
        """
        import librosa
        import soundfile as sf

        y, sr = librosa.load(audio_path, sr=16000, mono=True)

        # 대상자 구간만 추출
        target_audio_parts = []
        for segment in diarization_result.target_segments:
            start_sample = int(segment.start_time * sr)
            end_sample = int(segment.end_time * sr)
            target_audio_parts.append(y[start_sample:end_sample])

        if target_audio_parts:
            # 구간 간 짧은 무음 삽입하여 연결
            silence = np.zeros(int(0.1 * sr))  # 100ms 무음
            combined = []
            for i, part in enumerate(target_audio_parts):
                combined.append(part)
                if i < len(target_audio_parts) - 1:
                    combined.append(silence)
            target_audio = np.concatenate(combined)
        else:
            target_audio = y  # fallback: 전체 오디오

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, target_audio, sr)

        logger.info(f"Target audio extracted: {output_path}, duration={len(target_audio)/sr:.1f}s")
        return output_path

    def _perform_diarization(
        self,
        y: np.ndarray,
        sr: int,
        audio_path: str,
    ) -> List[SpeakerSegment]:
        """화자 분리 수행"""
        self._init_diarization_pipeline()

        if self._diarization_pipeline is not None:
            return self._pyannote_diarization(audio_path)
        else:
            return self._energy_based_diarization(y, sr)

    def _pyannote_diarization(self, audio_path: str) -> List[SpeakerSegment]:
        """pyannote 기반 화자 분리"""
        try:
            diarization = self._diarization_pipeline(
                audio_path,
                min_speakers=self.config.speaker.diarization_min_speakers,
                max_speakers=self.config.speaker.diarization_max_speakers,
            )

            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(
                    SpeakerSegment(
                        speaker_id=speaker,
                        start_time=turn.start,
                        end_time=turn.end,
                        confidence=0.85,
                    )
                )
            return segments

        except Exception as e:
            logger.error(f"Pyannote diarization failed: {e}")
            return []

    def _energy_based_diarization(
        self,
        y: np.ndarray,
        sr: int,
    ) -> List[SpeakerSegment]:
        """에너지 기반 간이 화자 분리 (fallback)"""
        import librosa

        # RMS 에너지 기반 발화 구간 탐지
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)

        # 발화 구간 (에너지 > threshold)
        threshold = -30
        is_speech = rms_db > threshold

        segments = []
        in_segment = False
        start_frame = 0

        hop_duration = 512 / sr

        for i, speech in enumerate(is_speech):
            if speech and not in_segment:
                start_frame = i
                in_segment = True
            elif not speech and in_segment:
                start_time = start_frame * hop_duration
                end_time = i * hop_duration

                if end_time - start_time >= 0.5:  # 최소 0.5초
                    # 간단한 스펙트럼 기반 화자 추정
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    segment_audio = y[start_sample:end_sample]

                    # 스펙트럼 중심 기반 화자 구분 (간이)
                    centroid = np.mean(
                        librosa.feature.spectral_centroid(y=segment_audio, sr=sr)
                    )
                    speaker_id = "speaker_0" if centroid < 2000 else "speaker_1"

                    segments.append(
                        SpeakerSegment(
                            speaker_id=speaker_id,
                            start_time=start_time,
                            end_time=end_time,
                            confidence=0.6,
                        )
                    )
                in_segment = False

        # 마지막 구간 처리
        if in_segment:
            start_time = start_frame * hop_duration
            end_time = len(is_speech) * hop_duration
            if end_time - start_time >= 0.5:
                segments.append(
                    SpeakerSegment(
                        speaker_id="speaker_0",
                        start_time=start_time,
                        end_time=end_time,
                        confidence=0.5,
                    )
                )

        return segments

    def _identify_target_speaker(
        self,
        y: np.ndarray,
        sr: int,
        segments: List[SpeakerSegment],
        user_id: Optional[str],
    ) -> str:
        """대상자 화자 식별"""
        # 등록된 프로필이 있으면 임베딩 매칭
        if user_id and user_id in self._speaker_profiles:
            profile = self._speaker_profiles[user_id]
            if profile.embedding is not None:
                return self._match_speaker_by_embedding(
                    y, sr, segments, profile.embedding
                )

        # 프로필이 없으면 가장 많이 말한 화자를 대상자로 추정
        speaker_durations: Dict[str, float] = {}
        for seg in segments:
            speaker_durations[seg.speaker_id] = (
                speaker_durations.get(seg.speaker_id, 0.0) + seg.duration
            )

        if speaker_durations:
            return max(speaker_durations, key=speaker_durations.get)

        return "speaker_0"

    def _match_speaker_by_embedding(
        self,
        y: np.ndarray,
        sr: int,
        segments: List[SpeakerSegment],
        target_embedding: np.ndarray,
    ) -> str:
        """임베딩 기반 화자 매칭"""
        speaker_ids = set(s.speaker_id for s in segments)
        best_speaker = None
        best_similarity = -1.0

        for speaker_id in speaker_ids:
            # 해당 화자의 구간 합치기
            speaker_segments = [s for s in segments if s.speaker_id == speaker_id]
            speaker_audio_parts = []
            for seg in speaker_segments[:5]:  # 최대 5개 구간
                start = int(seg.start_time * sr)
                end = int(seg.end_time * sr)
                speaker_audio_parts.append(y[start:end])

            if speaker_audio_parts:
                speaker_audio = np.concatenate(speaker_audio_parts)
                speaker_embedding = self._extract_speaker_embedding(speaker_audio, sr)

                similarity = self._cosine_similarity(target_embedding, speaker_embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_speaker = speaker_id

        if best_speaker and best_similarity >= self.config.speaker.speaker_similarity_threshold:
            logger.info(f"Speaker matched: {best_speaker}, similarity={best_similarity:.3f}")
            return best_speaker

        # 매칭 실패 시 가장 많이 말한 화자
        speaker_durations: Dict[str, float] = {}
        for seg in segments:
            speaker_durations[seg.speaker_id] = (
                speaker_durations.get(seg.speaker_id, 0.0) + seg.duration
            )
        return max(speaker_durations, key=speaker_durations.get)

    def _extract_speaker_embedding(
        self, y: np.ndarray, sr: int
    ) -> np.ndarray:
        """음성에서 화자 임베딩 추출 (MFCC 기반 간이 임베딩)"""
        import librosa

        # MFCC 기반 화자 특성 벡터
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)

        # 스펙트럼 특성
        spectral_centroid = np.mean(
            librosa.feature.spectral_centroid(y=y, sr=sr)
        )
        spectral_bandwidth = np.mean(
            librosa.feature.spectral_bandwidth(y=y, sr=sr)
        )

        # 피치 통계
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            idx = magnitudes[:, t].argmax()
            pitch = pitches[idx, t]
            if pitch > 0:
                pitch_values.append(pitch)

        pitch_mean = np.mean(pitch_values) if pitch_values else 0
        pitch_std = np.std(pitch_values) if pitch_values else 0

        # 임베딩 벡터 구성
        embedding = np.concatenate([
            mfcc_mean,
            mfcc_std,
            [spectral_centroid, spectral_bandwidth, pitch_mean, pitch_std]
        ])

        # L2 정규화
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """코사인 유사도 계산"""
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
