"""
Module 5: Feature Engine (특징 엔진)

전처리된 음성으로부터 음향·리듬 기반의 음성 특징을 중심으로 추출하며,
필요 시 개인정보 리스크를 고려하여 전사 기반 언어 특징을 선택적으로
추출하는 모듈.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from velora.config.settings import VeloraConfig, get_config
from velora.modules.preprocessing import SpeechSegment, UtteranceStatistics
from velora.utils.logging import setup_logger

logger = setup_logger("features")


@dataclass
class FeatureQualityMeta:
    """특징 품질 메타데이터"""
    total_features: int
    missing_features: int
    missing_ratio: float
    reliability_score: float  # 0~1
    warnings: List[str] = field(default_factory=list)


@dataclass
class FeatureExtractionResult:
    """특징 추출 결과"""
    feature_vector: np.ndarray
    feature_names: List[str]
    feature_groups: Dict[str, List[str]]
    quality_meta: FeatureQualityMeta
    spectrogram_image: Optional[np.ndarray] = None


class AcousticFeatureExtractor:
    """
    음성 기반 특징 추출기 (필수)

    텍스트 전사 없이 추출 가능한 음향 특징:
    - MFCC 통계
    - 스펙트럼 특징
    - 에너지/운율 특징
    - 발화 패턴 특징
    """

    def __init__(self, config: Optional[VeloraConfig] = None):
        self.config = config or get_config()

    def extract(
        self,
        audio_path: str,
        speech_segments: Optional[List[SpeechSegment]] = None,
        utterance_stats: Optional[UtteranceStatistics] = None,
    ) -> Tuple[Dict[str, float], List[str]]:
        """
        음향 특징 추출

        Returns:
            Tuple[Dict, List]: (특징 딕셔너리, 특징 이름 목록)
        """
        import librosa

        y, sr = librosa.load(
            audio_path,
            sr=self.config.audio.target_sample_rate,
            mono=True,
        )

        features = {}
        feature_names = []

        # 1. MFCC 특징
        mfcc_features = self._extract_mfcc_features(y, sr)
        features.update(mfcc_features)
        feature_names.extend(mfcc_features.keys())

        # 2. 스펙트럼 특징
        spectral_features = self._extract_spectral_features(y, sr)
        features.update(spectral_features)
        feature_names.extend(spectral_features.keys())

        # 3. 에너지/운율 특징
        prosodic_features = self._extract_prosodic_features(y, sr)
        features.update(prosodic_features)
        feature_names.extend(prosodic_features.keys())

        # 4. 발화 패턴 특징 (VAD 기반)
        if utterance_stats:
            temporal_features = self._extract_temporal_features(utterance_stats)
            features.update(temporal_features)
            feature_names.extend(temporal_features.keys())

        # 5. 구간별 특징 변동성
        if speech_segments:
            variability_features = self._extract_variability_features(
                y, sr, speech_segments
            )
            features.update(variability_features)
            feature_names.extend(variability_features.keys())

        logger.info(f"Extracted {len(features)} acoustic features")
        return features, feature_names

    def _extract_mfcc_features(
        self, y: np.ndarray, sr: int
    ) -> Dict[str, float]:
        """MFCC 통계 특징"""
        import librosa

        n_mfcc = self.config.feature.n_mfcc
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc,
            hop_length=self.config.feature.hop_length,
        )

        features = {}
        for i in range(n_mfcc):
            features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
            features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))
            features[f"mfcc_{i}_skew"] = float(
                np.mean(((mfccs[i] - np.mean(mfccs[i])) / (np.std(mfccs[i]) + 1e-8)) ** 3)
            )
            features[f"mfcc_{i}_kurtosis"] = float(
                np.mean(((mfccs[i] - np.mean(mfccs[i])) / (np.std(mfccs[i]) + 1e-8)) ** 4) - 3
            )

        # Delta MFCC (1차 미분)
        delta_mfccs = librosa.feature.delta(mfccs)
        for i in range(n_mfcc):
            features[f"delta_mfcc_{i}_mean"] = float(np.mean(delta_mfccs[i]))
            features[f"delta_mfcc_{i}_std"] = float(np.std(delta_mfccs[i]))

        # Delta-Delta MFCC (2차 미분)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        for i in range(n_mfcc):
            features[f"delta2_mfcc_{i}_mean"] = float(np.mean(delta2_mfccs[i]))
            features[f"delta2_mfcc_{i}_std"] = float(np.std(delta2_mfccs[i]))

        return features

    def _extract_spectral_features(
        self, y: np.ndarray, sr: int
    ) -> Dict[str, float]:
        """스펙트럼 특징"""
        import librosa

        hop = self.config.feature.hop_length

        # Spectral Centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]

        # Spectral Bandwidth
        bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop)[0]

        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop)[0]

        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop)

        # Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop)[0]

        # Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop)[0]

        features = {
            "spectral_centroid_mean": float(np.mean(centroid)),
            "spectral_centroid_std": float(np.std(centroid)),
            "spectral_bandwidth_mean": float(np.mean(bandwidth)),
            "spectral_bandwidth_std": float(np.std(bandwidth)),
            "spectral_rolloff_mean": float(np.mean(rolloff)),
            "spectral_rolloff_std": float(np.std(rolloff)),
            "spectral_flatness_mean": float(np.mean(flatness)),
            "spectral_flatness_std": float(np.std(flatness)),
            "zcr_mean": float(np.mean(zcr)),
            "zcr_std": float(np.std(zcr)),
        }

        # Spectral Contrast (per band)
        for i in range(contrast.shape[0]):
            features[f"spectral_contrast_{i}_mean"] = float(np.mean(contrast[i]))
            features[f"spectral_contrast_{i}_std"] = float(np.std(contrast[i]))

        return features

    def _extract_prosodic_features(
        self, y: np.ndarray, sr: int
    ) -> Dict[str, float]:
        """에너지/운율 특징 (피치, 에너지 변동성)"""
        import librosa

        # RMS 에너지
        rms = librosa.feature.rms(y=y, hop_length=self.config.feature.hop_length)[0]

        # 피치 추출
        pitches, magnitudes = librosa.piptrack(
            y=y, sr=sr, hop_length=self.config.feature.hop_length
        )

        pitch_values = []
        for t in range(pitches.shape[1]):
            idx = magnitudes[:, t].argmax()
            pitch = pitches[idx, t]
            if pitch > 50:  # 유효 피치만
                pitch_values.append(pitch)

        pitch_array = np.array(pitch_values) if pitch_values else np.array([0.0])

        features = {
            # 에너지 특징
            "rms_energy_mean": float(np.mean(rms)),
            "rms_energy_std": float(np.std(rms)),
            "rms_energy_max": float(np.max(rms)),
            "rms_energy_min": float(np.min(rms)),
            "rms_energy_range": float(np.max(rms) - np.min(rms)),
            "rms_energy_cv": float(np.std(rms) / (np.mean(rms) + 1e-8)),  # 변동계수

            # 피치(F0) 특징
            "pitch_mean": float(np.mean(pitch_array)),
            "pitch_std": float(np.std(pitch_array)),
            "pitch_max": float(np.max(pitch_array)),
            "pitch_min": float(np.min(pitch_array)) if len(pitch_values) > 0 else 0.0,
            "pitch_range": float(np.max(pitch_array) - np.min(pitch_array)),
            "pitch_cv": float(np.std(pitch_array) / (np.mean(pitch_array) + 1e-8)),

            # 운율 안정성 (피치 변동의 변동)
            "pitch_jitter": float(np.mean(np.abs(np.diff(pitch_array)))) if len(pitch_array) > 1 else 0.0,

            # 에너지 변동 패턴
            "energy_delta_mean": float(np.mean(np.abs(np.diff(rms)))) if len(rms) > 1 else 0.0,
        }

        return features

    def _extract_temporal_features(
        self, stats: UtteranceStatistics
    ) -> Dict[str, float]:
        """발화 시간 패턴 특징"""
        return {
            "silence_ratio": stats.silence_ratio,
            "speech_density": stats.speech_density,
            "avg_utterance_length": stats.avg_utterance_length,
            "max_utterance_length": stats.max_utterance_length,
            "min_utterance_length": stats.min_utterance_length,
            "num_utterances": float(stats.num_utterances),
            "num_pauses": float(stats.num_pauses),
            "avg_pause_length": stats.avg_pause_length,
            "speech_rate_estimate": stats.speech_rate_estimate,
            "speech_to_pause_ratio": (
                stats.speech_duration / (stats.silence_duration + 1e-8)
            ),
        }

    def _extract_variability_features(
        self,
        y: np.ndarray,
        sr: int,
        segments: List[SpeechSegment],
    ) -> Dict[str, float]:
        """구간별 특징 변동성"""
        import librosa

        if len(segments) < 2:
            return {
                "segment_energy_variability": 0.0,
                "segment_pitch_variability": 0.0,
                "segment_mfcc0_variability": 0.0,
            }

        segment_energies = []
        segment_pitches = []
        segment_mfcc0s = []

        for seg in segments:
            start = int(seg.start_time * sr)
            end = int(seg.end_time * sr)
            seg_audio = y[start:end]

            if len(seg_audio) < 1024:
                continue

            # 구간별 RMS
            rms = np.sqrt(np.mean(seg_audio ** 2))
            segment_energies.append(rms)

            # 구간별 MFCC[0]
            mfcc = librosa.feature.mfcc(y=seg_audio, sr=sr, n_mfcc=1)
            segment_mfcc0s.append(float(np.mean(mfcc[0])))

            # 구간별 피치
            pitches, mags = librosa.piptrack(y=seg_audio, sr=sr)
            pitch_vals = []
            for t in range(pitches.shape[1]):
                idx = mags[:, t].argmax()
                p = pitches[idx, t]
                if p > 50:
                    pitch_vals.append(p)
            if pitch_vals:
                segment_pitches.append(np.mean(pitch_vals))

        features = {
            "segment_energy_variability": float(np.std(segment_energies)) if segment_energies else 0.0,
            "segment_pitch_variability": float(np.std(segment_pitches)) if segment_pitches else 0.0,
            "segment_mfcc0_variability": float(np.std(segment_mfcc0s)) if segment_mfcc0s else 0.0,
        }

        return features


class LinguisticFeatureExtractor:
    """
    전사 기반 언어 특징 추출기 (선택적)

    ASR 오류 및 개인정보 리스크를 고려하여 선택적 활용:
    - 어휘 다양성 (TTR)
    - 반복 발화 비율
    - 문장 길이/복잡도
    - 의미 단절 지표
    - 비유창성 (disfluency)
    """

    def __init__(self, config: Optional[VeloraConfig] = None):
        self.config = config or get_config()

    def extract(self, text: str) -> Tuple[Dict[str, float], List[str]]:
        """
        텍스트 기반 언어 특징 추출

        Args:
            text: 전사 텍스트

        Returns:
            Tuple[Dict, List]: (특징 딕셔너리, 특징 이름 목록)
        """
        if not text.strip():
            return {}, []

        features = {}

        # 기본 텍스트 통계
        words = text.split()
        total_words = len(words)
        unique_words = len(set(w.lower() for w in words))

        if total_words == 0:
            return {}, []

        # 1. 어휘 다양성
        features["type_token_ratio"] = unique_words / total_words
        features["vocabulary_size"] = float(unique_words)
        features["total_word_count"] = float(total_words)

        # 2. 반복 발화 비율
        word_counts = {}
        for w in words:
            w_lower = w.lower()
            word_counts[w_lower] = word_counts.get(w_lower, 0) + 1
        repeated_words = sum(1 for c in word_counts.values() if c > 1)
        features["word_repetition_ratio"] = repeated_words / len(word_counts) if word_counts else 0.0

        # 인접 반복 (연속 반복)
        adjacent_repeats = 0
        for i in range(1, len(words)):
            if words[i].lower() == words[i - 1].lower():
                adjacent_repeats += 1
        features["adjacent_repetition_ratio"] = adjacent_repeats / total_words

        # 3. 문장 관련 특징
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        sentence_lengths = [len(s.split()) for s in sentences]

        features["sentence_count"] = float(len(sentences))
        features["avg_sentence_length"] = float(np.mean(sentence_lengths)) if sentence_lengths else 0.0
        features["max_sentence_length"] = float(np.max(sentence_lengths)) if sentence_lengths else 0.0
        features["sentence_length_std"] = float(np.std(sentence_lengths)) if sentence_lengths else 0.0

        # 4. 비유창성 마커 (한국어/영어)
        disfluency_markers_ko = ["음", "어", "그", "저", "에", "아"]
        disfluency_markers_en = ["um", "uh", "er", "ah", "like", "you know"]
        all_markers = disfluency_markers_ko + disfluency_markers_en

        disfluency_count = sum(
            1 for w in words if w.lower() in all_markers
        )
        features["disfluency_ratio"] = disfluency_count / total_words
        features["disfluency_count"] = float(disfluency_count)

        # 5. 평균 단어 길이 (글자수)
        word_lengths = [len(w) for w in words]
        features["avg_word_length"] = float(np.mean(word_lengths))

        feature_names = list(features.keys())
        logger.info(f"Extracted {len(features)} linguistic features")

        return features, feature_names


class SpectrogramExtractor:
    """Mel-spectrogram 이미지 생성기 (ViT 입력용)"""

    def __init__(self, config: Optional[VeloraConfig] = None):
        self.config = config or get_config()

    def create_mel_spectrogram(
        self,
        audio_path: str,
        target_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Mel-spectrogram 이미지 생성

        Args:
            audio_path: 오디오 파일 경로
            target_size: 출력 이미지 크기

        Returns:
            np.ndarray: RGB 이미지 (H, W, 3)
        """
        import librosa
        from PIL import Image

        y, sr = librosa.load(
            audio_path,
            sr=self.config.audio.target_sample_rate,
            mono=True,
        )

        # Mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr,
            n_mels=self.config.feature.n_mels,
            fmax=self.config.feature.fmax,
            hop_length=self.config.feature.hop_length,
            n_fft=self.config.feature.n_fft,
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.config.feature.hop_length)
        chroma_resized = np.resize(chroma, (mel_spec_db.shape[0], mel_spec_db.shape[1]))

        # Spectral Contrast
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=self.config.feature.hop_length)
        contrast_resized = np.resize(contrast, (mel_spec_db.shape[0], mel_spec_db.shape[1]))

        # 각 채널 정규화 (0-255)
        def normalize(data: np.ndarray) -> np.ndarray:
            d_min, d_max = data.min(), data.max()
            if d_max - d_min < 1e-8:
                return np.zeros_like(data, dtype=np.uint8)
            return ((data - d_min) / (d_max - d_min) * 255).astype(np.uint8)

        # RGB 채널: Mel / Chroma / Contrast
        rgb = np.stack([
            normalize(mel_spec_db),
            normalize(chroma_resized),
            normalize(contrast_resized),
        ], axis=-1)

        # 리사이즈
        img = Image.fromarray(rgb)
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        return np.array(img)


class FeatureEngine:
    """
    통합 특징 엔진

    음향 특징 (필수) + 언어 특징 (선택) + 스펙트로그램 (ViT용)
    """

    def __init__(self, config: Optional[VeloraConfig] = None):
        self.config = config or get_config()
        self.acoustic_extractor = AcousticFeatureExtractor(config)
        self.linguistic_extractor = LinguisticFeatureExtractor(config)
        self.spectrogram_extractor = SpectrogramExtractor(config)

    def extract_features(
        self,
        audio_path: str,
        speech_segments: Optional[List[SpeechSegment]] = None,
        utterance_stats: Optional[UtteranceStatistics] = None,
        transcript: Optional[str] = None,
        include_spectrogram: bool = True,
    ) -> FeatureExtractionResult:
        """
        통합 특징 추출

        Args:
            audio_path: 오디오 파일 경로
            speech_segments: 발화 구간 정보 (전처리 모듈 출력)
            utterance_stats: 발화 통계 (전처리 모듈 출력)
            transcript: 전사 텍스트 (선택)
            include_spectrogram: 스펙트로그램 생성 여부

        Returns:
            FeatureExtractionResult
        """
        all_features = {}
        all_names = []
        feature_groups = {}
        warnings = []

        # 1. 음향 특징 (필수)
        try:
            acoustic_features, acoustic_names = self.acoustic_extractor.extract(
                audio_path, speech_segments, utterance_stats
            )
            all_features.update(acoustic_features)
            all_names.extend(acoustic_names)
            feature_groups["acoustic"] = acoustic_names
        except Exception as e:
            logger.error(f"Acoustic feature extraction failed: {e}")
            warnings.append(f"음향 특징 추출 실패: {str(e)}")

        # 2. 언어 특징 (선택)
        if self.config.feature.use_linguistic_features and transcript:
            try:
                ling_features, ling_names = self.linguistic_extractor.extract(transcript)
                all_features.update(ling_features)
                all_names.extend(ling_names)
                feature_groups["linguistic"] = ling_names
            except Exception as e:
                logger.warning(f"Linguistic feature extraction failed: {e}")
                warnings.append(f"언어 특징 추출 실패: {str(e)}")

        # 3. 스펙트로그램 (ViT 입력용)
        spectrogram = None
        if include_spectrogram:
            try:
                spectrogram = self.spectrogram_extractor.create_mel_spectrogram(
                    audio_path, self.config.feature.vit_target_size
                )
            except Exception as e:
                logger.warning(f"Spectrogram creation failed: {e}")
                warnings.append(f"스펙트로그램 생성 실패: {str(e)}")

        # 특징 벡터 구성
        feature_vector = np.array(
            [all_features.get(name, 0.0) for name in all_names],
            dtype=np.float32,
        )

        # NaN/Inf 처리
        nan_mask = ~np.isfinite(feature_vector)
        missing_count = int(np.sum(nan_mask))
        feature_vector[nan_mask] = 0.0

        # 품질 메타
        quality_meta = FeatureQualityMeta(
            total_features=len(all_names),
            missing_features=missing_count,
            missing_ratio=missing_count / max(len(all_names), 1),
            reliability_score=1.0 - missing_count / max(len(all_names), 1),
            warnings=warnings,
        )

        logger.info(
            f"Feature extraction complete: {len(all_names)} features, "
            f"missing={missing_count}, reliability={quality_meta.reliability_score:.2f}"
        )

        return FeatureExtractionResult(
            feature_vector=feature_vector,
            feature_names=all_names,
            feature_groups=feature_groups,
            quality_meta=quality_meta,
            spectrogram_image=spectrogram,
        )
