"""
Module 7: Results & Guidance Module (결과·안내 모듈)

의료 진단이 아닌 비의료적 위험 신호 요약 결과를 제공하며,
필요 시 자가 관리·상담·검사 고려를 위한 안내 정보를 제시하되,
의료 판단을 대체하지 않음을 명확히 고지하는 모듈.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from velora.config.settings import VeloraConfig, get_config
from velora.modules.model_layer import ConfidenceScore, RiskAssessment
from velora.utils.logging import setup_logger

logger = setup_logger("results")


@dataclass
class RiskLevelDisplay:
    """위험 수준 표시 정보"""
    level: str              # "low", "caution", "high"
    label_ko: str           # 한국어 라벨
    label_en: str           # 영어 라벨
    color: str              # 표시 색상
    icon: str               # 아이콘
    description: str        # 설명
    recommendation: str     # 권고사항


@dataclass
class GuidanceItem:
    """안내 항목"""
    category: str           # "self_care", "consultation", "examination"
    title: str
    description: str
    priority: int           # 1=높음, 2=중간, 3=낮음
    action_type: str        # "info", "action", "warning"


@dataclass
class AnalysisReport:
    """분석 결과 리포트"""
    report_id: str
    timestamp: str
    user_id: Optional[str]

    # 위험 평가
    risk_score: float
    risk_level: str
    risk_display: RiskLevelDisplay
    confidence: ConfidenceScore

    # 주요 특징 요약
    key_features_summary: List[Dict]

    # 안내 사항
    guidance: List[GuidanceItem]
    disclaimers: List[str]

    # 이력 관리
    analysis_history: Optional[List[Dict]] = None


class ResultsModule:
    """
    결과·안내 모듈

    주요 기능:
    - 위험 구간 시각화 (낮음-주의-높음)
    - 비의료적 표현으로 결과 전달
    - 후속 행동 안내 (자가 관리, 상담, 검사)
    - 면책 고지
    - 결과 이력 관리
    """

    RISK_LEVELS = {
        "low": RiskLevelDisplay(
            level="low",
            label_ko="낮음",
            label_en="Low",
            color="#4CAF50",
            icon="shield-check",
            description=(
                "현재 분석된 음성 패턴에서 인지기능 변화와 연관될 수 있는 "
                "특이 신호가 뚜렷하게 관찰되지 않았습니다."
            ),
            recommendation=(
                "현재 상태를 유지하시되, 정기적인 건강 관리를 권장합니다."
            ),
        ),
        "caution": RiskLevelDisplay(
            level="caution",
            label_ko="주의",
            label_en="Caution",
            color="#FF9800",
            icon="alert-triangle",
            description=(
                "일부 음성 패턴에서 인지기능 변화와 연관될 수 있는 "
                "경향이 관찰되었습니다. 이는 참고용 정보이며, "
                "다양한 요인(컨디션, 환경 등)에 의해 영향받을 수 있습니다."
            ),
            recommendation=(
                "정기적인 자가 점검과 함께, 필요 시 전문 상담을 "
                "고려해 보시기 바랍니다."
            ),
        ),
        "high": RiskLevelDisplay(
            level="high",
            label_ko="높음",
            label_en="High",
            color="#F44336",
            icon="alert-circle",
            description=(
                "분석된 음성 패턴에서 인지기능 변화와 연관될 수 있는 "
                "신호가 비교적 뚜렷하게 관찰되었습니다. "
                "이는 AI 기반 비의료적 참고 정보입니다."
            ),
            recommendation=(
                "가까운 시일 내에 전문 의료기관을 방문하여 "
                "정밀 검사를 받으시는 것을 권장합니다."
            ),
        ),
    }

    DISCLAIMERS = [
        (
            "본 분석 결과는 AI 모델에 의한 비의료적 참고 정보이며, "
            "의료적 진단이나 치료 판단을 대체하지 않습니다."
        ),
        (
            "정확한 건강 상태 확인은 반드시 전문 의료기관에서 "
            "받으시기 바랍니다."
        ),
        (
            "분석 결과는 음성 품질, 환경, 컨디션 등 다양한 요인에 "
            "의해 영향받을 수 있으며, 단일 분석 결과만으로 "
            "건강 상태를 판단하지 마시기 바랍니다."
        ),
    ]

    def __init__(self, config: Optional[VeloraConfig] = None):
        self.config = config or get_config()
        self._analysis_history: Dict[str, List[Dict]] = {}

    def generate_report(
        self,
        risk_assessment: RiskAssessment,
        user_id: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
    ) -> AnalysisReport:
        """
        분석 결과 리포트 생성

        Args:
            risk_assessment: 모델 레이어 출력 (위험 평가)
            user_id: 사용자 ID
            feature_names: 특징 이름 목록

        Returns:
            AnalysisReport: 사용자에게 전달할 분석 리포트
        """
        import uuid

        report_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        # 위험 수준 표시 정보
        risk_display = self.RISK_LEVELS.get(
            risk_assessment.risk_level,
            self.RISK_LEVELS["caution"]
        )

        # 주요 특징 요약
        key_features = self._summarize_key_features(
            risk_assessment.feature_importance
        )

        # 안내 사항 생성
        guidance = self._generate_guidance(risk_assessment.risk_level)

        # 분석 이력 업데이트
        if user_id:
            self._update_history(user_id, {
                "report_id": report_id,
                "timestamp": timestamp,
                "risk_score": risk_assessment.risk_score,
                "risk_level": risk_assessment.risk_level,
                "confidence": risk_assessment.confidence.overall,
            })

        history = self._analysis_history.get(user_id, []) if user_id else None

        report = AnalysisReport(
            report_id=report_id,
            timestamp=timestamp,
            user_id=user_id,
            risk_score=risk_assessment.risk_score,
            risk_level=risk_assessment.risk_level,
            risk_display=risk_display,
            confidence=risk_assessment.confidence,
            key_features_summary=key_features,
            guidance=guidance,
            disclaimers=self.DISCLAIMERS,
            analysis_history=history,
        )

        logger.info(
            f"Report generated: {report_id}, "
            f"risk={risk_assessment.risk_level}, "
            f"score={risk_assessment.risk_score:.3f}"
        )

        return report

    def format_report_for_display(self, report: AnalysisReport) -> Dict:
        """리포트를 모바일 앱 표시용 딕셔너리로 변환"""
        return {
            "report_id": report.report_id,
            "timestamp": report.timestamp,
            "risk": {
                "score": round(report.risk_score, 2),
                "level": report.risk_level,
                "label": report.risk_display.label_ko,
                "color": report.risk_display.color,
                "icon": report.risk_display.icon,
                "description": report.risk_display.description,
                "recommendation": report.risk_display.recommendation,
            },
            "confidence": {
                "overall": round(report.confidence.overall, 2),
                "audio_quality": round(report.confidence.audio_quality, 2),
                "diarization": round(report.confidence.diarization_confidence, 2),
                "model_certainty": round(report.confidence.model_uncertainty, 2),
                "interpretation": self._interpret_confidence(report.confidence.overall),
            },
            "key_observations": report.key_features_summary,
            "guidance": [
                {
                    "category": g.category,
                    "title": g.title,
                    "description": g.description,
                    "priority": g.priority,
                    "type": g.action_type,
                }
                for g in report.guidance
            ],
            "disclaimers": report.disclaimers,
            "history_summary": self._format_history(report.analysis_history),
        }

    def _summarize_key_features(
        self, feature_importance: Dict[str, float]
    ) -> List[Dict]:
        """주요 특징을 사용자 친화적으로 요약"""
        feature_descriptions = {
            "silence_ratio": ("무음 비율", "대화 중 침묵 구간의 비율"),
            "speech_density": ("발화 밀도", "단위 시간당 발화 구간 수"),
            "avg_utterance_length": ("평균 발화 길이", "한 번 말할 때 평균 발화 시간"),
            "avg_pause_length": ("평균 쉼 길이", "발화 사이 쉬는 시간의 평균"),
            "pitch_cv": ("음높이 변동", "말할 때 음높이의 변화 정도"),
            "pitch_jitter": ("음성 떨림", "음높이의 불규칙한 변동"),
            "rms_energy_cv": ("음량 변동", "말할 때 음량의 변화 정도"),
            "spectral_centroid_mean": ("음색 특성", "전반적인 음색의 밝기"),
            "speech_rate_estimate": ("발화 속도", "추정 발화 속도"),
            "num_utterances": ("발화 횟수", "총 발화 구간 수"),
        }

        summaries = []
        for feature_name, importance in sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:5]:
            desc = feature_descriptions.get(
                feature_name,
                (feature_name, "음성 분석 특징")
            )
            summaries.append({
                "feature": desc[0],
                "description": desc[1],
                "contribution": round(importance, 3),
            })

        return summaries

    def _generate_guidance(self, risk_level: str) -> List[GuidanceItem]:
        """위험 수준에 따른 안내 생성"""
        guidance_items = []

        # 공통 안내
        guidance_items.append(
            GuidanceItem(
                category="info",
                title="분석 결과 안내",
                description=(
                    "본 결과는 AI 기반 비의료적 참고 정보입니다. "
                    "건강 상태에 대한 정확한 판단은 전문 의료인과 상담하시기 바랍니다."
                ),
                priority=1,
                action_type="info",
            )
        )

        if risk_level == "low":
            guidance_items.extend([
                GuidanceItem(
                    category="self_care",
                    title="인지 건강 관리 팁",
                    description="규칙적인 운동, 충분한 수면, 사회적 활동을 유지하세요.",
                    priority=3,
                    action_type="info",
                ),
                GuidanceItem(
                    category="self_care",
                    title="정기적 모니터링",
                    description="3~6개월마다 정기적으로 분석을 받아보시는 것을 권장합니다.",
                    priority=3,
                    action_type="action",
                ),
            ])

        elif risk_level == "caution":
            guidance_items.extend([
                GuidanceItem(
                    category="self_care",
                    title="생활 습관 점검",
                    description=(
                        "수면 패턴, 스트레스, 영양 상태 등을 점검하고 "
                        "개선이 필요한 부분을 찾아보세요."
                    ),
                    priority=2,
                    action_type="action",
                ),
                GuidanceItem(
                    category="consultation",
                    title="전문 상담 고려",
                    description=(
                        "필요 시 신경과 또는 정신건강의학과 전문의와 "
                        "상담을 고려해 보시기 바랍니다."
                    ),
                    priority=2,
                    action_type="action",
                ),
                GuidanceItem(
                    category="self_care",
                    title="추적 모니터링",
                    description="1~3개월 후 재분석을 통해 변화 추이를 확인하세요.",
                    priority=2,
                    action_type="action",
                ),
            ])

        elif risk_level == "high":
            guidance_items.extend([
                GuidanceItem(
                    category="examination",
                    title="전문 검사 권장",
                    description=(
                        "가까운 시일 내에 전문 의료기관을 방문하여 "
                        "인지기능 정밀 검사를 받으시는 것을 강력히 권장합니다."
                    ),
                    priority=1,
                    action_type="warning",
                ),
                GuidanceItem(
                    category="consultation",
                    title="가까운 의료기관 안내",
                    description=(
                        "신경과, 정신건강의학과, 또는 치매안심센터에서 "
                        "전문 상담을 받으실 수 있습니다."
                    ),
                    priority=1,
                    action_type="action",
                ),
                GuidanceItem(
                    category="self_care",
                    title="보호자와 공유",
                    description=(
                        "가족이나 가까운 분에게 본 결과를 공유하고, "
                        "함께 전문 상담을 받아보시는 것을 권장합니다."
                    ),
                    priority=1,
                    action_type="action",
                ),
            ])

        return guidance_items

    def _interpret_confidence(self, confidence: float) -> str:
        """신뢰도 해석"""
        if confidence >= 0.8:
            return "높음 - 분석 결과의 신뢰도가 높습니다"
        elif confidence >= 0.6:
            return "보통 - 분석 결과를 참고용으로 활용하세요"
        else:
            return "낮음 - 음성 품질 개선 후 재분석을 권장합니다"

    def _update_history(self, user_id: str, entry: Dict) -> None:
        """분석 이력 업데이트"""
        if user_id not in self._analysis_history:
            self._analysis_history[user_id] = []
        self._analysis_history[user_id].append(entry)
        # 최근 10개만 유지
        self._analysis_history[user_id] = self._analysis_history[user_id][-10:]

    def _format_history(self, history: Optional[List[Dict]]) -> Optional[List[Dict]]:
        """이력 포맷팅"""
        if not history:
            return None

        return [
            {
                "date": h.get("timestamp", ""),
                "risk_level": h.get("risk_level", ""),
                "risk_score": round(h.get("risk_score", 0), 2),
                "confidence": round(h.get("confidence", 0), 2),
            }
            for h in history
        ]
