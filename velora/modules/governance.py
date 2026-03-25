"""
Module 1: Data Governance & Consent (데이터 거버넌스·동의 모듈)

사용자의 음성 데이터 업로드 이전에 명시적 동의를 획득하고,
음성 내 개인정보를 자동 마스킹 처리하여 데이터 활용의
법적·윤리적 안전성을 확보하는 모듈.
"""

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from velora.config.settings import VeloraConfig, get_config
from velora.utils.logging import AuditLogger


class ConsentType(str, Enum):
    """동의 유형"""
    DATA_COLLECTION = "data_collection"           # 데이터 수집 동의
    DATA_ANALYSIS = "data_analysis"               # 데이터 분석 동의
    DATA_RETENTION = "data_retention"             # 데이터 보관 동의
    THIRD_PARTY_VOICE = "third_party_voice"       # 제3자 음성 포함 고지
    NON_MEDICAL_DISCLAIMER = "non_medical_disclaimer"  # 비의료적 성격 고지


@dataclass
class ConsentRecord:
    """동의 기록"""
    consent_id: str
    user_id: str
    consent_type: ConsentType
    granted: bool
    timestamp: str
    policy_version: str
    ip_address: Optional[str] = None
    details: Optional[str] = None


@dataclass
class ConsentToken:
    """동의 토큰 (분석 파이프라인 진입 전 검증용)"""
    token: str
    user_id: str
    granted_consents: List[ConsentType]
    created_at: str
    expires_at: str
    policy_version: str
    is_valid: bool = True


@dataclass
class PIIDetectionResult:
    """PII 탐지 결과"""
    original_text: str
    masked_text: str
    detections: List[Dict]
    total_pii_count: int
    masking_applied: bool


class DataGovernanceModule:
    """
    데이터 거버넌스·동의 모듈

    주요 기능:
    - 명시적 동의 절차 관리
    - PII (개인식별정보) 자동 탐지/마스킹
    - 비의료적 성격 고지 관리
    - 감사 로그 기록
    - 데이터 보관/삭제 정책 관리
    """

    POLICY_VERSION = "1.0.0"
    REQUIRED_CONSENTS = [
        ConsentType.DATA_COLLECTION,
        ConsentType.DATA_ANALYSIS,
        ConsentType.NON_MEDICAL_DISCLAIMER,
    ]

    # PII 패턴 (한국어 기준)
    PII_PATTERNS = {
        "resident_id": {
            "pattern": r"\d{6}[-\s]?\d{7}",
            "description": "주민등록번호",
            "mask": "***-*******"
        },
        "phone_number": {
            "pattern": r"(?:0\d{1,2})[-.\s]?\d{3,4}[-.\s]?\d{4}",
            "description": "전화번호",
            "mask": "***-****-****"
        },
        "card_number": {
            "pattern": r"\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}",
            "description": "카드번호",
            "mask": "****-****-****-****"
        },
        "email": {
            "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "description": "이메일",
            "mask": "****@****.***"
        },
        "address_postal": {
            "pattern": r"\d{5}",
            "description": "우편번호",
            "mask": "*****"
        },
    }

    # 비의료적 고지 문구
    NON_MEDICAL_NOTICES = {
        "pre_analysis": (
            "본 분석은 의료적 진단이 아닌 비의료적 위험 신호 선별 서비스입니다. "
            "분석 결과는 인지기능 변화와 연관될 수 있는 위험 신호를 참고용으로 "
            "제공하며, 전문 의료인의 진단을 대체하지 않습니다."
        ),
        "post_analysis": (
            "분석 결과는 AI 모델에 의한 비의료적 참고 정보입니다. "
            "정확한 의료적 판단은 반드시 전문 의료기관에서 받으시기 바랍니다."
        ),
        "data_usage": (
            "수집된 데이터는 AI 모델 고도화 및 품질 개선을 위한 "
            "기술적 참고 자료로만 활용됩니다."
        ),
    }

    def __init__(self, config: Optional[VeloraConfig] = None):
        self.config = config or get_config()
        self.audit_logger = AuditLogger(
            log_dir=f"{self.config.log_dir}/audit"
        )
        self._consent_store: Dict[str, List[ConsentRecord]] = {}
        self._token_store: Dict[str, ConsentToken] = {}

    def request_consent(
        self,
        user_id: str,
        consent_type: ConsentType,
        granted: bool,
        ip_address: Optional[str] = None
    ) -> ConsentRecord:
        """동의 요청 처리"""
        record = ConsentRecord(
            consent_id=str(uuid.uuid4()),
            user_id=user_id,
            consent_type=consent_type,
            granted=granted,
            timestamp=datetime.now(timezone.utc).isoformat(),
            policy_version=self.POLICY_VERSION,
            ip_address=ip_address,
        )

        if user_id not in self._consent_store:
            self._consent_store[user_id] = []
        self._consent_store[user_id].append(record)

        # 감사 로그
        self.audit_logger.log_consent(user_id, consent_type.value, granted)

        return record

    def process_all_consents(
        self,
        user_id: str,
        consents: Dict[ConsentType, bool],
        ip_address: Optional[str] = None
    ) -> ConsentToken:
        """모든 동의 항목을 일괄 처리하고 토큰 발급"""
        records = []
        for consent_type, granted in consents.items():
            record = self.request_consent(user_id, consent_type, granted, ip_address)
            records.append(record)

        # 필수 동의 항목 검증
        granted_types = [r.consent_type for r in records if r.granted]
        all_required_granted = all(
            req in granted_types for req in self.REQUIRED_CONSENTS
        )

        token = ConsentToken(
            token=str(uuid.uuid4()),
            user_id=user_id,
            granted_consents=granted_types,
            created_at=datetime.now(timezone.utc).isoformat(),
            expires_at="",  # 세션 기반
            policy_version=self.POLICY_VERSION,
            is_valid=all_required_granted,
        )

        self._token_store[token.token] = token
        return token

    def validate_consent_token(self, token_str: str) -> bool:
        """동의 토큰 유효성 검증"""
        token = self._token_store.get(token_str)
        if token is None:
            return False
        return token.is_valid

    def get_required_consents(self) -> List[Dict]:
        """필수 동의 항목 목록 반환"""
        consent_descriptions = {
            ConsentType.DATA_COLLECTION: {
                "title": "음성 데이터 수집 동의",
                "description": (
                    "통화 음성 데이터를 수집하여 인지기능 변화 관련 "
                    "위험 신호 분석에 활용하는 것에 동의합니다."
                ),
                "required": True,
            },
            ConsentType.DATA_ANALYSIS: {
                "title": "데이터 분석 동의",
                "description": (
                    "수집된 음성 데이터를 AI 모델을 통해 분석하여 "
                    "비의료적 위험 신호를 선별하는 것에 동의합니다."
                ),
                "required": True,
            },
            ConsentType.DATA_RETENTION: {
                "title": "데이터 보관 동의",
                "description": (
                    f"분석 완료 후 익명화된 특징 데이터를 최대 "
                    f"{self.config.security.data_retention_days}일간 "
                    f"보관하는 것에 동의합니다. 원본 음성은 분석 후 즉시 삭제됩니다."
                ),
                "required": False,
            },
            ConsentType.THIRD_PARTY_VOICE: {
                "title": "제3자 음성 포함 고지",
                "description": (
                    "통화 음성에 제3자의 발화가 포함될 수 있음을 인지하며, "
                    "이에 따른 법적·윤리적 책임 범위를 이해합니다."
                ),
                "required": False,
            },
            ConsentType.NON_MEDICAL_DISCLAIMER: {
                "title": "비의료적 서비스 고지 동의",
                "description": self.NON_MEDICAL_NOTICES["pre_analysis"],
                "required": True,
            },
        }

        return [
            {
                "consent_type": ct.value,
                **consent_descriptions[ct],
            }
            for ct in ConsentType
        ]

    def detect_pii(self, text: str) -> PIIDetectionResult:
        """텍스트 내 PII (개인식별정보) 탐지"""
        detections = []
        masked_text = text

        for pii_type, pii_info in self.PII_PATTERNS.items():
            matches = re.finditer(pii_info["pattern"], text)
            for match in matches:
                detections.append({
                    "type": pii_type,
                    "description": pii_info["description"],
                    "start": match.start(),
                    "end": match.end(),
                    "original": match.group(),
                })
                masked_text = masked_text.replace(
                    match.group(), pii_info["mask"]
                )

        return PIIDetectionResult(
            original_text=text,
            masked_text=masked_text,
            detections=detections,
            total_pii_count=len(detections),
            masking_applied=len(detections) > 0,
        )

    def mask_pii_in_text(self, text: str, user_id: Optional[str] = None) -> str:
        """텍스트 내 PII 마스킹 처리"""
        result = self.detect_pii(text)

        if result.masking_applied and user_id:
            self.audit_logger.log_pii_detection(
                user_id=user_id,
                file_id="text_input",
                pii_count=result.total_pii_count,
                masked=True,
            )

        return result.masked_text

    def get_data_retention_policy(self) -> Dict:
        """데이터 보관/삭제 정책 반환"""
        return {
            "policy_version": self.POLICY_VERSION,
            "retention_days": self.config.security.data_retention_days,
            "original_audio": "분석 완료 즉시 삭제",
            "feature_vectors": f"최대 {self.config.security.data_retention_days}일 보관 후 자동 삭제",
            "analysis_results": f"최대 {self.config.security.data_retention_days}일 보관",
            "audit_logs": "법적 요구에 따라 별도 보관",
            "deletion_method": "안전한 덮어쓰기 삭제",
            "user_rights": {
                "access": "자신의 데이터 열람 요청 가능",
                "deletion": "언제든 데이터 삭제 요청 가능",
                "portability": "데이터 이동 요청 가능",
            },
        }

    def get_disclaimer(self, context: str = "pre_analysis") -> str:
        """비의료적 면책 고지문 반환"""
        return self.NON_MEDICAL_NOTICES.get(context, self.config.disclaimer)
