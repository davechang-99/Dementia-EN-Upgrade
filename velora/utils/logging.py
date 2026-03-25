"""
VELORA Logging & Audit Log Utilities
감사 로그 및 시스템 로깅
"""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional


def setup_logger(name: str, log_dir: str = "./logs", level: int = logging.INFO) -> logging.Logger:
    """모듈별 로거 설정"""
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # 파일 핸들러
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{name}.log"),
            encoding="utf-8"
        )
        file_handler.setLevel(level)

        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)

        # 포맷터
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


class AuditLogger:
    """
    감사 로그 기록기 (Audit Log)
    모든 데이터 접근/처리 이력을 기록
    """

    def __init__(self, log_dir: str = "./logs/audit"):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.logger = setup_logger("audit", log_dir)

    def log_event(
        self,
        event_type: str,
        user_id: Optional[str],
        action: str,
        details: Optional[dict] = None,
        status: str = "success"
    ) -> str:
        """감사 이벤트 기록"""
        event_id = str(uuid.uuid4())
        event = {
            "event_id": event_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "action": action,
            "status": status,
            "details": details or {}
        }

        self.logger.info(json.dumps(event, ensure_ascii=False))

        # JSON 파일로도 저장
        audit_file = os.path.join(
            self.log_dir,
            f"audit_{datetime.now(timezone.utc).strftime('%Y%m%d')}.jsonl"
        )
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        return event_id

    def log_consent(self, user_id: str, consent_type: str, granted: bool) -> str:
        """동의 이벤트 기록"""
        return self.log_event(
            event_type="consent",
            user_id=user_id,
            action=f"consent_{consent_type}",
            details={"granted": granted, "consent_type": consent_type},
            status="granted" if granted else "denied"
        )

    def log_data_access(
        self,
        user_id: Optional[str],
        resource: str,
        action: str,
        details: Optional[dict] = None
    ) -> str:
        """데이터 접근 이벤트 기록"""
        return self.log_event(
            event_type="data_access",
            user_id=user_id,
            action=action,
            details={"resource": resource, **(details or {})}
        )

    def log_analysis(
        self,
        user_id: Optional[str],
        analysis_id: str,
        stage: str,
        result: Any = None
    ) -> str:
        """분석 파이프라인 이벤트 기록"""
        return self.log_event(
            event_type="analysis",
            user_id=user_id,
            action=f"analysis_{stage}",
            details={"analysis_id": analysis_id, "result_summary": str(result)[:200] if result else None}
        )

    def log_pii_detection(
        self,
        user_id: Optional[str],
        file_id: str,
        pii_count: int,
        masked: bool
    ) -> str:
        """PII 탐지/마스킹 이벤트 기록"""
        return self.log_event(
            event_type="pii_detection",
            user_id=user_id,
            action="pii_mask" if masked else "pii_detect",
            details={"file_id": file_id, "pii_count": pii_count, "masked": masked}
        )
