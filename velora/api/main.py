"""
VELORA FastAPI Backend

음성 업로드부터 결과 산출까지의 RESTful API 서버
"""

import os
import shutil
import tempfile
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from velora.config.settings import VeloraConfig, get_config
from velora.modules.governance import ConsentType, DataGovernanceModule
from velora.modules.pipeline import VeloraPipeline

# App 초기화
app = FastAPI(
    title="VELORA API",
    description=(
        "VELORA - AI 기반 비의료적 인지 건강 관리 지원 서비스 API. "
        "통화 음성 분석을 통해 인지기능 변화와 연관될 수 있는 위험 신호를 "
        "비의료적으로 선별하고 참고용 지표를 제공합니다."
    ),
    version="1.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 인스턴스
config = get_config()
pipeline = VeloraPipeline(config)
governance = DataGovernanceModule(config)


# ─── Pydantic Models ───


class ConsentRequest(BaseModel):
    """동의 요청"""
    user_id: str
    consents: Dict[str, bool]
    ip_address: Optional[str] = None


class ConsentResponse(BaseModel):
    """동의 응답"""
    token: str
    is_valid: bool
    granted_consents: List[str]
    policy_version: str
    disclaimer: str


class AnalysisResponse(BaseModel):
    """분석 응답"""
    analysis_id: str
    status: str
    message: Optional[str] = None
    report: Optional[Dict] = None
    processing_time_seconds: float = 0.0


class HealthResponse(BaseModel):
    """헬스 체크 응답"""
    status: str
    version: str
    service: str


# ─── API Endpoints ───


@app.get("/", response_model=HealthResponse)
async def root():
    """서비스 상태 확인"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        service="VELORA - AI 기반 비의료적 인지 건강 관리 지원 서비스",
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스 체크"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        service="VELORA",
    )


@app.get("/consent/required")
async def get_required_consents():
    """필수 동의 항목 조회"""
    return {
        "consents": governance.get_required_consents(),
        "disclaimer": governance.get_disclaimer("pre_analysis"),
        "data_retention_policy": governance.get_data_retention_policy(),
    }


@app.post("/consent", response_model=ConsentResponse)
async def submit_consent(request: ConsentRequest):
    """동의 제출 및 토큰 발급"""
    consent_map = {}
    for key, value in request.consents.items():
        try:
            consent_type = ConsentType(key)
            consent_map[consent_type] = value
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"유효하지 않은 동의 유형: {key}",
            )

    token = governance.process_all_consents(
        user_id=request.user_id,
        consents=consent_map,
        ip_address=request.ip_address,
    )

    if not token.is_valid:
        raise HTTPException(
            status_code=400,
            detail="필수 동의 항목이 모두 승인되지 않았습니다.",
        )

    return ConsentResponse(
        token=token.token,
        is_valid=token.is_valid,
        granted_consents=[c.value for c in token.granted_consents],
        policy_version=token.policy_version,
        disclaimer=governance.get_disclaimer("pre_analysis"),
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_audio(
    audio_file: UploadFile = File(..., description="통화 녹음 파일"),
    user_id: Optional[str] = Form(None),
    consent_token: Optional[str] = Form(None),
    age_group: Optional[str] = Form(None, description="연령대 (40s, 50s 등)"),
    language: str = Form("ko", description="언어 (ko, en)"),
    speaker_sample: Optional[UploadFile] = File(None, description="본인 음성 샘플 (5-10초)"),
):
    """
    음성 분석 API

    통화 음성 파일을 업로드하면 AI 모델이 분석하여
    인지기능 변화 관련 위험 신호를 비의료적으로 선별합니다.
    """
    # 임시 파일로 저장
    temp_dir = tempfile.mkdtemp()
    try:
        # 오디오 파일 저장
        audio_ext = os.path.splitext(audio_file.filename or "audio.wav")[1]
        audio_path = os.path.join(temp_dir, f"upload{audio_ext}")
        with open(audio_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)

        # 화자 샘플 저장 (있는 경우)
        speaker_sample_path = None
        if speaker_sample:
            sample_ext = os.path.splitext(speaker_sample.filename or "sample.wav")[1]
            speaker_sample_path = os.path.join(temp_dir, f"speaker_sample{sample_ext}")
            with open(speaker_sample_path, "wb") as f:
                sample_content = await speaker_sample.read()
                f.write(sample_content)

        # 파이프라인 실행
        result = pipeline.analyze(
            audio_path=audio_path,
            user_id=user_id,
            consent_token=consent_token,
            speaker_sample_path=speaker_sample_path,
            age_group=age_group,
        )

        if result.status == "completed":
            return AnalysisResponse(
                analysis_id=result.analysis_id,
                status="completed",
                message="분석이 완료되었습니다.",
                report=result.report_display,
                processing_time_seconds=result.processing_time_seconds,
            )
        elif result.status == "rejected":
            return AnalysisResponse(
                analysis_id=result.analysis_id,
                status="rejected",
                message=result.error_message,
                processing_time_seconds=result.processing_time_seconds,
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=result.error_message or "분석 중 오류가 발생했습니다.",
            )

    finally:
        # 임시 파일 정리
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/disclaimer")
async def get_disclaimer():
    """면책 고지문 조회"""
    return {
        "pre_analysis": governance.get_disclaimer("pre_analysis"),
        "post_analysis": governance.get_disclaimer("post_analysis"),
        "data_usage": governance.get_disclaimer("data_usage"),
    }


@app.get("/data-policy")
async def get_data_policy():
    """데이터 보관/삭제 정책 조회"""
    return governance.get_data_retention_policy()


# ─── Error Handlers ───


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """전역 에러 핸들러"""
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "서버 내부 오류가 발생했습니다.",
            "detail": str(exc) if config.debug else None,
        },
    )


def start_server():
    """서버 시작"""
    import uvicorn
    uvicorn.run(
        "velora.api.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=config.debug,
    )


if __name__ == "__main__":
    start_server()
