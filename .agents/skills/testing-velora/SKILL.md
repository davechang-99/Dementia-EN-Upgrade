# Testing VELORA System

## Overview
VELORA is a voice-based cognitive health screening service with a FastAPI backend. Testing involves verifying API endpoints, audio processing pipeline, PII masking, and consent management.

## Dependencies
```bash
pip install librosa soundfile numpy scipy scikit-learn fastapi uvicorn python-multipart pydantic
```

## Starting the Server
```bash
cd /home/ubuntu/repos/Dementia-EN-Upgrade
uvicorn velora.api.main:app --host 0.0.0.0 --port 8000 &
```

## Generating Test Audio Files
The system requires WAV files >= 30 seconds at 16kHz mono. Generate synthetic test files:
```python
import numpy as np
import soundfile as sf

sr = 16000
duration = 35  # Must be >= 30 seconds
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
signal = 0.3 * np.sin(2*np.pi*200*t) + 0.2 * np.sin(2*np.pi*400*t)
envelope = 0.5 + 0.5 * np.sin(2*np.pi*3*t)
signal = (signal * envelope + 0.02 * np.random.randn(len(signal)))
signal = signal / np.max(np.abs(signal)) * 0.8
sf.write('/tmp/test_audio.wav', signal, sr)
```

## Key API Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check (returns status, version) |
| `/health` | GET | Health check |
| `/consent/required` | GET | List required consent items (5 types) |
| `/consent` | POST | Submit consents, get token |
| `/analyze` | POST | Upload audio for full analysis pipeline |
| `/disclaimer` | GET | Get disclaimer texts |
| `/data-policy` | GET | Get data retention policy |

## Testing Consent Flow
```bash
# Valid consent (all 3 required = true)
curl -X POST http://localhost:8000/consent \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "consents": {"data_collection": true, "data_analysis": true, "non_medical_disclaimer": true}}'

# Invalid consent (should return 400)
curl -X POST http://localhost:8000/consent \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test", "consents": {"data_collection": false, "data_analysis": true, "non_medical_disclaimer": true}}'
```

## Testing Audio Analysis
```bash
curl -X POST http://localhost:8000/analyze \
  -F "audio_file=@/tmp/test_audio.wav" \
  -F "user_id=test_user"
```

### Expected Results (Without Trained Models)
- `risk_score`: 0.5 (default fallback when no models are loaded)
- `risk_level`: "caution"
- `risk.label`: "주의" (Korean)
- `risk.color`: "#FF9800"
- Processing time: ~15-20 seconds for a 35s audio file

### Audio Rejection
- Files shorter than 30 seconds are rejected with `status: "rejected"`
- Files with very low SNR may also be rejected

## Testing PII Detection (Direct Module)
```python
from velora.modules.governance import DataGovernanceModule
gov = DataGovernanceModule()

# Phone numbers: 010-1234-5678 -> ***-****-****
# Resident IDs: 900101-1234567 -> ***-*******
# Emails: test@example.com -> ****@****.***
result = gov.detect_pii('전화번호 010-1234-5678')
print(result.masked_text)  # Contains ***-****-****
```

## Important Notes
- Without pre-trained models, the risk score always defaults to 0.5
- The `/analyze` endpoint runs synchronously and takes ~15-20s per file
- ViT features require torch/transformers (commented out in requirements.txt)
- pyannote speaker diarization requires a HuggingFace token; fallback uses energy-based method
- All user-facing strings are in Korean
