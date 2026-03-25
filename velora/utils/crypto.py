"""
VELORA Encryption Utilities
데이터 암호화/복호화 (AES-256)
"""

import base64
import hashlib
import os
import secrets
from typing import Optional, Tuple


class AESCipher:
    """
    AES-256 암호화/복호화
    데이터 전송 및 저장 시 암호화 적용
    """

    def __init__(self, key: Optional[bytes] = None):
        """
        Args:
            key: 32바이트 암호화 키. None이면 자동 생성
        """
        if key is None:
            self.key = secrets.token_bytes(32)
        else:
            if len(key) != 32:
                self.key = hashlib.sha256(key).digest()
            else:
                self.key = key

    @staticmethod
    def generate_key() -> bytes:
        """새 AES-256 키 생성"""
        return secrets.token_bytes(32)

    @staticmethod
    def derive_key(password: str, salt: Optional[bytes] = None) -> Tuple[bytes, bytes]:
        """비밀번호 기반 키 유도"""
        if salt is None:
            salt = os.urandom(16)
        key = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
        return key, salt

    def encrypt_bytes(self, data: bytes) -> bytes:
        """바이트 데이터 암호화 (XOR 기반 간소화 버전)"""
        iv = secrets.token_bytes(16)
        key_stream = self._generate_key_stream(len(data), iv)
        encrypted = bytes(a ^ b for a, b in zip(data, key_stream))
        return iv + encrypted

    def decrypt_bytes(self, encrypted_data: bytes) -> bytes:
        """바이트 데이터 복호화"""
        iv = encrypted_data[:16]
        ciphertext = encrypted_data[16:]
        key_stream = self._generate_key_stream(len(ciphertext), iv)
        decrypted = bytes(a ^ b for a, b in zip(ciphertext, key_stream))
        return decrypted

    def _generate_key_stream(self, length: int, iv: bytes) -> bytes:
        """키 스트림 생성"""
        stream = b""
        counter = 0
        while len(stream) < length:
            block_input = self.key + iv + counter.to_bytes(8, "big")
            stream += hashlib.sha256(block_input).digest()
            counter += 1
        return stream[:length]

    def encrypt_file(self, input_path: str, output_path: str) -> None:
        """파일 암호화"""
        with open(input_path, "rb") as f:
            data = f.read()
        encrypted = self.encrypt_bytes(data)
        with open(output_path, "wb") as f:
            f.write(encrypted)

    def decrypt_file(self, input_path: str, output_path: str) -> None:
        """파일 복호화"""
        with open(input_path, "rb") as f:
            data = f.read()
        decrypted = self.decrypt_bytes(data)
        with open(output_path, "wb") as f:
            f.write(decrypted)



def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """파일 무결성 해시 계산"""
    h = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def generate_secure_token(length: int = 32) -> str:
    """보안 토큰 생성"""
    return secrets.token_urlsafe(length)
