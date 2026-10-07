"""
Cloud Stem Separation API Client for Suno Prep & Sanitizer.

Enables neural stem separation (vocals / instrumental) using remote cloud APIs
without downloading any heavy machine learning models, weights, or PyTorch checkpoints
to the local machine.

Supported Cloud Providers:
1. Replicate API (Demucs / UVR models via Replicate predictions API)
2. Hugging Face Inference & Spaces API (Remote endpoints via REST)
3. Custom REST Endpoint (Self-hosted or third-party stem separation webhook)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple
import soundfile as sf
import numpy as np


CONFIG_FILE_PATH = os.path.join(os.path.expanduser("~"), ".suno_sanitizer_cloud_config.json")


@dataclass
class CloudApiConfig:
    """Configuration for Cloud Stem Separation APIs."""
    provider: str = "replicate"                     # "replicate", "huggingface", or "custom"
    api_token: str = ""                              # API Key / Bearer token
    endpoint_url: str = ""                           # Custom endpoint URL or HF Space URL
    model_version: str = "cjwbw/demucs"              # Replicate model or HF model
    timeout_seconds: int = 180                       # Cloud job timeout
    save_config: bool = True

    @classmethod
    def load(cls) -> "CloudApiConfig":
        """Loads configuration from persistent encrypted .sn vault."""
        try:
            from config_manager import GhostWaveConfig
            sn_cfg = GhostWaveConfig.load()
            return cls(
                provider=sn_cfg.cloud_provider,
                api_token=sn_cfg.cloud_api_token,
                endpoint_url=sn_cfg.cloud_endpoint_url,
                model_version=sn_cfg.cloud_model_version,
                timeout_seconds=sn_cfg.cloud_timeout_seconds
            )
        except Exception:
            return cls()

    def save(self) -> None:
        """Saves current configuration to persistent encrypted .sn vault."""
        try:
            from config_manager import GhostWaveConfig
            sn_cfg = GhostWaveConfig.load()
            sn_cfg.cloud_provider = self.provider
            sn_cfg.cloud_api_token = self.api_token
            sn_cfg.cloud_endpoint_url = self.endpoint_url
            sn_cfg.cloud_model_version = self.model_version
            sn_cfg.cloud_timeout_seconds = self.timeout_seconds
            sn_cfg.save()
        except Exception:
            pass


@dataclass
class CloudSeparationResult:
    """Result of remote cloud stem separation."""
    success: bool
    instrumental_audio: Optional[np.ndarray] = None
    vocal_audio: Optional[np.ndarray] = None
    sample_rate: int = 44100
    error_message: Optional[str] = None
    logs: list[str] = field(default_factory=list)


class CloudStemClient:
    """
    Client for executing remote stem separation via Cloud APIs.
    Zero local model storage: only receives the rendered audio stem.
    """

    def __init__(self, config: Optional[CloudApiConfig] = None):
        self.config = config or CloudApiConfig.load()

    def separate_stems_remote(
        self,
        audio_file_path: str,
        progress_callback: Optional[Callable[[int, str], None]] = None
    ) -> CloudSeparationResult:
        """
        Submits audio to remote cloud API and downloads separated audio stems.
        """
        logs = []

        def log(pct: int, msg: str):
            logs.append(msg)
            if progress_callback:
                progress_callback(pct, msg)

        if not os.path.exists(audio_file_path):
            return CloudSeparationResult(
                success=False,
                error_message=f"File not found: {audio_file_path}",
                logs=logs
            )

        provider = self.config.provider.lower()
        log(10, f"Initializing remote stem separation via Cloud Provider: {provider.upper()}...")

        if provider == "replicate":
            return self._separate_replicate(audio_file_path, log, logs)
        elif provider == "huggingface":
            return self._separate_huggingface(audio_file_path, log, logs)
        elif provider == "custom":
            return self._separate_custom(audio_file_path, log, logs)
        else:
            err = f"Unsupported cloud provider: '{provider}'. Choose 'replicate', 'huggingface', or 'custom'."
            log(100, err)
            return CloudSeparationResult(success=False, error_message=err, logs=logs)

    def _separate_replicate(
        self,
        audio_file_path: str,
        log: Callable[[int, str], None],
        logs: list[str]
    ) -> CloudSeparationResult:
        """
        Sends audio to Replicate API for Demucs separation.
        Requires an active REPLICATE_API_TOKEN.
        """
        token = self.config.api_token or os.environ.get("REPLICATE_API_TOKEN", "")
        if not token:
            err = (
                "Replicate API Token is missing. Please configure your API token in Cloud API Settings "
                "or set the REPLICATE_API_TOKEN environment variable."
            )
            log(100, err)
            return CloudSeparationResult(success=False, error_message=err, logs=logs)

        import base64

        log(20, "Encoding audio payload for secure Cloud API transmission...")
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()

        ext = Path(audio_file_path).suffix.lower()
        mime = "audio/wav" if ext == ".wav" else ("audio/mpeg" if ext == ".mp3" else "application/octet-stream")
        data_uri = f"data:{mime};base64,{base64.b64encode(audio_bytes).decode('ascii')}"

        # Replicate model endpoint
        # Default model: "cjwbw/demucs" or user-configured
        create_url = "https://api.replicate.com/v1/predictions"
        headers = {
            "Authorization": f"Token {token}",
            "Content-Type": "application/json"
        }

        # Replicate request payload
        # Standard cjwbw/demucs deployment uses version or model-based predictions
        payload = {
            "version": "25a173c086e22410ed5e644737da3273b307586ce47304622790802f530c37a4",
            "input": {
                "audio": data_uri,
                "two_stems": "vocals"
            }
        }

        log(35, "Submitting task to Replicate Cloud API...")
        req = urllib.request.Request(create_url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as ex:
            err_body = ex.read().decode("utf-8", errors="ignore")
            err = f"Replicate API submission failed ({ex.code}): {err_body}"
            log(100, err)
            return CloudSeparationResult(success=False, error_message=err, logs=logs)
        except Exception as ex:
            err = f"Connection to Replicate failed: {str(ex)}"
            log(100, err)
            return CloudSeparationResult(success=False, error_message=err, logs=logs)

        pred_id = resp_data.get("id")
        get_url = resp_data.get("urls", {}).get("get", f"https://api.replicate.com/v1/predictions/{pred_id}")
        log(45, f"Prediction queued on Cloud (ID: {pred_id}). Polling remote status...")

        # Poll prediction status
        start_t = time.time()
        poll_headers = {"Authorization": f"Token {token}"}

        while time.time() - start_t < self.config.timeout_seconds:
            time.sleep(3)
            try:
                poll_req = urllib.request.Request(get_url, headers=poll_headers)
                with urllib.request.urlopen(poll_req, timeout=15) as poll_resp:
                    status_data = json.loads(poll_resp.read().decode("utf-8"))
            except Exception as poll_ex:
                log(50, f"Polling check retry: {poll_ex}")
                continue

            status = status_data.get("status")
            if status == "succeeded":
                log(80, "Remote stem separation completed in Cloud! Downloading separated audio stems...")
                output_urls = status_data.get("output", {})
                return self._download_and_load_stems(output_urls, log, logs)
            elif status == "failed":
                err = f"Replicate prediction failed: {status_data.get('error', 'Unknown remote error')}"
                log(100, err)
                return CloudSeparationResult(success=False, error_message=err, logs=logs)
            elif status == "canceled":
                err = "Replicate task was canceled."
                log(100, err)
                return CloudSeparationResult(success=False, error_message=err, logs=logs)

            elapsed = int(time.time() - start_t)
            log(55 + min(20, elapsed // 3), f"Cloud worker processing stem separation... ({elapsed}s elapsed)")

        timeout_err = f"Cloud stem separation timed out after {self.config.timeout_seconds} seconds."
        log(100, timeout_err)
        return CloudSeparationResult(success=False, error_message=timeout_err, logs=logs)

    def _separate_huggingface(
        self,
        audio_file_path: str,
        log: Callable[[int, str], None],
        logs: list[str]
    ) -> CloudSeparationResult:
        """
        Sends audio to Hugging Face Inference API or custom Space.
        """
        token = self.config.api_token or os.environ.get("HF_TOKEN", "")
        endpoint = self.config.endpoint_url.strip()
        if not endpoint:
            # Default to public Demucs HF Space / router
            endpoint = "https://api-inference.huggingface.co/models/facebook/demucs"

        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        log(25, f"Reading audio for Hugging Face Cloud transmission: {os.path.basename(audio_file_path)}...")
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()

        log(40, f"Sending request to Hugging Face Cloud endpoint ({endpoint})...")
        req = urllib.request.Request(endpoint, data=audio_bytes, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                content_type = resp.headers.get("Content-Type", "")
                resp_bytes = resp.read()

            log(80, "Received response from Hugging Face Cloud. Parsing audio output...")
            # If binary audio returned directly
            if "audio" in content_type or resp_bytes[:4] in (b"RIFF", b"OggS", b"ID3\x03", b"\xff\xfb"):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tf.write(resp_bytes)
                    temp_out = tf.name

                data, sr = sf.read(temp_out, dtype='float32')
                try:
                    os.remove(temp_out)
                except OSError:
                    pass
                log(100, "Successfully loaded remote separated stem!")
                return CloudSeparationResult(
                    success=True,
                    instrumental_audio=data,
                    sample_rate=sr,
                    logs=logs
                )
            else:
                # JSON with output links or base64
                parsed = json.loads(resp_bytes.decode("utf-8"))
                return self._download_and_load_stems(parsed, log, logs)

        except urllib.error.HTTPError as ex:
            err_body = ex.read().decode("utf-8", errors="ignore")
            err = f"Hugging Face Cloud API error ({ex.code}): {err_body}"
            log(100, err)
            return CloudSeparationResult(success=False, error_message=err, logs=logs)
        except Exception as ex:
            err = f"Hugging Face request failed: {str(ex)}"
            log(100, err)
            return CloudSeparationResult(success=False, error_message=err, logs=logs)

    def _separate_custom(
        self,
        audio_file_path: str,
        log: Callable[[int, str], None],
        logs: list[str]
    ) -> CloudSeparationResult:
        """
        Sends audio to a user-configured custom REST / Webhook endpoint.
        """
        endpoint = self.config.endpoint_url.strip()
        if not endpoint:
            err = "Custom endpoint URL is not configured. Please specify an endpoint in Cloud API Settings."
            log(100, err)
            return CloudSeparationResult(success=False, error_message=err, logs=logs)

        import mimetypes
        boundary = "----SunoSanitizerBoundary" + str(int(time.time()))
        headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
        if self.config.api_token:
            headers["Authorization"] = f"Bearer {self.config.api_token}"

        filename = os.path.basename(audio_file_path)
        mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        with open(audio_file_path, "rb") as f:
            file_data = f.read()

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="audio"; filename="{filename}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode("utf-8") + file_data + f"\r\n--{boundary}--\r\n".encode("utf-8")

        log(40, f"Posting audio to custom Cloud endpoint: {endpoint}...")
        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:
                data = resp.read()
                try:
                    json_data = json.loads(data.decode("utf-8"))
                    return self._download_and_load_stems(json_data, log, logs)
                except Exception:
                    # Binary response
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        tf.write(data)
                        tf_path = tf.name
                    audio_out, sr = sf.read(tf_path, dtype='float32')
                    try:
                        os.remove(tf_path)
                    except OSError:
                        pass
                    log(100, "Successfully loaded custom remote stem separation output!")
                    return CloudSeparationResult(
                        success=True,
                        instrumental_audio=audio_out,
                        sample_rate=sr,
                        logs=logs
                    )
        except Exception as ex:
            err = f"Custom endpoint request failed: {str(ex)}"
            log(100, err)
            return CloudSeparationResult(success=False, error_message=err, logs=logs)

    def _download_and_load_stems(
        self,
        output_data: dict | list | str,
        log: Callable[[int, str], None],
        logs: list[str]
    ) -> CloudSeparationResult:
        """
        Helper to download audio URLs returned by Cloud APIs and load as numpy arrays.
        """
        inst_url = None
        vocal_url = None

        if isinstance(output_data, dict):
            # Inspect keys like 'no_vocals', 'instrumental', 'vocals', etc.
            for k, v in output_data.items():
                k_lower = str(k).lower()
                if "no_vocals" in k_lower or "instrumental" in k_lower or "accompaniment" in k_lower:
                    inst_url = v
                elif "vocal" in k_lower and "no_vocal" not in k_lower:
                    vocal_url = v
            # Fallback if dictionary has integer keys or generic 'output'
            if not inst_url and "output" in output_data:
                inst_url = output_data["output"]
        elif isinstance(output_data, list):
            for item in output_data:
                item_str = str(item).lower()
                if "no_vocals" in item_str or "instrumental" in item_str:
                    inst_url = item
                elif "vocal" in item_str and "no_vocal" not in item_str:
                    vocal_url = item
            if not inst_url and len(output_data) > 0:
                inst_url = output_data[0]
        elif isinstance(output_data, str):
            inst_url = output_data

        if not inst_url and not vocal_url:
            err = f"Could not parse valid audio stem URLs from remote response: {output_data}"
            log(100, err)
            return CloudSeparationResult(success=False, error_message=err, logs=logs)

        def _fetch_url(url: str) -> Tuple[np.ndarray, int]:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                tmp_p = tf.name
            try:
                urllib.request.urlretrieve(url, tmp_p)
                d, sr = sf.read(tmp_p, dtype='float32')
                return d, sr
            finally:
                if os.path.exists(tmp_p):
                    try:
                        os.remove(tmp_p)
                    except OSError:
                        pass

        inst_data = None
        sr = 44100
        if inst_url:
            log(85, "Downloading separated instrumental stem from Cloud...")
            inst_data, sr = _fetch_url(inst_url)

        vocal_data = None
        if vocal_url:
            log(92, "Downloading separated vocal stem from Cloud...")
            vocal_data, sr = _fetch_url(vocal_url)

        log(100, "Remote stem download complete! Zero local model weights used.")
        return CloudSeparationResult(
            success=True,
            instrumental_audio=inst_data,
            vocal_audio=vocal_data,
            sample_rate=sr,
            logs=logs
        )
