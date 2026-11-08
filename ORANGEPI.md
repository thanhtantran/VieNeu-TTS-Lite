# VieNeu-TTS trên Orange Pi (local / LAN only)

Hướng dẫn chạy **độc lập trên Orange Pi** (RK3588 / `aarch64`), phục vụ mạng nội bộ.  
Deploy Instabox + URL online: xem [README.md](README.md).

## Yêu cầu

- Orange Pi / board ARM64 Linux
- Python ≥ 3.10, [`uv`](https://github.com/astral-sh/uv)
- Chỉ dùng **VieNeu-TTS v3 Nano @ 24 kHz** (CPU/ONNX)
- **Không** copy `.venv` từ Instabox (x86) sang Orange Pi (ARM) không dùng được

## 1) Clone và dependency

```bash
git clone https://github.com/thanhtantran/VieNeu-TTS-Lite
cd /home/orangepi/workspace/VieNeu-TTS-Lite
uv sync
```

`pyproject.toml` đã có `linux/aarch64`.

## 2) Tải model về `models/` (cần internet 1 lần)

```bash
uv run python scripts/download_models.py
uv run python scripts/download_models.py --offline-check
```

Kết quả mong đợi:

```text
models/
  hub/                             # HF cache
  VieNeu-TTS-v3-Nano/              # flat ONNX → VIENEU_ONNX_DIR
  MOSS-Audio-Tokenizer-Nano-ONNX/  # tokenizer phụ trợ
```

Sau bước này có thể chạy với `HF_HUB_OFFLINE=1` (không cần internet).

## 3) Chạy API LAN

```bash
# optional Bearer cho /v1/*
# printf 'vn_xxx' > .vieneu_api_key && chmod 600 .vieneu_api_key

export VieNeuTTS_WEB_ENABLED=true
export PORT=8000
./scripts/run_lan_api.sh
```

Mặc định script sẽ:

- `HOST=0.0.0.0`
- `VIENEU_MODE=v3nano`
- `VIENEU_ONNX_DIR=models/VieNeu-TTS-v3-Nano`
- `HF_HOME=models/`
- `HF_HUB_OFFLINE=1`

Trong LAN (ví dụ IP board `192.168.88.140`):

- Web UI: `http://192.168.88.140:8000/`
- Health: `http://192.168.88.140:8000/health`
- API: `POST http://192.168.88.140:8000/v1/audio/speech`

```bash
curl -s http://192.168.88.140:8000/health

curl -N http://192.168.88.140:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -H "Authorization: Bearer ${VIENEU_API_KEY}" \
  -d '{"model":"vieneu-v3-nano","voice":"Mai Anh","input":"Xin chào","response_format":"wav","sample_rate":24000}' \
  -o out.wav
```

Nếu chưa set `VIENEU_API_KEY` / `.vieneu_api_key` thì `/v1/*` mở (không Bearer). Web UI không dùng Bearer; tắt UI bằng `VieNeuTTS_WEB_ENABLED=false`.

## Biến môi trường

| Biến | Ý nghĩa | Mặc định script |
|---|---|---|
| `VIENEU_ONNX_DIR` | flat Nano ONNX | `models/VieNeu-TTS-v3-Nano` |
| `HF_HOME` | gốc cache/model | `models/` |
| `HF_HUB_OFFLINE` | không gọi Hugging Face | `1` |
| `VIENEU_API_KEY` | Bearer cho `/v1/*` | (rỗng / đọc `.vieneu_api_key`) |
| `VieNeuTTS_WEB_ENABLED` | bật/tắt UI `/` | `true` |
| `HOST` / `PORT` | bind LAN | `0.0.0.0` / `8000` |
| `VIENEU_WEB_HISTORY_DIR` | WAV history của web | `models/web-history` |

## Lưu ý

- Mở firewall/LAN cho cổng `8000` nếu gọi từ máy khác
- 1 stream CPU là phù hợp; đừng kỳ vọng throughput kiểu GPU
- Khi chạy lần đầu cần chạy file `download_models.py` để tải model về
