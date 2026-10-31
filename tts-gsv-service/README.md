# runtime.tts-gsv-core / tts_service（P1）

Windows 便携 GPT-SoVITS TTS 运行时与 headless 服务。P1 先打通
`Gateway -> tts.synthesize.gsv -> tts_service -> GPT-SoVITS`，UI/Agent 工具在 P2 接入。

## 1. 运行时目录（由 build_runtime.py 从官方 Windows 包生成）

```
runtime.tts-gsv-core/
├─ runtime/python.exe          # 官方/便携 Python + torch + GPT-SoVITS 依赖
├─ GPT_SoVITS/                 # 代码、configs、pretrained_models
├─ tools/                      # ffmpeg 等
├─ tts_service/tts_service.py  # AMS headless HTTP 服务
├─ runtime/Lib/site-packages/ams_gsv_paths.pth  # ASCII 相对路径，避免中文/编码问题
└─ manifest.json
```

构建（未来由 launcher feed 执行，P1 先手工验证）：

```powershell
python build_runtime.py --source F:\...\GPT-SoVITS-v2pro-20261006 `
  --out F:\AMS-Studio\data\components\runtime.tts-gsv-core --version 0.1.0
```

## 2. 环境变量

| 变量 | 含义 |
|---|---|
| `AMS_HOME` | AMS 安装数据根（launcher 注入） |
| `AMS_TTS_GSV_ROOT` | runtime.tts-gsv-core 根目录 |
| `AMS_TTS_GSV_HOME` | 音色/日志/tmp 根，默认 `<AMS_HOME>/tts-gsv` |

## 3. 音色包（`<AMS_TTS_GSV_HOME>/voices/<voice_id>/voice.json`）

```json
{
  "id": "xianyun_p0_v5turbo",
  "name": "闲云 V5turbo",
  "version": "v5turbo",
  "language": "zh",
  "sovits": "weights/xianyun_p0_v5turbo_e8_s4240_l32.pth",
  "gpt": "weights/xianyun_p0_v5turbo-e10.ckpt",
  "ref_audio": "ref/ref_01.wav",
  "ref_text": "与好友一起吃饭，无论是何种菜肴，都令人回味无穷。",
  "tags": ["原神", "闲云"],
  "created_at": "2026-10-31T21:00:00+08:00"
}
```

相对路径相对 `voice.json` 所在目录；参考音频必须是 3–10 秒。

## 4. Gateway 作业

```json
{
  "type": "tts.synthesize.gsv",
  "params": {
    "text": "机关术奥妙无穷，并非三言两语能道明。",
    "voice_id": "xianyun_p0_v5turbo",
    "speed": 1.0,
    "seed": 42
  }
}
```

P1 行为：每个作业启动一次 `tts_service`，加载权重→合成→`/shutdown`，确保
GPU 显存释放；后续 P1.5 可改常驻/空闲卸载。

## 5. 服务 API（仅回环）

- `GET /health`
- `POST /synthesize` `{text, speed, seed, output_path}`
- `POST /shutdown`

## 6. 安全/编码注意

- 只监听 `127.0.0.1`，token 由 Gateway 每次随机生成。
- 路径含中文时，服务内部走 Python 原生路径处理；`.pth` 必须保持 ASCII 相对路径。
- runtime 不写 C 盘；`TEMP/TMP` 指向 `AMS_TTS_GSV_HOME/tmp`。
