# AMS Gateway（MVP）

> 位置：`08_项目代码/ams-services/gateway`｜阶段：**M0.3 Gateway MVP**
> 依据：`06_方案设计/00_总体设计与约定.md` 第 3/6/8 节、`07_执行计划/M0_计划.md` M0.3/M0.4
> 技术栈：Python 3.12 + FastAPI + SQLite（stdlib sqlite3）+ SSE + Bearer token
> 状态：M0 技术验证版；只做**任务/服务/产物网关**，不含前端与真实推理后端。

---

## 1. 它是什么

AMS 的本地任务网关：所有插件/页面/启动器/CLI 都只通过它操作耗时任务与产物（00 第 3/8 节）。

- **唯一入口**：`http://127.0.0.1:<port>/api/v1`，端口动态分配并写入 `AMS_HOME/runtime/ports.json`；
- **鉴权**：所有 `/api/v1` 请求需 `Authorization: Bearer <token>`（包括 `/health`）；token 首次启动生成到 `AMS_HOME/runtime/gateway.token`；
- **任务引擎**：SQLite 持久化；状态机 `queued → running → succeeded/failed/cancelled`，支持 `stopping` 取消中间态；
- **GPU 锁**：GPU/显存型任务串行（单独工作线程），CPU 任务可并发（默认 2）；
- **SSE**：`GET /jobs/{id}/events` 先回放历史事件，再实时推送，终态后关闭；
- **音色库扫描**：`POST /voices/scan` 只读扫描 `voices/<名称>/<名称>.pth(+同名 .index)`，校验命名/重复/缺失，pth 元数据解析可替换（RVC runtime 子进程/待解析）。

M0 内置 mock 任务：`ping`（快速跑通全链路）与 `mock.long`（子进程 sleep；验证进度、取消杀进程树、GPU 串行）。训练/分离等真实后端的接入点是 `app/jobs/runner.py` 的 `RunnerRegistry`。

---

## 2. 目录结构

```text
gateway/
├─ pyproject.toml            # 包元数据 + 依赖（fastapi/uvicorn；dev: pytest/httpx）
├─ requirements.txt          # 运行时依赖（备用）
├─ requirements-dev.txt      # 开发依赖（备用）
├─ README.md                 # 本文档
├─ app/
│  ├─ main.py                # create_app() + uvicorn 启动（动态端口/ports.json）
│  ├─ config.py              # Settings：AMS_HOME 路径/回环校验/环境变量
│  ├─ db.py                  # SQLite：jobs/job_events/artifacts + 原子状态迁移
│  ├─ security.py            # Bearer token 读取/生成/常量时间校验
│  ├─ errors.py              # 统一错误响应 + 异常处理器
│  ├─ models.py              # pydantic 请求/响应模型
│  ├─ logging_setup.py       # 控制台 + 滚动文件日志
│  ├─ api/                   # health.py / jobs.py / voices.py
│  ├─ jobs/                  # events.py（SSE 总线）、runner.py（Runner 注册/mock）、service.py（引擎）
│  ├─ runners/               # 真实后端 Runner：msvc/rvc/pymss + scripts/（外部解释器脚本）+ registry.py
│  └─ voices/                # scanner.py（只读扫描）、pth_parser.py（可替换解析器）
├─ tests/                    # pytest（37 passed / 1 skipped；含 fake backend 与真实 smoke 开关）
├─ scripts/smoke.ps1         # 真实服务冒烟脚本（启动->health->ping->SSE->停止）
├─ .venv/                    # 本目录内独立虚拟环境（gitignore）
└─ .ams_home_dev/            # 未设置 AMS_HOME 时的开发默认数据目录（gitignore）
```

---

## 3. 快速开始（Windows PowerShell，Python 3.12）

> 不安装系统级包；所有依赖只进 `gateway/.venv`。

```powershell
Set-Location F:\Deepseek_harness\AI音乐处理\08_项目代码\ams-services\gateway

# 1) 用系统 Python 3.12 建 venv（已内置则跳过）
py -3.12 -m venv .venv

# 2) 安装（editable + dev 依赖）
.venv\Scripts\python.exe -m pip install -e ".[dev]"

# 3) 启动（推荐显式指定 AMS_HOME 与 0=动态端口）
.venv\Scripts\python.exe -m app --ams-home F:\tmp\AMS --port 0 --log-level INFO
# 等价：.venv\Scripts\ams-gateway.exe --ams-home F:\tmp\AMS --port 0
```

启动后：

| 文件 | 内容 |
|---|---|
| `AMS_HOME/runtime/ports.json` | `{gateway:{host,port,pid,api_prefix,token_file,started_at}}` |
| `AMS_HOME/runtime/gateway.token` | 64 字符随机 token（32 字节） |
| `AMS_HOME/config/state.db` | SQLite（jobs/job_events/artifacts） |
| `AMS_HOME/logs/gateway/gateway.log` | 滚动日志（10MB x5） |
| `AMS_HOME/artifacts/<job_id>/` | 任务产物（mock 任务写 JSON） |

> **开发默认 AMS_HOME**：若未传 `--ams-home` 且未设置环境变量，默认落在 `gateway/.ams_home_dev`（避免误写系统目录）；正式运行时由启动器显式注入 `AMS_HOME`。

---

## 4. HTTP API

前缀 `/api/v1`；请求/响应 JSON；除健康检查外均需 Bearer token；错误统一为：

```json
{"ok": false, "error": {"code": "JOB_NOT_FOUND", "message": "任务不存在：job_xxx", "detail": null}}
```

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康/版本/DB/队列/worker/job_types/voices 信息 |
| GET | `/jobs` | 任务列表，支持 `status`/`type`/`project_id`/`limit`/`offset` |
| POST | `/jobs` | 创建任务 `{type, params?, project_id?, agent_session_id?}`（201） |
| GET | `/jobs/{id}` | 任务详情（Job 对象） |
| POST | `/jobs/{id}/cancel` | 取消（排队直接 cancelled；运行中 → stopping → cancelled） |
| GET | `/jobs/{id}/events` | SSE：历史回放 + 实时事件（`event: status/progress/log/artifact`，`id:` 可作 Last-Event-ID） |
| GET | `/jobs/{id}/artifacts` | 产物列表 |
| GET | `/artifacts/{id}/download` | 下载产物（仅限 Gateway 产物目录内） |
| POST | `/voices/scan` | 只读音色库扫描（body 可选 `{"path": "..."}`） |
| GET | `/voices` | 只读扫描默认 voices 目录（便捷接口） |

**Job 对象**（字段对齐 00 第 8 节）：`id/type/status/progress/stage/params/project_id/agent_session_id/created_at/started_at/finished_at/error/engine/gpu_lock/cancel_requested/resumable`。

**内置任务类型**：

| type | GPU | 说明 |
|---|---|---|
| `ping` | 否 | 快速任务；产出 `ping-result.json` |
| `mock.long` | 是 | `{"seconds": 8}`（0.2–300）；子进程 sleep，用于验证进度/取消/GPU 串行 |
| `convert.msvc` | 否 | MSVC→RVC v2 pth；参数 `input_path/out_dir/[name]/[reference_pth]/[no_cache]`；产物 `.pth` + `conversion.json`（见 §9） |
| `convert.rvc` | 是 | RVC 换声；参数 `input_path/voice_name/[f0_up_key]/[index_rate]/[device]`；产物 wav + meta json |
| `separate.pymss` | 是 | pymss 分离；参数 `input_path/model/[device]/[chunk]`；产物每 stem wav + `separation-manifest.json` |

**SSE 示例事件**：

```text
id: 3
event: status
data: {"status":"running","progress":0.0,"stage":"starting",...}

id: 5
event: progress
data: {"progress":2.0,"stage":"mock.starting"}
```

---

## 5. curl / PowerShell 示例

### 5.1 bash / curl

```bash
AMS_HOME=/tmp/ams
TOKEN=$(cat "$AMS_HOME/runtime/gateway.token")
PORT=$(python -c "import json;print(json.load(open('$AMS_HOME/runtime/ports.json'))['gateway']['port'])")
BASE="http://127.0.0.1:$PORT/api/v1"

# 健康检查
curl -s -H "Authorization: Bearer $TOKEN" "$BASE/health"

# 创建 ping 任务
JOB=$(curl -s -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"type":"ping","params":{}}' "$BASE/jobs")
echo "$JOB"

# 查询 / SSE / 产物
curl -s -H "Authorization: Bearer $TOKEN" "$BASE/jobs/<job_id>"
curl -s -N -H "Authorization: Bearer $TOKEN" "$BASE/jobs/<job_id>/events"
curl -s -H "Authorization: Bearer $TOKEN" "$BASE/jobs/<job_id>/artifacts"
curl -s -OJ -H "Authorization: Bearer $TOKEN" "$BASE/artifacts/<artifact_id>/download"

# 取消长任务
curl -s -X POST -H "Authorization: Bearer $TOKEN" "$BASE/jobs/<job_id>/cancel"

# 扫描音色库
curl -s -X POST -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"path":"/path/to/AMS_HOME/voices"}' "$BASE/voices/scan"
```

### 5.2 PowerShell

```powershell
$home = "F:\tmp\AMS"
$ports = Get-Content -Raw (Join-Path $home "runtime\ports.json") | ConvertFrom-Json
$token = (Get-Content -Raw (Join-Path $home "runtime\gateway.token")).Trim()
$headers = @{ Authorization = "Bearer $token" }
$base = "http://127.0.0.1:$($ports.gateway.port)/api/v1"

Invoke-RestMethod "$base/health" -Headers $headers
$job = Invoke-RestMethod "$base/jobs" -Method Post -Headers $headers -ContentType "application/json" \
  -Body '{"type":"ping","params":{}}'
Invoke-RestMethod "$base/jobs/$($job.id)" -Headers $headers
```

---

## 6. 配置（环境变量）

| 变量 | 默认 | 说明 |
|---|---|---|
| `AMS_HOME` | （开发默认 `gateway/.ams_home_dev`） | 数据根目录；正式运行时必填 |
| `AMS_GATEWAY_HOST` | `127.0.0.1` | 仅允许回环（localhost/::1）；需显式 `AMS_GATEWAY_ALLOW_NON_LOOPBACK=1` 才允许其他地址（不推荐） |
| `AMS_GATEWAY_PORT` | `0` | 0 = 动态分配，实际写入 ports.json |
| `AMS_GATEWAY_CPU_CONCURRENCY` | `2` | CPU 工作线程数 |
| `AMS_GATEWAY_LOG_LEVEL` | `INFO` | DEBUG/INFO/WARNING/ERROR |
| `AMS_GATEWAY_EVENT_RING` | `500` | 每任务保留事件数（日志环，超出删最旧） |
| `AMS_GATEWAY_SHUTDOWN_GRACE` | `2.0` | 关闭宽限秒数 |
| `AMS_RVC_PYTHON` | 空 | RVC runtime 的 python.exe；`/voices/scan` 解析 pth + `convert.msvc/convert.rvc` 必需 |
| `AMS_RVC_ROOT` | 空 | RVC 整合包根目录（`assets/weights`、`assets/hubert`、ffmpeg）；`convert.rvc` 必需 |
| `AMS_PYMSS_PYTHON` | 空 | pymss 独立 venv 的 python.exe；`separate.pymss` 必需 |
| `AMS_PYMSS_ROOT` | 空 | pymss 环境根目录（模型目录 = `<root>/models`）；`separate.pymss` 必需 |
| `AMS_MSVC_TOOL` | 空 | `ams-tools/msvc2pth/msvc2pth.py` 路径；`convert.msvc` 必需 |

---

## 7. voices 扫描规则（只读）

对应 00 第 6 节；扫描器**不修改任何文件**：

```text
AMS_HOME/voices/<名称>/<名称>.pth          # 必需，同名
AMS_HOME/voices/<名称>/<名称>.index        # 可选，同名（无 index 仍可用，index_rate=0）
```

校验项（示例）：文件夹名非法字符/首尾空格/Windows 保留名；`.pth` 缺失、不同名、重复、为空、体积过小；`.index` 同名/缺失（缺失仅 info）；pth 元数据解析状态（ok / pending_parse / error）。返回结构含 `count/summary/issues/voices[]` 与 `readonly: true`。

pth 解析器：`app/voices/pth_parser.py` 中 `build_parser()`——配置了 `AMS_RVC_PYTHON` 时启用 `RvcSubprocessParser`（只读 torch.load），否则 `NullParser` 返回 `pending_parse`。测试可注入 FakeParser，无需真实权重。

---

## 8. 测试

```powershell
.venv\Scripts\python.exe -m pytest -q
# 37 passed, 1 skipped（真实后端 smoke 默认 skip）
```

覆盖：

| 文件 | 覆盖点 |
|---|---|
| `tests/test_health.py` | token 必需/错误 token/health 字段/token 文件 |
| `tests/test_jobs.py` | 创建/查询/列表/未知类型/非法参数/取消运行中任务/取消幂等/GPU 串行/重启后历史可查 |
| `tests/test_sse.py` | 终态任务历史回放、取消事件、404 |
| `tests/test_voices.py` | 命名规则、缺文件/不同名/小体积、解析器注入、pending_parse、只读性、API 扫描 |
| `tests/test_runners.py` | 三个 Runner：成功/失败/取消(杀进程树)/未配置/路径不存在/参数校验；App 注册与端到端 Job（fake backend） |

真实服务冒烟（会启动 uvicorn、写 `.ams_home_dev/`）：

```powershell
.\scripts\smoke.ps1
```

---

## 9. 真实后端 Runner（M1）

三个外部进程后端已接入（`app/runners/`），默认注册（`main.create_app`），`/health` 的 `job_types` 可见；
配置缺失时 Job 失败并返回可操作错误码 `BACKEND_NOT_CONFIGURED`（附 `hint`），**不会静默回退 mock**。

### 9.1 配置

| 变量 | 作用 | 必需任务 |
|---|---|---|
| `AMS_RVC_PYTHON` | RVC runtime python.exe | convert.msvc / convert.rvc（另供 /voices/scan 解析 pth） |
| `AMS_RVC_ROOT` | RVC 整合包根目录（assets/weights、hubert、rmvpe、ffmpeg） | convert.rvc |
| `AMS_PYMSS_PYTHON` | pymss 独立 venv 的 python.exe | separate.pymss |
| `AMS_PYMSS_ROOT` | pymss 环境根目录（模型目录 = `<root>/models`） | separate.pymss |
| `AMS_MSVC_TOOL` | `ams-tools/msvc2pth/msvc2pth.py` 路径 | convert.msvc |

配置在**每次任务执行时读取**（改环境变量后重启服务即可生效；测试可用 monkeypatch 注入）。
错误码：`BACKEND_NOT_CONFIGURED`（环境变量为空）、`BACKEND_PATH_NOT_FOUND`（python/目录/脚本不存在）、
`INPUT_NOT_FOUND`、`INVALID_PARAMS`、`BACKEND_TIMEOUT`、`BACKEND_FAILED`、`BACKEND_PROTOCOL_ERROR`。

### 9.2 任务类型与参数

| type | GPU | 必需参数 | 可选参数 | 产物 |
|---|---|---|---|---|
| `convert.msvc` | 否 | `input_path`, `out_dir` | `name`, `reference_pth`, `no_cache` | `<out>/<name>/<name>.pth`、`conversion.json` |
| `convert.rvc` | 是 | `input_path`, `voice_name` | `f0_up_key`(0), `index_rate`(0.75), `device`(cuda:0) | `<work_dir>/<stem>_<voice>.wav` + 同名 `.json` meta |
| `separate.pymss` | 是 | `input_path`, `model` | `device`(auto), `chunk` | `<work_dir>/stems/*.wav` + `separation-manifest.json` |

- `convert.msvc`：RVC runtime + `msvc2pth.py`，解析 `--json` 报告（失败回退读取 `conversion.json`）；实时把工具 stdout/stderr 转任务日志，并按 `[cache]/[calibrate]` 推进阶段进度。
- `convert.rvc`：RVC runtime + `app/runners/scripts/rvc_convert.py`；脚本内含 **faiss 非 ASCII 路径兼容补丁**（RVC 原生会静默忽略 index）；推理期间用心跳进度（RVC 无回调）。
- `separate.pymss`：pymss 独立环境 + `scripts/pymss_separate.py`；使用 pymss `progress_callback` 流式进度，产物为各 stem wav + manifest。
- 输出统一写在 Job 的 `work_dir`（`AMS_HOME/artifacts/<job_id>/`）；Gateway 负责落库、算大小/sha256、发 SSE artifact 事件。
- GPU 标记：`convert.rvc` / `separate.pymss` = true，`convert.msvc` = false；GPU 任务自动串行（`gpu_lock`）。

### 9.3 调用示例（PowerShell）

```powershell
$h = @{ Authorization = "Bearer $token" }
$base = "http://127.0.0.1:<port>"

# 1) RVC 换声
$body = @{
  type   = "convert.rvc"
  params = @{ input_path = "F:\work\vocal.wav"; voice_name = "manbo"; f0_up_key = 0; index_rate = 0.75; device = "cuda:0" }
} | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri "$base/api/v1/jobs" -Headers $h -Body $body -ContentType "application/json"

# 2) pymss 分离
$body = @{
  type   = "separate.pymss"
  params = @{ input_path = "F:\work\song.wav"; model = "scnet_checkpoint_musdb18"; device = "auto" }
} | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri "$base/api/v1/jobs" -Headers $h -Body $body -ContentType "application/json"

# 3) MSVC -> pth
$body = @{
  type   = "convert.msvc"
  params = @{ input_path = "F:\work\voice.msvc"; out_dir = "F:\AMS\voices"; name = "voice1" }
} | ConvertTo-Json -Depth 4
Invoke-RestMethod -Method Post -Uri "$base/api/v1/jobs" -Headers $h -Body $body -ContentType "application/json"
```

错误示例（未配置后端时 Job 失败，不是 RUNNER_ERROR）：

```json
{"status":"failed","error":{"code":"BACKEND_NOT_CONFIGURED","message":"未配置 RVC runtime Python（环境变量 AMS_RVC_PYTHON 为空）","hint":"配置 AMS_RVC_PYTHON（RVC 整合包 runtime\\python.exe）"}}
```

### 9.4 子进程协议（脚本开发约定）

`app/runners/scripts/` 下脚本由外部解释器执行，stdout 每行一个 JSON：

```text
{"type":"progress","percent":0-100,"stage":"...","message":"..."}
{"type":"log","level":"info|warning|error","message":"..."}
{"type":"artifact","name":"...","path":"绝对路径","kind":"audio|model|report","meta":{}}
{"type":"result","ok":true,"artifacts":[...]?,"meta":{}}
{"type":"result","ok":false,"error":{"code":"...","message":"...","hint":"..."}}
```

Runner 行为：stderr 逐行转任务日志；取消时 `kill_process_tree`（含子进程）；超时杀进程树并返回 `BACKEND_TIMEOUT`。
RVC 脚本用 RVC runtime（Python 3.9）执行，pymss 脚本用 pymss venv（Python 3.12）执行——**Gateway venv 不安装 torch**。

### 9.5 测试

- 默认 `pytest -q`：**37 passed, 1 skipped**（真实 smoke 默认 skip）；fake backend 覆盖成功/失败/取消/未配置/路径不存在/参数校验。
- fake 脚本：`tests/fake_protocol_backend.py`（JSONL 协议）、`tests/fake_msvc2pth.py`（msvc CLI + JSON 报告）。
- 真实后端 smoke（手动）：设置 `AMS_TEST_REAL_BACKENDS=1`、`AMS_TEST_REAL_INPUT`（音频路径）、可选 `AMS_TEST_REAL_VOICE` 后运行：

```powershell
.venv\Scripts\python.exe -m pytest tests/test_runners.py -q -k real
```

- 本次实现已手工验证：`convert.rvc` 真机 **7.1s**（40k 单声道 wav，index 生效，RTF 0.096）；`separate.pymss` 真机 **6.4s**（GPU，4 stems + manifest，进度回调正常）。msvc 真实 `.msvc` 样例本机未找到，使用 fake tool 验证 CLI/JSON 解析。

### 9.6 与 mock 的关系

`ping` / `mock.long` 保持不变；真实后端注册在默认 registry 中（`register_backend_runners`）。
测试/实验可用 `register_backend_runners(registry, rvc_script=..., pymss_script=...)` 注入 fake 脚本。

---

## 10. 已知限制与后续（M0.4+）

| 限制 | 说明 / 计划 |
|---|---|
| 真实后端需外部环境 | 依赖 `AMS_RVC_PYTHON`/`AMS_RVC_ROOT`/`AMS_PYMSS_PYTHON`/`AMS_PYMSS_ROOT`/`AMS_MSVC_TOOL`（启动器注入）；未配置时任务失败并给 hint，不静默走 mock |
| 断点续跑未实现 | 重启会把 running/stopping 标 failed（code=GATEWAY_RESTARTED）；queued 自动重新入队 |
| pth 解析依赖外部 RVC runtime | 配置 `AMS_RVC_PYTHON` 后启用；M0.4 由启动器注入 |
| 无前端/无 WebUI | 插件与启动器作为客户端接入（M0.6） |
| 无 GPU 状态探测（nvidia-smi） | GPU 锁目前只做任务串行；M1 增加显存探测与排队原因细化 |
| 事件上限 | 每任务保留 `AMS_GATEWAY_EVENT_RING` 条（默认 500），超出删最旧 |
| 与 00 的差异 | 新增 `AMS_HOME/artifacts/`（产物）；开发默认 AMS_HOME 落在 gateway 内（避免误写系统目录）；新增 `GET /voices` 便捷接口 |

---

## 11. 安全与数据

- 只监听回环地址；token 常量时间比较；错误响应不泄露 token；
- 产物下载限制在 `AMS_HOME/artifacts/` 内（路径穿越防护）；
- 服务日志记录请求方法/路径/状态/耗时与 request-id，不记录请求体与 Authorization；
- 不访问外网、不下载模型；不修改 `voices/` 与用户源目录；
- 无遥测；数据全部在 `AMS_HOME` 本地。
