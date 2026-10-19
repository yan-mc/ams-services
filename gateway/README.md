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
│  └─ voices/                # scanner.py（只读扫描）、pth_parser.py（可替换解析器）
├─ tests/                    # pytest（22 个用例）
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
| `AMS_RVC_PYTHON` | 空 | RVC runtime 的 python.exe；配置后 `/voices/scan` 用其子进程解析 pth 元数据 |

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
# 22 passed
```

覆盖：

| 文件 | 覆盖点 |
|---|---|
| `tests/test_health.py` | token 必需/错误 token/health 字段/token 文件 |
| `tests/test_jobs.py` | 创建/查询/列表/未知类型/非法参数/取消运行中任务/取消幂等/GPU 串行/重启后历史可查 |
| `tests/test_sse.py` | 终态任务历史回放、取消事件、404 |
| `tests/test_voices.py` | 命名规则、缺文件/不同名/小体积、解析器注入、pending_parse、只读性、API 扫描 |

真实服务冒烟（会启动 uvicorn、写 `.ams_home_dev/`）：

```powershell
.\scripts\smoke.ps1
```

---

## 9. 接入真实后端（Runner 注册点）

`app/jobs/runner.py`：

```python
from app.jobs.runner import RunnerRegistry, ArtifactSpec, build_default_registry

registry = build_default_registry()
# 真实后端示例（M0.7 后）：训练/分离服务以子进程或 HTTP 客户端形式实现 Runner
# registry.register("separate.pymss", lambda: PymssRunner(...), gpu=True,
#                   description="pymss 分离（显存型）")
```

约定：

- `Runner.run(ctx)` 内通过 `await ctx.progress(percent, stage, message?)` 上报进度、`ctx.log(level, msg)` 写任务日志；
- 长任务用子进程时，把 `kill_process_tree(pid)` 注册到 `ctx.add_kill_callback(...)`，取消时 Gateway 会杀进程树；
- 返回 `ArtifactSpec` 列表，Gateway 负责落库、算大小/sha256、发 SSE artifact 事件；
- GPU 任务注册时传 `gpu=True`，自动进 GPU 串行队列（`gpu_lock` 字段可见 waiting/held）。

---

## 10. 已知限制与后续（M0.4+）

| 限制 | 说明 / 计划 |
|---|---|
| mock 任务无真实后端 | M0.7 接 RVC/pymss 冒烟；Runner 已留注册点 |
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
