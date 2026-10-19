# ams-services

AMS（AI Music Studio）本地服务：Gateway + 各后端服务。

## 当前内容（M0）

| 服务 | 说明 |
|---|---|
| `gateway/` | FastAPI + SQLite 任务网关：jobs/SSE/artifacts/voices scan、token 鉴权、RunnerRegistry 扩展点 |

## 约定

- 每个服务独立 Python 3.12 + `.venv`（uv/pip），不共用旧整合包 3.9 环境。
- 所有服务只监听 127.0.0.1，由启动器/Gateway 管理。
- 不提交 `.venv/`、运行时数据、模型、音频。
