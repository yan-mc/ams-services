# ams-services

> **AMS Studio 子项目 · 本地服务层**  
> AMS 的本地 Gateway 与各个 AI 音频后端（RVC、pymss、MSVC 等）的 HTTP 任务入口，只监听 127.0.0.1，由启动器管理生命周期。  
> 总仓库：https://github.com/yan-mc/AMS-Studio


AMS（AI Music Studio）本地服务：Gateway + 各后端服务。

## 当前内容（M0）

| 服务 | 说明 |
|---|---|
| `gateway/` | FastAPI + SQLite 任务网关：jobs/SSE/artifacts/voices scan、token 鉴权、RunnerRegistry 扩展点 |

## 约定

- 每个服务独立 Python 3.12 + `.venv`（uv/pip），不共用旧整合包 3.9 环境。
- 所有服务只监听 127.0.0.1，由启动器/Gateway 管理。
- 不提交 `.venv/`、运行时数据、模型、音频。
