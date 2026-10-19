# experiments —— M0.7 后端冒烟实验目录

> 用途：RVC 换声与 pymss 分离的**可复现冒烟实验**（M0.7）。正式代码在 `ams-services/gateway` 等仓库；本目录只放实验脚本、环境与证据。
> 报告：`07_执行计划/验证/M0_后端冒烟.md`；输出音频：`07_执行计划/验证/audio/`。
> 约束：本目录内创建独立环境（uv/venv），不安装系统级软件；RVC 源目录只读（不修改/不写 pycache/不写日志）。

## 1. 目录

```text
experiments/
├─ rvc_smoke.py              # RVC 换声冒烟脚本（RVC runtime 执行；含 faiss 非 ASCII 兼容补丁）
├─ pymss_smoke.ps1           # pymss 环境搭建 + 最小分离一键脚本（CPU 默认，-Gpu 切换 cu128）
├─ README.md                 # 本文件
├─ results/                  # rvc_smoke.json / rvc_smoke_noindex.json / smoke_summary.json
├─ logs/                     # RVC/pymss 运行日志
├─ work/                     # 输入片段、对比脚本、源目录文件快照
├─ pymss/                    # pymss 独立环境（.venv）与模型（models/，均由脚本生成，可删）
├─ tools-uv/                 # 临时 uv（仅本目录，可删）
├─ cache/uv/                 # uv 下载缓存（可删）
└─ tmp/                      # TMP/TEMP 重定向（避免写系统临时目录）
```

## 2. 环境

| 项 | 值 |
|---|---|
| RVC 整合包 | `F:\ai歌曲\RVC20260311`（runtime Python 3.9.13 + torch 2.0.0+cu118） |
| 系统 Python | 3.12.10（仅用于创建 venv；不安装任何包到系统） |
| uv | 0.13.1（`experiments/tools-uv/.venv` 内） |
| pymss 环境 | `experiments/pymss/.venv`：Python 3.12.10 + torch 2.7.1+cpu + pymss 2.1.9 + pymss-core 0.1.11 |
| GPU | RTX 4080 Laptop 12GB（驱动 610.88） |
| 模型 | RVC：manbo.pth（55.2MB）+ manbo.index（141.4MB）；pymss：scnet_checkpoint_musdb18（42.4MB） |

## 3. RVC 换声冒烟

```powershell
$exp = "F:\Deepseek_harness\AI音乐处理\08_项目代码\ams-services\experiments"
$env:TMP = "$exp\tmp"; $env:TEMP = "$exp\tmp"; $env:PYTHONDONTWRITEBYTECODE = "1"
Set-Location $exp
& "F:\ai歌曲\RVC20260311\runtime\python.exe" -B "$exp\rvc_smoke.py" ^
  --rvc-root "F:\ai歌曲\RVC20260311" ^
  --input "$exp\work\input_vocal_35s.wav" ^
  --output "F:\Deepseek_harness\AI音乐处理\07_执行计划\验证\audio\rvc_manbo40k_35s.wav" ^
  --model manbo.pth --index assets/indices/manbo.index ^
  --f0-method rmvpe --index-rate 0.75 --f0up-key 0 --protect 0.33 ^
  --rms-mix-rate 0.25 --filter-radius 3 --is-half --device cuda:0 ^
  --result-json "$exp\results\rvc_smoke.json"
```

实测（35s 干声 / 4080 Laptop）：模型加载 0.46s、推理 4.49s（RTF≈0.128）、端到端 7.8s、显存峰值 allocated 1.3GB、输出 40k 单声道、时长差 20ms。详见报告 §2。

## 4. pymss 最小分离

一键（首次会创建环境 + 下载 CPU torch/pymss + 40MB 模型，约 2–4 分钟）：

```powershell
pwsh -File "F:\Deepseek_harness\AI音乐处理\08_项目代码\ams-services\experiments\pymss_smoke.ps1"
# GPU 版（下载 torch cu128 ≈3.05GiB；网络/时间允许时）：
# pwsh -File ...\pymss_smoke.ps1 -Gpu
```

手动分步：

```powershell
$exp = "F:\Deepseek_harness\AI音乐处理\08_项目代码\ams-services\experiments"
py -3.12 -m venv "$exp\tools-uv\.venv"
& "$exp\tools-uv\.venv\Scripts\python.exe" -m pip install -q uv
$uv = "$exp\tools-uv\.venv\Scripts\uv.exe"
& $uv venv "$exp\pymss\.venv" --python (Get-Command python).Source
& $uv pip install --python "$exp\pymss\.venv\Scripts\python.exe" --index-url https://download.pytorch.org/whl/cpu "torch==2.7.1"
& $uv pip install --python "$exp\pymss\.venv\Scripts\python.exe" pymss psutil
$env:PYMSS_MODEL_DIR = "$exp\pymss\models"
& "$exp\pymss\.venv\Scripts\python.exe" -m pymss.cli download scnet_checkpoint_musdb18 --model-dir $env:PYMSS_MODEL_DIR --source hf-mirror
& "$exp\pymss\.venv\Scripts\python.exe" -m pymss.cli infer scnet_checkpoint_musdb18 -i "$exp\work\input_music_20s.wav" -o "F:\Deepseek_harness\AI音乐处理\07_执行计划\验证\audio\pymss_scnet_20s" --save-as-folder --device cpu --format wav --wav-bit-depth FLOAT --model-dir $env:PYMSS_MODEL_DIR --source hf-mirror --param batch_size=1
```

实测（20s 音乐 / CPU）：首次 37s（RTF 0.95），热缓存后 20–22s（RTF 0.50–0.53）；CPU 时间 ~220s（≈11 核）；峰值工作集 928MB；输出 4 stems（44.1k 立体声 float wav）。

## 5. 关键坑（务必看）

| 坑 | 现象 | 处理 |
|---|---|---|
| **faiss + 非 ASCII 路径** | RVC 的 `faiss.read_index` 打不开中文路径，pipeline 静默忽略 index → `index_rate` 失效、音色变差且无报错 | `rvc_smoke.py` 已 monkeypatch：失败时 `open(path,'rb')` + `faiss.deserialize_index(np.frombuffer(...))`；或把 index 放纯 ASCII 路径/subst 盘符 |
| dotenv 找不到 .env | 脚本模式下 `load_dotenv()` 从脚本目录向上找，不读 cwd | 显式 `load_dotenv(dotenv_path=RVC_ROOT/".env")`（脚本已处理） |
| cwd 必须 = RVC 根目录 | 模型/hubert/rmvpe/ffmpeg 均按相对路径 | 脚本内 `os.chdir(rvc_root)` |
| `sid` 类型 | `vc_single` 第一参数是 int 说话人 id，不是模型名 | 传 `0`（脚本 `--speaker-id`） |
| 模型名 | `get_vc` 拼 `weight_root/<sid>` | 必须传 `manbo.pth`（带扩展名） |
| index 自动发现 | 只在 `logs/` 目录里找 | 显式传 `--index assets/indices/manbo.index` |
| RVC 输出 dtype | pipeline 返回 int16 | 写 wav 前 /32768 归一为 float（脚本已处理） |
| uv venv 无 pip | 不能 `python -m pip install` | 用 `uv pip install --python <venv python>` |
| torch wheel 体积 | CPU 206MiB / cu128 3.05GiB | 默认 CPU；GPU 准备足量磁盘/时间 |
| TMP/TEMP | RVC/ffmpeg 可能写系统临时目录 | 运行前重定向到 `experiments/tmp` |

## 6. 验证与清理

- 对比 index / 无 index 输出：`work/compare_rvc_outputs.py`（corr=0.873、mean|Δ|=0.0146）；
- pymss 一致性：sum(stems) vs mix corr=1.0000、相对残差 0.93%；
- 清理（回收约 3.2GB）：删除 `pymss/.venv`、`tools-uv`、`cache`（模型可按需保留）；
- RVC 源目录未改动证据：`work/rvc_files_before.txt` 与 `rvc_files_after.txt`（39,770 个文件一致）。

## 7. 相关文档

- 冒烟报告：`07_执行计划/验证/M0_后端冒烟.md`
- Gateway MVP：`08_项目代码/ams-services/gateway/README.md`
- 插件 02 人声分离 / 插件 06 数据集工作室方案：`06_方案设计/03_插件/`
