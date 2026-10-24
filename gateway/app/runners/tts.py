"""tts.synthesize：受管 Piper 离线 TTS（runtime.tts）。"""

from __future__ import annotations

import asyncio
import os
import wave
from pathlib import Path
from typing import Any

from ..jobs.runner import ArtifactSpec, JobContext
from .config import RunnerError

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0



def split_tts_text(text: str, limit: int = 1800) -> list[str]:
    """Split long text at sentence punctuation, then hard-wrap if needed."""
    text = text.strip()
    if not text:
        return []
    chunks = []
    current = ""
    for char in text:
        current += char
        if len(current) >= limit:
            chunks.append(current)
            current = ""
        elif char in "。！？!?；;\n" and len(current) >= 400:
            chunks.append(current)
            current = ""
    if current.strip():
        chunks.append(current)
    return [chunk for chunk in chunks if chunk.strip()]

class TtsSynthesizeRunner:
    job_type = "tts.synthesize"
    gpu = False
    description = "Piper 离线语音合成（runtime.tts）"

    def _root(self) -> Path:
        value = os.environ.get("AMS_TTS_ROOT", "").strip()
        if not value:
            raise RunnerError(
                "BACKEND_NOT_CONFIGURED",
                "未配置 TTS 运行时（AMS_TTS_ROOT 为空）",
                hint="安装 runtime.tts 组件后重启 Gateway",
            )
        root = Path(value).expanduser()
        if not root.is_dir():
            raise RunnerError("BACKEND_PATH_NOT_FOUND", f"TTS 运行时目录不存在：{root}")
        return root

    def _find(self, root: Path, *candidates: str) -> Path:
        for candidate in candidates:
            path = root / candidate
            if path.is_file():
                return path
        raise RunnerError(
            "BACKEND_PATH_NOT_FOUND",
            "TTS 运行时缺少 piper.exe 或模型文件",
            detail={"root": str(root) },
            hint="重新安装 runtime.tts 组件",
        )

    async def run(self, ctx: JobContext) -> list[ArtifactSpec]:
        text = str(ctx.params.get("text") or "").strip()
        if not text:
            raise RunnerError("INVALID_PARAMS", "缺少参数 text")
        if len(text) > 20000:
            raise RunnerError("INVALID_PARAMS", "text 过长（最多 20000 字）")
        root = self._root()
        exe = self._find(root, "piper/piper.exe", "piper.exe")
        model = Path(str(ctx.params.get("model") or root / "models" / "zh_CN-huayan-medium.onnx")).expanduser()
        config = Path(str(ctx.params.get("config") or model.with_name(model.name + ".json"))).expanduser()
        if not model.is_file():
            raise RunnerError("BACKEND_PATH_NOT_FOUND", f"TTS 模型不存在：{model}")
        if not config.is_file():
            raise RunnerError("BACKEND_PATH_NOT_FOUND", f"TTS 模型配置不存在：{config}")
        output = ctx.work_dir / "tts.wav"
        chunks = split_tts_text(text)
        if not chunks:
            raise RunnerError("INVALID_PARAMS", "缺少可合成文本")
        ctx.log("info", f"[tts.synthesize] Piper 合成 {len(text)} 字，分 {len(chunks)} 段")
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        part_paths = []
        for index, chunk in enumerate(chunks):
            part = output if len(chunks) == 1 else (ctx.work_dir / ("tts_part_%03d.wav" % index))
            proc = await asyncio.create_subprocess_exec(
                str(exe),
                "--model", str(model),
                "--config", str(config),
                "--output_file", str(part),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            ctx.add_kill_callback(lambda proc=proc: proc.kill())
            try:
                _, stderr = await asyncio.wait_for(proc.communicate(chunk.encode("utf-8")), timeout=180)
            except asyncio.TimeoutError as exc:
                proc.kill()
                raise RunnerError("BACKEND_TIMEOUT", f"TTS 第 {index + 1} 段合成超时") from exc
            if proc.returncode != 0:
                raise RunnerError(
                    "TTS_FAILED",
                    f"Piper 第 {index + 1} 段合成失败",
                    detail={"stderr": stderr.decode("utf-8", "replace")[-1000:]},
                )
            if not part.is_file() or part.stat().st_size == 0:
                raise RunnerError("TTS_FAILED", f"Piper 第 {index + 1} 段未生成有效 WAV")
            part_paths.append(part)
            await ctx.progress(10.0 + 85.0 * (index + 1) / len(chunks), "tts", f"{index + 1}/{len(chunks)}")
        if len(part_paths) > 1:
            with wave.open(str(part_paths[0]), "rb") as first:
                params = first.getparams()
                frames = [first.readframes(first.getnframes())]
            for part in part_paths[1:]:
                with wave.open(str(part), "rb") as reader:
                    if reader.getnchannels() != params.nchannels or reader.getsampwidth() != params.sampwidth or reader.getframerate() != params.framerate:
                        raise RunnerError("TTS_FAILED", "TTS 分段格式不一致，无法拼接")
                    frames.append(reader.readframes(reader.getnframes()))
            with wave.open(str(output), "wb") as writer:
                writer.setparams(params)
                for frame in frames:
                    writer.writeframes(frame)
        if not output.is_file() or output.stat().st_size == 0:
            raise RunnerError("TTS_FAILED", "Piper 未生成有效 WAV")
        await ctx.progress(100, "done")
        return [ArtifactSpec(name="tts.wav", path=output, kind="audio", meta={"voice": model.stem, "text_length": len(text), "chunks": len(chunks)})]
