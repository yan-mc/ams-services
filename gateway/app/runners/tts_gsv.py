"""tts.synthesize.gsv：通过 runtime.tts-gsv-core 的 headless 服务合成。"""
from __future__ import annotations

from pathlib import Path

from ..jobs.runner import ArtifactSpec, JobContext
from ..tts_gsv.manager import GsvError, load_voice, run_synthesis
from .config import RunnerError


class TtsGsvSynthesizeRunner:
    job_type = "tts.synthesize.gsv"
    gpu = True
    description = "GPT-SoVITS 语音合成（V5turbo / V2ProPlus，headless service）"

    async def run(self, ctx: JobContext) -> list[ArtifactSpec]:
        params = dict(ctx.params or {})
        text = str(params.get("text") or "").strip()
        if not text:
            raise RunnerError("INVALID_PARAMS", "缺少参数 text")
        if len(text) > 20000:
            raise RunnerError("INVALID_PARAMS", "text 过长（最多 20000 字）")
        voice_id = str(params.get("voice_id") or "").strip()
        voice_dir = str(params.get("voice_dir") or "").strip()
        if not voice_id and not voice_dir:
            raise RunnerError("INVALID_PARAMS", "缺少 voice_id 或 voice_dir")
        try:
            spec = load_voice(
                voice_id=voice_id,
                voice_dir=voice_dir,
                overrides={
                    "version": str(params.get("version") or ""),
                    "sovits": str(params.get("sovits_path") or ""),
                    "gpt": str(params.get("gpt_path") or ""),
                    "ref_audio": str(params.get("ref_audio") or ""),
                    "ref_text": str(params.get("ref_text") or ""),
                    "language": str(params.get("language") or ""),
                },
            )
        except GsvError as exc:
            raise RunnerError(exc.code, exc.message, hint=exc.hint, detail=exc.detail) from exc

        speed = max(0.5, min(2.0, float(params.get("speed") or 1.0)))
        seed = int(params.get("seed") if params.get("seed") is not None else 42)
        output = ctx.work_dir / "tts_gsv.wav"
        ctx.log("info", f"[tts.synthesize.gsv] voice={spec.voice_id} engine={spec.version} chars={len(text)}")
        await ctx.progress(8, "tts.load", f"加载 {spec.version}")
        try:
            result = await run_synthesis(spec, text=text, output_path=output, speed=speed, seed=seed)
        except GsvError as exc:
            raise RunnerError(exc.code, exc.message, hint=exc.hint, detail=exc.detail) from exc
        await ctx.progress(100, "done")
        if not output.is_file() or output.stat().st_size == 0:
            raise RunnerError("TTS_FAILED", "tts_service 未生成有效 WAV")
        ctx.log(
            "info",
            f"[tts.synthesize.gsv] done sr={result.get('sample_rate')} "
            f"dur={result.get('duration_s')}s rtf={result.get('rtf')}",
        )
        return [
            ArtifactSpec(
                name="tts_gsv.wav",
                path=Path(output),
                kind="audio",
                meta={
                    "voice": spec.voice_id,
                    "voice_name": spec.display_name,
                    "engine": spec.version,
                    "sample_rate": result.get("sample_rate"),
                    "duration_s": result.get("duration_s"),
                    "rtf": result.get("rtf"),
                    "gen_s": result.get("gen_s"),
                    "model_load_s": result.get("model_load_s"),
                    "service_log": result.get("service_log"),
                },
            )
        ]
