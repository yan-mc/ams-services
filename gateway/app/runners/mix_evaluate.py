"""F: evaluate.audio and mix.master runners."""
from __future__ import annotations
from pathlib import Path
from typing import Any
from ..jobs.runner import JobContext
from .base import ProtocolProcessRunner
from .config import BackendConfig, RunnerError


class EvaluateAudioRunner(ProtocolProcessRunner):
    job_type="evaluate.audio"; gpu=False; timeout_sec=300.0
    description="音频客观评测（时长/采样率/声道/削波/静音/RMS/SNR）"
    def default_script_path(self)->Path: return Path(__file__).resolve().parent/"scripts"/"evaluate_audio.py"
    def validate_params(self,params:dict[str,Any])->dict[str,Any]:
        value=str(params.get("input_path") or "").strip()
        if not value: raise RunnerError("INVALID_PARAMS","缺少 input_path")
        path=Path(value).expanduser()
        if not path.is_file(): raise RunnerError("INPUT_NOT_FOUND",f"音频不存在：{path}")
        ref=str(params.get("reference_path") or "").strip()
        if ref and not Path(ref).expanduser().is_file(): raise RunnerError("INPUT_NOT_FOUND",f"参考音频不存在：{ref}")
        return {"input_path":str(path),"reference_path":ref}
    def build_command(self,cfg:BackendConfig,params:dict[str,Any],ctx:JobContext):
        python,_=cfg.require_rvc(); script=self.resolved_script_path()
        cmd=[str(python),"-B",str(script),"--input",params["input_path"],"--output-dir",str(ctx.work_dir)]
        if params.get("reference_path"): cmd += ["--reference",params["reference_path"]]
        return cmd, None, None


class MixMasterRunner(ProtocolProcessRunner):
    job_type="mix.master"; gpu=False; timeout_sec=1800.0
    description="独立多轨混音/母带（ffmpeg，不进入歌曲换源）"
    def default_script_path(self)->Path: return Path(__file__).resolve().parent/"scripts"/"mix_master.py"
    def validate_params(self,params:dict[str,Any])->dict[str,Any]:
        raw=str(params.get("input_paths") or "").replace("\n",";")
        paths=[x.strip() for x in raw.split(";") if x.strip()]
        if len(paths)<1: raise RunnerError("INVALID_PARAMS","至少一个输入音频")
        for value in paths:
            if not Path(value).expanduser().is_file(): raise RunnerError("INPUT_NOT_FOUND",f"音频不存在：{value}")
        loud=str(params.get("loudness") or "-14")
        peak=str(params.get("true_peak") or "-1")
        return {"input_paths":";".join(paths),"loudness":loud,"true_peak":peak}
    def build_command(self,cfg:BackendConfig,params:dict[str,Any],ctx:JobContext):
        python,root=cfg.require_rvc(); script=self.resolved_script_path(); ff=Path(root)/"ffmpeg.exe"
        if not ff.is_file(): raise RunnerError("BACKEND_NOT_CONFIGURED",f"ffmpeg 不存在：{ff}")
        output=ctx.work_dir/"mix.wav"
        return [str(python),"-B",str(script),"--inputs",params["input_paths"],"--output",str(output),"--ffmpeg",str(ff),"--loudness",params["loudness"],"--true-peak",params["true_peak"]], None, None
