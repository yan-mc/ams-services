"""tts.dataset.prepare: TTS dataset validation/import."""
from __future__ import annotations
import os,re
from pathlib import Path
from typing import Any
from ..jobs.runner import JobContext
from .base import ProtocolProcessRunner
from .config import BackendConfig, RunnerError

class TtsDatasetPrepareRunner(ProtocolProcessRunner):
    job_type="tts.dataset.prepare"
    gpu=False
    timeout_sec=600.0
    description="TTS 数据集导入/校验（metadata.csv + wav -> data/datasets/tts）"
    def default_script_path(self): return Path(__file__).resolve().parent/"scripts"/"tts_dataset.py"
    def validate_params(self,p):
        src=str(p.get("source_path") or p.get("source") or "").strip()
        if not src: raise RunnerError("INVALID_PARAMS","缺少 source_path")
        name=str(p.get("dataset_name") or p.get("name") or "").strip()
        if not re.match(r"^[A-Za-z0-9_\-\u4e00-\u9fff]{1,64}$",name): raise RunnerError("INVALID_PARAMS","数据集名称非法")
        return {"source_path":str(Path(src).expanduser().resolve()),"dataset_name":name}
    def build_command(self,cfg:BackendConfig,p:dict[str,Any],ctx:JobContext):
        ams=Path(os.environ.get("AMS_HOME","")).expanduser()
        independent=ams/"runtime"/"tts-train"/"0.1.0"/"python"/"python.exe"
        python=independent if independent.is_file() else cfg.require_rvc()[0]
        script=self.resolved_script_path()
        return [str(python),"-u","-B",str(script),"--ams-home",str(ams),"--source",p["source_path"],"--name",p["dataset_name"]],None,None
