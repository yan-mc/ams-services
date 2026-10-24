"""dataset.prepare: import + mono/resample + clip + validate."""
from __future__ import annotations
import os, re
from pathlib import Path
from typing import Any
from ..jobs.runner import JobContext
from .base import ProtocolProcessRunner
from .config import BackendConfig, RunnerError

class DatasetPrepareRunner(ProtocolProcessRunner):
    job_type="dataset.prepare"; gpu=False; timeout_sec=1800.0
    description="数据集导入/切片/校验"
    def default_script_path(self)->Path: return Path(__file__).resolve().parent/"scripts"/"dataset_prepare.py"
    def validate_params(self,params:dict[str,Any])->dict[str,Any]:
        source=str(params.get("source_path") or "").strip()
        if not source: raise RunnerError("INVALID_PARAMS","缺少 source_path")
        path=Path(source).expanduser()
        if not path.exists(): raise RunnerError("INPUT_NOT_FOUND",f"源路径不存在：{path}")
        name=str(params.get("dataset_name") or "").strip()
        if not re.match(r"^[A-Za-z0-9_\-\u4e00-\u9fff]{1,64}$",name): raise RunnerError("INVALID_PARAMS","数据集名需 1-64 字符中英文/数字/下划线/连字符")
        sr=int(params.get("target_sr") or 48000)
        if sr not in {32000,40000,48000}: raise RunnerError("INVALID_PARAMS","target_sr 必须是 32000/40000/48000")
        mn=float(params.get("min_sec") or 3.0); mx=float(params.get("max_sec") or 10.0)
        if mn<1 or mx>30 or mn>=mx: raise RunnerError("INVALID_PARAMS","min_sec/max_sec 范围不合法")
        return {"source_path":str(path),"dataset_name":name,"target_sr":sr,"min_sec":mn,"max_sec":mx}
    def build_command(self,cfg:BackendConfig,params:dict[str,Any],ctx:JobContext):
        python,_=cfg.require_rvc(); script=self.resolved_script_path()
        if not script.is_file(): raise RunnerError("BACKEND_PATH_NOT_FOUND",f"数据集脚本不存在：{script}")
        home=os.environ.get("AMS_HOME","")
        if not home: raise RunnerError("BACKEND_NOT_CONFIGURED","AMS_HOME 为空")
        return [str(python),"-B",str(script),"--source-path",params["source_path"],"--dataset-name",params["dataset_name"],"--home",home,"--target-sr",str(params["target_sr"]),"--min-sec",str(params["min_sec"]),"--max-sec",str(params["max_sec"])], None, None
