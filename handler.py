import os
import time
import json
import atexit
import subprocess
from typing import Any, Dict, Optional

import requests
import runpod

ACESTEP_API_HOST = os.getenv("ACESTEP_API_HOST", "127.0.0.1")
ACESTEP_API_PORT = int(os.getenv("ACESTEP_API_PORT", "8001"))
ACESTEP_API_URL = f"http://{ACESTEP_API_HOST}:{ACESTEP_API_PORT}"

HEALTH_URL = f"{ACESTEP_API_URL}/health"
RELEASE_TASK_URL = f"{ACESTEP_API_URL}/release_task"
QUERY_RESULT_URL = f"{ACESTEP_API_URL}/query_result"

STARTUP_TIMEOUT_SEC = int(os.getenv("STARTUP_TIMEOUT_SEC", "600"))
REQUEST_TIMEOUT_SEC = int(os.getenv("REQUEST_TIMEOUT_SEC", "180"))
POLL_INTERVAL_SEC = float(os.getenv("POLL_INTERVAL_SEC", "2.5"))
JOB_TIMEOUT_SEC = int(os.getenv("JOB_TIMEOUT_SEC", "3600"))

ACESTEP_START_CMD = os.getenv(
    "ACESTEP_START_CMD",
    f"python -m acestep.api_server --host 0.0.0.0 --port {ACESTEP_API_PORT}"
)

_api_process: Optional[subprocess.Popen] = None
_api_ready = False


def _cleanup() -> None:
    global _api_process
    if _api_process and _api_process.poll() is None:
        try:
            _api_process.terminate()
            _api_process.wait(timeout=15)
        except Exception:
            try:
                _api_process.kill()
            except Exception:
                pass


atexit.register(_cleanup)


def _healthcheck() -> bool:
    try:
        r = requests.get(HEALTH_URL, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def _wait_for_api() -> None:
    start = time.time()
    while time.time() - start < STARTUP_TIMEOUT_SEC:
        if _healthcheck():
            return

        if _api_process and _api_process.poll() is not None:
            output = ""
            try:
                if _api_process.stdout:
                    output = _api_process.stdout.read()
            except Exception:
                pass
            raise RuntimeError(
                f"ACE-Step API exited early with code {_api_process.returncode}. Output:\n{output}"
            )

        time.sleep(2)

    raise TimeoutError(f"ACE-Step API did not become healthy within {STARTUP_TIMEOUT_SEC} seconds.")


def _start_api_if_needed() -> None:
    global _api_process, _api_ready

    if _api_ready and _healthcheck():
        return

    if _api_process and _api_process.poll() is None:
        _wait_for_api()
        _api_ready = True
        return

    env = os.environ.copy()
    _api_process = subprocess.Popen(
        ACESTEP_START_CMD,
        shell=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    _wait_for_api()
    _api_ready = True


def _strip_none(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}


def _build_release_payload(job_input: Dict[str, Any]) -> Dict[str, Any]:
    prompt = job_input.get("prompt") or job_input.get("lyrics") or job_input.get("description")
    if not prompt:
        raise ValueError("Missing required input: prompt")

    payload = {
        "prompt": prompt,
        "lyrics": job_input.get("lyrics"),
        "title": job_input.get("title"),
        "tags": job_input.get("tags"),
        "genre": job_input.get("genre"),
        "negative_prompt": job_input.get("negative_prompt"),
        "seed": job_input.get("seed"),
        "steps": job_input.get("steps"),
        "seconds": job_input.get("seconds") or job_input.get("duration"),
        "duration": job_input.get("duration"),
        "bpm": job_input.get("bpm"),
        "cfg_scale": job_input.get("cfg_scale"),
        "guidance_scale": job_input.get("guidance_scale"),
        "temperature": job_input.get("temperature"),
        "top_p": job_input.get("top_p"),
        "instrumental": job_input.get("instrumental"),
        "model_name": job_input.get("model_name"),
        "audio_path": job_input.get("audio_path"),
        "response_format": job_input.get("response_format", "json"),
    }
    return _strip_none(payload)


def _submit_release_task(payload: Dict[str, Any]) -> Dict[str, Any]:
    r = requests.post(RELEASE_TASK_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    if r.status_code != 200:
        raise RuntimeError(f"/release_task failed with {r.status_code}: {r.text}")

    body = r.json()
    if body.get("error"):
        raise RuntimeError(f"/release_task returned error: {body['error']}")

    return body


def _extract_task_id(submit_resp: Dict[str, Any]) -> Optional[str]:
    data = submit_resp.get("data", {})
    if isinstance(data, dict):
        task_id = data.get("task_id")
        if task_id:
            return str(task_id)
    return None


def _query_result(task_id: str) -> Dict[str, Any]:
    payload = {"task_id_list": [task_id]}
    r = requests.post(QUERY_RESULT_URL, json=payload, timeout=REQUEST_TIMEOUT_SEC)

    if r.status_code != 200:
        raise RuntimeError(f"/query_result failed with {r.status_code}: {r.text}")

    body = r.json()
    if body.get("error"):
        raise RuntimeError(f"/query_result returned error: {body['error']}")

    return body


def _parse_result_field(result_field: Any) -> Any:
    if isinstance(result_field, str):
        try:
            return json.loads(result_field)
        except Exception:
            return result_field
    return result_field


def _poll_until_done(task_id: str) -> Dict[str, Any]:
    start = time.time()
    last_item = None

    while time.time() - start < JOB_TIMEOUT_SEC:
        body = _query_result(task_id)
        data = body.get("data", [])

        if not isinstance(data, list) or not data:
            time.sleep(POLL_INTERVAL_SEC)
            continue

        item = data[0]
        last_item = item
        status = item.get("status")
        progress_text = item.get("progress_text")
        parsed_result = _parse_result_field(item.get("result"))

        if status == 1:
            return {
                "task_id": task_id,
                "status": status,
                "progress_text": progress_text,
                "result": parsed_result,
                "raw": item,
            }

        if status == 2:
            raise RuntimeError(
                f"ACE-Step task failed. progress_text={progress_text}, result={parsed_result}"
            )

        time.sleep(POLL_INTERVAL_SEC)

    raise TimeoutError(
        f"Timed out waiting for task {task_id}. Last item: {json.dumps(last_item, indent=2)}"
    )


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    job_input = job.get("input", {})

    try:
        _start_api_if_needed()

        release_payload = _build_release_payload(job_input)
        submit_resp = _submit_release_task(release_payload)

        task_id = _extract_task_id(submit_resp)
        if not task_id:
            return {
                "ok": False,
                "error": "Could not extract task_id from /release_task response",
                "submit_response": submit_resp,
            }

        final_result = _poll_until_done(task_id)

        return {
            "ok": True,
            "submit_response": submit_resp,
            "task_id": task_id,
            "result": final_result,
        }

    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }


runpod.serverless.start({"handler": handler})
