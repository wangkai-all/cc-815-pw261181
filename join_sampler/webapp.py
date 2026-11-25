"""网页应用：基于标准库 http.server 的单文件后端。

提供：
* POST /api/synthetic      生成合成两表并抽样
* POST /api/upload         上传两侧 CSV（multipart/form-data）并抽样
* POST /api/runs           用已有运行的快照以新配置抽样
* GET  /api/runs           运行列表
* GET  /api/runs/<id>      运行详情（含每对入选依据、键分布、内存）
* POST /api/runs/<id>/append   从检查点恢复后追加一批（同种子，随机状态延续）
* POST /api/compare        比较两版运行

所有运行在服务端保留完整抽样器状态（内存中 + 可选检查点文件），
因此 append 走的是真恢复路径而非重新抽样。
"""

from __future__ import annotations

import io
import json
import re
import threading
import time
import uuid
from email.parser import BytesParser
from email.policy import default as email_default_policy
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from .checkpoint import ResumeError
from .conditions import Condition, ConditionError
from .data import load_csv_side, make_synthetic_tables, side_from_rows
from .sampler import JoinSampler, SamplerConfig, SamplingError, ResourceLimitExceeded
from .snapshot import build_snapshot
from .stats import compare_runs
from .store import SideData


class RunRegistry:
    """线程安全的运行注册表：持有抽样器、快照与结果。"""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._runs: dict[str, dict[str, Any]] = {}

    def create(
        self,
        *,
        left: SideData,
        right: SideData,
        condition_expr: str | None,
        config: SamplerConfig,
        label: str,
    ) -> dict[str, Any]:
        try:
            condition = Condition.compile(condition_expr)
        except ConditionError as exc:
            raise WebError(str(exc), HTTPStatus.BAD_REQUEST) from exc
        snapshot = build_snapshot(left, right, condition)
        if not snapshot.keys():
            raise WebError(
                "两侧没有任何共同的非空键，关联结果为空", HTTPStatus.BAD_REQUEST
            )
        sampler = JoinSampler(snapshot, config)
        try:
            sampler.build()
            sampler.draw_more(config.sample_size)
        except ResourceLimitExceeded as exc:
            raise WebError(
                f"资源上限被突破: {exc.limit}（预算 {exc.budget}，已用 {exc.used}）",
                HTTPStatus.UNPROCESSABLE_ENTITY,
                payload={"partial": exc.partial, "snapshot": snapshot.describe()},
            ) from exc
        except SamplingError as exc:
            raise WebError(str(exc), HTTPStatus.BAD_REQUEST) from exc

        run_id = uuid.uuid4().hex[:12]
        entry = {
            "id": run_id,
            "label": label,
            "created_at": time.time(),
            "sampler": sampler,
            "snapshot": snapshot,
            "left": left,
            "right": right,
            "history": [
                {
                    "at": time.time(),
                    "event": "created",
                    "sample_size": config.sample_size,
                    "mode": config.mode,
                    "seed": config.rng_seed,
                }
            ],
        }
        with self._lock:
            self._runs[run_id] = entry
        return self.describe(run_id)

    def append(self, run_id: str, more: int) -> dict[str, Any]:
        with self._lock:
            entry = self._require(run_id)
            sampler: JoinSampler = entry["sampler"]
            try:
                sampler.draw_more(more)
            except SamplingError as exc:
                raise WebError(str(exc), HTTPStatus.BAD_REQUEST) from exc
            entry["history"].append(
                {
                    "at": time.time(),
                    "event": "appended",
                    "more": more,
                    "total": len(sampler._records),
                    "seed_unchanged": True,
                }
            )
            return self.describe(run_id)

    def describe(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            entry = self._require(run_id)
            sampler: JoinSampler = entry["sampler"]
            result = sampler.result().to_jsonable()
            result["id"] = run_id
            result["label"] = entry["label"]
            result["history"] = entry["history"]
            result["key_build"] = [
                self._key_info_jsonable(info) for info in sampler._infos
            ]
            return result

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            out = []
            for run_id, entry in self._runs.items():
                sampler: JoinSampler = entry["sampler"]
                out.append(
                    {
                        "id": run_id,
                        "label": entry["label"],
                        "created_at": entry["created_at"],
                        "sample_size": len(sampler._records),
                        "mode": sampler.config.mode,
                        "seed": sampler.config.rng_seed,
                        "snapshot_fingerprint": sampler.snapshot.fingerprint,
                    }
                )
            return sorted(out, key=lambda r: r["created_at"], reverse=True)

    def get_entry(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            return self._require(run_id)

    def compare(self, base_id: str, other_id: str) -> dict[str, Any]:
        with self._lock:
            base = self.describe(base_id)
            other = self.describe(other_id)
            report = compare_runs(base, other)
            report["base_id"] = base_id
            report["other_id"] = other_id
            report["base_label"] = base["label"]
            report["other_label"] = other["label"]
            return report

    def _require(self, run_id: str) -> dict[str, Any]:
        entry = self._runs.get(run_id)
        if entry is None:
            raise WebError(f"运行 {run_id!r} 不存在", HTTPStatus.NOT_FOUND)
        return entry

    @staticmethod
    def _key_info_jsonable(info: Any) -> dict[str, Any]:
        return {
            "key": info.key,
            "pair_count": info.pair_count,
            "left_rows": info.left_rows,
            "right_rows": info.right_rows,
            "acceptance_rate": info.acceptance_rate,
            "rate_is_estimate": info.rate_is_estimate,
            "rate_smoothed": info.rate_smoothed,
            "weight_mode": info.weight_mode,
            "weight": info.weight,
            "accepted_count": info.accepted_count,
            "accepted_observed": info.accepted_observed,
            "probes_used": info.probes_used,
            "draws": info.draws,
        }


class WebError(Exception):
    def __init__(self, message: str, status: HTTPStatus, payload: dict | None = None):
        super().__init__(message)
        self.status = status
        self.payload = payload


def config_from_json(body: dict[str, Any]) -> SamplerConfig:
    try:
        return SamplerConfig(
            sample_size=int(body.get("sample_size", 100)),
            mode=body.get("mode", "WR"),
            rng_seed=body.get("seed"),
            max_pairs_per_key=int(body.get("max_pairs_per_key", 100_000)),
            estimator_probes=int(body.get("estimator_probes", 2000)),
            max_build_evaluations=body.get("max_build_evaluations"),
            max_build_seconds=body.get("max_build_seconds"),
            max_memory_bytes=body.get("max_memory_bytes"),
            max_accepted_cache_pairs=body.get("max_accepted_cache_pairs", 200_000),
        )
    except (TypeError, ValueError) as exc:
        raise WebError(f"配置参数非法: {exc}", HTTPStatus.BAD_REQUEST) from exc


class Handler(BaseHTTPRequestHandler):
    registry: RunRegistry

    # -- HTTP 基础 ------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # 安静一点
        pass

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise WebError(f"请求体不是合法 JSON: {exc}", HTTPStatus.BAD_REQUEST) from exc
        if not isinstance(body, dict):
            raise WebError("请求体必须是 JSON 对象", HTTPStatus.BAD_REQUEST)
        return body

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/runs":
                self._send_json({"runs": self.registry.list_runs()})
                return
            if path.startswith("/api/runs/"):
                run_id = path.split("/")[3]
                self._send_json(self.registry.describe(run_id))
                return
            if path == "/api/health":
                self._send_json({"ok": True})
                return
            self._serve_static(path)
        except WebError as exc:
            self._send_json(
                {"error": str(exc), **(exc.payload or {})}, status=exc.status
            )

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/synthetic":
                self._handle_synthetic()
            elif path == "/api/upload":
                self._handle_upload()
            elif path == "/api/compare":
                body = self._read_json()
                report = self.registry.compare(body["base_id"], body["other_id"])
                self._send_json(report)
            elif path.startswith("/api/runs/") and path.endswith("/append"):
                run_id = path.split("/")[3]
                body = self._read_json()
                more = int(body.get("more", 100))
                if more <= 0:
                    raise WebError("more 必须为正整数", HTTPStatus.BAD_REQUEST)
                self._send_json(self.registry.append(run_id, more))
            else:
                raise WebError(f"未知接口: {path}", HTTPStatus.NOT_FOUND)
        except WebError as exc:
            self._send_json(
                {"error": str(exc), **(exc.payload or {})}, status=exc.status
            )
        except ResumeError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.CONFLICT)

    # -- 业务接口 -------------------------------------------------------
    def _handle_synthetic(self) -> None:
        body = self._read_json()
        left, right = make_synthetic_tables(
            left_rows=int(body.get("left_rows", 50_000)),
            right_rows=int(body.get("right_rows", 50_000)),
            distinct_keys=int(body.get("distinct_keys", 500)),
            hot_key_multiplier=int(body.get("hot_key_multiplier", 200)),
            null_key_fraction=float(body.get("null_key_fraction", 0.02)),
            value_max=int(body.get("value_max", 10_000)),
            rng_seed=int(body.get("data_seed", 42)),
        )
        cfg = config_from_json(body)
        result = self.registry.create(
            left=left,
            right=right,
            condition_expr=body.get("condition") or None,
            config=cfg,
            label=body.get("label") or f"合成数据抽样 {time.strftime('%H:%M:%S')}",
        )
        self._send_json(result, HTTPStatus.CREATED)

    def _handle_upload(self) -> None:
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            raise WebError("上传接口要求 multipart/form-data", HTTPStatus.BAD_REQUEST)
        length = int(self.headers.get("Content-Length", 0))
        if length <= 0:
            raise WebError("上传请求体为空", HTTPStatus.BAD_REQUEST)
        form = _parse_multipart(ctype, self.rfile.read(length))

        def file_field(name: str) -> io.StringIO:
            payload = form.files.get(name)
            if payload is None:
                raise WebError(f"缺少文件字段 {name}", HTTPStatus.BAD_REQUEST)
            return io.StringIO(payload.decode("utf-8-sig"))

        key = form.fields.get("key", "k")
        condition = form.fields.get("condition") or None
        label = form.fields.get("label") or f"CSV 上传抽样 {time.strftime('%H:%M:%S')}"
        left = _csv_stream_to_side(file_field("left_file"), "left", key)
        right = _csv_stream_to_side(file_field("right_file"), "right", key)
        cfg = config_from_json(
            {
                "sample_size": form.fields.get("sample_size", "100"),
                "mode": form.fields.get("mode", "WR"),
                "seed": form.fields.get("seed") or None,
                "max_pairs_per_key": form.fields.get("max_pairs_per_key", "100000"),
                "estimator_probes": form.fields.get("estimator_probes", "2000"),
            }
        )
        result = self.registry.create(
            left=left, right=right, condition_expr=condition, config=cfg, label=label
        )
        self._send_json(result, HTTPStatus.CREATED)

    # -- 静态资源 -------------------------------------------------------
    def _serve_static(self, path: str) -> None:
        if path in ("/", "/index.html"):
            path = "/index.html"
        static_dir = _static_dir()
        target = (static_dir / path.lstrip("/")).resolve()
        if not str(target).startswith(str(static_dir.resolve())) or not target.is_file():
            self._send_json({"error": "not found"}, status=HTTPStatus.NOT_FOUND)
            return
        suffix = target.suffix
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".svg": "image/svg+xml",
        }.get(suffix, "application/octet-stream")
        data = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _MultipartForm:
    """解析后的 multipart 表单：文本字段与文件字段分开存放。"""

    def __init__(self) -> None:
        self.fields: dict[str, str] = {}
        self.files: dict[str, bytes] = {}


_BOUNDARY_RE = re.compile(r'boundary=(?:"([^"]+)"|([^;]+))', re.IGNORECASE)
_DISPO_NAME_RE = re.compile(r'name="([^"]*)"')
_DISPO_FILE_RE = re.compile(r'filename="([^"]*)"')


def _parse_multipart(content_type: str, body: bytes) -> _MultipartForm:
    """用标准库 email 解析 multipart/form-data，替代已废弃的 cgi 模块。

    只处理本应用需要的两种 part：普通文本字段与带 filename 的文件字段；
    不执行任何 part 内的内容，CSV 内容按字节取出后交由 csv 模块解析。
    """
    match = _BOUNDARY_RE.search(content_type)
    if not match:
        raise WebError("Content-Type 缺少 boundary", HTTPStatus.BAD_REQUEST)
    boundary = (match.group(1) or match.group(2)).strip()
    # email 解析器需要完整的 MIME 文档头 + 空行 + 各 part。
    document = (
        f"MIME-Version: 1.0\r\nContent-Type: {content_type}\r\n\r\n"
    ).encode("ascii") + body
    try:
        message = BytesParser(policy=email_default_policy).parsebytes(document)
    except Exception as exc:  # email 内部异常类型较杂，统一收敛为 400
        raise WebError(f"multipart 报文解析失败: {exc}", HTTPStatus.BAD_REQUEST) from exc
    if not message.is_multipart():
        raise WebError("请求体不是合法的 multipart 表单", HTTPStatus.BAD_REQUEST)

    form = _MultipartForm()
    for part in message.iter_parts():
        disposition = part.get("Content-Disposition", "")
        if not disposition:
            continue
        name_match = _DISPO_NAME_RE.search(disposition)
        if not name_match:
            continue
        name = name_match.group(1)
        payload = part.get_payload(decode=True)
        if payload is None:
            payload = b""
        if _DISPO_FILE_RE.search(disposition):
            form.files[name] = payload
        else:
            charset = part.get_content_charset() or "utf-8"
            form.fields[name] = payload.decode(charset, errors="replace")
    return form


def _csv_stream_to_side(stream: io.StringIO, name: str, key: str) -> SideData:
    import csv

    reader = csv.DictReader(stream)
    if reader.fieldnames is None:
        raise WebError(f"{name} CSV 缺少表头", HTTPStatus.BAD_REQUEST)
    if key not in reader.fieldnames:
        raise WebError(f"{name} CSV 中找不到键列 {key!r}", HTTPStatus.BAD_REQUEST)
    from .data import infer_scalar

    columns = tuple(reader.fieldnames)
    rows = [
        {col: infer_scalar(raw[col]) for col in columns} for raw in reader
    ]
    if not rows:
        raise WebError(f"{name} CSV 没有数据行", HTTPStatus.BAD_REQUEST)
    return side_from_rows(rows, name=name, key=key)


def _static_dir():
    from pathlib import Path

    return Path(__file__).resolve().parent / "web_static"


def make_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    registry = RunRegistry()

    class BoundHandler(Handler):
        pass

    BoundHandler.registry = registry
    server = ThreadingHTTPServer((host, port), BoundHandler)
    return server


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="关联抽样 Web 应用")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = make_server(args.host, args.port)
    print(f"关联抽样 Web 已启动: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
