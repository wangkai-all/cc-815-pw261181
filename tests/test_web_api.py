"""Web API 端到端检验（线程内启动真实 HTTP 服务）。

覆盖：合成数据抽样、CSV 上传、追加批次（同种子延续）、两版比较、
资源上限通过 API 返回 422、静态页面可访问。
"""

import io
import json
import threading
import unittest
import urllib.request
import urllib.error
import uuid

from join_sampler.webapp import make_server


class WebApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = make_server("127.0.0.1", 0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def _post_json(self, path, payload):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read().decode("utf-8"))

    def _get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=30) as resp:
            return resp.status, resp.read()

    def test_full_flow_synthetic_append_compare(self):
        status, run = self._post_json("/api/synthetic", {
            "left_rows": 5000, "right_rows": 5000, "distinct_keys": 40,
            "hot_key_multiplier": 80, "null_key_fraction": 0.03,
            "condition": "abs(L.v - R.v) <= 200",
            "sample_size": 150, "mode": "WOR", "seed": 42,
            "max_pairs_per_key": 5000, "estimator_probes": 300,
        })
        self.assertEqual(status, 201)
        self.assertEqual(len(run["records"]), 150)
        self.assertIn("p_joint", run["records"][0])
        self.assertEqual(run["memory_report"]["materialized_join_rows"], 0)
        self.assertTrue(any(k["weight_mode"] == "estimate" for k in run["key_build"]))
        run_id = run["id"]

        # 追加一批：随机状态延续，总量增长且无重复对（WOR）。
        status, appended = self._post_json(f"/api/runs/{run_id}/append", {"more": 100})
        self.assertEqual(status, 200)
        self.assertEqual(len(appended["records"]), 250)
        pairs = [(r["left_row_id"], r["right_row_id"]) for r in appended["records"]]
        self.assertEqual(len(set(pairs)), 250)
        self.assertEqual(appended["history"][-1]["event"], "appended")
        self.assertTrue(appended["history"][-1]["seed_unchanged"])

        # 第二版运行用于比较。
        status, run2 = self._post_json("/api/synthetic", {
            "left_rows": 5000, "right_rows": 5000, "distinct_keys": 40,
            "hot_key_multiplier": 80, "null_key_fraction": 0.03,
            "condition": "abs(L.v - R.v) <= 200",
            "sample_size": 150, "mode": "WOR", "seed": 99,
            "max_pairs_per_key": 5000, "estimator_probes": 300,
        })
        self.assertEqual(status, 201)
        status, report = self._post_json("/api/compare", {
            "base_id": run_id, "other_id": run2["id"],
        })
        self.assertEqual(status, 200)
        self.assertIn("key_distribution", report)
        self.assertIn("shared_pair_count", report)
        # 两版数据参数一致 → 同一快照指纹；抽样种子不同 → 抽到的对可不同。
        self.assertTrue(report["same_snapshot"])

        # 运行列表与详情。
        status, listing = self._get("/api/runs")
        self.assertEqual(status, 200)
        ids = {r["id"] for r in json.loads(listing.decode())["runs"]}
        self.assertIn(run_id, ids)

    def test_csv_upload_multipart(self):
        boundary = "----b" + uuid.uuid4().hex
        left_csv = "k,v\n1,10\n1,11\n2,20\n,99\n"
        right_csv = "k,v\n1,12\n1,13\n2,21\n,88\n"

        def part(name, filename, content):
            return (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                "Content-Type: text/csv\r\n\r\n"
                f"{content}\r\n"
            ).encode()

        fields = (
            part("left_file", "l.csv", left_csv)
            + part("right_file", "r.csv", right_csv)
            + f"--{boundary}\r\nContent-Disposition: form-data; name=\"key\"\r\n\r\nk\r\n"
              f"--{boundary}\r\nContent-Disposition: form-data; name=\"sample_size\"\r\n\r\n50\r\n"
              f"--{boundary}\r\nContent-Disposition: form-data; name=\"mode\"\r\n\r\nWR\r\n"
              f"--{boundary}\r\nContent-Disposition: form-data; name=\"seed\"\r\n\r\n7\r\n"
              f"--{boundary}--\r\n".encode()
        )
        req = urllib.request.Request(
            self.base + "/api/upload",
            data=fields,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            run = json.loads(resp.read().decode())
        self.assertEqual(len(run["records"]), 50)
        # NULL 键行不参与关联，样本中只应出现键 1、2。
        self.assertTrue(all(r["key"] in (1, 2) for r in run["records"]))
        self.assertEqual(run["snapshot_describe"]["left_null_key_rows"], 1)

    def test_resource_limit_returns_422(self):
        status, body = self._post_json("/api/synthetic", {
            "left_rows": 30000, "right_rows": 30000, "distinct_keys": 100,
            "condition": "L.v <= R.v",
            "sample_size": 10, "seed": 1,
            "max_pairs_per_key": 10_000_000, "max_build_evaluations": 100,
        })
        self.assertEqual(status, 422)
        self.assertIn("资源上限", body["error"])
        self.assertIn("partial", body)

    def test_bad_condition_returns_400(self):
        status, body = self._post_json("/api/synthetic", {
            "left_rows": 100, "right_rows": 100, "distinct_keys": 5,
            "condition": "__import__('os')",
            "sample_size": 10, "seed": 1,
        })
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_static_index_served(self):
        status, content = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn("关联抽样".encode("utf-8"), content)
        status, js = self._get("/app.js")
        self.assertEqual(status, 200)
        self.assertIn(b"api/synthetic", js)
        status, _ = self._get("/styles.css")
        self.assertEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
