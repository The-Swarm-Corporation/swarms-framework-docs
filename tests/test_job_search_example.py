"""Offline tests of the documented function, without API or model calls.

Run from the repository root: python tests/test_job_search_example.py -q
Optionally pass an MDX path before unittest options to check another revision.
"""

import ast
import http.client
import json
import os
import sys
import unittest
import urllib.parse
from pathlib import Path
from unittest.mock import patch


def load(path):
    text = Path(path).read_text(encoding="utf-8")
    start = text.index("```python\n") + 10
    end = text.index("\n```", start)
    tree = ast.parse(text[start:end])
    fn = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "get_jobs"
        )
    )
    namespace = {
        "http": __import__("http"),
        "json": json,
        "os": os,
        "urllib": __import__("urllib"),
    }
    # Execute only the reviewed tool function from this checkout, not the agents.
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["get_jobs"]


get_jobs = load(
    sys.argv.pop(1)
    if len(sys.argv) > 1 and (not sys.argv[1].startswith("-"))
    else Path(__file__).resolve().parents[1] / "examples/applications/job-finding.mdx"
)


class Contract(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"RAPIDAPI_KEY": "test-key-not-a-credential"})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.factory = patch("http.client.HTTPSConnection")
        self.connect = self.factory.start()
        self.addCleanup(self.factory.stop)
        self.conn = self.connect.return_value
        self.response = self.conn.getresponse.return_value
        self.response.status = 200
        self.body({"status": "OK", "data": []})

    def body(self, obj):
        self.response.read.return_value = json.dumps(obj).encode()

    def test_empty_success(self):
        self.assertEqual(json.loads(get_jobs("engineer")), [])
        self.conn.close.assert_called_once()

    def test_auth_failure_not_empty_result(self):
        self.response.status = 401
        self.body({"message": "not authorized"})
        with self.assertRaisesRegex(RuntimeError, "401"):
            get_jobs("engineer")
        self.conn.close.assert_called_once()

    def test_rate_limit_not_empty_result(self):
        self.response.status = 429
        with self.assertRaisesRegex(RuntimeError, "429"):
            get_jobs("engineer")

    def test_timeout_closes_connection(self):
        self.conn.request.side_effect = TimeoutError("do not echo raw diagnostic")
        with self.assertRaisesRegex(RuntimeError, "connectivity"):
            get_jobs("engineer")
        self.conn.close.assert_called_once()

    def test_malformed_json_not_returned(self):
        self.response.read.return_value = b"<html>service error</html>"
        with self.assertRaisesRegex(RuntimeError, "invalid JSON"):
            get_jobs("engineer")

    def test_api_error_status(self):
        self.body({"status": "ERROR", "data": []})
        with self.assertRaises(RuntimeError):
            get_jobs("engineer")

    def test_bad_shapes(self):
        for obj in [
            [],
            None,
            {"status": "OK", "data": None},
            {"status": "OK", "data": ["bad"]},
        ]:
            with self.subTest(obj=obj):
                self.body(obj)
                with self.assertRaises(RuntimeError):
                    get_jobs("engineer")

    def test_null_experience_and_limit(self):
        self.body(
            {
                "status": "OK",
                "data": [
                    {
                        "job_title": "Developer",
                        "job_required_experience": None,
                        "job_country": "US",
                    },
                    {"job_title": "Second"},
                ],
            }
        )
        result = json.loads(get_jobs("engineer", 1))
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0]["experience"])
        self.assertEqual(result[0]["location"], "US")

    def test_experience_zero_preserved(self):
        self.body(
            {
                "status": "OK",
                "data": [
                    {"job_required_experience": {"required_experience_in_months": 0}}
                ],
            }
        )
        self.assertEqual(json.loads(get_jobs("engineer"))[0]["experience"], 0)

    def test_missing_key_before_network(self):
        with patch.dict(os.environ, {"RAPIDAPI_KEY": " "}):
            with self.assertRaises(ValueError):
                get_jobs("engineer")
        self.connect.assert_not_called()

    def test_invalid_input_before_network(self):
        for query, limit in [
            ("", 10),
            (None, 10),
            ("x", 0),
            ("x", 101),
            ("x", True),
            ("x", 1.5),
        ]:
            with self.subTest(query=query, limit=limit):
                with self.assertRaises(ValueError):
                    get_jobs(query, limit)
        self.connect.assert_not_called()

    def test_query_encoding_and_key(self):
        get_jobs("C++ & data / München?")
        path = self.conn.request.call_args.args[1]
        self.assertEqual(
            urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)["query"],
            ["C++ & data / München?"],
        )
        self.assertEqual(
            self.conn.request.call_args.kwargs["headers"]["x-rapidapi-key"],
            "test-key-not-a-credential",
        )
        self.connect.assert_called_once_with("jsearch.p.rapidapi.com", timeout=30)

    def test_response_bound(self):
        self.response.read.return_value = b"x" * 2000001
        with self.assertRaisesRegex(RuntimeError, "2 MB"):
            get_jobs("engineer")
        self.response.read.assert_called_once_with(2000001)
        self.conn.close.assert_called_once()

    def test_full_result_preserves_fields(self):
        self.body(
            {
                "status": "OK",
                "data": [
                    {
                        "job_title": "Développeur",
                        "employer_name": "Example",
                        "job_city": "Paris",
                        "job_country": "FR",
                        "job_required_experience": {
                            "required_experience_in_months": 24
                        },
                        "job_apply_link": "https://example.com/jobs/1",
                    }
                ],
            }
        )
        self.assertEqual(
            json.loads(get_jobs("developer")),
            [
                {
                    "title": "Développeur",
                    "company": "Example",
                    "location": "Paris",
                    "experience": 24,
                    "url": "https://example.com/jobs/1",
                }
            ],
        )

    def test_response_read_failure_closes_connection(self):
        self.response.read.side_effect = http.client.IncompleteRead(b"partial")
        with self.assertRaisesRegex(RuntimeError, "connectivity"):
            get_jobs("developer")
        self.conn.close.assert_called_once()

    def test_exact_response_size_allowed(self):
        self.response.read.return_value = b'{"status":"OK","data":[]}'.ljust(2000000)
        self.assertEqual(json.loads(get_jobs("developer")), [])


if __name__ == "__main__":
    unittest.main()
