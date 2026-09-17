"""Exercise the documented query function without importing Swarms or calling an LLM.

Run: python -m unittest discover -s tests -p test_smart_database_query.py -v
"""
import ast
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import unittest
from unittest.mock import patch


def load_query():
    page = Path(__file__).resolve().parents[1] / "examples/applications/smart-database.mdx"
    blocks = re.findall(r"```python\n(.*?)\n```", page.read_text(encoding="utf-8"), re.S)
    functions = [
        node
        for block in blocks
        for node in ast.parse(block).body
        if isinstance(node, ast.FunctionDef) and node.name == "query_database"
    ]
    if len(functions) != 1:
        raise AssertionError("Expected exactly one documented query_database function")
    namespace = {"json": json, "Path": Path, "sqlite3": sqlite3}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(page), "exec"), namespace)
    return namespace["query_database"]


class QuerySerializationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = str(Path(self.directory.name) / "example.sqlite")
        connection = sqlite3.connect(self.path)
        try:
            connection.executescript(
                "CREATE TABLE customers (id INTEGER, name TEXT);"
                "CREATE TABLE orders (id INTEGER, customer_id INTEGER);"
                "INSERT INTO customers VALUES (7, '客户');"
                "INSERT INTO orders VALUES (42, 7);"
            )
            connection.commit()
        finally:
            connection.close()
        self.query = load_query()

    def run_query(self, sql, params="[]"):
        return json.loads(self.query(self.path, sql, params))

    def assert_alias_error(self, result):
        self.assertEqual(result["status"], "error")
        self.assertIn("alias", result["error"].lower())
        self.assertNotIn("results", result)

    def test_duplicate_join_labels_are_not_silently_dropped(self):
        self.assert_alias_error(self.run_query(
            "SELECT customers.id, orders.id FROM customers "
            "JOIN orders ON customers.id = orders.customer_id"
        ))

    def test_duplicate_expression_aliases_are_rejected(self):
        self.assert_alias_error(self.run_query("SELECT 1 AS value, 2 AS value"))

    def test_duplicate_labels_are_detected_even_without_rows(self):
        self.assert_alias_error(self.run_query(
            "SELECT id AS value, name AS value FROM customers WHERE 0"
        ))

    def test_case_distinct_labels_preserve_both_values(self):
        result = self.run_query("SELECT 1 AS value, 2 AS VALUE")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["columns"], ["value", "VALUE"])
        self.assertEqual(result["results"], [{"value": 1, "VALUE": 2}])

    def test_explicit_aliases_preserve_join_values(self):
        result = self.run_query(
            "SELECT customers.id AS customer_id, orders.id AS order_id "
            "FROM customers JOIN orders ON customers.id = orders.customer_id"
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["results"], [{"customer_id": 7, "order_id": 42}])
        self.assertEqual(result["row_count"], 1)

    def test_parameters_unicode_and_null_remain_compatible(self):
        result = self.run_query(
            "SELECT id, name, NULL AS optional FROM customers WHERE id = ?", "[7]"
        )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["results"], [{"id": 7, "name": "客户", "optional": None}])
        self.assertEqual(result["columns"], ["id", "name", "optional"])

    def test_duplicate_label_error_closes_connection(self):
        opened = []
        real_connect = sqlite3.connect

        def track_connection(*args, **kwargs):
            connection = real_connect(*args, **kwargs)
            opened.append(connection)
            return connection

        with patch.object(sqlite3, "connect", side_effect=track_connection):
            result = self.run_query("SELECT 1 AS value, 2 AS value")
        self.assert_alias_error(result)
        self.assertEqual(len(opened), 1)
        with self.assertRaises(sqlite3.ProgrammingError):
            opened[0].execute("SELECT 1")

    def test_empty_result_keeps_column_metadata(self):
        result = self.run_query("SELECT id, name FROM customers WHERE id = ?", "[999]")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["results"], [])
        self.assertEqual(result["row_count"], 0)
        self.assertEqual(result["columns"], ["id", "name"])


if __name__ == "__main__":
    unittest.main()
