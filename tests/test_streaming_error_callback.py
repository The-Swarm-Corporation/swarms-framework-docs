"""Run the documented callback offline, without constructing an Agent."""
import ast
import contextlib
import io
import logging
from pathlib import Path
import re
import unittest


class CallbackTests(unittest.TestCase):
    def setUp(self):
        text = Path("examples/streaming.mdx").read_text()
        section = text.split("### 2. Error Handling", 1)[1].split("### 3.", 1)[0]
        code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
        tree = ast.parse(code)
        # Execute the example's definitions/imports, stopping before Agent creation.
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "agent" for t in node.targets):
                break
            nodes.append(node)
        namespace = {"__name__": "streaming_example_test"}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "documented_callback", "exec"), namespace)
        self.callback = namespace["safe_streaming_callback"]
        self.nodes = nodes
        self.text = text

    def test_previous_file_logger_does_not_supply_error_method(self):
        section = self.text.split("### 3. File Logging", 1)[1].split("## Best Practices", 1)[0]
        code = re.search(r"```python\n(.*?)```", section, re.S).group(1)
        tree = ast.parse(code)
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)]
        namespace = {"__name__": "streaming_example_test"}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "previous_example", "exec"), namespace)
        namespace["logger"] = namespace["StreamLogger"]("unused.log")
        exec(compile(ast.Module(body=self.nodes, type_ignores=[]), "documented_callback", "exec"), namespace)
        class BrokenOutput:
            def write(self, value):
                raise OSError("output unavailable")
            def flush(self):
                pass
        with self.assertLogs("streaming_example_test", level="ERROR") as logs:
            with contextlib.redirect_stdout(BrokenOutput()):
                namespace["safe_streaming_callback"]("token")
        self.assertIn("output unavailable", logs.output[0])

    def test_normal_token_is_printed_unchanged(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.callback("hello 🌍")
        self.assertEqual(output.getvalue(), "hello 🌍")

    def test_output_failure_is_logged_and_callback_can_continue(self):
        class BrokenOutput:
            def write(self, value):
                raise OSError("output unavailable")
            def flush(self):
                pass
        with self.assertLogs("streaming_example_test", level="ERROR") as logs:
            with contextlib.redirect_stdout(BrokenOutput()):
                self.callback("first")
        self.assertIn("Streaming error: output unavailable", logs.output[0])
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.callback("second")
        self.assertEqual(output.getvalue(), "second")

if __name__ == "__main__":
    unittest.main(verbosity=2)
