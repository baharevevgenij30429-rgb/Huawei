import asyncio
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import run_agent as agent


class BaselineTests(unittest.TestCase):
    def test_labels_never_enter_task(self):
        with tempfile.TemporaryDirectory() as root:
            task = Path(root) / 'one'
            task.mkdir()
            (task / 'task.json').write_text(json.dumps(dict(
                instruction='calculate', reward=2, expected='SECRET', check='exact_int')))
            loaded = agent.load_tasks(Path(root))
            self.assertNotIn('SECRET', repr(loaded))
            self.assertNotIn('expected', loaded[0])

    def test_full_loop_saves_only_answer_and_charges(self):
        class FakeClient:
            def __init__(self, name):
                self.calls = 0

            async def call(self, messages, cap):
                self.calls += 1
                if self.calls == 1:
                    return json.dumps({'action': 'python', 'code': 'print(6 * 7)'}), {'input_tokens': 100, 'output_tokens': 20}
                assert '42' in messages[-1]['content']
                return json.dumps({'action': 'answer', 'answer': '42'}), {'input_tokens': 100, 'output_tokens': 10}

        with tempfile.TemporaryDirectory() as root:
            task = Path(root) / 'one'
            task.mkdir()
            (task / 'task.json').write_text(json.dumps(dict(instruction='6 * 7', reward=1)))
            args = SimpleNamespace(tasks_dir=Path(root), dry_run=False, model='fake',
                seconds=10, budget=10000, max_steps=3, max_output=100,
                answer_location='task')
            output = io.StringIO()
            with patch.object(agent, 'JiuwenClient', FakeClient), contextlib.redirect_stdout(output):
                asyncio.run(agent.run(args))
            self.assertEqual((task / 'answer.txt').read_text(), '42\n')
            self.assertEqual(json.loads(output.getvalue())['charged_or_reserved'], 50)

    def test_usage_including_reasoning_is_not_added_twice(self):
        self.assertEqual(agent.token_cost({'prompt_tokens': 100,
            'completion_tokens': 30, 'completion_tokens_details': {'reasoning_tokens': 20}}), 40)
        self.assertIsNone(agent.token_cost({}))


if __name__ == '__main__':
    unittest.main()
