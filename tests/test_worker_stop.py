import subprocess
import sys
import time
import unittest

from backend.resident_worker import stop_process


class StopProcessTests(unittest.TestCase):
    def test_hung_worker_tree_is_ended(self):
        # Parent waits on a child, like the Windows venv launcher and its interpreter.
        code = 'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); time.sleep(60)'
        process = subprocess.Popen([sys.executable, '-c', code])
        time.sleep(1)
        started = time.time()
        stop_process(process, timeout=1)
        self.assertIsNotNone(process.poll())
        self.assertLess(time.time() - started, 20)

    def test_exited_worker_returns_immediately(self):
        process = subprocess.Popen([sys.executable, '-c', 'pass'])
        stop_process(process, timeout=10)
        self.assertEqual(process.returncode, 0)


if __name__ == '__main__':
    unittest.main()
