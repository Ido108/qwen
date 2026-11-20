import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import portrait_engine


def run(*arguments, cwd):
    return subprocess.run(['git', *arguments], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


class EngineCheckoutTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        self.source = root / 'Multifck'
        (self.source / 'portrait_app').mkdir(parents=True)
        (self.source / 'Install.ps1').write_text('installer\n')
        (self.source / 'portrait_app' / 'server.py').write_text('server\n')
        run('init', '--quiet', '--initial-branch=main', cwd=self.source)
        run('-c', 'user.name=t', '-c', 'user.email=t@t', 'add', '.', cwd=self.source)
        run('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '--quiet', '-m', 'one', cwd=self.source)
        self.first = run('rev-parse', 'HEAD', cwd=self.source)
        self.engine = root / 'runtime' / 'engines' / 'multifck'
        self.patches = [patch.object(portrait_engine, 'ENGINE_DIR', self.engine),
                        patch.object(portrait_engine, 'local_source', lambda: self.source)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.folder.cleanup()

    def checkout(self, commit):
        with patch.object(portrait_engine, 'ENGINE_COMMIT', commit):
            portrait_engine.checkout(lambda **_: None)

    def test_first_download_of_the_newest_commit_has_the_files(self):
        self.checkout(self.first)
        self.assertEqual((self.engine / 'Install.ps1').read_text(), 'installer\n')
        self.assertTrue((self.engine / 'portrait_app' / 'server.py').is_file())
        self.assertEqual(portrait_engine.head(), self.first)

    def test_first_download_of_an_older_commit_has_the_files(self):
        (self.source / 'Install.ps1').write_text('installer two\n')
        run('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '--quiet', '-am', 'two', cwd=self.source)
        self.checkout(self.first)
        self.assertEqual((self.engine / 'Install.ps1').read_text(), 'installer\n')

    def test_a_new_pinned_commit_updates_and_a_changed_file_is_kept(self):
        self.checkout(self.first)
        (self.source / 'Install.ps1').write_text('installer two\n')
        run('-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '--quiet', '-am', 'two', cwd=self.source)
        second = run('rev-parse', 'HEAD', cwd=self.source)
        self.checkout(second)
        self.assertEqual((self.engine / 'Install.ps1').read_text(), 'installer two\n')
        (self.engine / 'Install.ps1').write_text('my change\n')
        with self.assertRaises(portrait_engine.EngineError):
            self.checkout(self.first)
        self.assertEqual((self.engine / 'Install.ps1').read_text(), 'my change\n')


if __name__ == '__main__':
    unittest.main()
