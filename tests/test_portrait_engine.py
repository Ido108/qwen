import subprocess
import tempfile
import unittest
import json
import importlib.util
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch

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


class EngineInstallTests(unittest.TestCase):
    def test_engine_does_not_inherit_the_launchers_python_preference(self):
        with patch.dict(os.environ, UV_PYTHON_PREFERENCE='only-managed', UV_LINK_MODE='copy'):
            env = portrait_engine.engine_env()
            self.assertNotIn('UV_PYTHON_PREFERENCE', env)
            self.assertEqual(env['UV_LINK_MODE'], 'copy')
            self.assertEqual(os.environ['UV_PYTHON_PREFERENCE'], 'only-managed')

    def test_installer_failure_reports_the_tool_error_before_wrapper_output(self):
        process = Mock(stdout=['error: incompatible Python options\n', *['more output\n'] * 15,
                               'Installation failed: command exited with code 2\n'])
        process.wait.return_value = 2
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (patch.object(portrait_engine, 'ENGINE_MODELS', root / 'models'),
                  patch.object(portrait_engine, 'ENGINE_DATA', root / 'data'),
                  patch.object(portrait_engine, 'INSTALL_LOG', root / 'install.log'),
                  patch.object(portrait_engine, 'checkout'), patch.object(portrait_engine, 'seed_models'),
                  patch.object(portrait_engine.subprocess, 'Popen', return_value=process),
                  patch.object(portrait_engine, 'quick_check') as check,
                  patch('builtins.print')):
                with self.assertRaisesRegex(portrait_engine.EngineError, 'error: incompatible Python options'):
                    portrait_engine.install(lambda **_: None)
                check.assert_not_called()
                self.assertIn('command exited with code 2', (root / 'install.log').read_text())


class PortraitInputTests(unittest.TestCase):
    def test_separate_drive_copies_selected_checkpoints_and_shared_dependencies(self):
        request = {'kind': 'animate', 'options': {'animation_engine': 'mrfa', 'mrfa_checkpoint': 'celebvhq',
                                                 'lipsync': True, 'sync_model': 'tbdub_teacher'}}
        for name in ('tbdub/tbdub_teacher.safetensors', 'tbdub/Wan2.2_VAE.safetensors', 'mrfa/celebvhq.safetensors',
                     'liveportrait/landmark.onnx'):
            self.assertTrue(portrait_engine.copy_needed(Path(name), request))
        for name in ('tbdub/tbdub_student.safetensors', 'personalive/motion.pth', 'mrfa/vox.safetensors'):
            self.assertFalse(portrait_engine.copy_needed(Path(name), request))

    @unittest.skipUnless(importlib.util.find_spec('av') and importlib.util.find_spec('imageio_ffmpeg'), 'media runtime')
    def test_trimmed_audio_uses_local_ffmpeg_without_preparing_models(self):
        import imageio_ffmpeg
        from backend import portrait_worker
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / 'with-audio.mp4'
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-loglevel', 'error', '-f', 'lavfi', '-i',
                            'color=size=32x32:duration=1', '-f', 'lavfi', '-i', 'sine=duration=1',
                            '-c:v', 'libx264', '-c:a', 'aac', str(source)], check=True)
            job = {'id': 'sample', 'started_at': time.time()}
            with patch.object(portrait_worker, 'resolve_video', return_value=source), patch.object(portrait_engine, 'ensure') as ensure:
                for fmt in ('wav', 'mp3'):
                    request = dict(kind='extract-audio', source_video='id', start=.45, duration=.2,
                                   audio_format=fmt, output_dir=str(folder / 'output'))
                    portrait_worker.run(request, job, folder, lambda **values: job.update(values))
                    self.assertEqual(job['status'], 'succeeded')
                    self.assertEqual(job['audio_format'], fmt)
                    self.assertAlmostEqual(job['duration_seconds'], .2, delta=.055)
                    self.assertTrue(Path(job['output_path']).is_file())
                ensure.assert_not_called()

    def test_repeated_upload_reuses_only_an_existing_unchanged_input_of_the_same_kind(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            uploads = folder / 'uploads'
            uploads.mkdir()
            source = folder / 'clip.mp4'
            source.write_bytes(b'original')
            engine = portrait_engine.Engine(folder)
            engine.client = Mock()
            response = Mock(status_code=200)
            identifier = 'a' * 64
            response.json.return_value = {'id': identifier}
            engine.client.post.return_value = response
            with patch.object(portrait_engine, 'ENGINE_DATA', folder):
                engine.upload('driving', source)
                (uploads / (identifier + '.json')).write_text(json.dumps({'kind':'driving'}))
                stored = uploads / (identifier + '.mp4')
                stored.write_bytes(source.read_bytes())
                self.assertEqual(engine.upload('driving', source), identifier)
                self.assertEqual(engine.client.post.call_count, 1)
                engine.upload('audio', source)
                source.write_bytes(b'updated clip')
                engine.upload('driving', source)
                stored.unlink()
                engine.upload('driving', source)
                self.assertEqual(engine.client.post.call_count, 4)


if __name__ == '__main__':
    unittest.main()
