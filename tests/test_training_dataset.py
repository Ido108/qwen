import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch
from PIL import Image
from backend import training_data as data, training_captions as captions, training_profiles, training


def picture(color='blue'):
    stream = io.BytesIO(); Image.new('RGB', (20, 30), color).save(stream, 'PNG'); return stream.getvalue()


def archive(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        for name, content in files: z.writestr(name, content)
    return stream.getvalue()


def source(name, raw):
    return name, lambda: io.BytesIO(raw), len(raw)


def save(image, name):
    return {'id':hashlib.sha256(image.tobytes()).hexdigest(), 'name':name, 'url':'/image'}


class DatasetTests(unittest.TestCase):
    def test_nested_zip_matches_caption_and_preserves_text(self):
        text='MyToken, a cup.\nA second sentence.'
        raw=archive([('one/a.PNG',picture()),('one/a.txt',text),('two/a.png',picture('red')),('two/a.caption','Red cup'),('notes.txt','orphan')])
        result=data.import_files([source('dataset.zip',raw)],save)
        self.assertEqual([i['caption'] for i in result['items']], [text,'Red cup'])
        self.assertEqual((result['captions'],result['unpaired_captions']), (2,1))

    def test_duplicate_pixels_keep_distinct_captions(self):
        result=data.import_files([source('a.png',picture()),source('a.txt',b'First'),source('b.png',picture()),source('b.txt',b'Second'),source('c.png',picture()),source('c.txt',b'First')],save)
        self.assertEqual(len(result['items']),2)
        self.assertEqual(result['duplicates'],1)
        self.assertNotEqual(result['items'][0]['dataset_key'],result['items'][1]['dataset_key'])

    def test_conflicting_captions_and_unsafe_archives_fail(self):
        with self.assertRaisesRegex(ValueError,'Different captions'):
            data.import_files([source('a.png',picture()),source('a.txt',b'One'),source('a.caption',b'Two')],save)
        for name in ('../escape.png','C:/escape.png','/escape.png'):
            with self.assertRaises(ValueError): data.import_files([source('input.zip',archive([(name,picture())]))],save)

    def test_chunk_retry_and_staging_cleanup_do_not_remove_shared_images(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(data,'ROOT',Path(folder)/'parts'):
            raw=archive([('a.png',picture()),('a.txt',b'A cup')])
            ident=data.create([{'name':'set.zip','size':len(raw)}])['id']
            half=len(raw)//2
            self.assertEqual(data.append(ident,0,0,raw[:half])['offset'],half)
            self.assertEqual(data.append(ident,0,0,raw[:half])['offset'],half)
            with self.assertRaises(ValueError): data.append(ident,0,-1,b'x')
            data.append(ident,0,half,raw[half:])
            self.assertEqual(data.finish(ident,save)['captions'],1)
            self.assertFalse(data.directory(ident).exists())
            data.discard(ident)
            with self.assertRaises(ValueError): data.discard('../outside')

    def test_failed_zip_releases_windows_file_handles_and_removes_parts(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(data,'ROOT',Path(folder)/'parts'):
            raw=archive([('a.png',b'bad image')]); ident=data.create([{'name':'set.zip','size':len(raw)}])['id']
            data.append(ident,0,0,raw)
            with self.assertRaises(OSError): data.finish(ident,save)
            self.assertFalse(data.directory(ident).exists())

    def test_each_architecture_gets_bounded_dataset_specific_values(self):
        for arch in training.ARCHS:
            small=training_profiles.recommend(arch,'character',[{'width':640,'height':768}]*5)
            large=training_profiles.recommend(arch,'style',[{'width':1600,'height':1200}]*50)
            self.assertEqual(small['resolutions'],[512])
            self.assertEqual(large['resolutions'],[512,768,1024])
            self.assertLess(small['steps'],large['steps'])
            self.assertEqual((small['learning_rate'],large['learning_rate']),(1e-4,1e-4))  # every ai-toolkit source uses 1e-4
            self.assertEqual(small['steps'],1500)
            self.assertLessEqual(large['steps'],2000 if training.ARCHS[arch].get('assistant_lora_path') else 3000)
            self.assertEqual(small['rank'],32 if training.ARCHS[arch]['arch'] in ('krea2','moss_image') else 16)

    def test_caption_identity_style_and_cache_keys_are_separate(self):
        character={'name':'Maya','trigger_word':'mayax','subject':'character'}
        style={**character,'subject':'style'}
        self.assertIn('the subject word stands for them',captions.prompt(character))
        self.assertIn('visual style itself implicit',captions.prompt(style))
        self.assertEqual(captions.clean('<think>hidden</think>A cup.',style),'In the style of mayax, A cup.')
        with tempfile.TemporaryDirectory() as folder, patch('backend.models.entry',return_value={'revision':'v1'}):
            image=Path(folder)/'image.png';image.write_bytes(picture())
            with patch.object(captions,'resolve_image',return_value=image):
                self.assertEqual(captions.fingerprint('a',character),captions.fingerprint('a',character))
                self.assertNotEqual(captions.fingerprint('a',character),captions.fingerprint('a',style))
                self.assertNotEqual(captions.fingerprint('a',character),captions.fingerprint('a',{**character,'trigger_word':'other'}))

    def test_existing_caption_cache_skips_model_download_and_process(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cache=root/'cache';cache.mkdir();job=root/'job';job.mkdir()
            (cache/'key.json').write_text(json.dumps({'caption':'Saved cup','key':'key'}))
            request={'images':[{'id':'a','caption':''},{'id':'b','caption':'User text'}]}
            with patch.object(captions,'CACHE',cache),patch.object(captions,'fingerprint',return_value='key'),patch('backend.models.ensure_installed') as download,patch.object(captions.subprocess,'Popen') as spawn:
                result=captions.run(request,job,Mock(),lambda:False,Mock())
            self.assertEqual(result['a']['caption'],'Saved cup')
            self.assertNotIn('b',result)
            download.assert_not_called();spawn.assert_not_called()


class DatasetNotesTests(unittest.TestCase):
    def test_blur_size_and_near_copies_are_reported_from_the_images(self):
        from PIL import Image, ImageDraw, ImageFilter
        from backend.training_profiles import dataset_notes, image_facts
        def photo(seed):
            # Large blocks in a different layout per seed (the hash), fine lines on top (the sharpness).
            image = Image.new('RGB', (1024, 1024), 'white'); draw = ImageDraw.Draw(image)
            for i in range(8):
                if (seed >> (i % 4)) & 1 ^ i % 2: draw.rectangle((i * 128, 0, i * 128 + 127, 1024), fill=(40 * (i % 3), 90, 160))
            for i in range(40): draw.line((i * 25, 0, i * 25 + seed * 9, 1024), fill='black', width=3)
            return image
        sharp = [image_facts(photo(seed)) for seed in range(4)]
        blurred = image_facts(photo(9).filter(ImageFilter.GaussianBlur(8)))
        small = image_facts(photo(3).resize((600, 600)))
        notes = ' '.join(dataset_notes('character', sharp + [blurred, small]))
        self.assertIn('blurrier', notes)
        self.assertIn('smaller than 768 px', notes)
        self.assertIn('nearly the same', notes)  # the small copy of photo 3
        self.assertIn('15 to 40', notes)
        self.assertNotIn('15 to 40', ' '.join(dataset_notes('style', sharp)))
