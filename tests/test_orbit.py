import unittest
from unittest.mock import patch
from PIL import Image
from backend import models
from backend.orbit import OrbitRequest, video_request, prepare_image, PROMPT
from backend.video_sampling import video_sampling


class OrbitTests(unittest.TestCase):
    def test_embedded_fl2va_turbo_uses_its_steps_without_an_extra_turbo_lora(self):
        real_entry=models.entry
        turbo={'id':'test-turbo','kind':'video','family':'minimax-h3','modes':['text','image'],
               'dependencies':['h3-encoder','h3-video-vae','h3-audio-vae'],
               'generation_defaults':{'steps':4,'embedded_turbo':True,'sampler':'euler'}}
        with patch.object(models,'entry',side_effect=lambda key:turbo if key=='test-turbo' else real_entry(key)):
            request=video_request(OrbitRequest(image='photo.png',model_id='test-turbo'))
            needed=models.required_video(request)
            self.assertEqual(request.steps,4)
            self.assertEqual(request.first_frame,request.last_frame)
            self.assertNotIn('h3-turbo',needed)
            self.assertEqual(video_sampling(turbo,request.quality,request.steps)['sampler'],'euler')
            for family in ('fasth3','minimax-h3-ref2va'):
                turbo['family']=family
                with self.assertRaises(ValueError):video_request(OrbitRequest(image='photo.png',model_id='test-turbo'))

    def test_new_standard_model_appears_by_capability_and_author_steps_win(self):
        from backend.orbit import profile
        lora=models.entry('h3-360-orbit')
        item={'id':'a-new-checkpoint-not-in-any-list','kind':'video','family':'minimax-h3','modes':['text','image'],
              'generation_defaults':{'steps':20}}
        self.assertEqual(profile(item,lora)['steps'],28)
        item['generation_defaults'].update(steps=12,field_sources={'steps':'version'})
        self.assertEqual(profile(item,lora)['steps'],12)
        item['modes']=['text']
        self.assertIsNone(profile(item,lora))

    def test_author_contract_reaches_queue_dependencies_and_sampler(self):
        request=video_request(OrbitRequest(image='photo.png'))
        self.assertEqual((request.first_frame,request.last_frame),('photo.png','photo.png'))
        self.assertEqual((request.width,request.height,request.frames,request.steps,request.audio),(768,768,73,28,False))
        self.assertEqual(request.prompt,PROMPT)
        needed=models.required_video(request)
        self.assertEqual(set(needed),{'h3-int8','h3-encoder','h3-video-vae','h3-audio-vae','h3-360-orbit'})
        profile=video_sampling(models.entry(request.model_id),request.quality,request.steps,True)
        self.assertEqual((profile['steps'],profile['cfg']),(28,1))
        self.assertNotIn('h3-turbo',needed)

    def test_generic_endpoint_cannot_silently_change_orbit_recipe(self):
        request=video_request(OrbitRequest(image='photo.png'))
        for update in ({'last_frame':'other.png'},{'steps':20},{'model_id':'fasth3-q4_k_m'},{'audio':True},{'width':1024},{'prompt':'other'}):
            with self.subTest(update=update),self.assertRaises(ValueError):
                models.required_video(request.model_copy(update=update))

    def test_verified_q4_uses_same_recipe_without_turbo(self):
        request=video_request(OrbitRequest(image='photo.png',model_id='h3-gguf-q4'))
        needed=models.required_video(request)
        self.assertIn('h3-gguf-q4',needed)
        self.assertIn('h3-360-orbit',needed)
        self.assertNotIn('h3-int8',needed)
        self.assertNotIn('h3-turbo',needed)
        self.assertEqual(request.steps,28)

    def test_center_crop_preserves_shape_and_input(self):
        image=Image.new('RGB',(1600,800),'red')
        image.paste('blue',(400,0,1200,800))
        result=prepare_image(image)
        self.assertEqual(result.size,(768,768))
        self.assertEqual(result.getpixel((384,384)),(0,0,255))
        self.assertEqual(image.size,(1600,800))

    def test_strength_seed_survive_roundtrip(self):
        request=video_request(OrbitRequest(image='photo.png',strength=.75,seed=123,randomize_seed=False))
        self.assertEqual(request.loras[0].strength,.75)
        self.assertEqual(request.seed,123)
        self.assertFalse(request.randomize_seed)


if __name__=='__main__': unittest.main()
