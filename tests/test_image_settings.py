import unittest
from types import SimpleNamespace
from backend.image_settings import output_dimensions


class ImageSizingTests(unittest.TestCase):
    def test_auto_1k_and_2k_square_and_input_portrait(self):
        for resolution, square, portrait in [(1024,(1024,1024),(864,1216)),(2048,(2048,2048),(1728,2432))]:
            request={'custom_size':False,'auto_resolution':resolution}
            self.assertEqual(output_dimensions(request,[],{}),square)
            self.assertEqual(output_dimensions(request,[SimpleNamespace(width=1728,height=2432)],{}),portrait)

    def test_rewrite_ratio_and_follow_image_apply_to_both_sizes(self):
        images=[SimpleNamespace(width=100,height=200),SimpleNamespace(width=200,height=100)]
        for resolution in (1024,2048):
            request={'custom_size':False,'auto_resolution':resolution}
            portrait=output_dimensions(request,images,{'ratio_follow':'<image1>'})
            landscape=output_dimensions(request,images,{})
            rewritten=output_dimensions(request,images,{'wh_ratio':'1:1','ratio_follow':'<image1>'})
            self.assertLess(portrait[0],portrait[1])
            self.assertGreater(landscape[0],landscape[1])
            self.assertEqual(rewritten,(resolution,resolution))
            self.assertAlmostEqual(portrait[0]*portrait[1]/resolution**2,1,delta=.04)
            self.assertTrue(all(d%32==0 for d in portrait+landscape))

    def test_custom_and_legacy_sizes_are_preserved(self):
        self.assertEqual(output_dimensions({'custom_size':False},[],{}),(2048,2048))
        self.assertEqual(output_dimensions({'custom_size':True,'auto_resolution':1024,'width':1000,'height':888},[],{'wh_ratio':'1:1'}),(1000,888))

    def test_invalid_ratio_falls_back_to_input(self):
        self.assertEqual(output_dimensions({'custom_size':False,'auto_resolution':1024},[SimpleNamespace(width=100,height=100)],{'wh_ratio':'0:0'}),(1024,1024))
