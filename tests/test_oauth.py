"""Connection boundary tests. Google exchange is mocked; live sign-in is separate."""
import base64
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlsplit
import httpx
from PIL import Image
from backend import antigravity as auth
from backend.prompt_enhancer import enhance

class OAuthTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix='qwen-oauth-test-')
        folder = Path(self.directory.name)
        self.patches = [patch.object(auth, 'AUTH_DIR', folder), patch.object(auth, 'TOKEN_FILE', folder/'tokens.dpapi'),
                        patch.object(auth, 'SETTINGS_FILE', folder/'settings.json'), patch.object(auth, '_flows', {}),
                        patch.object(auth, '_callback', None)]
        for item in self.patches: item.start()

    def tearDown(self):
        if auth._callback:
            auth._callback.shutdown()
            auth._callback.server_close()
        for item in reversed(self.patches): item.stop()
        self.directory.cleanup()

    def token(self, expires=None):
        return {'accessToken':'test-access-secret','refreshToken':'test-refresh-secret','projectId':'example-project',
                'projectsPool':['example-project'], 'expiresAt': expires or (time.time()+3600)*1000}

    def test_tokens_are_encrypted_and_status_contains_no_secrets(self):
        auth.save_tokens(self.token())
        self.assertNotIn(b'test-access-secret', auth.TOKEN_FILE.read_bytes())
        self.assertEqual(auth.load_tokens(), self.token(expires=auth.load_tokens()['expiresAt']))
        status=auth.status()
        self.assertTrue(status['connected'])
        self.assertNotIn('secret', json.dumps(status))
        auth.disconnect()
        self.assertFalse(auth.TOKEN_FILE.exists())
        self.assertFalse(auth.status()['connected'])

    def test_pkce_callback_state_single_exchange_and_expiry(self):
        with patch.object(auth,'_start_listener',return_value=False):
            started=auth.start()
        args=parse_qs(urlsplit(started['authUrl']).query)
        state=started['state'];verifier=auth._flows[state]['verifier']
        digest=base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip('=')
        self.assertEqual(args['code_challenge'],[digest])
        self.assertEqual(args['redirect_uri'],[auth.REDIRECT_URI])
        with patch.object(auth,'_token_request',return_value=self.token()) as exchange:
            for url in ('http://evil.example:1455/auth/callback?code=x&state='+state,auth.REDIRECT_URI+'?code=x&state=wrong'):
                with self.assertRaises(ValueError):auth.exchange(url)
            exchange.assert_not_called()
            url=auth.REDIRECT_URI+'?code=fake&state='+state
            auth.exchange(url);auth.exchange(url)
            self.assertEqual(exchange.call_count,1)
            self.assertEqual(exchange.call_args.args[0]['code_verifier'],verifier)
        self.assertEqual(auth.poll(state),{'status':'completed','success':True})
        auth._flows[state]['createdAt']-=601
        self.assertEqual(auth.poll(state),{'status':'expired'})

    def test_refresh_rechecks_disk_and_preserves_refresh_token(self):
        auth.save_tokens(self.token(expires=(time.time()+30)*1000))
        response=httpx.Response(200,json={'access_token':'new-access','expires_in':3600})
        with patch.object(auth.httpx,'post',return_value=response) as post, patch.object(auth,'discover_projects',return_value=['new-project']):
            first=auth.ensure_tokens();second=auth.ensure_tokens()
        self.assertEqual(post.call_count,1)
        self.assertEqual(first,second)
        self.assertEqual(first['refreshToken'],'test-refresh-secret')
        self.assertEqual(first['projectId'],'new-project')

    def test_model_wire_labels_and_thinking_contract(self):
        body=auth.request_body('gemini-3.6-flash','system',[{'text':'user'}],'project')
        self.assertEqual(body['model'],'gemini-3.6-flash-tiered')
        self.assertEqual(body['request']['labels']['model_enum'],'MODEL_PLACEHOLDER_M196')
        self.assertNotIn('thinkingBudget',json.dumps(body))
        self.assertTrue(body['requestId'].startswith('agent/'))
        pro=auth.request_body('gemini-3.1-pro-low','system',[],'project')
        self.assertEqual(pro['request']['generationConfig']['thinkingConfig'],{'thinkingLevel':'low'})

    def test_sse_ignores_thinking_and_surfaces_refusal(self):
        lines=[{'response':{'candidates':[{'content':{'parts':[{'thought':True,'text':'private thought'},{'text':'{"ok":'}]}}]}},
               {'response':{'candidates':[{'content':{'parts':[{'text':'true}'}]},'finishReason':'STOP'}]}}]
        response=httpx.Response(200,text='\n'.join('data: '+json.dumps(line) for line in lines))
        self.assertEqual(auth._read_stream(response),'{"ok":true}')
        with self.assertRaises(auth.ConnectionError):
            auth._read_stream(httpx.Response(200,text='data: {"response":{"candidates":[{"finishReason":"SAFETY"}]}}'))

    def test_edit_prompt_images_keep_order_and_unicode_text(self):
        output=json.dumps({'rewritten_prompt':'Change the label on <image1> to "Café Zürich". Keep <image2> as the background.','wh_ratio':'','ratio_follow':'<image1>'},ensure_ascii=False)
        with patch.object(auth,'generate_text',return_value=output) as call:
            result=enhance('Change the label to Café Zürich',[Image.new('RGBA',(20,30),'red'),Image.new('RGBA',(30,20),'blue')],'gemini-3.7-flash')
        parts=call.call_args.args[2]
        self.assertEqual([parts[0]['text'],parts[2]['text']],['<image1>','<image2>'])
        self.assertIn('inlineData',parts[1])
        self.assertIn('Café Zürich',result['positive_prompt'])
        self.assertEqual(result['ratio_follow'],'<image1>')

    def test_transparency_contract_survives_rewrite(self):
        with patch.object(auth,'generate_text',return_value='{"rewritten_prompt":"A red flower.","wh_ratio":"1:1"}'):
            result=enhance('This is an RGBA image with transparency. A red flower.',[],'gemini-3.7-flash')
        self.assertTrue(result['positive_prompt'].startswith('This is an RGBA image with transparency.'))
        self.assertTrue(result['positive_prompt'].endswith('The image has alpha channel and the background is transparent.'))

    def test_listener_roundtrip_without_google(self):
        # Exercises the real local HTTP callback, with only Google exchange mocked.
        self.assertTrue(auth._start_listener(port=0))
        port=auth._callback.server_address[1]
        started=auth.start()
        with patch.object(auth,'_token_request',return_value=self.token()):
            response=httpx.get(f'http://127.0.0.1:{port}/auth/callback',params={'code':'fake','state':started['state']})
        self.assertEqual(response.status_code,200)
        self.assertTrue(auth.poll(started['state'])['success'])
        self.assertNotIn('fake',response.text)

    def test_callback_port_is_exclusive(self):
        self.assertTrue(auth._start_listener(port=0))
        server=auth._callback
        with patch.object(auth,'_callback',None):
            self.assertFalse(auth._start_listener(port=server.server_address[1]))

if __name__=='__main__':unittest.main()
