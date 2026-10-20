import sys
import tempfile
import unittest
import os
import io
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from watch_logs import LogTail, ActivityRows, activity_console, render_activity
from rich.console import Console


class ActivityTests(unittest.TestCase):
    def test_long_stage_history_scrolls_without_hiding_current_progress(self):
        job={'id':'a'*32,'kind':'video','status':'running','stage':'generating','step':2,'total_steps':4,
             'stage_started_at':95,'started_at':10,'request':{},'stage_history':[
                 {'stage':'generating','seconds':1},
                 *[{'stage':f'phase_{n}','seconds':n} for n in range(20)]]}
        def render(offset):
            output=io.StringIO()
            Console(file=output,width=120,color_system=None).print(render_activity(job,{},[],now=100,history_limit=3,history_offset=offset))
            return output.getvalue()
        latest=render(0);earlier=render(17)
        self.assertIn('Phase 19',latest);self.assertNotIn('Phase 0 ',latest)
        self.assertIn('Phase 0',earlier);self.assertNotIn('Phase 19',earlier)
        self.assertIn('2/4 | 50%',latest);self.assertIn('6.0s',latest)
        self.assertEqual(latest.count('Generating'),1)

    def test_completed_stage_times_remain_and_live_unknown_stage_has_no_fake_percent(self):
        job={'id':'a'*32,'kind':'video','status':'running','stage':'encoding_prompt','stage_started_at':90,
             'started_at':70,'stage_seconds':{'loading_encoder':20},'request':{'model_id':'h3-int8'}}
        output=io.StringIO();console=Console(file=output,width=110,color_system=None)
        console.print(render_activity(job,{'pending_count':2},[],now=100))
        text=output.getvalue()
        self.assertIn('20.0s',text);self.assertIn('10.0s',text);self.assertIn('2 waiting',text)
        self.assertIn('Working',text);self.assertNotIn('%',text)
        job.update(status='succeeded',stage='complete',finished_at=100,stage_seconds={'loading_encoder':20,'encoding_prompt':10})
        output=io.StringIO();Console(file=output,width=110,color_system=None).print(render_activity(job,{},[],now=150))
        text=output.getvalue();self.assertIn('Total 30.0s',text);self.assertIn('20.0s',text);self.assertIn('10.0s',text)
        self.assertIn('These times stay visible',text);self.assertNotIn('Working',text)

    @unittest.skipUnless(os.name=='nt','Windows Terminal environment')
    def test_real_terminal_renders_even_with_inherited_dumb_environment(self):
        stream=io.StringIO()
        stream.isatty=lambda:True
        with patch.dict(os.environ,{'WT_SESSION':'test','TERM':'dumb','TTY_INTERACTIVE':'0'}), patch('sys.stdout',stream):
            console=activity_console()
            self.assertTrue(console.is_terminal)
            self.assertTrue(console.is_interactive)
            self.assertFalse(console.is_dumb_terminal)

    def test_http_success_and_repeated_progress_stay_hidden_but_errors_remain(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'server.log'
            path.write_text('INFO: "POST /api/uploads HTTP/1.1" 200 OK\nINFO: "DELETE /api/people/id HTTP/1.1" 204 No Content\n[Civitai] downloading: 2 of 10 GB\nINFO: "GET /api/models HTTP/1.1" 500 Error\nModel load failed\n',encoding='utf-8')
            tail=LogTail(path,'Server')
            lines=tail.read()
            self.assertEqual(lines,['INFO: "GET /api/models HTTP/1.1" 500 Error','Model load failed'])
            self.assertEqual(tail.read(),[])

    def test_parallel_tasks_and_stage_changes_keep_separate_rows(self):
        state={'active':{'id':'job-a','kind':'image','stage':'generating','status':'running','step':2,'total_steps':8},
               'downloads':[{'requested_id':'civitai:1','title':'One','stage':'downloading','status':'running','completed_bytes':20,'total_bytes':100},
                            {'requested_id':'civitai:2','title':'Two','stage':'downloading','status':'running','completed_bytes':60,'total_bytes':200}]}
        tracker=ActivityRows();rows,events=tracker.update(state)
        self.assertEqual(len(rows),3);self.assertEqual(events,[])
        state['downloads'][0]['completed_bytes']=40
        rows,events=tracker.update(state)
        self.assertEqual(events,[]);self.assertEqual([r['done'] for r in rows],[2,40,60])
        state['active']['stage']='saving'
        rows,events=tracker.update(state)
        self.assertEqual(len(events),1);self.assertIn('Generating finished',events[0])
        self.assertEqual(rows[0]['total'],0)

    def test_initial_tail_drops_a_partial_success_line(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'server.log'
            path.write_text('INFO: "GET /'+('x'*13000)+' HTTP/1.1" 200 OK\nServer ready\n',encoding='utf-8')
            self.assertEqual(LogTail(path,'Server').read(),['Server ready'])
