"""Live activity rows for independent Qwen jobs and downloads."""
from __future__ import annotations
import argparse, codecs, hashlib, json, os, re, sys, time
from pathlib import Path
from urllib.request import urlopen
from urllib.error import HTTPError
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.progress_bar import ProgressBar
from rich.panel import Panel

HTTP_SUCCESS=re.compile(r'"[A-Z]+ .* HTTP/\S+" [23]\d\d\b')
TRANSFER_LINE=re.compile(r'^\[Civitai\] (downloading|verifying|preparing):')
BAR_LINE=re.compile(r'\d+%\|.*\||\d+(?:\.\d+)?\s*(?:it/s|s/it)')
# Red only for real problems: error words, or an HTTP 4xx/5xx status ("GET /x HTTP/1.1" 500). A bare 4xx/5xx
# number also matched millisecond timestamps such as 16:00:38.441.
ALERT=re.compile(r'(?:error|exception|traceback|failed)\b|HTTP/[\d.]+" [45]\d\d\b',re.I)
STAGES={'preparing':'Preparing','loading':'Loading tools','loading_model':'Loading model','loading_encoder':'Loading text encoder','encoding':'Encoding inputs','rewriting':'Preparing prompt','downloading_models':'Downloading required models','generating':'Generating','saving':'Saving result','verifying':'Checking file','downloading':'Downloading','upscaling':'Upscaling','training':'Training','muxing_audio':'Adding audio','complete':'Complete','failed':'Failed','cancelled':'Cancelled'}
STAGES.update(initializing_video='Start video tools', loading_video_vae='Load video encoder',
    preparing_conditioning='Prepare inputs', encoding_prompt='Encode prompt',
    encoding_reference_image='Encode input images', encoding_source_video='Encode source video',
    encoding_source_audio='Encode source audio', loading_audio_vae='Load audio encoder',
    loading_model_gpu='Load model to GPU + apply LoRAs', loading_encoder_gpu='Load text encoder to GPU',
    loading_vae_gpu='Load input encoder to GPU', loading_decoder_gpu='Load video decoder to GPU',
    loading_video_decoder='Prepare video decoder', decoding='Decode result',
    loading_prompt_model='Load prompt model',loading_vae='Load image encoder',sampling='Create sample images',
    loading_audio_encoder_gpu='Move audio encoder to GPU', loading_audio_decoder='Prepare audio decoder',
    loading_audio_decoder_gpu='Move audio decoder to GPU',
    decoding_audio='Decode audio', preview='Prepare preview', releasing_model='Release video model')


def duration(seconds):
    seconds=max(0,float(seconds or 0))
    return f'{int(seconds//60)}m {seconds%60:04.1f}s' if seconds>=60 else f'{seconds:.1f}s'


def load_job(data_dir,job_id):
    try:
        if not re.fullmatch(r'[a-f0-9]{32}',job_id):return None
        return json.loads((data_dir/'jobs'/job_id/'status.json').read_text(encoding='utf-8'))
    except (OSError,ValueError):return None


def render_activity(job,snapshot,rows,online=True,now=None,history_limit=None,history_offset=0):
    """Keep completed timings visible. Unknown progress never gets a made-up percent."""
    now=time.time() if now is None else now
    count=snapshot.get('pending_count',len(snapshot.get('pending') or []))
    connection='Connected' if online else 'Connection lost; retrying'
    queue=f'{count} waiting'+(' | Queue paused' if snapshot.get('paused') else '')
    parts=[Text(f'QWEN   {connection}   |   {queue}',style='bold cyan')]
    if job:
        ended=job.get('status') in ('succeeded','failed','cancelled')
        end=job.get('finished_at',now) if ended else now
        elapsed=max(0,end-job.get('started_at',end))
        status={'succeeded':'Complete','failed':'Failed','cancelled':'Cancelled'}.get(job.get('status'),'Running')
        request=job.get('request') or {}
        name=f"{(job.get('kind') or 'Job').replace('-',' ').title()} {job['id'][:8]}"
        parts.append(Text(f'{name}   |   {status}   |   Total {duration(elapsed)}',style='bold'))
        parts.append(Text(job.get('_model_title') or request.get('model_id',''),style='dim',overflow='ellipsis',no_wrap=True))
        profile=' | '.join(str(v) for v in (
            f"{request['width']} x {request['height']}" if request.get('width') else None,
            f"{request['frames']} frames" if request.get('frames') else None,
            f"{job.get('total_steps') or request.get('steps')} steps" if job.get('total_steps') or request.get('steps') else None) if v)
        parts.append(Text(profile,style='dim',overflow='ellipsis',no_wrap=True))
        table=Table(expand=True,box=None,padding=(0,1),show_header=True)
        table.add_column('Stage',ratio=4);table.add_column('Time',justify='right',width=12)
        table.add_column('Progress',ratio=3);table.add_column('State',width=18)
        history=job.get('stage_history') or [{'stage':key,'seconds':value} for key,value in (job.get('stage_seconds') or {}).items()]
        grouped={}
        for entry in history:
            key=(entry.get('phase',''),entry['stage'])
            if key not in grouped:grouped[key]={**entry,'seconds':0,'_order':entry.get('started_at',len(grouped)),'_measured':False}
            if entry['seconds']>=.1 and not grouped[key]['_measured']:
                grouped[key]['_order']=entry.get('started_at',grouped[key]['_order']);grouped[key]['_measured']=True
            grouped[key]['seconds']+=entry['seconds']
        # A stage can pause for GPU placement and then resume. Display one total
        # for that stage, rather than calling its first fragment complete.
        active_key=(job.get('phase',''),job.get('stage'))
        prior_active=grouped.pop(active_key,None) if not ended else None
        entries=sorted(grouped.values(),key=lambda entry:entry['_order'])
        if history_limit is not None and len(entries)>history_limit:
            offset=min(max(0,history_offset),len(entries)-history_limit)
            stop=len(entries)-offset;start=max(0,stop-history_limit)
            entries=entries[start:stop]
            parts.append(Text(f'Stages {start+1}–{stop} of {len(grouped)} | ↑ / ↓ scroll stages | End: latest',style='dim'))
        for entry in entries:
            label=STAGES.get(entry['stage'],entry['stage'].replace('_',' ').capitalize())
            if entry['stage']=='generating' and job.get('kind') in ('video','reference-video') and not job.get('stage_history'):
                label='Generate + GPU load + preview'
            if entry.get('phase'):label=entry['phase']+' / '+label
            steps=job.get('total_steps',0)
            details=f'{steps}/{steps} steps' if entry['stage']=='generating' and steps else ''
            table.add_row(Text(label),duration(entry['seconds']),Text('Complete',style='green'),details)
        if not ended:
            stage=job.get('stage','preparing');step=job.get('step',0);total=job.get('total_steps',0)
            measured=stage in ('generating','training','upscaling') and total>0
            bar=ProgressBar(total=total,completed=min(step,total)) if measured else ProgressBar(total=None,pulse=True)
            label=STAGES.get(stage,stage.replace('_',' ').capitalize())
            if job.get('phase'):label=job['phase']+' / '+label
            elapsed=now-job.get('stage_started_at',now)+(prior_active or {}).get('seconds',0)
            table.add_row(Text(label,style='bold cyan'),duration(elapsed),bar,
                f'{step}/{total} | {min(100,round(step*100/total))}%' if measured else 'Working')
        parts.append(table)
        if job.get('preview_seconds_total'):
            parts.append(Text(f"Preview time: {duration(job['preview_seconds_total'])} (included in Generate)",style='dim'))
        if not ended:
            parts.append(Text(job.get('message','Working'),style='cyan'))
        elif job.get('status') in ('failed','cancelled'):
            parts.append(Text(job.get('message',status),style='red' if job['status']=='failed' else 'yellow'))
        else:
            parts.append(Text('Result saved. These times stay visible until the next job.',style='green'))
        if job.get('output_path'):parts.append(Text(job['output_path'],style='dim',overflow='ellipsis',no_wrap=True))
    else:
        parts.extend([Text('Ready',style='bold green'),Text('No job is running. Stage times and progress appear here when a job starts.',style='dim')])
    downloads=[r for r in rows if r['units']=='bytes']
    if downloads:parts.extend([Text('Downloads',style='bold'),render_rows(downloads,snapshot,online)])
    parts.append(Text('Closing a Qwen terminal tab stops Qwen and active jobs. Closing the browser does not.',style='dim'))
    return Panel(Group(*parts),title='Activity',border_style='cyan',padding=(1,1))

class LogTail:
    def __init__(self,path,label,from_end=False):
        self.path,self.label=path,label;self.offset=None;self.pending='';self.from_end=from_end
        self.decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')
    def read(self):
        try:
            with self.path.open('rb') as file:
                file.seek(0,2);size=file.tell()
                if self.offset is None or size<self.offset:
                    at_end=self.offset is None and self.from_end
                    self.offset=size if at_end else max(0,size-12000);self.decoder.reset();self.pending=''
                    if self.offset and not at_end:
                        # A bounded initial tail may start inside a UTF-8 line or
                        # progress bar. Drop that fragment before filtering.
                        file.seek(self.offset);file.readline();self.offset=file.tell()
                file.seek(self.offset);data=file.read(128*1024);self.offset=file.tell()
            content=self.pending+self.decoder.decode(data)
            lines=content.splitlines(keepends=True)
            self.pending=lines.pop() if lines and not lines[-1].endswith(('\n','\r')) else ''
            output=[]
            for line in lines:
                if 'Targets are not supported by the new graph engine for now' in line:continue
                if not line.strip() or HTTP_SUCCESS.search(line) or TRANSFER_LINE.search(line) or BAR_LINE.search(line):continue
                if '"GET /api/activity HTTP/' in line and '404' in line:continue
                try:
                    event=json.loads(line)
                    if isinstance(event,dict):
                        if event.get('status')=='failed' or event.get('error'):output.append(str(event.get('error') or event.get('message')))
                        continue
                except ValueError:pass
                output.append(line.rstrip())
            return output
        except (FileNotFoundError,PermissionError):return []

class ActivityRows:
    def __init__(self):self.previous={};self.samples={}
    def update(self,snapshot):
        rows=[]
        active=snapshot.get('active')
        if active:
            rows.append({'key':'job:'+active['id'],'name':(active.get('kind') or 'Image').replace('-',' ').title()+' '+active['id'][:8],
                'stage':active.get('stage','preparing'),'phase':active.get('phase',''),'status':active.get('status','running'),
                'done':active.get('step',0),'total':active.get('total_steps',0) if active.get('stage') in ('generating','training','upscaling') else 0,
                'units':'steps','started':active.get('stage_started_at') or active.get('started_at')})
        transfer=snapshot.get('transfer') or {}
        downloads=list(snapshot.get('downloads') or [])
        if transfer.get('status')=='running':downloads.insert(0,{**transfer,'requested_id':'model:'+str(transfer.get('model_id',''))})
        for item in downloads:
            key='download:'+item.get('requested_id','model')
            if item.get('status')!='running':continue
            rows.append({'key':key,'name':item.get('title') or item.get('model_id') or 'Model download','stage':item.get('stage','preparing'),'status':item['status'],
                'done':item.get('completed_bytes',0),'total':item.get('total_bytes',0) if item.get('stage')=='downloading' else 0,'units':'bytes','started':None})
        events=[];now=time.time();current={}
        for row in rows:
            key=row['key'];old=self.previous.get(key)
            if old and (old['stage']!=row['stage'] or old.get('phase')!=row.get('phase')):
                events.append(f"{old['name']} | {STAGES.get(old['stage'],old['stage'].replace('_',' ').title())} finished")
            points=self.samples.setdefault((key,row['stage'],row.get('phase','')),[]);points.append((now,row['done']));points[:]=[point for point in points if now-point[0]<12]
            row['speed']=max(0,(points[-1][1]-points[0][1])/(points[-1][0]-points[0][0])) if len(points)>1 and points[-1][0]>points[0][0] else 0
            row['started']=row['started'] or (old['started'] if old and old['stage']==row['stage'] else now)
            current[key]=row
        for key,old in self.previous.items():
            if key not in current:
                ended=next((d for d in downloads if 'download:'+d.get('requested_id','')==key),None)
                terminal=(ended or {}).get('status')
                events.append(f"{old['name']} | {terminal.capitalize() if terminal else 'Finished; see result in Qwen'}")
        self.previous=current
        return rows,events


def render_rows(rows,snapshot,online=True):
    table=Table(expand=True,box=None,padding=(0,1),show_header=True)
    table.add_column('Task',ratio=3,overflow='ellipsis',no_wrap=True);table.add_column('Stage',ratio=2,no_wrap=True);table.add_column('Progress',ratio=2);table.add_column('Details',ratio=4,no_wrap=True)
    now=time.time()
    for row in rows:
        total=row['total'];done=row['done'];seconds=max(0,int(now-row['started']))
        progress=ProgressBar(total=total,completed=min(done,total),width=None) if total else Text('Working',style='cyan')
        if row['units']=='bytes':
            detail=f"{done/1e9:.2f} / {total/1e9:.2f} GB" if total else f"{seconds}s"
            if row['speed'] and total:detail+=f" | {row['speed']/1e6:.0f} MB/s"
        else:detail=(f"{done}/{total} steps | " if total else '')+f'{seconds}s'
        if total:detail=f'{min(100,int(done*100/total))}% | '+detail
        table.add_row(Text(row['name']),Text(STAGES.get(row['stage'],row['stage'].replace('_',' ').title())),progress,Text(detail))
    count=snapshot.get('pending_count',len(snapshot.get('pending') or []))
    footer='Reconnecting to Qwen' if not online else f"{count} waiting"+(' | Queue paused' if snapshot.get('paused') else '') if rows or count else 'Ready'
    return Group(table if rows else Text('Qwen activity',style='bold'),Text(footer,style='yellow' if not online else 'dim'))


def viewer_name(data_dir,channel,session=''):
    identity=hashlib.sha256(str(data_dir.resolve()).lower().encode()).hexdigest()[:24]
    return 'Local\\Qwen-viewer-'+identity+'-'+channel+('-'+session if session else '')


def activity_console():
    # A launcher can inherit TERM=dumb from an editor/automation process. A real
    # Windows Terminal tab still supports cursor movement. Without this, Rich
    # Live suppresses every update until the viewer exits, leaving a blank panel.
    if os.name=='nt' and os.environ.get('WT_SESSION') and sys.stdout.isatty():
        os.environ['TERM']='xterm-256color'
        return Console(force_terminal=True,force_interactive=True)
    return Console()


def watch_server(url,data_dir,*,once=False,stop_event=None):
    console=activity_console()
    tails=[LogTail(data_dir/'logs/server.out.log','Server'),LogTail(data_dir/'logs/server.err.log','Server')]
    console.print(Text('Qwen - Server',style='bold'))
    console.print(Text('Recent server messages, warnings, and errors. Successful HTTP requests stay hidden.'))
    console.print(Text(str(data_dir/'logs'),style='dim'))
    console.print(Text('Watching for new messages. Generation and download progress appear in Activity.\n',style='cyan'))
    worker=None;worker_id=None
    while not stop_event or not stop_event.is_set():
        try:
            with urlopen(url+'/api/activity',timeout=3) as response:current=json.load(response).get('active')
            if current and current['id']!=worker_id:
                worker_id=current['id'];worker=LogTail(data_dir/'jobs'/worker_id/'worker.log','Generation')
                console.print(Text('Job '+worker_id[:8],style='bold cyan'))
        except (OSError,ValueError):pass
        for tail in tails+([worker] if worker else []):
            for line in tail.read():
                console.print(Text(line,style='red' if ALERT.search(line) else None))
        if once:return
        if stop_event:stop_event.wait(.6)
        else:time.sleep(.6)


def watch(url,data_dir,*,once=False,stop_event=None,server_messages=True):
    console=activity_console()
    tracker=ActivityRows();snapshot={};last_models=0;has_activity=True;online=True;last_job=None;titles={};history_offset=0
    try:
        with urlopen(url+'/api/queue',timeout=3) as response:initial=json.load(response)
        recent=initial.get('recent') or []
        if recent:last_job=load_job(data_dir,recent[0]['id'])
        with urlopen(url+'/api/models',timeout=3) as response:catalog=json.load(response)
        titles={item['id']:item['title'] for item in catalog.get('models',[])}
    except (OSError,ValueError):pass
    with Live(console=console,auto_refresh=False,vertical_overflow='crop') as live:
        while not stop_event or not stop_event.is_set():
            try:
                if has_activity:
                    try:
                        with urlopen(url+'/api/activity',timeout=3) as response:snapshot=json.load(response)
                    except HTTPError as error:
                        if error.code!=404:raise
                        has_activity=False
                if not has_activity:
                    with urlopen(url+'/api/queue',timeout=3) as response:queue=json.load(response)
                    snapshot.update(queue)
                    if time.monotonic()-last_models>5:
                        with urlopen(url+'/api/models',timeout=4) as response:models=json.load(response)
                        snapshot.update(downloads=models.get('downloads',[]),transfer=models.get('transfer',{}));last_models=time.monotonic()
                online=True;rows,_=tracker.update(snapshot)
                current=snapshot.get('active')
                if current:
                    if not last_job or current['id']!=last_job['id']:history_offset=0
                    last_job=current
                elif last_job:last_job=load_job(data_dir,last_job['id']) or last_job
                if last_job:last_job['_model_title']=titles.get((last_job.get('request') or {}).get('model_id'))
            except (OSError,ValueError):online=False;rows=list(tracker.previous.values())
            if os.name=='nt' and sys.stdin.isatty():
                import msvcrt
                while msvcrt.kbhit():
                    key=msvcrt.getwch()
                    if key in ('\x00','\xe0'):
                        key=msvcrt.getwch()
                        if key in ('H','I'):history_offset+=1 if key=='H' else 5
                        elif key in ('P','Q'):history_offset=max(0,history_offset-(1 if key=='P' else 5))
                        elif key=='O':history_offset=0
            limit=max(1,console.size.height-17-3*len([r for r in rows if r['units']=='bytes']))
            live.update(render_activity(last_job,snapshot,rows,online,history_limit=limit,history_offset=history_offset),refresh=True)
            if once:return
            if stop_event:stop_event.wait(.6)
            else:time.sleep(.6)


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:47821');parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--once',action='store_true');parser.add_argument('--channel',choices=('activity','server'),default='activity')
    parser.add_argument('--owner-session',default='')
    parser.add_argument('--owner-pid',type=int)
    args=parser.parse_args(argv)
    instance=None
    owner=None
    try:
        if os.name=='nt' and not args.once:
            from windows_processes import Instance, open_process, close_handle, wait_handle, set_event
            if bool(args.owner_session) != bool(args.owner_pid):
                raise ValueError('The Qwen terminal needs both its server session and owner.')
            if args.owner_pid:
                owner=open_process(0x100000,False,args.owner_pid)
                if not owner or wait_handle(owner,0)!=258:
                    return
            instance=Instance(viewer_name(args.data_dir,args.channel,args.owner_session))
            if not instance.acquire():return
            if args.owner_session:
                set_event(instance.focus)
        print(f'Qwen | {args.url}\nClosing this tab stops Qwen and its active jobs.\n',flush=True)
        class Stop:
            def is_set(self):
                return bool(instance and instance.stopped() or owner and wait_handle(owner,0)!=258)
            def wait(self,seconds):time.sleep(seconds)
        if args.channel=='server':watch_server(args.url,args.data_dir,once=args.once,stop_event=Stop())
        else:watch(args.url,args.data_dir,once=args.once,stop_event=Stop(),server_messages=False)
    finally:
        if instance:instance.close()
        if owner:close_handle(owner)

if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('\nQwen terminal closed. Stopping Qwen.')
