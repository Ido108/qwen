import {useEffect,useRef,useState,type ReactNode} from 'react';
import type {Api} from './ModelSettings';
import type {SavedJob} from './JobQueue';
import {FilePicker} from './FilePicker';
import {uploadVideo,type UploadedVideo} from './mediaUpload';
import {useQueuedVideo} from './useQueuedVideo';

export const VIDEO_ACCEPT='video/mp4,video/quicktime,video/x-matroska,video/webm';
export const AUDIO_ACCEPT='.mp3,.wav,.m4a,.flac,.ogg,audio/*';
export type AudioUpload={id:string;name:string;url:string};

const seconds=(value:number|null|undefined)=>value?`${value.toFixed(1)}s`:'';

/** A video input: select, drop, or receive one from another tool (Send to). */
export function VideoPick({title,hint,video,onVideo,token,disabled,children}:{title:string;hint:string;video:UploadedVideo|null;onVideo:(video:UploadedVideo|null)=>void;token:string;disabled?:boolean;children?:ReactNode}){
  const [progress,setProgress]=useState<number|null>(null),[error,setError]=useState('');
  async function put(file?:File){if(!file)return;setProgress(0);setError('');try{onVideo(await uploadVideo(file,token,setProgress));}catch(e){setError((e as Error).message);}finally{setProgress(null);}}
  return <section className="panel vs-panel portrait-input">
    <div className="section-heading"><h2>{title}</h2><small>{video?`${video.width} × ${video.height} · ${seconds(video.duration_seconds)}${video.has_audio?' · with audio':' · no audio'}`:hint}</small></div>
    <div className="reference-video-drop" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();if(!disabled)put(e.dataTransfer.files[0]);}}>
      {video?<><video controls playsInline muted preload="metadata" poster={video.poster_url} src={video.url}/><div><strong>{video.name}</strong><small>{video.fps?.toFixed(1)} fps</small>
        <FilePicker className="text-button" label={`Replace ${title.toLowerCase()}`} accept={VIDEO_ACCEPT} disabled={disabled||progress!==null} onFiles={files=>put(files[0])}>Replace video</FilePicker>
        <button className="text-button" disabled={disabled} onClick={()=>onVideo(null)}>Remove</button></div></>
        :<FilePicker className="portrait-drop-button" label={`Select ${title.toLowerCase()}`} accept={VIDEO_ACCEPT} disabled={disabled||progress!==null} onFiles={files=>put(files[0])}><strong>Select a video</strong><small>MP4, MOV, MKV or WebM</small></FilePicker>}
    </div>
    {progress!==null?<progress value={progress} max={1} aria-label="Video upload"/>:null}
    {error?<div className="message error" role="alert">{error}</div>:null}
    {children}
  </section>;
}

export type AudioChoice={mode:string;file:AudioUpload|null;video:UploadedVideo|null};

/** The audio of a job: the video's own audio, an audio file, or the audio track of another video. */
export function AudioPick({api,token,value,onChange,ownLabel,ownMode,disabled}:{api:Api;token:string;value:AudioChoice;onChange:(value:AudioChoice)=>void;ownLabel:string;ownMode:string;disabled?:boolean}){
  const [busy,setBusy]=useState(false),[progress,setProgress]=useState<number|null>(null),[error,setError]=useState('');
  async function putFile(file?:File){if(!file)return;setBusy(true);setError('');try{const body=new FormData();body.append('file',file);const item:AudioUpload=await api('/api/audio-uploads',{method:'POST',body});onChange({...value,mode:'file',file:item});}catch(e){setError((e as Error).message);}finally{setBusy(false);}}
  async function putVideo(file?:File){if(!file)return;setProgress(0);setError('');try{const video=await uploadVideo(file,token,setProgress);if(!video.has_audio)throw new Error('This video has no audio track.');onChange({...value,mode:'video',video});}catch(e){setError((e as Error).message);}finally{setProgress(null);}}
  return <section className="panel vs-panel portrait-audio">
    <div className="section-heading"><h2>Audio</h2></div>
    <div className="portrait-segmented" role="radiogroup" aria-label="Audio source">
      {[[ownMode,ownLabel],['file','Audio file'],['video','Audio of another video']].map(([mode,label])=><label key={mode} className={value.mode===mode?'active':''}><input type="radio" name="portrait-audio" checked={value.mode===mode} disabled={disabled} onChange={()=>onChange({...value,mode})}/>{label}</label>)}
    </div>
    {value.mode==='file'?<div className="portrait-audio-row">{value.file?<><audio controls preload="metadata" src={value.file.url}/><span className="portrait-file-name" title={value.file.name}>{value.file.name}</span></>:null}
      <FilePicker className="button secondary" label="Select audio file" accept={AUDIO_ACCEPT} disabled={disabled||busy} onFiles={files=>putFile(files[0])}>{busy?'Uploading…':value.file?'Replace audio':'Select audio'}</FilePicker><small>MP3, WAV, M4A, FLAC or OGG, under 100 MB</small></div>:null}
    {value.mode==='video'?<div className="portrait-audio-row">{value.video?<><video className="portrait-audio-video" controls playsInline preload="metadata" poster={value.video.poster_url} src={value.video.url}/><span className="portrait-file-name" title={value.video.name}>{value.video.name}</span></>:null}
      <FilePicker className="button secondary" label="Select video for its audio" accept={VIDEO_ACCEPT} disabled={disabled||progress!==null} onFiles={files=>putVideo(files[0])}>{value.video?'Replace video':'Select video'}</FilePicker><small>Only the audio track of this video is used.</small></div>:null}
    {progress!==null?<progress value={progress} max={1} aria-label="Video upload"/>:null}
    {error?<div className="message error" role="alert">{error}</div>:null}
  </section>;
}

/** Clip start and length; empty length means to the end. */
export function ClipFields({start,duration,onStart,onDuration,audioStart,onAudioStart,prefix}:{start:number;duration:number|null;onStart:(v:number)=>void;onDuration:(v:number|null)=>void;audioStart?:number;onAudioStart?:(v:number)=>void;prefix:string}){
  return <div className="portrait-fields portrait-clip">
    <div className="field"><label htmlFor={`${prefix}-start`}>Start (seconds)</label><input id={`${prefix}-start`} type="number" min={0} max={7200} step={.1} value={start} onChange={e=>onStart(Math.max(0,e.target.valueAsNumber||0))}/></div>
    <div className="field"><label htmlFor={`${prefix}-duration`}>Length (seconds)</label><input id={`${prefix}-duration`} type="number" min={.1} max={7200} step={.1} placeholder="To the end" value={duration??''} onChange={e=>onDuration(Number.isFinite(e.target.valueAsNumber)&&e.target.valueAsNumber>0?e.target.valueAsNumber:null)}/></div>
    {onAudioStart?<div className="field"><label htmlFor={`${prefix}-audio-start`}>Audio start (seconds)</label><input id={`${prefix}-audio-start`} type="number" min={0} max={7200} step={.1} value={audioStart??0} onChange={e=>onAudioStart(Math.max(0,e.target.valueAsNumber||0))}/><small>For an audio file or the audio of another video.</small></div>:null}
  </div>;
}

/** The queue job of a tool: follows it, keeps its history and reattaches after a reload. */
export function usePortraitJob(api:Api,kind:string,onQueued:()=>void){
  const [job,setJob]=useState<SavedJob|null>(null),[result,setResult]=useState<SavedJob|null>(null);
  const [history,setHistory]=useState<SavedJob[]>([]),[pollId,setPollId]=useState<string|null>(null);
  const [error,setError]=useState(''),[notice,setNotice]=useState(''),[clock,setClock]=useState(Date.now());
  const queued=useRef(onQueued);queued.current=onQueued;
  useQueuedVideo(api,kind,pollId,current=>{setJob(current);setPollId(current.id);});
  useEffect(()=>{let alive=true;api(`/api/history?kind=${kind}`).then((items:SavedJob[])=>{if(alive){setHistory(items);setResult(r=>r||items[0]||null);}}).catch(()=>{});return()=>{alive=false;};},[api,kind]);
  useEffect(()=>{if(!pollId)return;let alive=true,timer:number;async function poll(){try{const current:SavedJob=await api(`/api/jobs/${pollId}`);if(!alive)return;setJob(current);if(['succeeded','failed','cancelled'].includes(current.status)){setPollId(null);queued.current();if(current.status==='succeeded'){setResult(current);setHistory(items=>[current,...items.filter(item=>item.id!==current.id)]);setError('');setNotice('');}else if(current.status==='failed')setError(current.message);else setNotice('Job stopped.');return;}}catch{}if(alive)timer=window.setTimeout(poll,1200);}poll();return()=>{alive=false;clearTimeout(timer);};},[api,pollId]);
  useEffect(()=>{if(!pollId)return;const timer=window.setInterval(()=>setClock(Date.now()),1000);return()=>clearInterval(timer);},[pollId]);
  async function submit(path:string,body:unknown){
    setError('');setNotice('');
    try{const created:SavedJob=await api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});if(!pollId){setJob(created);setPollId(created.id);}setNotice('Added to queue.');queued.current();return true;}
    catch(e){setError((e as Error).message);return false;}
  }
  const stop=()=>pollId&&api(`/api/jobs/${pollId}/cancel`,{method:'POST'}).then(()=>queued.current()).catch(e=>setError((e as Error).message));
  return {job,result,setResult,history,busy:!!pollId,clock,error,setError,notice,setNotice,submit,stop};
}

/** First use installs the portrait engine; this tells the user before they start. */
export function usePortraitStatus(api:Api,visible:boolean){
  const [status,setStatus]=useState<{installed:boolean;ready:boolean}|null>(null);
  useEffect(()=>{if(!visible)return;let alive=true;api('/api/portrait/status').then(s=>{if(alive)setStatus(s);}).catch(()=>{});return()=>{alive=false;};},[api,visible]);
  return status;
}

export function InstallNote({status}:{status:{installed:boolean;ready:boolean}|null}){
  if(!status||status.ready)return null;
  return <div className="notice">The first job installs the animation and lip sync tools (about 12 GB) and tests them on this GPU. This takes 10 to 20 minutes once. Progress appears in the queue.</div>;
}
