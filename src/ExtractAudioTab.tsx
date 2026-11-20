import {useEffect,useState} from 'react';
import type {Api} from './ModelSettings';
import type {SavedJob} from './JobQueue';
import type {VideoInput} from './SendVideoTo';
import type {UploadedVideo} from './mediaUpload';
import {ClipFields,InstallNote,VideoPick,usePortraitJob,usePortraitStatus} from './PortraitInputs';
import {VideoProgress} from './VideoProgress';
import {ShowInFolder} from './ShowInFolder';
import './video-studio.css';
import './portrait.css';

export type AudioTarget='lipsync'|'animate';
export type SendAudio=(jobId:string,target:AudioTarget)=>Promise<void>;
type Draft={video:string;audio_format:'wav'|'mp3';start:number;duration:number|null;output_dir:string};
const KEY='qwen-extract-audio-draft';
const blank:Draft={video:'',audio_format:'wav',start:0,duration:null,output_dir:''};
function load():Draft{try{return {...blank,...JSON.parse(localStorage.getItem(KEY)||'{}')};}catch{return blank;}}
const length=(seconds?:number)=>seconds?`${Math.floor(seconds/60)}:${String(Math.round(seconds%60)).padStart(2,'0')}`:'';

export function SendAudioTo({jobId,onSend}:{jobId:string;onSend:SendAudio}){
  const [busy,setBusy]=useState(false),[error,setError]=useState('');
  async function send(target:AudioTarget){setBusy(true);setError('');try{await onSend(jobId,target);}catch(e){setError((e as Error).message);}finally{setBusy(false);}}
  return <div className="send-video-control"><label className="send-video-select"><select aria-label="Send audio to" value="" disabled={busy} onChange={e=>{if(e.target.value)void send(e.target.value as AudioTarget);}}>
    <option value="" disabled>{busy?'Preparing audio…':'Send to'}</option><option value="lipsync">Lip sync</option><option value="animate">Animate image</option></select></label>
    {error?<small role="alert" className="send-video-error">{error}</small>:null}</div>;
}

export function ExtractAudioTab({api,token,visible,videoInput,onSendAudio,onQueued,restoreRequest}:{
  api:Api;token:string;visible:boolean;videoInput:VideoInput|null;onSendAudio:SendAudio;onQueued:()=>void;restoreRequest:{job:SavedJob;key:number}|null;
}){
  const [draft,setDraft]=useState<Draft>(load),[video,setVideo]=useState<UploadedVideo|null>(null),[submitting,setSubmitting]=useState(false);
  const run=usePortraitJob(api,'extract-audio',onQueued);
  const status=usePortraitStatus(api,visible);
  useEffect(()=>{localStorage.setItem(KEY,JSON.stringify(draft));},[draft]);
  useEffect(()=>{if(!videoInput)return;setVideo(videoInput.video);setDraft(d=>({...d,video:videoInput.video.id}));run.setNotice('Video added.');},[videoInput]);
  useEffect(()=>{if(!draft.video||video?.id===draft.video)return;let alive=true;api(`/api/video-uploads/${draft.video}/metadata`).then((v:UploadedVideo)=>{if(alive)setVideo(v);}).catch(()=>{if(alive)setDraft(d=>({...d,video:''}));});return()=>{alive=false;};},[api,draft.video]);
  useEffect(()=>{if(!restoreRequest)return;const r=restoreRequest.job.request;setDraft(d=>({...d,video:r.source_video||'',audio_format:r.audio_format||'wav',start:r.start||0,duration:r.duration??null}));if(restoreRequest.job.status==='succeeded')run.setResult(restoreRequest.job);},[restoreRequest]);
  const ready=!!video&&video.has_audio;
  async function add(){if(!ready)return;setSubmitting(true);await run.submit('/api/extract-audio/generate',{source_video:video!.id,audio_format:draft.audio_format,start:draft.start,duration:draft.duration,output_dir:draft.output_dir});setSubmitting(false);}
  const result=run.result;
  return <main className="video-studio portrait-page" hidden={!visible}>
    <div className="vs-grid"><div className="vs-controls"><div className="vs-scroll">
      <VideoPick title="Video" hint="The audio track of this video is saved" video={video} token={token} onVideo={v=>{setVideo(v);setDraft(d=>({...d,video:v?.id||''}));}}/>
      <section className="panel vs-panel portrait-panel"><div className="section-heading"><h2>Export</h2></div>
        <div className="portrait-fields"><div className="field"><label htmlFor="extract-format">Audio format</label><select id="extract-format" value={draft.audio_format} onChange={e=>setDraft(d=>({...d,audio_format:e.target.value as Draft['audio_format']}))}><option value="wav">WAV</option><option value="mp3">MP3 (320 kbps)</option></select></div></div>
        <ClipFields prefix="extract" start={draft.start} duration={draft.duration} onStart={start=>setDraft(d=>({...d,start}))} onDuration={duration=>setDraft(d=>({...d,duration}))}/>
        <div className="field"><label htmlFor="extract-folder">Output folder</label><input id="extract-folder" placeholder="Videos folder of the app" value={draft.output_dir} onChange={e=>setDraft(d=>({...d,output_dir:e.target.value}))}/></div>
      </section>
    </div>
    <div className="vs-action">
      <InstallNote status={status}/>
      {run.error?<div className="message error" role="alert"><span>{run.error}</span><button className="icon-button" onClick={()=>run.setError('')} aria-label="Dismiss error">×</button></div>:null}
      {run.notice?<div className="notice" role="status">{run.notice}</div>:null}
      <div className="vs-action-row"><span className="vs-summary">{!video?'Add a video':!video.has_audio?'This video has no audio':draft.audio_format.toUpperCase()}<small>No AI model</small></span>
        {run.busy?<button className="text-button" onClick={run.stop}>{run.job?.status==='queued'?'Remove from queue':'Stop this job'}</button>:null}
        <button className="button primary vs-submit" disabled={!ready||submitting} onClick={add}>{submitting?<span className="train-spinner" aria-hidden="true"/>:null}Add to queue</button></div>
    </div></div>
    <section className="panel vs-result">
      <div className="section-heading"><h2>Audio</h2><small>{result?`${String(result.audio_format||'').toUpperCase()} · ${length(result.duration_seconds)}`:''}</small></div>
      {run.busy&&run.job?<VideoProgress job={run.job} clock={run.clock}/>:null}
      {result?.audio_url?<div className="portrait-audio-result"><audio key={result.id} controls preload="metadata" src={result.audio_url}/>
        <div className="result-actions"><SendAudioTo jobId={result.id} onSend={onSendAudio}/><a className="button secondary" href={`${result.audio_url}?download=true`}>Download {String(result.audio_format||'audio').toUpperCase()}</a><ShowInFolder jobId={result.id}/></div></div>
        :!run.busy?<div className="video-empty"><strong>The audio will appear here</strong></div>:null}
      {run.history.length?<details className="disclosure vs-history"><summary>Recent audio <span className="summary-hint">{run.history.length}</span></summary><ul className="portrait-audio-list">{run.history.slice(0,20).map(item=><li key={item.id}><button className={item.id===result?.id?'active':''} onClick={()=>run.setResult(item)}>{new Date((item.finished_at||item.created_at)*1000).toLocaleString(undefined,{day:'numeric',month:'short',hour:'2-digit',minute:'2-digit'})} · {String(item.audio_format||'').toUpperCase()} · {length(item.duration_seconds)}</button></li>)}</ul></details>:null}
    </section></div>
  </main>;
}
