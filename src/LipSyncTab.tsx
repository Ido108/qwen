import {useEffect,useState} from 'react';
import type {Api,ModelState} from './ModelSettings';
import type {SavedJob} from './JobQueue';
import {VideoOutput} from './VideoOutput';
import type {SendVideo,VideoInput} from './SendVideoTo';
import type {UploadedVideo} from './mediaUpload';
import {portraitDefaults,portraitGroups,portraitOptions,syncTitle,type PortraitSettings} from './portraitControls';
import {PortraitFields} from './PortraitFields';
import {AudioPick,ClipFields,InstallNote,VideoPick,usePortraitJob,usePortraitStatus,type AudioChoice,type AudioUpload} from './PortraitInputs';
import './video-studio.css';
import './portrait.css';

type Draft={video:string;settings:PortraitSettings;audio:AudioChoice;start:number;duration:number|null;audio_start:number;output_dir:string};
const KEY='qwen-lipsync-draft';
const blank=():Draft=>({video:'',settings:portraitDefaults('sync',{sync_model:'musetalk'}),audio:{mode:'file',file:null,video:null},start:0,duration:null,audio_start:0,output_dir:''});
function load():Draft{try{const saved=JSON.parse(localStorage.getItem(KEY)||'{}');const base=blank();return {...base,...saved,settings:portraitDefaults('sync',{...base.settings,...saved.settings}),audio:{...base.audio,...saved.audio}};}catch{return blank();}}

export function LipSyncTab({api,token,visible,models,videoInput,audioInput,onSendVideo,onQueued,restoreRequest}:{
  api:Api;token:string;visible:boolean;models:ModelState|null;videoInput:VideoInput|null;audioInput:{audio:AudioUpload;key:number}|null;
  onSendVideo:SendVideo;onQueued:()=>void;restoreRequest:{job:SavedJob;key:number}|null;
}){
  const [draft,setDraft]=useState<Draft>(load),[video,setVideo]=useState<UploadedVideo|null>(null),[submitting,setSubmitting]=useState(false);
  const run=usePortraitJob(api,'lipsync',onQueued);
  const status=usePortraitStatus(api,visible);
  const groups=portraitGroups('sync',draft.settings);
  const set=(key:string,value:unknown)=>setDraft(d=>({...d,settings:portraitDefaults('sync',{...d.settings,[key]:value as never})}));
  useEffect(()=>{localStorage.setItem(KEY,JSON.stringify(draft));},[draft]);
  useEffect(()=>{if(!videoInput)return;const incoming=videoInput.video;
    if(videoInput.target==='lipsync-audio'){setDraft(d=>({...d,audio:{...d.audio,mode:'video',video:incoming}}));run.setNotice('The audio of this video is selected.');return;}
    setVideo(incoming);setDraft(d=>({...d,video:incoming.id}));run.setNotice('Video added. Select the audio for the new mouth movement.');},[videoInput]);
  useEffect(()=>{if(!audioInput)return;setDraft(d=>({...d,audio:{...d.audio,mode:'file',file:audioInput.audio}}));run.setNotice('Audio added.');},[audioInput]);
  useEffect(()=>{if(!draft.video||video?.id===draft.video)return;let alive=true;api(`/api/video-uploads/${draft.video}/metadata`).then((v:UploadedVideo)=>{if(alive)setVideo(v);}).catch(()=>{if(alive)setDraft(d=>({...d,video:''}));});return()=>{alive=false;};},[api,draft.video]);
  useEffect(()=>{if(!restoreRequest)return;const r=restoreRequest.job.request;setDraft(d=>({...d,video:r.source_video||'',settings:portraitDefaults('sync',{...blank().settings,...r.options}),start:r.options?.start??0,duration:r.options?.duration??null,audio_start:r.options?.audio_start??0,
    audio:{mode:r.audio_mode||'file',file:r.audio_file?{id:r.audio_file,name:'Audio file',url:`/api/audio-uploads/${r.audio_file}`}:null,video:null}}));
    if(r.audio_mode==='video'&&r.audio_video)api(`/api/video-uploads/${r.audio_video}/metadata`).then((v:UploadedVideo)=>setDraft(d=>({...d,audio:{...d.audio,video:v}}))).catch(()=>{});
    if(restoreRequest.job.status==='succeeded')run.setResult(restoreRequest.job);run.setNotice('Settings restored.');},[restoreRequest]);
  const audioMissing=(draft.audio.mode==='file'&&!draft.audio.file)||(draft.audio.mode==='video'&&!draft.audio.video)||(draft.audio.mode==='own'&&video!==null&&!video.has_audio);
  const ready=!!video&&!audioMissing;
  async function add(){
    if(!ready)return;setSubmitting(true);
    const options={...portraitOptions('sync',draft.settings),start:draft.start,...(draft.duration?{duration:draft.duration}:{}),...(draft.audio.mode!=='own'?{audio_start:draft.audio_start}:{})};
    await run.submit('/api/lipsync/generate',{source_video:video!.id,audio_mode:draft.audio.mode,audio_file:draft.audio.file?.id||'',audio_video:draft.audio.video?.id||'',options,output_dir:draft.output_dir});
    setSubmitting(false);
  }
  return <main className="video-studio portrait-page" hidden={!visible}>
    <div className="vs-grid"><div className="vs-controls"><div className="vs-scroll">
      <VideoPick title="Video" hint="A video with a visible face" video={video} token={token} onVideo={v=>{setVideo(v);setDraft(d=>({...d,video:v?.id||''}));}}/>
      <AudioPick api={api} token={token} value={draft.audio} onChange={audio=>setDraft(d=>({...d,audio}))} ownMode="own" ownLabel="Audio of this video"/>
      <PortraitFields groups={groups} settings={draft.settings} onChange={set} prefix="lipsync"/>
      <details className="disclosure"><summary>Part of the video and output folder</summary><div className="details-body">
        <ClipFields prefix="lipsync" start={draft.start} duration={draft.duration} onStart={start=>setDraft(d=>({...d,start}))} onDuration={duration=>setDraft(d=>({...d,duration}))} audioStart={draft.audio_start} onAudioStart={audio_start=>setDraft(d=>({...d,audio_start}))}/>
        <div className="field"><label htmlFor="lipsync-folder">Output folder</label><input id="lipsync-folder" placeholder="Videos folder of the app" value={draft.output_dir} onChange={e=>setDraft(d=>({...d,output_dir:e.target.value}))}/></div>
      </div></details>
    </div>
    <div className="vs-action">
      <InstallNote status={status}/>
      {run.error?<div className="message error" role="alert"><span>{run.error}</span><button className="icon-button" onClick={()=>run.setError('')} aria-label="Dismiss error">×</button></div>:null}
      {run.notice?<div className="notice" role="status">{run.notice}</div>:null}
      <div className="vs-action-row"><span className="vs-summary">{!video?'Add a video':draft.audio.mode==='own'&&!video.has_audio?'This video has no audio':audioMissing?'Select the audio':syncTitle(String(draft.settings.sync_model))}<small>Local GPU</small></span>
        {run.busy?<button className="text-button" onClick={run.stop}>{run.job?.status==='queued'?'Remove from queue':'Stop this job'}</button>:null}
        <button className="button primary vs-submit" disabled={!ready||submitting} onClick={add}>{submitting?<span className="train-spinner" aria-hidden="true"/>:null}Add to queue</button></div>
    </div></div>
    <section className="panel vs-result"><VideoOutput job={run.job} result={run.result} busy={run.busy} clock={run.clock} models={models} onSendVideo={onSendVideo} frames/>
      {run.history.length?<details className="disclosure vs-history"><summary>Recent lip sync videos <span className="summary-hint">{run.history.length}</span></summary><div className="vs-history-grid">{run.history.slice(0,12).map(item=><button key={item.id} onClick={()=>run.setResult(item)}><img src={item.poster_url} alt=""/><span>{syncTitle(item.request.options?.sync_model||'musetalk')}</span></button>)}</div></details>:null}
    </section></div>
  </main>;
}
