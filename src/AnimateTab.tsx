import {useEffect,useState} from 'react';
import type {Api,ModelState} from './ModelSettings';
import type {SavedJob} from './JobQueue';
import type {InputImage,Person} from './InputLibrary';
import {FilePicker} from './FilePicker';
import {VideoImageLibrary} from './VideoImageLibrary';
import {useImageDestination} from './UseImageAs';
import {VideoOutput} from './VideoOutput';
import type {SendVideo,VideoInput} from './SendVideoTo';
import type {UploadedVideo} from './mediaUpload';
import {ENGINES,portraitDefaults,portraitGroups,portraitOptions,type PortraitSettings} from './portraitControls';
import {PortraitFields} from './PortraitFields';
import {AudioPick,ClipFields,InstallNote,VideoPick,usePortraitJob,usePortraitStatus,type AudioChoice,type AudioUpload} from './PortraitInputs';
import './video-studio.css';
import './portrait.css';

type Draft={image:InputImage|null;video:string;settings:PortraitSettings;audio:AudioChoice;start:number;duration:number|null;audio_start:number;output_dir:string};
const KEY='qwen-animate-draft';
const blank=():Draft=>({image:null,video:'',settings:portraitDefaults('animate',{animation_engine:'imtalker'}),audio:{mode:'driving',file:null,video:null},start:0,duration:null,audio_start:0,output_dir:''});
function load():Draft{try{const saved=JSON.parse(localStorage.getItem(KEY)||'{}');const base=blank();return {...base,...saved,settings:portraitDefaults('animate',{...base.settings,...saved.settings}),audio:{...base.audio,...saved.audio}};}catch{return blank();}}

export function AnimateTab({api,token,visible,models,videoInput,audioInput,onSendVideo,onQueued,restoreRequest,library}:{
  api:Api;token:string;visible:boolean;models:ModelState|null;videoInput:VideoInput|null;audioInput:{audio:AudioUpload;key:number}|null;onSendVideo:SendVideo;onQueued:()=>void;
  restoreRequest:{job:SavedJob;key:number}|null;library:{people:Person[];onPeople:()=>void;results:{id:string;request:{prompt:string}}[]};
}){
  const [draft,setDraft]=useState<Draft>(load),[video,setVideo]=useState<UploadedVideo|null>(null),[uploading,setUploading]=useState(false),[submitting,setSubmitting]=useState(false);
  const run=usePortraitJob(api,'animate',onQueued);
  const status=usePortraitStatus(api,visible);
  const engine=String(draft.settings.animation_engine);
  const info=ENGINES.find(item=>item.id===engine);
  const groups=portraitGroups('animate',draft.settings);
  const set=(key:string,value:unknown)=>setDraft(d=>({...d,settings:portraitDefaults('animate',{...d.settings,[key]:value as never})}));
  useImageDestination('animate',image=>{setDraft(d=>({...d,image}));run.setNotice('Picture added.');});
  useEffect(()=>{localStorage.setItem(KEY,JSON.stringify(draft));},[draft]);
  useEffect(()=>{if(!videoInput)return;const incoming=videoInput.video;
    if(videoInput.target==='animate-audio'){setDraft(d=>({...d,audio:{...d.audio,mode:'video',video:incoming}}));run.setNotice('The audio of this video is selected.');return;}
    setVideo(incoming);setDraft(d=>({...d,video:incoming.id}));run.setNotice('Video added as the driving video.');},[videoInput]);
  useEffect(()=>{if(!audioInput)return;setDraft(d=>({...d,audio:{...d.audio,mode:'file',file:audioInput.audio}}));run.setNotice('Audio added.');},[audioInput]);
  useEffect(()=>{if(!draft.video||video?.id===draft.video)return;let alive=true;api(`/api/video-uploads/${draft.video}/metadata`).then((v:UploadedVideo)=>{if(alive)setVideo(v);}).catch(()=>{if(alive)setDraft(d=>({...d,video:''}));});return()=>{alive=false;};},[api,draft.video]);
  useEffect(()=>{if(!restoreRequest)return;const r=restoreRequest.job.request;setDraft(d=>({...d,image:r.source_image?{id:r.source_image,name:'Picture',url:`/api/uploads/${r.source_image}`}:null,video:r.driving_video||'',
    settings:portraitDefaults('animate',{...blank().settings,...r.options}),start:r.options?.start??0,duration:r.options?.duration??null,audio_start:r.options?.audio_start??0,audio:{mode:r.audio_mode||'driving',file:r.audio_file?{id:r.audio_file,name:'Audio file',url:`/api/audio-uploads/${r.audio_file}`}:null,video:null}}));
    if(r.audio_mode==='video'&&r.audio_video)api(`/api/video-uploads/${r.audio_video}/metadata`).then((v:UploadedVideo)=>setDraft(d=>({...d,audio:{...d.audio,video:v}}))).catch(()=>{});
    if(restoreRequest.job.status==='succeeded')run.setResult(restoreRequest.job);run.setNotice('Settings restored.');},[restoreRequest]);
  async function upload(files:File[]){if(!files[0])return;setUploading(true);run.setError('');try{const body=new FormData();body.append('files',files[0]);const images=await api('/api/uploads',{method:'POST',body});setDraft(d=>({...d,image:images[0]}));}catch(e){run.setError((e as Error).message);}finally{setUploading(false);}}
  const audioMissing=(draft.audio.mode==='file'&&!draft.audio.file)||(draft.audio.mode==='video'&&!draft.audio.video);
  const ready=!!draft.image&&!!video&&!audioMissing&&!uploading;
  async function add(preview:boolean){
    if(!ready)return;setSubmitting(true);
    const options={...portraitOptions('animate',draft.settings),start:draft.start,...(draft.duration?{duration:draft.duration}:{}),...(draft.audio.mode!=='driving'?{audio_start:draft.audio_start}:{})};
    await run.submit('/api/animate/generate',{source_image:draft.image!.id,driving_video:video!.id,audio_mode:draft.audio.mode,audio_file:draft.audio.file?.id||'',audio_video:draft.audio.video?.id||'',preview,options,output_dir:draft.output_dir});
    setSubmitting(false);
  }
  return <main className="video-studio portrait-page" hidden={!visible}>
    <div className="vs-grid"><div className="vs-controls"><div className="vs-scroll">
      <section className="panel vs-panel portrait-input"><div className="section-heading"><h2>Picture</h2><small>The face to animate</small></div>
        <div className="portrait-photo" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();upload(Array.from(e.dataTransfer.files));}}>
          {draft.image?<img src={draft.image.url} alt="Picture to animate"/>:<FilePicker className="portrait-drop-button" label="Upload picture" disabled={uploading} onFiles={upload}><strong>Upload a picture</strong><small>JPG, PNG, WebP or HEIC, with one clear face</small></FilePicker>}
        </div>
        <div className="portrait-input-actions">{draft.image?<><FilePicker className="text-button" label="Replace picture" disabled={uploading} onFiles={upload}>Replace picture</FilePicker><button className="text-button" onClick={()=>setDraft(d=>({...d,image:null}))}>Remove</button></>:null}
          <VideoImageLibrary api={api} library={library} value={draft.image} onChoose={image=>setDraft(d=>({...d,image}))} onUpload={upload} disabled={uploading}/></div>
      </section>
      <VideoPick title="Driving video" hint="The motion and expressions come from this video" video={video} token={token} onVideo={v=>{setVideo(v);setDraft(d=>({...d,video:v?.id||''}));}}/>
      <section className="panel vs-panel portrait-panel"><div className="section-heading"><h2>Animation engine</h2></div>
        <div className="field"><label htmlFor="animate-engine">Engine</label><select id="animate-engine" value={engine} onChange={e=>set('animation_engine',e.target.value)}>{ENGINES.map(item=><option key={item.id} value={item.id}>{item.title}</option>)}</select>
          {info?<small>{info.detail} <a href={info.url} target="_blank" rel="noreferrer">Model page</a></small>:null}</div>
      </section>
      <PortraitFields groups={groups} settings={draft.settings} onChange={set} prefix="animate"/>
      <AudioPick api={api} token={token} value={draft.audio} onChange={audio=>setDraft(d=>({...d,audio}))} ownMode="driving" ownLabel="Audio of the driving video"/>
      <details className="disclosure"><summary>Part of the video and output folder</summary><div className="details-body">
        <ClipFields prefix="animate" start={draft.start} duration={draft.duration} onStart={start=>setDraft(d=>({...d,start}))} onDuration={duration=>setDraft(d=>({...d,duration}))} audioStart={draft.audio_start} onAudioStart={audio_start=>setDraft(d=>({...d,audio_start}))}/>
        <div className="field"><label htmlFor="animate-folder">Output folder</label><input id="animate-folder" placeholder="Videos folder of the app" value={draft.output_dir} onChange={e=>setDraft(d=>({...d,output_dir:e.target.value}))}/></div>
      </div></details>
    </div>
    <div className="vs-action">
      <InstallNote status={status}/>
      {run.error?<div className="message error" role="alert"><span>{run.error}</span><button className="icon-button" onClick={()=>run.setError('')} aria-label="Dismiss error">×</button></div>:null}
      {run.notice?<div className="notice" role="status">{run.notice}</div>:null}
      <div className="vs-action-row"><span className="vs-summary">{!draft.image?'Add a picture':!video?'Add a driving video':audioMissing?'Select the audio':'Ready'}<small>Local GPU</small></span>
        {run.busy?<button className="text-button" onClick={run.stop}>{run.job?.status==='queued'?'Remove from queue':'Stop this job'}</button>:null}
        <button className="button secondary" disabled={!ready||submitting} onClick={()=>add(true)} title="The first 3 seconds">Preview 3 s</button>
        <button className="button primary vs-submit" disabled={!ready||submitting} onClick={()=>add(false)}>{submitting?<span className="train-spinner" aria-hidden="true"/>:null}Add to queue</button></div>
    </div></div>
    <section className="panel vs-result"><VideoOutput job={run.job} result={run.result} busy={run.busy} clock={run.clock} models={models} onSendVideo={onSendVideo} frames/>
      {run.history.length?<details className="disclosure vs-history"><summary>Recent animations <span className="summary-hint">{run.history.length}</span></summary><div className="vs-history-grid">{run.history.slice(0,12).map(item=><button key={item.id} onClick={()=>run.setResult(item)}><img src={item.poster_url} alt=""/><span>{ENGINES.find(e=>e.id===item.request.options?.animation_engine)?.title.split(':')[0]||''}</span></button>)}</div></details>:null}
    </section></div>
  </main>;
}
