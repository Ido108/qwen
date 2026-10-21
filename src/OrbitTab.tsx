import {ModelDefaultsScope,bindSettings} from './ModelDefaults';
import {useImageDestination} from './UseImageAs';
import {SettingLabel} from './SettingLabel';
import {VideoImageLibrary} from './VideoImageLibrary';
import {useEffect,useState} from 'react';
import {CircleHelp,Images,RotateCcw,X} from 'lucide-react';
import type {Api,ModelState} from './ModelSettings';
import {type InputImage,type Person} from './InputLibrary';
import type {QueueState,SavedJob} from './JobQueue';
import {FilePicker} from './FilePicker';
import {VideoOutput} from './VideoOutput';
import type {SendVideo} from './SendVideoTo';
import './orbit.css';
import {useLastSeed} from './lastSeed';

type Draft={model_id:string;image:InputImage|null;strength:number;seed:number;randomize_seed:boolean};
const initial:Draft={model_id:'h3-int8',image:null,strength:1,seed:0,randomize_seed:true};
function load():Draft{try{return {...initial,...JSON.parse(localStorage.getItem('qwen-orbit-draft')||'{}')};}catch{return initial;}}
const isOrbit=(job:SavedJob)=>job.kind==='video'&&job.request.workflow==='orbit';

export function OrbitTab({api,visible,models,queue,onQueued,restoreRequest,people,onPeople,results,onSendVideo}:{
  api:Api;visible:boolean;models:ModelState|null;queue:QueueState;onQueued:()=>void;restoreRequest:{job:SavedJob;key:number}|null;
  people:Person[];onPeople:()=>void;results:{id:string;request:{prompt:string}}[];onSendVideo:SendVideo;
}){
  const [draft,setDraft]=useState(load),[job,setJob]=useState<SavedJob|null>(null),[result,setResult]=useState<SavedJob|null>(null);
  const [history,setHistory]=useState<SavedJob[]>([]),[pollId,setPollId]=useState<string|null>(null);
  useLastSeed(history,seed=>setDraft(p=>p.randomize_seed&&p.seed!==seed?{...p,seed}:p));
  const [error,setError]=useState(''),[notice,setNotice]=useState(''),[acting,setActing]=useState(false),[clock,setClock]=useState(Date.now());
  useImageDestination('orbit',image=>setDraft(p=>({...p,image})));
  const busy=!!pollId;
  useEffect(()=>{localStorage.setItem('qwen-orbit-draft',JSON.stringify(draft));},[draft]);
  useEffect(()=>{if(!visible)return;let alive=true;api('/api/history?kind=video').then(items=>{if(!alive)return;const own=items.filter(isOrbit);setHistory(own);setResult(p=>p||own[0]||null);}).catch(e=>alive&&setError(e.message));return()=>{alive=false;};},[api,visible]);
  useEffect(()=>{if(pollId)return;const next=[queue.active,...queue.pending].find((j):j is SavedJob=>!!j&&isOrbit(j));if(next){setJob(next);setPollId(next.id);}},[queue,pollId]);
  useEffect(()=>{if(!pollId)return;let alive=true,timer:number;async function poll(){try{const next:SavedJob=await api(`/api/jobs/${pollId}`);if(!alive)return;setJob(next);if(['succeeded','failed','cancelled'].includes(next.status)){setPollId(null);onQueued();if(next.status==='succeeded'){setResult(next);setHistory(p=>[next,...p.filter(j=>j.id!==next.id)]);setNotice('Orbit complete.');}else if(next.status==='failed')setError(next.message);else setNotice('Orbit cancelled.');return;}}catch{if(alive)setNotice('Reconnecting to Qwen…');}if(alive)timer=window.setTimeout(poll,1200);}poll();return()=>{alive=false;clearTimeout(timer);};},[api,pollId]);
  useEffect(()=>{if(!busy)return;const timer=setInterval(()=>setClock(Date.now()),1000);return()=>clearInterval(timer);},[busy]);
  function restore(saved:SavedJob){const id=saved.request.first_frame;setDraft({model_id:saved.request.model_id,image:id?{id,name:'Orbit image',url:id.startsWith('example:')?`/examples/${id.slice(8)}`:`/api/uploads/${id}`}:null,strength:saved.request.loras?.[0]?.strength??1,seed:saved.seed,randomize_seed:false});if(saved.status==='succeeded')setResult(saved);setNotice('Image, strength and seed loaded.');}
  useEffect(()=>{if(restoreRequest)restore(restoreRequest.job);},[restoreRequest]);
  async function upload(files:File[]){if(!files[0]||acting)return;setActing(true);setError('');try{const body=new FormData();body.append('files',files[0]);const images=await api('/api/uploads',{method:'POST',body});setDraft(p=>({...p,image:images[0]}));}catch(e){setError((e as Error).message);}finally{setActing(false);}}
  async function generate(){if(acting||!draft.image)return;setActing(true);setError('');setNotice('');try{const next=await api('/api/orbit/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...draft,image:draft.image.id})});if(!pollId){setJob(next);setPollId(next.id);}setNotice('Added to queue. You can prepare another orbit.');onQueued();}catch(e){setError((e as Error).message);}finally{setActing(false);}}
  const choices=(models?.models||[]).filter(m=>m.orbit_profile);
  const selected=choices.find(m=>m.id===draft.model_id);
  const profile=selected?.orbit_profile;
  const needed=[draft.model_id,...(selected?.dependencies||[]),'h3-360-orbit'];
  const missing=needed.map(id=>models?.models.find(m=>m.id===id)).filter(m=>m&&!m.installed);
  const downloadGB=missing.reduce((sum,m)=>sum+(m?.bytes||0),0)/1e9;
  const savedBindings=bindSettings(draft,patch=>setDraft(p=>({...p,...patch})),{"Orbit strength": "strength", "Seed": ["seed", "randomize_seed"]});
  return <ModelDefaultsScope model={draft.model_id} scope="orbit" bindings={savedBindings} draftKey="qwen-orbit-draft" restoreKey={restoreRequest?.key} restoreModel={restoreRequest?.job.request.model_id}><main className="video-page video-studio orbit-page" hidden={!visible}>
    <div className="vs-grid"><div className="vs-controls"><div className="vs-scroll">
      <section className="panel orbit-intro"><div className="orbit-heading"><span className="orbit-symbol" aria-hidden="true">360°</span><div><h1>360 Orbit</h1><p>One photo. A full circle around the scene.</p></div></div>
        <p>The camera circles the subject and returns to the starting view. People and objects should stay still. Small movements can occur.</p>
        <div className="orbit-profile"><span>768 × 768</span><span>3.04 seconds</span><span>{profile?.steps ?? "..."} steps</span><span>No audio</span></div>
      </section>
      <section className="panel orbit-input"><div className="section-heading"><h2>Photo to orbit</h2>{draft.image?<button className="icon-button" aria-label="Remove orbit photo" disabled={acting} onClick={()=>setDraft(p=>({...p,image:null}))}><X size={18}/></button>:null}</div>
        <div className="orbit-photo" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();upload(Array.from(e.dataTransfer.files));}}>
          {draft.image?<img src={draft.image.url} alt="Square crop used for the start and end of the orbit"/>:<FilePicker label="Upload orbit photo" className="orbit-upload" disabled={acting} onFiles={upload}><Images size={32}/><strong>Upload photo</strong><span>JPG, PNG, WebP or HEIC</span></FilePicker>}
          {draft.image?<span className="orbit-photo-label">Start and end frame</span>:null}
        </div><div className="orbit-input-actions">{draft.image?<FilePicker className="text-button" label="Replace orbit photo" disabled={acting} onFiles={upload}>Replace photo</FilePicker>:null}<VideoImageLibrary api={api} library={{people,onPeople,results}} value={draft.image} onChoose={image=>setDraft(p=>({...p,image}))} onUpload={upload} disabled={acting}/></div>
        <small>Non-square photos use the center square shown above. Keep the whole subject inside it.</small>
      </section>
      <details className="disclosure orbit-settings"><summary>Settings<span className="summary-hint">Strength {draft.strength.toFixed(2)}</span></summary><div className="details-body"><div className="field"><label htmlFor="orbit-model">Model</label><select id="orbit-model" disabled={acting} value={draft.model_id} onChange={e=>setDraft(p=>({...p,model_id:e.target.value as Draft['model_id']}))}>{!selected?<option value={draft.model_id}>Select a compatible model</option>:null}{choices.map(model=><option key={model.id} value={model.id}>{model.title} - {model.orbit_profile?.steps} steps{model.installed?" - Installed":""}</option>)}</select><small>{profile?.steps_source==='model'?"Uses the recommended steps for this model.":"Uses the recommended steps for the Orbit LoRA."} Compatible models you add appear here automatically.</small></div><SettingLabel htmlFor="orbit-strength" label="Orbit strength" value={1} onReset={()=>setDraft(p=>({...p,strength:models?.models.find(m=>m.id==="h3-360-orbit")?.default_strength??1}))}>Orbit strength <strong>{draft.strength.toFixed(2)}</strong></SettingLabel><input id="orbit-strength" type="range" min="0.5" max="1.5" step="0.05" disabled={acting} value={draft.strength} onChange={e=>setDraft(p=>({...p,strength:Number(e.target.value)}))}/><small>1.00 is recommended. A higher value can change the scene more.</small><label className="checkbox-label"><input type="checkbox" checked={draft.randomize_seed} onChange={e=>setDraft(p=>({...p,randomize_seed:e.target.checked}))}/>Randomize seed</label><div className="field"><SettingLabel htmlFor="orbit-seed" label="Seed" value={'Random'} onReset={()=>setDraft(p=>({...p,seed:0,randomize_seed:true}))}>Seed</SettingLabel><input id="orbit-seed" type="number" min="0" max="2147483647" disabled={draft.randomize_seed||acting} value={draft.seed} onChange={e=>setDraft(p=>({...p,seed:Number(e.target.value)}))}/></div><p className="muted">{selected?.title}<br/>360 Orbit LoRA · 73 frames · 24 fps<br/>Guidance 1 · No negative prompt<br/>{profile?.sampler} / {profile?.scheduler} · Video shift {profile?.video_shift}</p><a className="text-button" href="https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA" target="_blank" rel="noreferrer"><CircleHelp size={15}/>Model guide</a></div></details>
      {history.length?<details className="disclosure"><summary>History<span className="summary-hint">{history.length}</span></summary><div className="details-body history-grid">{history.map(item=><button className="history-item" key={item.id} onClick={()=>setResult(item)}><img src={item.poster_url} alt="Orbit result" loading="lazy"/></button>)}</div>{result?<button className="text-button" onClick={()=>restore(result)}><RotateCcw size={15}/>Use these settings</button>:null}</details>:null}
    </div><div className="vs-action">{error?<div className="message error" role="alert">{error}</div>:null}{notice?<div className="notice" role="status">{notice}</div>:null}{missing.length?<small className="orbit-download">The first run downloads the required models ({downloadGB.toFixed(1)} GB). Progress appears in the queue.</small>:null}<div className="vs-action-row">{busy?<button className="button secondary" onClick={()=>api(`/api/jobs/${pollId}/cancel`,{method:'POST'}).then(onQueued).catch(e=>setError(e.message))}>{job?.status==='queued'?'Remove from queue':'Stop this job'}</button>:null}<button className="button primary vs-submit" disabled={acting||!draft.image||!selected} onClick={generate}>{acting?<span className="train-spinner" aria-hidden="true"/>:null} {acting?'Adding…':queue.active||queue.pending.length||queue.paused?'Add orbit to queue':'Generate orbit'}</button></div></div></div>
    <section className="panel result-panel vs-result"><VideoOutput job={job} result={result} busy={busy} clock={clock} models={models} onSendVideo={onSendVideo}/></section></div>

  </main></ModelDefaultsScope>;
}
