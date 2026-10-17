import {useEffect,useRef,useState} from 'react';
import type {Api,ModelState} from './ModelSettings';
import type {SavedJob} from './JobQueue';
import {InputLibrary,type InputImage,type Person} from './InputLibrary';
import {LibraryPopover} from './LibraryPopover';
import {formatDuration} from './duration';

type Arch={id:string;label:string;gate?:string|null;families:string[];turbo:boolean};
type Setup={status:string;ready:boolean;stage?:string;message?:string;directory:string};
type TrainState={setup:Setup;archs:Arch[];jobs:SavedJob[]};
type Item=InputImage&{caption:string};
type Draft={name:string;arch:string;subject:'character'|'style';trigger_word:string;items:Item[];steps:number|null;rank:number;learning_rate:number;resolutions:number[];save_every:number;sample_prompts:string[]};

const DRAFT_KEY='qwen-train-draft-v1';
const empty:Draft={name:'',arch:'z-image-turbo',subject:'character',trigger_word:'',items:[],steps:null,rank:32,learning_rate:1e-4,resolutions:[512,768,1024],save_every:250,sample_prompts:[]};
function loadDraft():Draft{try{return {...empty,...JSON.parse(localStorage.getItem(DRAFT_KEY)||'{}')};}catch{return empty;}}
/** About 100 steps per image, from 1000 to 3000: the common range for a character or style LoRA. */
const autoSteps=(count:number)=>Math.min(3000,Math.max(1000,count*100));
const samplePrompts=(trigger:string,subject:Draft['subject'])=>subject==='character'?[`${trigger}, close-up portrait photo, soft window light`,`${trigger} walking in a city street at night, full body photo`]:[`a lighthouse on a cliff in the style of ${trigger}`,`a portrait of an old sailor in the style of ${trigger}`];

/** Train a LoRA with ai-toolkit: images and captions, the base model, then a queue job with progress and samples. */
export function TrainPage({api,visible,models,results,people,onPeople,onUse}:{api:Api;visible:boolean;models:ModelState|null;results:{id:string;request:{prompt:string}}[];people:Person[];onPeople:()=>void;onUse:(id:string)=>void}){
  const [state,setState]=useState<TrainState|null>(null);const [draft,setDraft]=useState<Draft>(loadDraft);
  const [error,setError]=useState('');const [busy,setBusy]=useState('');const [libraryOpen,setLibraryOpen]=useState(false);const [revision,setRevision]=useState(0);
  const panel=useRef<HTMLElement>(null);const upload=useRef<HTMLInputElement>(null);
  const change=<K extends keyof Draft>(key:K,value:Draft[K])=>setDraft(d=>({...d,[key]:value}));
  useEffect(()=>{try{localStorage.setItem(DRAFT_KEY,JSON.stringify(draft));}catch{}},[draft]);
  const running=!!state&&(state.setup.status==='running'||state.jobs.some(j=>['queued','running'].includes(j.status)));
  useEffect(()=>{if(!visible)return;let alive=true;let timer:number;
    async function poll(){try{const next=await api('/api/training');if(alive)setState(next);}catch(e){if(alive)setError((e as Error).message);}if(alive)timer=window.setTimeout(poll,running?2000:6000);}
    poll();return()=>{alive=false;clearTimeout(timer);};},[visible,api,running]);
  const arch=state?.archs.find(a=>a.id===draft.arch);
  const trigger=draft.trigger_word.trim();const steps=draft.steps??autoSteps(draft.items.length);
  const prompts=draft.sample_prompts.length?draft.sample_prompts:samplePrompts(trigger||'the subject',draft.subject);
  const missingCaptions=draft.items.filter(i=>!i.caption.trim()).length;
  const nameTaken=!!state?.jobs.some(j=>j.request?.name?.trim().toLowerCase()===draft.name.trim().toLowerCase()&&j.request?.arch===draft.arch&&j.status==='succeeded');
  const canStart=!!state?.setup.ready&&!!draft.name.trim()&&draft.items.length>0&&!busy;
  function setItems(next:InputImage[]){setDraft(d=>({...d,items:next.map(i=>({...i,caption:d.items.find(o=>o.id===i.id)?.caption||''}))}));}
  async function addFiles(files:File[]){if(!files.length)return;setBusy('upload');setError('');try{const data=new FormData();files.forEach(f=>data.append('files',f));const added:InputImage[]=await api('/api/uploads',{method:'POST',body:data});setDraft(d=>({...d,items:[...d.items,...added.filter(a=>!d.items.some(i=>i.id===a.id)).map(a=>({...a,caption:''}))]}));setRevision(r=>r+1);}catch(e){setError((e as Error).message);}finally{setBusy('');if(upload.current)upload.current.value='';}}
  async function writeCaptions(all:boolean){const targets=draft.items.filter(i=>all||!i.caption.trim());if(!targets.length)return;setBusy('captions');setError('');
    try{const answer=await api('/api/training/captions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({images:targets.map(i=>i.id),trigger_word:trigger,subject:draft.subject})});
      const found=new Map<string,{caption:string;error?:string}>(answer.captions.map((c:{id:string;caption:string;error?:string})=>[c.id,c]));
      setDraft(d=>({...d,items:d.items.map(i=>found.get(i.id)?.caption?{...i,caption:found.get(i.id)!.caption}:i)}));
      const failed=answer.captions.filter((c:{error?:string})=>c.error);if(failed.length)setError(`${failed.length} caption${failed.length>1?'s':''} failed: ${failed[0].error}`);}
    catch(e){setError((e as Error).message);}finally{setBusy('');}}
  async function start(){setBusy('start');setError('');try{await api('/api/training/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:draft.name.trim(),arch:draft.arch,subject:draft.subject,trigger_word:trigger,steps,rank:draft.rank,learning_rate:draft.learning_rate,resolutions:draft.resolutions,save_every:draft.save_every,sample_prompts:prompts,images:draft.items.map(i=>({id:i.id,caption:i.caption}))})});setState(await api('/api/training'));}catch(e){setError((e as Error).message);}finally{setBusy('');}}
  async function install(){setError('');try{await api('/api/training/setup',{method:'POST'});setState(await api('/api/training'));}catch(e){setError((e as Error).message);}}
  async function act(path:string){setError('');try{await api(path,{method:'POST'});setState(await api('/api/training'));}catch(e){setError((e as Error).message);}}
  const setup=state?.setup;
  return <main className="train-page" hidden={!visible}>
    <div className="civitai-top"><div><h1>Train a LoRA</h1><p>Teach a model a person, a character or a style from your images. The LoRA works with the models of the type you train on, also Civitai models of that type. Training uses ai-toolkit by Ostris and runs on your GPU as a queue job.</p></div></div>
    {error?<div className="message error" role="alert"><span>{error}</span><button className="text-button" onClick={()=>setError('')}>Close</button></div>:null}
    {setup&&!setup.ready?<section className="panel train-setup"><h2>Install the trainer</h2><p>One time: about 9 GB in {setup.directory} (ai-toolkit and its own PyTorch). Your models and images are not changed.</p>
      {setup.status==='running'?<p className="train-live"><span className="train-spinner" aria-hidden="true"/>{setup.message}</p>:<button className="button primary" onClick={install}>{setup.status==='failed'?'Try again':'Install the trainer'}</button>}
      {setup.status==='failed'?<p className="train-failure">{setup.message}</p>:null}</section>:null}
    <div className="train-grid">
      <section ref={panel} className="panel train-form" aria-label="New training">
        <h2>New training</h2>
        <div className="train-fields">
          <div className="field"><label htmlFor="train-name">Name</label><input id="train-name" value={draft.name} maxLength={60} placeholder="For example: Maya" onChange={e=>change('name',e.target.value)}/>{nameTaken?<small>A finished LoRA has this name. A new training with it continues from the saved steps.</small>:null}</div>
          <div className="field"><label htmlFor="train-arch">Base model</label><select id="train-arch" value={draft.arch} onChange={e=>change('arch',e.target.value)}>{(state?.archs||[]).map(a=><option key={a.id} value={a.id}>{a.label}</option>)}</select>
            <small>{arch?.turbo?'Uses the official training adapter, so the LoRA keeps Turbo fast. ':''}The LoRA works with {arch?.label} models. The first training downloads the base model.{arch?.gate?<> Needs access on <a href={arch.gate} target="_blank" rel="noreferrer">Hugging Face</a>.</>:null}</small></div>
          <div className="field"><label>What it learns</label><div className="segmented" role="radiogroup" aria-label="What it learns">{([['character','Person or character'],['style','Style']] as const).map(([id,label])=><button key={id} type="button" role="radio" aria-checked={draft.subject===id} aria-selected={draft.subject===id} onClick={()=>change('subject',id)}>{label}</button>)}</div></div>
          <div className="field"><label htmlFor="train-trigger">Trigger word</label><input id="train-trigger" value={draft.trigger_word} maxLength={40} placeholder="For example: mayaxz" onChange={e=>change('trigger_word',e.target.value.replace(/\s+/g,''))}/><small>A rare word. Put it in your prompt to call the LoRA.</small></div>
        </div>
        <div className="section-heading train-images-heading"><h3>Images <span className="count">{draft.items.length}</span></h3>
          <button type="button" className="text-button" disabled={!!busy} onClick={()=>setLibraryOpen(o=>!o)}>Add from library</button>
          <button type="button" className="text-button" disabled={!!busy} onClick={()=>upload.current?.click()}>{busy==='upload'?'Uploading…':'Upload'}</button>
          {draft.items.length?<button type="button" className="text-button" disabled={!!busy} onClick={()=>change('items',[])}>Clear</button>:null}
          <input ref={upload} type="file" accept="image/png,image/jpeg,image/webp" multiple hidden onChange={e=>addFiles(Array.from(e.target.files||[]))}/></div>
        <LibraryPopover open={libraryOpen} onClose={()=>setLibraryOpen(false)} anchor={panel}>
          <InputLibrary api={api} inputs={draft.items} onChange={setItems} results={results} people={people} onPeople={onPeople} revision={revision} disabled={!!busy} limit={300} replace={false} target={0} onUpload={()=>upload.current?.click()} onClose={()=>setLibraryOpen(false)}/>
        </LibraryPopover>
        <p className="train-hint">{draft.subject==='character'?'Use 15 to 40 sharp images of the same person: different clothes, places, light and angles, and some close-ups. Avoid other people in the images.':'Use 15 to 40 images in the same style with different subjects.'}</p>
        {draft.items.length?<>
          <div className="train-caption-tools"><button type="button" className="button secondary" disabled={!!busy} onClick={()=>writeCaptions(false)}>{busy==='captions'?'Writing captions…':missingCaptions?`Write ${missingCaptions} caption${missingCaptions>1?'s':''} with Antigravity`:'Captions are written'}</button>{missingCaptions<draft.items.length?<button type="button" className="text-button" disabled={!!busy} onClick={()=>writeCaptions(true)}>Write all again</button>:null}<small>{draft.subject==='character'?'Captions describe clothes, place and pose, not the face: the trigger word learns the face.':'Captions describe the content, not the style.'}</small></div>
          <div className="train-items">{draft.items.map((item,index)=><div className="train-item" key={item.id}><img src={item.url} alt={`Image ${index+1}`}/><div><span>{index+1}</span><textarea dir="auto" value={item.caption} placeholder={trigger?`${trigger}, …`:'Caption'} onChange={e=>setDraft(d=>({...d,items:d.items.map(i=>i.id===item.id?{...i,caption:e.target.value}:i)}))}/></div><button type="button" className="text-button" aria-label={`Remove image ${index+1}`} onClick={()=>setDraft(d=>({...d,items:d.items.filter(i=>i.id!==item.id)}))}>Remove</button></div>)}</div>
        </>:<div className="train-empty">No images yet. Add them from the library (uploads, results, People) or upload new ones.</div>}
        <details className="disclosure train-advanced"><summary>Training settings<span className="summary-hint">{steps} steps · rank {draft.rank} · {draft.resolutions.join(', ')} px</span></summary><div className="details-body train-fields">
          <div className="field"><label htmlFor="train-steps">Steps</label><input id="train-steps" type="number" min="50" max="20000" step="50" value={steps} onChange={e=>change('steps',Number(e.target.value)||null)}/>{draft.steps===null?<small>Auto: about 100 per image, 1000 to 3000.</small>:<button type="button" className="text-button" onClick={()=>change('steps',null)}>Auto ({autoSteps(draft.items.length)})</button>}</div>
          <div className="field"><label htmlFor="train-rank">Rank</label><select id="train-rank" value={draft.rank} onChange={e=>change('rank',Number(e.target.value))}>{[8,16,32,64,128].map(r=><option key={r} value={r}>{r}{r===32?' (default)':''}</option>)}</select><small>Higher learns more detail and makes a larger file.</small></div>
          <div className="field"><label htmlFor="train-lr">Learning rate</label><select id="train-lr" value={draft.learning_rate} onChange={e=>change('learning_rate',Number(e.target.value))}>{[5e-5,1e-4,2e-4,3e-4].map(v=><option key={v} value={v}>{v.toExponential(0).replace('e-','e-')}{v===1e-4?' (default)':''}</option>)}</select></div>
          <div className="field"><label htmlFor="train-save">Save and sample every</label><select id="train-save" value={draft.save_every} onChange={e=>change('save_every',Number(e.target.value))}>{[100,250,500].map(v=><option key={v} value={v}>{v} steps</option>)}</select></div>
          <div className="field span-2"><label>Image sizes</label><div className="train-sizes">{[512,768,1024,1280].map(size=><label key={size} className="checkbox-label"><input type="checkbox" checked={draft.resolutions.includes(size)} onChange={e=>change('resolutions',e.target.checked?[...draft.resolutions,size].sort((a,b)=>a-b):draft.resolutions.filter(r=>r!==size).length?draft.resolutions.filter(r=>r!==size):draft.resolutions)}/>{size}</label>)}</div><small>Images are trained at these sizes. 1024 and up needs more time and memory.</small></div>
          <div className="field span-2"><label htmlFor="train-prompts">Sample prompts</label><textarea id="train-prompts" dir="auto" rows={3} value={prompts.join('\n')} onChange={e=>change('sample_prompts',e.target.value.split('\n').filter(l=>l.trim()).slice(0,6))}/><small>One per line. The trainer makes these images at every save, so you can see the LoRA learn.</small></div>
        </div></details>
        <div className="train-actions"><button className="button primary" disabled={!canStart} onClick={start}>{busy==='start'?'Adding…':'Start training'}</button><small>{!state?.setup.ready?'Install the trainer first.':!draft.name.trim()?'Enter a name.':!draft.items.length?'Add images.':missingCaptions?`${missingCaptions} image${missingCaptions>1?'s have':' has'} no caption: the trainer uses only the trigger word for ${missingCaptions>1?'them':'it'}.`:`${steps} steps on ${arch?.label}. Image jobs wait until the training ends.`}</small></div>
      </section>
      <section className="panel train-jobs" aria-label="Trainings"><h2>Trainings</h2>
        {!state?.jobs.length?<p className="train-empty">No trainings yet.</p>:state.jobs.map(job=><TrainingCard key={job.id} job={job} archs={state.archs} onStop={()=>act(`/api/jobs/${job.id}/cancel`)} onRetry={()=>act(`/api/jobs/${job.id}/retry`)} onUse={onUse} installed={!!models?.models.some(m=>m.id===job.lora_id)}/>)}
      </section>
    </div>
  </main>;
}

/** ai-toolkit names samples [time]__[step, 9 digits]_[index].jpg. */
const sampleStep=(name:string)=>Number(name.split('__')[1]?.split('_')[0]||0);

function TrainingCard({job,archs,onStop,onRetry,onUse,installed}:{job:SavedJob;archs:Arch[];onStop:()=>void;onRetry:()=>void;onUse:(id:string)=>void;installed:boolean}){
  const r=job.request;const active=['queued','running'].includes(job.status);const samples:string[]=job.samples||[];
  const percent=job.total_steps?Math.round(100*(job.step||0)/job.total_steps):0;
  return <article className={`train-job ${job.status}`}>
    <div className="train-job-head"><strong>{r.name}</strong><span>{archs.find(a=>a.id===r.arch)?.label||r.arch} · {r.images?.length} images · {r.steps} steps</span></div>
    <p className="train-job-status">{job.status==='queued'?'Waiting in queue':job.status==='running'?job.message:job.status==='succeeded'?`Done in ${formatDuration(job.elapsed_seconds||0)}`:job.status==='cancelled'?job.message||'Stopped':job.message}</p>
    {job.status==='running'?<><progress value={job.stage==='training'?job.step:undefined} max={job.total_steps||1}/><small className="train-job-meta">{job.stage==='training'?`${percent}%${job.loss!=null?` · loss ${Number(job.loss).toFixed(4)}`:''}${job.eta_seconds?` · ${formatDuration(job.eta_seconds)} left`:''}`:job.trainer_message||'Starting…'}</small></>:null}
    {samples.length?<div className="train-samples">{samples.slice(-6).map(name=><a key={name} href={`/api/training/${job.id}/samples/${encodeURIComponent(name)}`} target="_blank" rel="noreferrer" title={`Step ${sampleStep(name)}`}><img src={`/api/training/${job.id}/samples/${encodeURIComponent(name)}`} alt="" loading="lazy"/><span>{sampleStep(name)}</span></a>)}</div>:null}
    <div className="train-job-actions">{active?<button className="text-button" onClick={onStop}>{job.status==='queued'?'Remove':'Stop'}</button>:null}
      {job.status==='succeeded'&&job.lora_id?installed?<button className="button primary" onClick={()=>onUse(job.lora_id)}>Use this LoRA</button>:<span>The LoRA was removed.</span>:null}
      {['failed','cancelled'].includes(job.status)?<button className="text-button" onClick={onRetry}>{job.status==='cancelled'?'Continue':'Retry'}</button>:null}</div>
  </article>;
}
