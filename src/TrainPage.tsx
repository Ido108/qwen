import {ModelDefaultsScope,bindSettings} from './ModelDefaults';
import {useImageDestination} from './UseImageAs';
import {SettingLabel,ResetButton} from './SettingLabel';
import {FilePicker} from './FilePicker';
import {useEffect,useRef,useState} from 'react';
import type {Api,ModelState} from './ModelSettings';
import type {SavedJob} from './JobQueue';
import {InputLibrary,type InputImage,type Person} from './InputLibrary';
import {LibraryPopover} from './LibraryPopover';
import {formatDuration} from './duration';
import {inputImage} from './Gallery';

type Arch={id:string;label:string;gate?:string|null;families:string[];turbo:boolean};
type Setup={status:string;ready:boolean;stage?:string;message?:string;directory:string};
type TrainState={setup:Setup;archs:Arch[];jobs:SavedJob[]};
type Item=InputImage&{caption:string;dataset_key?:string;relative_path?:string;caption_source?:string;caption_error?:string};
type Advice={steps:number;rank:number;learning_rate:number;resolutions:number[];save_every:number;notes:string[];warnings?:string[];version:string};
type Draft={auto_caption:boolean;settings_mode:'recommended'|'custom';caption_job?:string;name:string;arch:string;subject:'character'|'style';trigger_word:string;items:Item[];steps:number|null;rank:number;learning_rate:number;resolutions:number[];save_every:number;sample_prompts:string[]};

const DRAFT_KEY='qwen-train-draft-v1';
const empty:Draft={auto_caption:true,settings_mode:'recommended',name:'',arch:'qwen-image-2.1',subject:'character',trigger_word:'',items:[],steps:null,rank:32,learning_rate:1e-4,resolutions:[512,768,1024],save_every:250,sample_prompts:[]};
function loadDraft():Draft{try{const saved=JSON.parse(localStorage.getItem(DRAFT_KEY)||'{}');return {...empty,...saved,settings_mode:saved.settings_mode||(Object.keys(saved).length?'custom':'recommended')};}catch{return empty;}}
/** About 100 steps per image, from 1000 to 3000: the common range for a character or style LoRA. */
const autoSteps=(count:number)=>Math.min(3000,Math.max(1000,count*100));
const samplePrompts=(trigger:string,subject:Draft['subject'])=>subject==='character'?[`${trigger}, close-up portrait photo, soft window light`,`${trigger} walking in a city street at night, full body photo`]:[`a lighthouse on a cliff in the style of ${trigger}`,`a portrait of an old sailor in the style of ${trigger}`];

/** Train a LoRA with ai-toolkit: images and captions, the base model, then a queue job with progress and samples. */
export function TrainPage({api,visible,models,results,people,onPeople,onUse}:{api:Api;visible:boolean;models:ModelState|null;results:{id:string;request:{prompt:string}}[];people:Person[];onPeople:()=>void;onUse:(id:string)=>void}){
  const [state,setState]=useState<TrainState|null>(null);const [draft,setDraft]=useState<Draft>(loadDraft);
  const [advice,setAdvice]=useState<Advice|null>(null);const [captionState,setCaptionState]=useState<SavedJob|null>(null);const [uploadProgress,setUploadProgress]=useState(0);const [importNote,setImportNote]=useState('');
  const [error,setError]=useState('');const [busy,setBusy]=useState('');const [libraryOpen,setLibraryOpen]=useState(false);const [revision,setRevision]=useState(0);
  const panel=useRef<HTMLElement>(null);
  const change=<K extends keyof Draft>(key:K,value:Draft[K])=>setDraft(d=>({...d,[key]:value}));
  useEffect(()=>{try{localStorage.setItem(DRAFT_KEY,JSON.stringify(draft));}catch{}},[draft]);
  const running=!!state&&(state.setup.status==='running'||state.jobs.some(j=>['queued','running'].includes(j.status)));
  useEffect(()=>{if(!visible)return;let alive=true;let timer:number;
    async function poll(){try{const next=await api('/api/training');if(alive)setState(next);}catch(e){if(alive)setError((e as Error).message);}if(alive)timer=window.setTimeout(poll,running?2000:6000);}
    poll();return()=>{alive=false;clearTimeout(timer);};},[visible,api,running]);
  const arch=state?.archs.find(a=>a.id===draft.arch);
  const trigger=draft.trigger_word.trim();const recommended=draft.settings_mode==='recommended';const effective=recommended&&advice?advice:draft;const steps=effective.steps??autoSteps(draft.items.length);
  const datasetIds=draft.items.map(i=>i.id).join('|');
  const resetPending=draft.items.length>0&&!advice;
  useEffect(()=>{if(!visible||!draft.items.length){setAdvice(null);return;}let alive=true;setAdvice(null);const timer=window.setTimeout(()=>{api('/api/training/recommendations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({arch:draft.arch,subject:draft.subject,images:datasetIds.split('|').map(id=>({id}))})}).then(value=>{if(alive)setAdvice(value);}).catch(e=>{if(alive)setError(e.message);});},200);return()=>{alive=false;clearTimeout(timer);};},[api,visible,draft.arch,draft.subject,datasetIds]);
  useEffect(()=>{if(!draft.caption_job)return;let alive=true;let timer:number;const id=draft.caption_job;async function poll(){try{const result=await api(`/api/training/captions/${id}`);if(!alive)return;setCaptionState(result.job);setDraft(d=>({...d,items:d.items.map(item=>{const answer=result.captions[item.id];return !item.caption.trim()&&answer?{...item,caption:answer.caption||'',caption_source:answer.caption?'automatic':item.caption_source,caption_error:answer.error}:item;}),...(['succeeded','failed','cancelled'].includes(result.job.status)?{caption_job:undefined}:{})}));if(['succeeded','failed','cancelled'].includes(result.job.status)){if(result.job.status!=='succeeded')setError(result.job.message);return;}}catch(e){if(alive)setError((e as Error).message);}if(alive)timer=window.setTimeout(poll,1500);}poll();return()=>{alive=false;clearTimeout(timer);};},[api,draft.caption_job]);
  function custom<K extends keyof Draft>(key:K,value:Draft[K]){setDraft(d=>({...d,...(advice&&d.settings_mode==='recommended'?advice:{}),settings_mode:'custom',[key]:value}));}
  const prompts=draft.sample_prompts.length?draft.sample_prompts:samplePrompts(trigger||'the subject',draft.subject);
  const missingCaptions=draft.items.filter(i=>!i.caption.trim()).length;
  const nameTaken=!!state?.jobs.some(j=>j.request?.name?.trim().toLowerCase()===draft.name.trim().toLowerCase()&&j.request?.arch===draft.arch&&j.status==='succeeded');
  const canStart=!!state?.setup.ready&&!!draft.name.trim()&&draft.items.length>0&&!busy&&!draft.caption_job&&(!recommended||!!advice)&&steps>=50&&steps<=20000;
  useImageDestination('training',image=>{if(draft.caption_job||busy)throw new Error('Wait for the dataset update to finish.');if(draft.items.some(i=>i.id===image.id))return;if(draft.items.length>=300)throw new Error('The dataset has 300 images. Remove one before adding another.');setDraft(d=>({...d,items:[...d.items,{...image,caption:''}]}));});
  function setItems(next:InputImage[]){setDraft(d=>({...d,items:next.map(i=>({...i,caption:typeof (i as Item).caption==='string'?(i as Item).caption:d.items.find(o=>o.id===i.id)?.caption||''}))}));}
  async function addFiles(files:File[]){
    if(!files.length)return;setBusy('upload');setError('');setImportNote('');setUploadProgress(0);let uploadId='';
    try{
      const upload=await api('/api/training/imports',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({files:files.map(f=>({name:f.webkitRelativePath||f.name,size:f.size}))})});uploadId=upload.id;
      const total=files.reduce((n,f)=>n+f.size,0);let done=0;
      for(let index=0;index<files.length;index++){const file=files[index];let offset=0;
        do{const part=file.slice(offset,offset+upload.chunk_bytes);const answer=await api(`/api/training/imports/${uploadId}/files/${index}?offset=${offset}`,{method:'PUT',headers:{'Content-Type':'application/octet-stream'},body:part});done+=answer.offset-offset;offset=answer.offset;setUploadProgress(total?Math.round(100*done/total):100);}while(offset<file.size);
      }
      const result=await api(`/api/training/imports/${uploadId}/finish`,{method:'POST'});
      const additions:Item[]=result.items.filter((i:Item)=>!draft.items.some(old=>old.id===i.id&&old.caption===i.caption));
      if(draft.items.length+additions.length>300)throw new Error('This import exceeds 300 images. Remove images from the current dataset or use a smaller import.');
      setDraft(d=>({...d,items:[...d.items,...additions]}));setRevision(r=>r+1);
      setImportNote(`${additions.length} images added. ${result.captions} captions read.${result.duplicates?` ${result.duplicates} exact duplicates skipped.`:''}${result.unpaired_captions?` ${result.unpaired_captions} caption files have no matching image.`:''}${result.ignored?` ${result.ignored} other files skipped.`:''}`);
    }catch(e){setError((e as Error).message);}finally{if(uploadId)await api(`/api/training/imports/${uploadId}`,{method:'DELETE'}).catch(()=>{});setBusy('');}
  }
  async function writeCaptions(){const targets=draft.items.filter(i=>!i.caption.trim());if(!targets.length)return;setBusy('captions');setError('');
    try{const job=await api('/api/training/captions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({images:targets.map(i=>i.id),name:draft.name,trigger_word:trigger,subject:draft.subject,arch:draft.arch})});change('caption_job',job.id);setCaptionState(job);}
    catch(e){setError((e as Error).message);}finally{setBusy('');}
  }
  async function start(){setBusy('start');setError('');try{await api('/api/training/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:draft.name.trim(),arch:draft.arch,subject:draft.subject,trigger_word:trigger,steps,rank:effective.rank,learning_rate:effective.learning_rate,resolutions:effective.resolutions,save_every:effective.save_every,auto_caption:draft.auto_caption,settings_mode:draft.settings_mode,sample_prompts:prompts,images:draft.items.map(i=>({id:i.id,caption:i.caption}))})});setState(await api('/api/training'));}catch(e){setError((e as Error).message);}finally{setBusy('');}}
  async function install(){setError('');try{await api('/api/training/setup',{method:'POST'});setState(await api('/api/training'));}catch(e){setError((e as Error).message);}}
  async function act(path:string){setError('');try{await api(path,{method:'POST'});setState(await api('/api/training'));}catch(e){setError((e as Error).message);}}
  async function useDataset(job:SavedJob){setError('');try{const result=await api(`/api/training/captions/${job.id}`);const r=job.request;setDraft({...empty,...r,settings_mode:'custom',items:r.images.map((item:{id:string;caption?:string},index:number)=>({...inputImage(item.id),caption:item.caption?.trim()?item.caption:result.captions[item.id]?.caption||'',dataset_key:`${item.id}-${index}`,caption_error:result.captions[item.id]?.error}))});panel.current?.scrollIntoView({behavior:'smooth',block:'start'});}catch(e){setError((e as Error).message);}}
  const setup=state?.setup;
  const savedBindings=bindSettings({...draft,...effective,steps},patch=>setDraft(d=>({...d,...patch,...(Object.keys(patch).some(k=>['steps','rank','learning_rate','resolutions','save_every'].includes(k))?{settings_mode:'custom' as const}:{})})),{"Write missing captions before training": "auto_caption", "Settings": "settings_mode", "Steps": "steps", "Rank": "rank", "Learning rate": "learning_rate", "Save and sample every": "save_every", "Image sizes": "resolutions"});
  return <ModelDefaultsScope model={draft.arch} scope="training" bindings={savedBindings} draftKey="qwen-train-draft-v1"><main className="train-page" hidden={!visible}>
    <div className="civitai-top"><div><h1>Train a LoRA</h1><p>Teach a model a person, a character or a style from your images. The LoRA works with the models of the type you train on, also Civitai models of that type. Training runs on this computer. Other queued jobs wait until it finishes.</p></div></div>
    {error?<div className="message error" role="alert"><span>{error}</span><button className="text-button" onClick={()=>setError('')}>Close</button></div>:null}
    {setup&&!setup.ready?<section className="panel train-setup"><h2>Install the trainer</h2><p>Install once to train LoRAs on this computer. The training tools need about 9 GB of disk space.</p>
      {setup.status==='running'?<p className="train-live"><span className="train-spinner" aria-hidden="true"/>{setup.message}</p>:<button className="button primary" onClick={install}>{setup.status==='failed'?'Try again':'Install the trainer'}</button>}
      {setup.status==='failed'?<p className="train-failure">{setup.message}</p>:null}</section>:null}
    <div className="train-grid">
      <section ref={panel} className="panel train-form" aria-label="New training">
        <h2>New training</h2>
        <div className="train-fields">
          <div className="field"><label htmlFor="train-name">Name</label><input id="train-name" value={draft.name} maxLength={60} placeholder="For example: Maya" onChange={e=>change('name',e.target.value)} onBlur={()=>{if(!draft.trigger_word.trim()&&draft.name.trim())change('trigger_word',draft.name.replace(/[^\p{L}\p{N}]/gu,'').toLowerCase().slice(0,30)+'x');}}/>{nameTaken?<small>A finished LoRA has this name. A new training with it continues from the saved steps.</small>:null}</div>
          <div className="field"><label htmlFor="train-arch">Base model</label><select id="train-arch" value={draft.arch} onChange={e=>change('arch',e.target.value)}>{(state?.archs||[]).map(a=><option key={a.id} value={a.id}>{a.label}</option>)}</select>
            <small>{arch?.turbo?"Uses the model's training adapter to help retain Turbo behavior. ":''}The LoRA works with {draft.arch==='flux3-klein-base'?'FLUX.3 Klein Base and Distilled':draft.arch==='moss-image'?'MOSS Image SFT, Base and Base finetunes such as Serenity (not Turbo)':draft.arch.startsWith('krea-2')?'Krea 2 Raw and Turbo':arch?.label} models.{draft.arch==='krea-2-turbo'?' Krea recommends training on Raw and running the LoRA on Turbo.':draft.arch==='qwen-image-2.1'?' Qwen-Image-2.1 is the default because the LoRA also runs on Qwen-Image-2.1 Uncensored, the model you use most.':''} The first training downloads the base model.{arch?.gate?<> Needs access on <a href={arch.gate} target="_blank" rel="noreferrer">Hugging Face</a>.</>:null}</small></div>
          <div className="field"><label>What it learns</label><div className="segmented" role="radiogroup" aria-label="What it learns">{([['character','Person or character'],['style','Style']] as const).map(([id,label])=><button key={id} type="button" role="radio" aria-checked={draft.subject===id} aria-selected={draft.subject===id} onClick={()=>change('subject',id)}>{label}</button>)}</div></div>
          <div className="field"><label htmlFor="train-trigger">Trigger word</label><input id="train-trigger" value={draft.trigger_word} maxLength={40} placeholder="For example: mayaxz" onChange={e=>change('trigger_word',e.target.value.replace(/\s+/g,''))}/><small>A rare word that identifies this LoRA. Added automatically when the LoRA is on.</small></div>
        </div>
        <div className="section-heading train-images-heading"><h3>Images <span className="count">{draft.items.length}</span></h3>
          <button type="button" className="text-button" disabled={!!busy} onClick={()=>setLibraryOpen(o=>!o)}>Add from library</button>
          <FilePicker className="text-button" disabled={!!busy||!!draft.caption_job} multiple accept="image/*,.heic,.heif,.txt,.caption" onFiles={addFiles}>{busy==='upload'?'Uploading…':'Upload images'}</FilePicker>
          <FilePicker className="text-button" disabled={!!busy||!!draft.caption_job} directory multiple accept="" onFiles={addFiles}>Select folder</FilePicker>
          <FilePicker className="text-button" disabled={!!busy||!!draft.caption_job} accept=".zip,application/zip" onFiles={addFiles}>Upload ZIP</FilePicker>
          {draft.items.length?<button type="button" className="text-button" disabled={!!busy} onClick={()=>change('items',[])}>Clear</button>:null}
          </div>
        {busy==='upload'?<div role="status"><p>Uploading dataset: {uploadProgress}%</p><progress value={uploadProgress} max={100}/></div>:null}
        {importNote?<p role="status" className="train-hint">{importNote}</p>:null}
        <p className="train-hint">Select a folder or ZIP with images and matching TXT or CAPTION files. Up to 300 images and 1 GB.</p>
        <LibraryPopover open={libraryOpen} onClose={()=>setLibraryOpen(false)} anchor={panel}>
          <InputLibrary multiple api={api} inputs={draft.items} onChange={setItems} results={results} people={people} onPeople={onPeople} revision={revision} disabled={!!busy} limit={300} replace={false} target={0} onUpload={addFiles} onClose={()=>setLibraryOpen(false)}/>
        </LibraryPopover>
        <p className="train-hint">{draft.subject==='character'?'Use 15 to 40 sharp images of the same person: different clothes, places, light and angles, and some close-ups. Avoid other people in the images.':'Use 15 to 40 images in the same style with different subjects.'}</p>
        {draft.items.length?<>
          <div className="train-caption-tools"><button type="button" className="button secondary" disabled={!!busy||!!draft.caption_job||!missingCaptions} onClick={writeCaptions}>{draft.caption_job?'Writing captions...':missingCaptions?`Write ${missingCaptions} missing captions`:'All images have captions'}</button><div className="setting-toggle"><label className="checkbox-label"><input type="checkbox" checked={draft.auto_caption} onChange={e=>change('auto_caption',e.target.checked)}/>Write missing captions before training</label><ResetButton label="Write missing captions before training" value="On" onReset={()=>change('auto_caption',empty.auto_caption)}/></div><small>Existing captions stay unchanged. Captions run on this computer. The caption model downloads on first use if needed.</small></div>
          {advice?.warnings?.length?<div className="train-warnings" role="status"><strong>Dataset check</strong><ul>{advice.warnings.map(note=><li key={note}>{note}</li>)}</ul></div>:null}
          {draft.caption_job?<div className="train-caption-progress" role="status"><p>{captionState?.message||'Waiting in queue'}</p><progress value={captionState?.stage==='captioning'?captionState.step:undefined} max={captionState?.total_steps||1}/><button className="text-button" onClick={()=>api(`/api/jobs/${draft.caption_job}/cancel`,{method:'POST'}).catch(e=>setError(e.message))}>Stop captions</button></div>:null}
          <div className="train-items">{draft.items.map((item,index)=><div className="train-item" key={item.dataset_key||`${item.id}-${index}`}><img src={item.url} alt={`Image ${index+1}`} loading="lazy"/><div><span>{index+1}{item.caption_source==='file'?': From caption file':item.caption_source==='automatic'?': Automatic':''}</span><textarea dir="auto" value={item.caption} maxLength={8000} aria-label={`Caption ${index+1}`} placeholder={trigger?`${trigger}, ...`:'Caption'} onChange={e=>setDraft(d=>({...d,items:d.items.map((i,n)=>n===index?{...i,caption:e.target.value,caption_source:'manual',caption_error:undefined}:i)}))}/>{item.caption_error?<small role="alert">{item.caption_error}</small>:null}</div><button type="button" className="text-button" aria-label={`Remove image ${index+1}`} onClick={()=>setDraft(d=>({...d,items:d.items.filter((_,n)=>n!==index)}))}>Remove</button></div>)}</div>
        </>:<div className="train-empty">No images yet. Add them from the library (uploads, results, People) or upload new ones.</div>}
        <details className="disclosure train-advanced"><summary>Training settings<span className="summary-hint">{steps} steps · rank {effective.rank} · {effective.resolutions.join(', ')} px</span></summary><div className="details-body train-fields"><div className="field span-2"><SettingLabel htmlFor="train-mode" label="Settings" onReset={()=>change('settings_mode','recommended')}>Settings</SettingLabel><select id="train-mode" value={draft.settings_mode} onChange={e=>change('settings_mode',e.target.value as Draft['settings_mode'])}><option value="recommended">Suggested for this dataset</option><option value="custom">Custom</option></select><small>Starting values based on the model, image count, size, and training subject. Review the samples as training runs.</small>{recommended&&advice?advice.notes.map(note=><small key={note}>{note}</small>):null}</div>
          <div className="field"><SettingLabel disabled={resetPending} htmlFor="train-steps" label="Steps" onReset={()=>custom('steps',advice?.steps??autoSteps(draft.items.length))}>Steps</SettingLabel><input id="train-steps" type="number" min="50" max="20000" step="50" value={steps} onChange={e=>custom('steps',Number(e.target.value)||null)}/>{steps<50||steps>20000?<small role="alert">Choose 50 to 20000 steps.</small>:recommended?<small>Suggested for this dataset.</small>:<button type="button" className="text-button" onClick={()=>change('settings_mode','recommended')}>Use suggested settings</button>}</div>
          <div className="field"><SettingLabel disabled={resetPending} htmlFor="train-rank" label="Rank" onReset={()=>custom('rank',advice?.rank??empty.rank)}>Rank</SettingLabel><select id="train-rank" value={effective.rank} onChange={e=>custom('rank',Number(e.target.value))}>{[8,16,32,64,128].map(r=><option key={r} value={r}>{r}{r===(advice?.rank??empty.rank)?' (suggested)':''}</option>)}</select><small>Higher learns more detail and makes a larger file.</small></div>
          <div className="field"><SettingLabel disabled={resetPending} htmlFor="train-lr" label="Learning rate" onReset={()=>custom('learning_rate',advice?.learning_rate??empty.learning_rate)}>Learning rate</SettingLabel><select id="train-lr" value={effective.learning_rate} onChange={e=>custom('learning_rate',Number(e.target.value))}>{[5e-5,1e-4,2e-4,3e-4].map(v=><option key={v} value={v}>{v.toExponential(0).replace('e-','e-')}{v===(advice?.learning_rate??empty.learning_rate)?' (suggested)':''}</option>)}</select></div>
          <div className="field"><SettingLabel disabled={resetPending} htmlFor="train-save" label="Save and sample every" onReset={()=>custom('save_every',advice?.save_every??empty.save_every)}>Save and sample every</SettingLabel><select id="train-save" value={effective.save_every} onChange={e=>custom('save_every',Number(e.target.value))}>{[...new Set([100,250,500,effective.save_every])].sort((a,b)=>a-b).map(v=><option key={v} value={v}>{v} steps</option>)}</select></div>
          <div className="field span-2"><SettingLabel disabled={resetPending} label="Image sizes" onReset={()=>custom('resolutions',advice?.resolutions??empty.resolutions)}>Image sizes</SettingLabel><div className="train-sizes">{[512,768,1024,1280].map(size=><label key={size} className="checkbox-label"><input type="checkbox" checked={effective.resolutions.includes(size)} onChange={e=>custom('resolutions',e.target.checked?[...effective.resolutions,size].sort((a,b)=>a-b):effective.resolutions.filter(r=>r!==size).length?effective.resolutions.filter(r=>r!==size):effective.resolutions)}/>{size}</label>)}</div><small>Images are trained at these sizes. 1024 and up needs more time and memory.</small></div>
          <div className="field span-2"><SettingLabel htmlFor="train-prompts" label="Sample prompts" onReset={()=>change('sample_prompts',[])}>Sample prompts</SettingLabel><textarea id="train-prompts" dir="auto" rows={3} value={prompts.join('\n')} onChange={e=>change('sample_prompts',e.target.value.split('\n').filter(l=>l.trim()).slice(0,6))}/><small>One per line. The trainer makes these images at every save, so you can see the LoRA learn.</small></div>
        </div></details>
        <div className="train-actions"><button className="button primary" disabled={!canStart} onClick={start}>{busy==='start'?'Adding…':'Start training'}</button><small>{!state?.setup.ready?'Install the trainer first.':!draft.name.trim()?'Enter a name.':!draft.items.length?'Add images.':missingCaptions?draft.auto_caption?`${missingCaptions} missing captions will be written first.`:`${missingCaptions} images use only the trigger word.`:`${steps} steps on ${arch?.label}. Image jobs wait until the training ends.`}</small></div>
      </section>
      <section className="panel train-jobs" aria-label="Trainings"><h2>Trainings</h2>
        {!state?.jobs.length?<p className="train-empty">No trainings yet.</p>:state.jobs.map(job=><TrainingCard key={job.id} job={job} archs={state.archs} onStop={()=>act(`/api/jobs/${job.id}/cancel`)} onRetry={()=>act(`/api/jobs/${job.id}/retry`)} onDataset={()=>useDataset(job)} onUse={onUse} installed={!!models?.models.some(m=>m.id===job.lora_id)}/>)}
      </section>
    </div>
  </main></ModelDefaultsScope>;
}

/** ai-toolkit names samples [time]__[step, 9 digits]_[index].jpg. */
const sampleStep=(name:string)=>Number(name.split('__')[1]?.split('_')[0]||0);

function TrainingCard({job,archs,onStop,onRetry,onDataset,onUse,installed}:{job:SavedJob;archs:Arch[];onStop:()=>void;onRetry:()=>void;onDataset:()=>void;onUse:(id:string)=>void;installed:boolean}){
  const r=job.request;const active=['queued','running'].includes(job.status);const samples:string[]=job.samples||[];
  const percent=job.total_steps?Math.round(100*(job.step||0)/job.total_steps):0;
  return <article className={`train-job ${job.status}`}>
    <div className="train-job-head"><strong>{r.name}</strong><span>{archs.find(a=>a.id===r.arch)?.label||r.arch} · {r.images?.length} images · {r.steps} steps</span></div>
    <p className="train-job-status">{job.status==='queued'?'Waiting in queue':job.status==='running'?job.message:job.status==='succeeded'?`Done in ${formatDuration(job.elapsed_seconds||0)}`:job.status==='cancelled'?job.message||'Stopped':job.message}</p>
    {job.status==='running'?<><progress value={job.stage==='training'?job.step:undefined} max={job.total_steps||1}/><small className="train-job-meta">{job.stage==='training'?`${percent}%${job.loss!=null?` · loss ${Number(job.loss).toFixed(4)}`:''}${job.eta_seconds?` · ${formatDuration(job.eta_seconds)} left`:''}`:'Elapsed '+formatDuration(Math.max(0,(Date.now()/1000)-(job.started_at||job.created_at)))}</small></>:null}
    {samples.length?<div className="train-samples">{samples.slice(-6).map(name=><a key={name} href={`/api/training/${job.id}/samples/${encodeURIComponent(name)}`} target="_blank" rel="noreferrer" title={`Step ${sampleStep(name)}`}><img src={`/api/training/${job.id}/samples/${encodeURIComponent(name)}`} alt="" loading="lazy"/><span>{sampleStep(name)}</span></a>)}</div>:null}
    <div className="train-job-actions"><button className="text-button" onClick={onDataset}>Use dataset and captions</button>{active?<button className="text-button" onClick={onStop}>{job.status==='queued'?'Remove':'Stop'}</button>:null}
      {job.status==='succeeded'&&job.lora_id?installed?<button className="button primary" onClick={()=>onUse(job.lora_id)}>Use this LoRA</button>:<span>The LoRA was removed.</span>:null}
      {['failed','cancelled'].includes(job.status)?<button className="text-button" onClick={onRetry}>{job.status==='cancelled'?'Continue':'Retry'}</button>:null}</div>
  </article>;
}
