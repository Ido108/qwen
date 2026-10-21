import {useEffect,useId,useRef,useState,type ReactNode} from 'react';
import type {Api,ModelState} from './ModelSettings';
import type {SavedJob} from './JobQueue';
import type {InputImage} from './InputLibrary';
import {elapsedLabel,upscaleFraction,upscaleStageLabel} from './upscaleProgress';
import './image-upscale.css';

/** An image to upscale: an uploaded input (image) or a finished gallery result (job). */
export type UpscaleSource={image?:string;job?:string;preview:string;label:string;inputId?:string};
export type UpscaleSettings={model_id:string;size_mode:'scale'|'long_side';scale:2|4;long_side:number;color_correction:string;seed:number;
  input_noise_scale:number;latent_noise_scale:number;vae_tile_size:number;vae_tile_overlap:number;memory_mode:'auto'|'low_memory';gan_tile_size:number};
/** An input image being upscaled; when it is ready, the input panel offers to replace that input. */
export type InputUpscale={jobId:string;inputId:string};
type Plan={source_width:number;source_height:number;width:number|null;height:number|null;error?:string};
const KEY='qwen-image-upscale-v1';
// Advanced defaults are the numz ComfyUI-SeedVR2_VideoUpscaler CLI defaults: seed 42, LAB colors, no noise, 1024 px tiles, 128 px overlap.
const ADVANCED={color_correction:'lab',seed:42,input_noise_scale:0,latent_noise_scale:0,vae_tile_size:1024,vae_tile_overlap:128,memory_mode:'auto' as const,gan_tile_size:1024};
export const upscaleDefaults:UpscaleSettings={model_id:'seedvr2-3b-fp16',size_mode:'scale',scale:2,long_side:4096,...ADVANCED};
export const UPSCALE_KEYS=Object.keys(upscaleDefaults) as (keyof UpscaleSettings)[];
function saved():UpscaleSettings{try{return {...upscaleDefaults,...JSON.parse(localStorage.getItem(KEY)||'{}')};}catch{return upscaleDefaults;}}
const gb=(bytes:number)=>`${(bytes/1024**3).toFixed(1)} GB`;
/** The image upscalers offered, in this order, each with one line on when to choose it. */
const CHOICES:[string,string][]=[
  ['seedvr2-3b-fp16','Default. Adds real detail. 15 to 50 seconds on this GPU.'],
  ['seedvr2-3b-fp8','The same 3B model in a file half the size. Slightly less precise.'],
  ['seedvr2-7b-fp16','Larger model with more detail. Slower and needs more GPU memory.'],
  ['seedvr2-7b-fp8-mixed','7B in a file half the size. The mixed file avoids the plain FP8 quality problem.'],
  ['seedvr2-7b-sharp-fp16','7B tuned for sharper edges. Can over-sharpen clean AI images.'],
  ['seedvr2-7b-sharp-fp8-mixed','Sharp 7B in a file half the size.'],
  ['nomos2-hq-mosr','Fast and faithful: sharpens what is there and adds no new content. Good for text and logos. Up to 4×.'],
];
const COLORS:[string,string][]=[['lab','Match source colors (LAB)'],['wavelet','Wavelet'],['wavelet_adaptive','Wavelet with saturation fix'],['hsv','Hue based (HSV)'],['adain','Color statistics (AdaIN)'],['none','None']];

function Field({label,help,onDefault,isDefault,children,htmlFor}:{label:string;help:string;onDefault:()=>void;isDefault:boolean;children:ReactNode;htmlFor:string}){
  return <div className="upscale-field">
    <div className="upscale-field-head"><label htmlFor={htmlFor}>{label}</label><button type="button" className="upscale-default" disabled={isDefault} onClick={onDefault} aria-label={`Set ${label} to its default`}>Default</button></div>
    {children}<small>{help}</small>
  </div>;
}

/** Stage, bar, stage count and elapsed time of a running image upscale. */
export function UpscaleProgress({job,compact=false}:{job:SavedJob;compact?:boolean}){
  const [now,setNow]=useState(Date.now());
  useEffect(()=>{if(job.status!=='running')return;const timer=window.setInterval(()=>setNow(Date.now()),500);return()=>clearInterval(timer);},[job.status]);
  const fraction=upscaleFraction(job,now);const stage=upscaleStageLabel(job);const elapsed=elapsedLabel(job,now);
  const sizes=job.source_width&&job.width?`${job.source_width} × ${job.source_height} → ${job.width} × ${job.height}`:'';
  const text=job.status==='queued'?`Waiting in queue${job.position?` · #${job.position}`:''}`:job.stage==='downloading_models'?job.message:job.message||'Starting…';
  return <div className={`upscale-progress ${compact?'compact':''}`}>
    <div className="upscale-progress-line"><strong>{text}</strong><span>{[stage,elapsed].filter(Boolean).join(' · ')}</span></div>
    <progress aria-label="Upscale progress" value={job.status==='running'?Math.round(fraction*100):undefined} max={100}/>
    {sizes&&!compact?<small>{sizes}</small>:null}
  </div>;
}

function usePolledJob(api:Api,jobId:string|null){
  const [job,setJob]=useState<SavedJob|null>(null);const [gone,setGone]=useState(false);
  useEffect(()=>{if(!jobId)return;let alive=true;let timer:number;
    async function poll(){try{const current:SavedJob=await api(`/api/jobs/${jobId}`);if(!alive)return;setJob(current);if(['succeeded','failed','cancelled'].includes(current.status))return;}
      catch(e){if(!alive)return;if((e as Error).message.toLowerCase().includes('not found')){setGone(true);return;}}
      timer=window.setTimeout(poll,1000);}
    poll();return()=>{alive=false;clearTimeout(timer);};
  },[api,jobId]);
  return {job,gone};
}

export function ImageUpscaleDialog({source,preset,api,models,onClose,onQueued,onView}:{source:UpscaleSource;preset?:Partial<UpscaleSettings>;api:Api;models:ModelState|null;onClose:()=>void;onQueued:(job:SavedJob)=>void;onView?:(job:SavedJob)=>void}){
  const dialog=useRef<HTMLDialogElement>(null);const heading=useId();const id=useId();
  const [settings,setSettings]=useState<UpscaleSettings>(()=>({...saved(),...preset}));
  const [longText,setLongText]=useState(()=>String({...saved(),...preset}.long_side));
  const [plan,setPlan]=useState<Plan|null>(null);const [planError,setPlanError]=useState('');
  const [submitting,setSubmitting]=useState(false);const [error,setError]=useState('');const [queuedId,setQueuedId]=useState<string|null>(null);
  const {job:queued}=usePolledJob(api,queuedId);
  const all=models?.models||[];
  const choices=CHOICES.map(([choice,note])=>({model:all.find(m=>m.id===choice),note})).filter((c):c is {model:NonNullable<typeof c.model>;note:string}=>!!c.model);
  const selected=choices.find(c=>c.model.id===settings.model_id);
  const gan=selected?.model.family==='spandrel';
  const longValid=Number.isInteger(settings.long_side)&&settings.long_side>=256&&settings.long_side<=8192;
  const change=(patch:Partial<UpscaleSettings>)=>setSettings(old=>({...old,...patch}));
  const locked=!!queuedId;
  useEffect(()=>{if(!dialog.current?.open)dialog.current?.showModal();},[]);
  useEffect(()=>{try{localStorage.setItem(KEY,JSON.stringify(settings));}catch{/* the form still works */}},[settings]);
  useEffect(()=>{if(settings.size_mode==='long_side'&&!longValid){setPlan(old=>old&&{...old,width:null,height:null,error:undefined});return;}
    let alive=true;
    const query=new URLSearchParams({...(source.job?{source_job:source.job}:{source_image:source.image||''}),size_mode:settings.size_mode,scale:String(settings.scale),long_side:String(settings.long_side),model_id:settings.model_id});
    const timer=window.setTimeout(()=>api(`/api/image-upscale/plan?${query}`).then((next:Plan)=>{if(alive){setPlan(next);setPlanError('');}}).catch((e:Error)=>{if(alive){setPlan(null);setPlanError(e.message);}}),settings.size_mode==='long_side'?250:0);
    return()=>{alive=false;clearTimeout(timer);};
  },[api,source.image,source.job,settings.size_mode,settings.scale,settings.long_side,settings.model_id,longValid]);
  const ready=!!plan?.width&&!planError&&!!selected&&(settings.size_mode==='scale'||longValid);
  async function submit(){
    setSubmitting(true);setError('');
    try{const job:SavedJob=await api('/api/image-upscale/generate',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({...settings,...(source.job?{source_job:source.job}:{source_image:source.image})})});setQueuedId(job.id);onQueued(job);}
    catch(e){setError((e as Error).message);}finally{setSubmitting(false);}
  }
  const size=(w?:number|null,h?:number|null)=>w&&h?`${w} × ${h}`:'…';
  const changed=(gan?['gan_tile_size']:['seed','color_correction','input_noise_scale','latent_noise_scale','vae_tile_size','vae_tile_overlap','memory_mode']).filter(key=>settings[key as keyof UpscaleSettings]!==ADVANCED[key as keyof typeof ADVANCED]).length;
  const field=<K extends keyof typeof ADVANCED>(key:K)=>({isDefault:settings[key]===ADVANCED[key],onDefault:()=>change({[key]:ADVANCED[key]} as Partial<UpscaleSettings>),htmlFor:`${id}-${key}`});
  const done=queued?.status==='succeeded',ended=queued&&['failed','cancelled'].includes(queued.status);
  return <dialog ref={dialog} className="upscale-dialog" aria-labelledby={heading} onCancel={e=>{e.preventDefault();onClose();}} onClick={e=>{if(e.target===dialog.current)onClose();}} onKeyDown={e=>e.stopPropagation()}>
    <header><h2 id={heading}>Upscale image</h2><button type="button" className="upscale-close" aria-label="Close" onClick={onClose}>×</button></header>
    <div className="upscale-dialog-body">
      <figure className="upscale-dialog-preview"><img className="checker" src={done&&queued?.thumbnail_url?queued.thumbnail_url:source.preview} alt="Image to upscale"/><figcaption>{done?'Upscaled result':source.label}</figcaption></figure>
      <div className="upscale-dialog-form">
        <div className="upscale-choice" role="group" aria-label="Result size">
          <button type="button" aria-pressed={settings.size_mode==='scale'&&settings.scale===2} disabled={locked} onClick={()=>change({size_mode:'scale',scale:2})}>2×</button>
          <button type="button" aria-pressed={settings.size_mode==='scale'&&settings.scale===4} disabled={locked} onClick={()=>change({size_mode:'scale',scale:4})}>4×</button>
          <button type="button" aria-pressed={settings.size_mode==='long_side'} disabled={locked} onClick={()=>change({size_mode:'long_side'})}>Long side</button>
        </div>
        {settings.size_mode==='long_side'?<label className="upscale-long"><span>Long side (px)</span><input type="number" inputMode="numeric" min={256} max={8192} step={1} value={longText} disabled={locked} aria-invalid={!longValid}
          onChange={e=>{setLongText(e.target.value);change({long_side:Number(e.target.value)});}}/>{!longValid?<small>Enter 256 to 8192.</small>:null}</label>:null}
        <dl className="upscale-plan" aria-live="polite">
          <div><dt>Source</dt><dd>{size(plan?.source_width,plan?.source_height)}</dd></div>
          <div className="upscale-plan-arrow" aria-hidden="true">→</div>
          <div><dt>Result</dt><dd>{plan?.error?'Too large':size(plan?.width,plan?.height)}</dd></div>
        </dl>
        {plan?.error||planError?<p className="upscale-problem" role="alert">{plan?.error||planError}</p>:null}
        <div className="upscale-field"><label htmlFor={`${id}-model`}>Upscaler</label><select id={`${id}-model`} value={settings.model_id} disabled={locked} onChange={e=>change({model_id:e.target.value})}>
          <optgroup label="SeedVR2: adds detail">{choices.filter(c=>c.model.family==='seedvr2').map(({model})=><option key={model.id} value={model.id}>{model.title}{model.installed?'':` · ${gb(model.bytes)} download`}</option>)}</optgroup>
          <optgroup label="Fast: no new detail">{choices.filter(c=>c.model.family!=='seedvr2').map(({model})=><option key={model.id} value={model.id}>{model.title}{model.installed?'':` · ${gb(model.bytes)} download`}</option>)}</optgroup></select>
          <small>{!selected?'Loading upscalers…':`${selected.note} ${selected.model.installed?'Installed.':`Downloads ${gb(selected.model.bytes)} when the job starts.`}`}</small></div>
        <details className="upscale-advanced"><summary>Advanced<span>{changed?`${changed} changed`:'Defaults'}</span></summary>
          <p className="upscale-advanced-note">{gan?'Default: 1024 px tiles.':'Defaults follow the SeedVR2 README (numz ComfyUI node and CLI).'}</p>
          {gan?<Field label="Tile size" help="The image is processed in pieces of this size. Larger pieces are a little faster and use more GPU memory." {...field('gan_tile_size')}>
            <select id={`${id}-gan_tile_size`} value={settings.gan_tile_size} disabled={locked} onChange={e=>change({gan_tile_size:Number(e.target.value)})}>{[256,384,512,768,1024].map(n=><option key={n} value={n}>{n} px</option>)}</select></Field>
          :<>
          <Field label="Seed" help="The same seed and settings give the same result. Change it for a slightly different version." {...field('seed')}>
            <input id={`${id}-seed`} type="number" min={0} max={2147483647} step={1} value={settings.seed} disabled={locked} onChange={e=>change({seed:Math.max(0,Math.floor(Number(e.target.value)||0))})}/></Field>
          <Field label="Color correction" help="Keeps the colors of the source. Match source colors (LAB) is the recommended choice." {...field('color_correction')}>
            <select id={`${id}-color_correction`} value={settings.color_correction} disabled={locked} onChange={e=>change({color_correction:e.target.value})}>{COLORS.map(([value,label])=><option key={value} value={value}>{label}</option>)}</select></Field>
          <Field label={`Input noise: ${settings.input_noise_scale.toFixed(2)}`} help="Adds noise to the image before upscaling. Higher values make more new texture and a less faithful result. 0 keeps it faithful." {...field('input_noise_scale')}>
            <input id={`${id}-input_noise_scale`} type="range" min={0} max={1} step={0.05} value={settings.input_noise_scale} disabled={locked} onChange={e=>change({input_noise_scale:Number(e.target.value)})}/></Field>
          <Field label={`Latent noise: ${settings.latent_noise_scale.toFixed(2)}`} help="Adds noise inside the model. Higher values change more fine detail. 0 keeps it faithful." {...field('latent_noise_scale')}>
            <input id={`${id}-latent_noise_scale`} type="range" min={0} max={1} step={0.05} value={settings.latent_noise_scale} disabled={locked} onChange={e=>change({latent_noise_scale:Number(e.target.value)})}/></Field>
          <Field label="Tile size" help="The image is read and written in tiles of this size. Smaller tiles use less GPU memory." {...field('vae_tile_size')}>
            <select id={`${id}-vae_tile_size`} value={settings.vae_tile_size} disabled={locked} onChange={e=>change({vae_tile_size:Number(e.target.value)})}>{[512,768,1024,1536,2048].map(n=><option key={n} value={n}>{n} px</option>)}</select></Field>
          <Field label="Tile overlap" help="Pixels shared by neighbor tiles. More overlap hides seams better and takes a little longer." {...field('vae_tile_overlap')}>
            <select id={`${id}-vae_tile_overlap`} value={settings.vae_tile_overlap} disabled={locked} onChange={e=>change({vae_tile_overlap:Number(e.target.value)})}>{[32,64,128,256].map(n=><option key={n} value={n}>{n} px</option>)}</select></Field>
          <Field label="GPU memory" help="Save memory keeps part of the model in system memory. It is slower. Use it only if a job runs out of GPU memory." {...field('memory_mode')}>
            <select id={`${id}-memory_mode`} value={settings.memory_mode} disabled={locked} onChange={e=>change({memory_mode:e.target.value as UpscaleSettings['memory_mode']})}><option value="auto">Automatic</option><option value="low_memory">Save memory</option></select></Field>
          </>}
        </details>
      </div>
    </div>
    <footer>
      {error?<p className="upscale-problem" role="alert">{error}</p>:null}
      {queued&&!done&&!ended?<div className="upscale-footer-progress"><UpscaleProgress job={queued}/></div>:null}
      {ended?<p className="upscale-problem" role="alert">{queued?.status==='cancelled'?'The upscale was cancelled.':queued?.message}</p>:null}
      {queuedId?<>{done?<p className="upscale-queued" role="status">Done in {queued?.elapsed_seconds}s. The image is in Recent and in the Gallery.{source.inputId?' The input panel offers to replace the input.':''}</p>:!queued||queued.status==='queued'?<p className="upscale-queued" role="status">Added to the queue. You can close this window; progress also shows in the queue bar.</p>:null}
        {!done&&!ended?<button type="button" className="button secondary" onClick={()=>api(`/api/jobs/${queuedId}/cancel`,{method:'POST'}).catch(e=>setError((e as Error).message))}>{queued?.status==='running'?'Stop':'Remove'}</button>:null}
        {done&&onView&&queued?<button type="button" className="button secondary" onClick={()=>{onView(queued);onClose();}}>View result</button>:null}
        <button type="button" className="button primary" onClick={onClose}>Close</button></>
        :<><button type="button" className="button secondary" onClick={onClose}>Cancel</button><button type="button" className="button primary" disabled={!ready||submitting} onClick={submit}>{submitting?'Adding…':'Add to queue'}</button></>}
    </footer>
  </dialog>;
}

/** Progress of an input image upscale, then the choice to replace that input with the result. */
export function InputUpscaleStatus({api,track,inputs,onReplace,onAdd,onDismiss}:{api:Api;track:InputUpscale;inputs:InputImage[];onReplace:(inputId:string,image:InputImage&{width?:number;height?:number})=>void;onAdd:(image:InputImage)=>void;onDismiss:()=>void}){
  const {job,gone}=usePolledJob(api,track.jobId);const [busy,setBusy]=useState(false);const [error,setError]=useState('');
  const index=inputs.findIndex(item=>item.id===track.inputId);
  const name=index>=0?`image ${index+1}`:'the input image';
  const sizes=job?.source_width&&job?.width?`${job.source_width} × ${job.source_height} → ${job.width} × ${job.height}`:'';
  async function use(replace:boolean){
    setBusy(true);setError('');
    try{const image=await api(`/api/jobs/${track.jobId}/reuse`,{method:'POST'});if(replace)onReplace(track.inputId,image);else onAdd(image);onDismiss();}
    catch(e){setError((e as Error).message);}finally{setBusy(false);}
  }
  const done=job?.status==='succeeded',ended=job&&['failed','cancelled'].includes(job.status);
  return <div className={`input-upscale-status ${done?'ready':ended?'ended':''}`} role="status">
    {done?<>{job?.thumbnail_url?<img src={job.thumbnail_url} alt=""/>:null}<div><strong>Upscaled {name} is ready</strong><span>{sizes}{job?.elapsed_seconds?` · ${job.elapsed_seconds}s`:''}</span></div>
      <div className="input-upscale-actions">{index>=0?<button type="button" className="button primary" disabled={busy} onClick={()=>use(true)}>Replace input {index+1}</button>:null}<button type="button" className="button secondary" disabled={busy||inputs.length>=10} onClick={()=>use(false)}>Add as new input</button><button type="button" className="text-button" onClick={onDismiss}>Keep original</button></div></>
    :ended||gone?<><div><strong>{gone?'This upscale job is no longer available':job?.status==='cancelled'?'Upscale cancelled':'Upscale failed'}</strong><span>{job?.message}</span></div><div className="input-upscale-actions"><button type="button" className="text-button" onClick={onDismiss}>Dismiss</button></div></>
    :<><div className="input-upscale-running"><strong>Upscaling {name}</strong>{job?<UpscaleProgress job={job}/>:<span>Starting…</span>}</div>
      <div className="input-upscale-actions"><button type="button" className="text-button" disabled={!job} onClick={()=>api(`/api/jobs/${track.jobId}/cancel`,{method:'POST'}).catch(e=>setError((e as Error).message))}>{job?.status==='queued'?'Remove':'Stop'}</button></div></>}
    {error?<p className="upscale-problem" role="alert">{error}</p>:null}
  </div>;
}

/** Running and waiting image upscales above the Result image, with their source picture. */
export function ResultUpscales({jobs}:{jobs:SavedJob[]}){
  if(!jobs.length)return null;
  return <div className="result-upscales">{jobs.map(job=><div className="result-upscale" key={job.id}>
    {job.request.source_image?<img className="checker" src={`/api/uploads/${job.request.source_image}/thumb`} alt=""/>:null}
    <div><strong>Upscaling{job.source_width?` ${job.source_width} × ${job.source_height} to ${job.width} × ${job.height}`:''}</strong><UpscaleProgress job={job} compact/></div>
  </div>)}</div>;
}
