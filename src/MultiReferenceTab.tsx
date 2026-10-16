import {SendVideoTo,type SendVideo} from './SendVideoTo';
import {GenerationTiming} from './GenerationTiming';
import {useEffect,useRef,useState} from 'react';
import {Download,Film,ImagePlus,Loader2,Plus,Sparkles,X} from 'lucide-react';
import type {Api,ModelState} from './ModelSettings';
import type {SavedJob,QueueState} from './JobQueue';
import {inputImage} from './Gallery';
import {VideoProgress,VramFitHint,useGpuMemory} from './VideoProgress';
import {PromptHistory,rememberPrompt} from './PromptHistory';
import './video-studio.css';
import './multi-reference.css';
import {LoraStack,type LoraChoice} from './LoraStack';
import {familyLabel,fitSize,groupByFamily,isRef2va,nearestFrames,REF2VA_FRAMES,sizeFits,supports,videoModelsFor} from './videoModels';

type Params={model_id:string;prompt:string;reference_images:string[];background:string|null;audio_references:(string|null)[];width:number;height:number;frames:number;fps:24|25;seed:number;randomize_seed:boolean;reference_strength:number;lora_strength:number;reference_frames:25|33;refine:boolean;audio:boolean;enhance:false;output_dir:string;steps:number|null;quality:'standard'|'turbo'|null;loras:LoraChoice[]};
const initial:Params={model_id:'ltx25-msr-q4_k_m',prompt:'',reference_images:[],background:null,audio_references:[],width:768,height:448,frames:121,fps:24,seed:0,randomize_seed:true,reference_strength:1,lora_strength:1,reference_frames:33,refine:false,audio:true,enhance:false,output_dir:'',steps:null,quality:null,loras:[]};
function loadDraft():Params{try{return {...initial,...JSON.parse(localStorage.getItem('qwen-multi-reference-v1')||'{}'),randomize_seed:true};}catch{return initial;}}
const allSizes=[[768,448],[448,768],[640,640],[1024,576],[576,1024],[768,768],[1280,704],[704,1280],[1024,1024],[1344,768],[768,1344]] as const;
const LTX_REFERENCE_FRAMES=[73,121,193,241,361];
// LTX first pass: up to one megapixel. MiniMax H3 Ref2VA: up to 1344 x 768.
const ltxRule={step:32,maxSide:1536,maxPixels:1048576},h3Rule={step:32,maxSide:1344,maxPixels:1344*768};

export function MultiReferenceTab({api,onSendVideo,visible,models,queue,onQueued,restoreRequest,onView}:{api:Api;onSendVideo:SendVideo;visible:boolean;models:ModelState|null;queue:QueueState;onQueued:()=>void;restoreRequest:{job:SavedJob;key:number}|null;onView:(job:SavedJob)=>void}){
  const [params,setParams]=useState<Params>(loadDraft);
  const [uploading,setUploading]=useState(false),[submitting,setSubmitting]=useState(false),[error,setError]=useState(''),[notice,setNotice]=useState('');
  const [history,setHistory]=useState<SavedJob[]>([]),[clock,setClock]=useState(Date.now()),[job,setJob]=useState<SavedJob|null>(null);
  const promptRef=useRef<HTMLTextAreaElement>(null),subjectInput=useRef<HTMLInputElement>(null),backgroundInput=useRef<HTMLInputElement>(null);
  const options=videoModelsFor(models?.models,'references');
  const selected=options.find(m=>m.id===params.model_id);const h3=isRef2va(selected);
  const rule=h3?h3Rule:ltxRule;const sizes=allSizes.filter(([w,h])=>sizeFits(w,h,rule));const frameOptions=h3?REF2VA_FRAMES:LTX_REFERENCE_FRAMES;
  const profiles=selected?.generation_defaults?.profiles;const quality=params.quality||selected?.generation_defaults?.quality||'standard';const sampling=profiles?.[quality];
  const defaultSteps=sampling?.steps||selected?.generation_defaults?.steps||20;const totalSteps=h3?(params.steps??defaultSteps):8+(params.refine?3:0);
  const stepsValid=params.steps===null||Number.isInteger(params.steps)&&params.steps>=1&&params.steps<=60;
  const videoLoras=(models?.models||[]).filter(m=>m.kind==='video-lora');
  const previousModel=useRef(params.model_id);
  const active=queue.active?.kind==='multi-reference'?queue.active:null;
  const running=active||(job&&['queued','running'].includes(job.status)?job:null);
  const gpu=useGpuMemory(api,visible&&!models?.remote&&!running);
  const recentKey=queue.recent.filter(j=>j.kind==='multi-reference').map(j=>j.id+':'+j.status).join(',');
  useEffect(()=>{localStorage.setItem('qwen-multi-reference-v1',JSON.stringify(params));},[params]);
  useEffect(()=>{if(!visible)return;const timer=window.setInterval(()=>setClock(Date.now()),1000);return()=>clearInterval(timer);},[visible]);
  useEffect(()=>{if(!visible)return;let alive=true;api('/api/history?kind=multi-reference').then((items:SavedJob[])=>{if(alive)setHistory(items);}).catch(e=>{if(alive)setError(e.message);});return()=>{alive=false;};},[api,visible,recentKey]);
  useEffect(()=>{if(!job||!['queued','running'].includes(job.status))return;let alive=true;const refresh=()=>api(`/api/jobs/${job.id}`).then(next=>{if(alive)setJob(next);}).catch(e=>{if(alive)setError(e.message);});const timer=window.setInterval(refresh,1500);return()=>{alive=false;clearInterval(timer);};},[api,job?.id,job?.status]);
  useEffect(()=>{api('/api/config').then(cfg=>setParams(p=>({...p,output_dir:p.output_dir||cfg.video_output_dir}))).catch(()=>{});},[api]);
  useEffect(()=>{if(!restoreRequest)return;setParams({...initial,...restoreRequest.job.request,seed:restoreRequest.job.seed,randomize_seed:true});setNotice('Settings restored.');},[restoreRequest]);
  const result=job?.status==='succeeded'?job:history[0];
  const missing=selected&&!models?.remote?[selected.id,...selected.dependencies,...(!h3&&params.refine?['ltx25-spatial-upscaler']:[]),...(h3&&sampling?.lora_id?[sampling.lora_id]:[])].filter(id=>!models?.models.find(m=>m.id===id)?.installed):[];
  const validSize=sizeFits(params.width,params.height,rule);
  const ready=!!selected&&(!models?.remote||selected.ready)&&!!params.prompt.trim()&&!!params.reference_images.length&&validSize&&stepsValid&&frameOptions.includes(params.frames);
  // Keep a model this page cannot run from staying selected (for example from an old draft).
  useEffect(()=>{if(models&&options.length&&!selected)setParams(p=>({...p,model_id:(options.find(m=>m.ready)||options[0]).id}));},[models,params.model_id]);
  // Selecting a model applies its own rules and defaults: frame grid, frame rate, size, steps and quality.
  useEffect(()=>{if(!selected)return;const switched=previousModel.current!==selected.id;previousModel.current=selected.id;setParams(p=>{const next={...p};const ref2va=isRef2va(selected);const frames=ref2va?REF2VA_FRAMES:LTX_REFERENCE_FRAMES;if(!frames.includes(p.frames))next.frames=nearestFrames(p.frames,frames);if(ref2va){next.fps=24;next.refine=false;}const r=ref2va?h3Rule:ltxRule;if(!sizeFits(p.width,p.height,r))Object.assign(next,fitSize(p.width,p.height,r));if(switched){next.steps=null;next.quality=null;}return next;});},[selected?.id]);
  function change<K extends keyof Params>(key:K,value:Params[K]){setParams(p=>({...p,[key]:value}));}
  function insertTag(tag:string){const field=promptRef.current;const start=field?.selectionStart??params.prompt.length,end=field?.selectionEnd??start;change('prompt',params.prompt.slice(0,start)+tag+params.prompt.slice(end));requestAnimationFrame(()=>{field?.focus();field?.setSelectionRange(start+tag.length,start+tag.length);});}
  async function upload(files:File[],background=false){
    if(!files.length)return;setUploading(true);setError('');
    try{const ids:string[]=[];for(const file of files.slice(0,background?1:4-params.reference_images.length)){const body=new FormData();body.append('files',file);const items=await api('/api/uploads',{method:'POST',body});ids.push(items[0].id);}if(background)change('background',ids[0]);else setParams(p=>({...p,reference_images:[...p.reference_images,...ids].slice(0,4)}));}
    catch(e){setError((e as Error).message);}finally{setUploading(false);}
  }
  async function uploadVoice(index:number,file?:File){if(!file)return;setUploading(true);setError('');try{const body=new FormData();body.append('file',file);const item=await api('/api/audio-uploads',{method:'POST',body});setParams(p=>{const voices=[...p.audio_references];voices[index]=item.id;return {...p,audio_references:voices};});}catch(e){setError((e as Error).message);}finally{setUploading(false);}}
  async function generate(text?:string){const prompt=text??params.prompt;if(!prompt.trim()){setError('Describe the video first.');return;}setSubmitting(true);setError('');setNotice('');try{const next=await api('/api/multi-reference/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...params,prompt,loras:params.loras.filter(l=>videoLoras.find(m=>m.id===l.id)?.families?.includes(selected?.family||''))})});rememberPrompt('multi-reference',prompt);setJob(next);setNotice('Added to queue. You can prepare another video.');onQueued();}catch(e){setError((e as Error).message);}finally{setSubmitting(false);}}
  return <main hidden={!visible} className="msr-workbench video-studio">
    <div className="vs-grid">
    <div className="vs-controls">
      <div className="vs-scroll">
      <section className="panel vs-panel msr-inputs">
      <label className="msr-model">Model<select aria-label="Reference images model" value={params.model_id} onChange={e=>change('model_id',e.target.value)}>{groupByFamily(options,'LTX 2.5').map(([family,items])=><optgroup key={family} label={family}>{items.map(m=><option key={m.id} value={m.id}>{m.title}{m.ready?'':' · download'}</option>)}</optgroup>)}</select>{selected?<small className="vs-model-supports">Supports: {supports(selected).join(', ')}</small>:null}</label>
      {!running&&!models?.remote?<VramFitHint models={models?.models||[]} selected={selected} gpu={gpu} onUse={id=>change('model_id',id)}/>:null}
      <div className="section-heading"><h2>References</h2><span>Up to 4 subjects + background</span></div>
      <input ref={subjectInput} hidden type="file" accept="image/png,image/jpeg,image/webp" multiple onChange={e=>{upload(Array.from(e.target.files||[]));e.target.value='';}}/>
      <input ref={backgroundInput} hidden type="file" accept="image/png,image/jpeg,image/webp" onChange={e=>{upload(Array.from(e.target.files||[]),true);e.target.value='';}}/>
      <div className="msr-reference-list">{params.reference_images.map((id,index)=><div className="msr-reference-card" key={id+index}>
        <img src={inputImage(id).url} alt={`Image ${index+1}`}/><button className="msr-tag" onClick={()=>insertTag(`Image ${index+1}`)} title="Insert into description">Image {index+1}</button>
        <button className="msr-remove icon-button" aria-label={`Remove image ${index+1}`} disabled={uploading} onClick={()=>setParams(p=>({...p,reference_images:p.reference_images.filter((_,i)=>i!==index),audio_references:p.audio_references.filter((_,i)=>i!==index).slice(0,2)}))}><X size={14}/></button>
      </div>)}{params.reference_images.length<4?<button className="msr-add" disabled={uploading} onClick={()=>subjectInput.current?.click()}><ImagePlus size={22}/><span>Add reference</span></button>:null}
      <div className="msr-background">{params.background?<><img src={inputImage(params.background).url} alt="Background reference"/><button className="msr-tag" onClick={()=>insertTag(`Image ${params.reference_images.length+1} (background)`)}>Background</button><button className="msr-remove icon-button" aria-label="Remove background" onClick={()=>change('background',null)}><X size={14}/></button></>:<button className="msr-add" disabled={uploading} onClick={()=>backgroundInput.current?.click()}><Plus size={20}/><span>Background</span><small>Optional</small></button>}</div></div>
      <p className="msr-hint">Click an image label to mention it in the description. Reference images do not set the first frame.</p>
      <div className="prompt-row"><label className="msr-prompt-label" htmlFor="msr-prompt">Describe the video</label><PromptHistory scope="multi-reference" disabled={submitting||uploading} onLoad={text=>{change('prompt',text);promptRef.current?.focus();}} onRun={text=>{change('prompt',text);generate(text);}}/></div><textarea id="msr-prompt" ref={promptRef} value={params.prompt} onChange={e=>change('prompt',e.target.value)} onKeyDown={e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'&&ready&&!submitting&&!uploading){e.preventDefault();generate();}}} placeholder="Image 1 and Image 2 walk through the scene in Image 3. Describe the action, camera and dialogue."/>
      <div className="msr-output-controls"><label>{params.refine?'Generation size (px)':'Output size (px)'}<select aria-label="LTX output size" value={sizes.some(s=>s[0]===params.width&&s[1]===params.height)?`${params.width}x${params.height}`:'custom'} onChange={e=>{if(e.target.value==='custom'){change('width',736);return;}const [width,height]=e.target.value.split('x').map(Number);setParams(p=>({...p,width,height}));}}>{sizes.map(([w,h])=><option key={`${w}x${h}`} value={`${w}x${h}`}>{w} × {h}</option>)}<option value="custom">Enter width and height</option></select></label><label>Duration<select aria-label="Duration" value={params.frames} onChange={e=>change('frames',Number(e.target.value))}>{frameOptions.map(frames=><option key={frames} value={frames}>{(frames/params.fps).toFixed(1)} seconds</option>)}</select></label></div>
      {!sizes.some(s=>s[0]===params.width&&s[1]===params.height)?<div className="msr-output-controls"><label>Width (px)<input type="number" min="256" max={rule.maxSide} step="32" value={params.width} onChange={e=>change('width',Number(e.target.value))}/></label><label>Height (px)<input type="number" min="256" max={rule.maxSide} step="32" value={params.height} onChange={e=>change('height',Number(e.target.value))}/></label></div>:null}
      {!validSize?<p className="message error">{h3?'Use multiples of 32, 256 to 1344 pixels per side, up to 1,032,192 pixels, aspect ratio 1:4 to 4:1.':'Use multiples of 32, 256 to 1536 pixels per side, up to 1,048,576 pixels before optional 2× enlargement.'}</p>:null}
      </section>
      <details className="disclosure msr-voices"><summary>Voice references <span>Optional · first 5 seconds</span></summary><div>{[0,1].map(index=><div className="msr-voice" key={index}><strong>Image {index+1}</strong>{params.audio_references[index]?<><audio controls preload="metadata" src={`/api/audio-uploads/${params.audio_references[index]}`}/><button className="icon-button" aria-label={`Remove voice ${index+1}`} onClick={()=>setParams(p=>{const voices=[...p.audio_references];voices[index]=null;return {...p,audio_references:voices};})}><X size={15}/></button></>:<label className="button secondary">Add voice<input type="file" accept=".mp3,.wav,.m4a,.flac,.ogg" disabled={uploading||!params.reference_images[index]} onChange={e=>{uploadVoice(index,e.target.files?.[0]);e.target.value='';}}/></label>}</div>)}</div></details>
      <details className="disclosure msr-options" open={params.loras.length?true:undefined}><summary>LoRAs <span>{params.loras.length?`${params.loras.length} of 4`:'Optional'}</span></summary><div><LoraStack value={params.loras} onChange={next=>change('loras',next)} loras={videoLoras} family={selected?.family||''} prompt={params.prompt} onPrompt={next=>change('prompt',next)} onBrowse={()=>{window.location.hash='#civitai';}} disabled={submitting}/></div></details>
      <details className="disclosure msr-options"><summary>Advanced settings <span>{h3?`${totalSteps} steps`:`8 steps${params.refine?' + 3 refinement steps':''}`}</span></summary><div className="msr-settings">
        {h3?<>{profiles&&Object.keys(profiles).length>1?<label>Generation mode<select aria-label="Generation mode" value={quality} onChange={e=>setParams(p=>({...p,quality:e.target.value as 'standard'|'turbo',steps:null}))}>{Object.entries(profiles).map(([key,profile])=><option key={key} value={key}>{key==='turbo'?'Turbo':'Standard'} · {profile!.steps} steps</option>)}</select></label>:null}<label>Steps<input aria-describedby="msr-steps-help" type="number" min="1" max="60" value={params.steps??''} placeholder={String(defaultSteps)} onChange={e=>change('steps',e.target.value===''?null:Number(e.target.value))}/><small id="msr-steps-help">Empty uses the model default ({defaultSteps}).</small></label>
        <label className="checkbox-label"><input type="checkbox" checked={params.audio} onChange={e=>change('audio',e.target.checked)}/>Generate audio</label></>:<>
        <label>Frame rate<select aria-label="Frame rate" value={params.fps} onChange={e=>change('fps',Number(e.target.value) as 24|25)}><option value="24">24 fps</option><option value="25">25 fps</option></select></label><label>Image influence<input aria-describedby="msr-image-influence-help" type="number" min="0.1" max="1" step=".05" value={params.reference_strength} onChange={e=>change('reference_strength',Number(e.target.value))}/><small id="msr-image-influence-help">How strongly the images guide the video. 1 = full influence.</small></label>
        <label>Reference model strength (LoRA)<input aria-describedby="msr-lora-help" type="number" min="0.1" max="1.5" step=".05" value={params.lora_strength} onChange={e=>change('lora_strength',Number(e.target.value))}/><small id="msr-lora-help">The LoRA is the add-on that lets LTX use multiple reference images. This scales its effect on the model; 1 is the default.</small></label><label>Frames used per reference<select aria-label="Reference frames" aria-describedby="msr-reference-frames-help" value={params.reference_frames} onChange={e=>change('reference_frames',Number(e.target.value) as 25|33)}><option value="33">33 frames</option><option value="25">25 frames</option></select><small id="msr-reference-frames-help">Frames used to represent each still image. This does not change the output duration.</small></label>
        <label className="checkbox-label"><input type="checkbox" checked={params.audio} onChange={e=>change('audio',e.target.checked)}/>Generate audio</label><label className="checkbox-label"><input type="checkbox" checked={params.refine} onChange={e=>change('refine',e.target.checked)}/>Double width and height (2×)</label>
        {params.refine?<p className="msr-wide">Final output: {params.width*2} × {params.height*2} px. Adds 3 processing steps and uses more GPU memory.</p>:null}</>}
        <label className="checkbox-label"><input type="checkbox" checked={params.randomize_seed} onChange={e=>change('randomize_seed',e.target.checked)}/>Random seed</label>{!params.randomize_seed?<label>Seed<input type="number" min="0" max="2147483647" value={params.seed} onChange={e=>change('seed',Number(e.target.value))}/></label>:null}
        <label className="msr-wide">Output folder<input value={params.output_dir} onChange={e=>change('output_dir',e.target.value)}/></label>
      </div></details>
      </div>
      <div className="vs-action">
        {missing.length?<p className="msr-download-note">Required model files download automatically when you add a job.{h3?null:<> <a href="https://huggingface.co/Lightricks/LTX-2.5" target="_blank" rel="noreferrer">LTX model access</a></>}</p>:null}
        {error?<div role="alert" className="message error"><span>{error}</span><button className="icon-button" onClick={()=>setError('')} aria-label="Dismiss error"><X size={16}/></button></div>:null}{notice?<p role="status" className="notice">{notice}</p>:null}
        <div className="vs-action-row"><span className="vs-summary">{params.width*(params.refine?2:1)} × {params.height*(params.refine?2:1)} · {(params.frames/params.fps).toFixed(1)} seconds<small>{models?.remote?'RunPod GPU':'Local GPU'} · {totalSteps} steps</small></span>{running?<button className="text-button" onClick={()=>api(`/api/jobs/${running.id}/cancel`,{method:'POST'}).then(onQueued).catch(e=>setError(e.message))}>{running.status==='queued'?'Remove from queue':'Stop this job'}</button>:null}<button className="button primary vs-submit" disabled={!ready||submitting||uploading} onClick={()=>generate()}>{submitting||uploading?<Loader2 className="spin" size={17}/>:<Sparkles size={17}/>}Add to queue</button></div>
      </div>
    </div>
    <section className="panel vs-result msr-viewer"><div className="section-heading"><h2>{running?.preview_url?'Live preview':'Result'}</h2><span>{running?'In progress':result?'MP4':selected?familyLabel(selected):''}</span></div>
      <div className="msr-result vs-stage">{running?<div className="msr-draft">{running.preview_url?<img src={`${running.preview_url}?v=${running.preview_updated_at}`} alt="Live generation preview"/>:<><Film size={38}/><p>{running.stage==='downloading_models'?'Downloading required model files':running.status==='queued'?'Waiting in queue':running.preview_unavailable?'Preview is unavailable for this run.':'The preview appears after the first generation step.'}</p></>}</div>:result?<video src={result.video_url} poster={result.poster_url} controls playsInline/>:<div className="msr-draft"><Film size={38}/><strong>Your video will appear here</strong><p>Add references and describe their action.</p></div>}</div>
      {running?<VideoProgress job={running} clock={clock}/>:null}
      {!running&&result?<GenerationTiming job={result} models={models}/>:null}
      {!running&&result?<div className="result-actions"><SendVideoTo jobId={result.id} onSend={onSendVideo}/><a className="button secondary" href={`${result.video_url}?download=true`}><Download size={15}/>Download video</a></div>:null}
      {job?.status==='failed'?<div role="alert" className="message error">{job.message}</div>:null}
      {history.length?<details className="disclosure vs-history"><summary>Recent videos <span>{history.length}</span></summary><div className="vs-history-grid">{history.slice(0,12).map(item=><button key={item.id} onClick={()=>onView(item)}><img src={item.poster_url} alt=""/>{item.request.reference_images?.[0]?<img className="msr-source-badge" src={inputImage(item.request.reference_images[0]).url} alt="Source reference"/>:null}<span>{item.request.prompt}</span></button>)}</div></details>:null}
    </section>
    </div>
  </main>;
}
