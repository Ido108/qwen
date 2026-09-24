import React, {useCallback, useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {UploadCloud, Image as ImageIcon, Download, X, ChevronDown, ChevronLeft, ChevronRight, Sparkles, FolderOpen, Expand, Copy, Check, RotateCcw, CircleHelp, Loader2, Plus, ArrowUpRight, Settings} from 'lucide-react';
import {ModelSettings,type ModelState} from './ModelSettings';
import {VideoTab} from './VideoTab';
import {ReferenceTab} from './ReferenceTab';
import {UpscaleTab} from './UpscaleTab';
import {ImageSettings,type ImageOptions} from './ImageSettings';
import {Gallery,inputImage} from './Gallery';
import {JobQueue,useQueue,type SavedJob} from './JobQueue';
import {promptSettingsCopy,type EnhancementState} from './PromptSettings';
import {GenerationTiming,type Timing} from './GenerationTiming';
import {en,type TextKey} from './i18n';
import './style.css';
import './workspace.css';
import './reference.css';

type InputImage = {id:string; name:string; url:string};
type Params = ImageOptions & {model_id:string;text_encoder:string;prompt:string; images:string[]; enhance:boolean; seed:number; randomize_seed:boolean; negative_prompt:string; output_dir:string};
type Job = Timing & {id:string; status:string; stage:TextKey; message:string; seed:number; step:number; total_steps:number; request:Params; rewritten_prompt:string; created_at:number; started_at?:number; stage_started_at?:number; finished_at?:number; elapsed_seconds?:number; image_url?:string; width?:number; height?:number; tokens?:number; has_transparency?:boolean};
type Example = {id:string; title:string; title_en:string; prompt:string; prompt_extend:boolean; inputs:string[]; outputs:string[]; category:string};
type Config = {token:string; models:Record<string,boolean>; output_dir:string; active_job:string|null; device:string};
const defaults:Params = {model_id:'gguf-q8_0',text_encoder:'auto',prompt:'',images:[],enhance:true,custom_size:false,auto_resolution:2048,width:2688,height:1536,seed:0,randomize_seed:true,negative_prompt:'',output_dir:'',steps:null};
function loadDraft():Params {try {const saved=JSON.parse(localStorage.getItem('image-studio-draft-v1')||'{}');return {...defaults,...saved,steps:saved.auto_resolution===undefined&&saved.steps===40?null:(saved.steps??null)};} catch {return defaults;}}
function App(){
  const [tab,setTab]=useState<'image'|'video'|'reference'|'upscale'|'gallery'>(()=>location.hash==='#gallery'?'gallery':location.hash==='#video'?'video':location.hash==='#reference'?'reference':location.hash==='#upscale'?'upscale':'image');
  const [queueOpen,setQueueOpen]=useState(false);
  const [videoRestore,setVideoRestore]=useState<{job:SavedJob;key:number}|null>(null);
  const [referenceRestore,setReferenceRestore]=useState<{job:SavedJob;key:number}|null>(null);
  const [upscaleRestore,setUpscaleRestore]=useState<{job:SavedJob;key:number}|null>(null);
  const [galleryView,setGalleryView]=useState<{job:SavedJob;key:number}|null>(null);
  const [videoBusy,setVideoBusy]=useState(false);
  const [videoModel,setVideoModel]=useState(()=>{try{return JSON.parse(localStorage.getItem('qwen-video-draft')||'{}').model_id||'fasth3-q4_k_m';}catch{return 'fasth3-q4_k_m';}});
  const lang='en';
  const t=(key:TextKey)=>en[key];
  const [params,setParams]=useState<Params>(loadDraft);
  const [inputs,setInputs]=useState<InputImage[]>(()=>loadDraft().images.map(inputImage));
  const [config,setConfig]=useState<Config|null>(null);
  const [modelState,setModelState]=useState<ModelState|null>(null);
  const [showSettings,setShowSettings]=useState(()=>window.location.hash==='#settings');
  const [enhancement,setEnhancement]=useState<EnhancementState|null>(null);
  const [examples,setExamples]=useState<Example[]>([]);
  const [history,setHistory]=useState<Job[]>([]);
  const [job,setJob]=useState<Job|null>(null);
  const [result,setResult]=useState<Job|null>(null);
  const [pollId,setPollId]=useState<string|null>(null);
  const [error,setError]=useState(''); const [notice,setNotice]=useState('');
  const [submitting,setSubmitting]=useState(false); const [uploading,setUploading]=useState(false);
  const [allExamples,setAllExamples]=useState(false); const [activeExample,setActiveExample]=useState<Example|null>(null);
  const [lightbox,setLightbox]=useState<string|null>(null); const [showHelp,setShowHelp]=useState(false);
  const [copied,setCopied]=useState(false); const [dragging,setDragging]=useState(false); const [clock,setClock]=useState(Date.now());
  const uploadRef=useRef<HTMLInputElement>(null); const promptRef=useRef<HTMLTextAreaElement>(null);
  const token=useRef(''); const modal=useRef<HTMLDialogElement>(null);
  const busy=!!pollId; const locked=submitting||uploading;
  const chosenModel=modelState?.models.find(m=>m.id===params.model_id);
  const validSteps=params.steps===null||(Number.isInteger(params.steps)&&params.steps>=1&&params.steps<=80);
  const chosenEncoder=params.text_encoder==='auto'?(chosenModel?.format==='GGUF'?'encoder-int8':'encoder-bf16'):params.text_encoder;
  const requiredIds=[params.model_id,chosenEncoder,'vae'];
  const imageReady=modelState?.remote?!!chosenModel?.ready:requiredIds.every(id=>modelState?.models.find(m=>m.id===id)?.installed);
  const ready=imageReady&&!!chosenModel?.generation_defaults&&validSteps&&(!params.enhance||!!enhancement?.connected);
  const missingModels=modelState?.remote?[]:requiredIds.map(id=>modelState?.models.find(m=>m.id===id)).filter(m=>m&&!m.installed);
  const change=<K extends keyof Params>(key:K,value:Params[K])=>setParams(p=>({...p,[key]:value}));
  const api=useCallback(async (path:string,init:RequestInit={})=>{
    const response=await fetch(path,{...init,headers:{...init.headers,'X-Studio-Token':token.current}});
    if(!response.ok){const data=await response.json().catch(()=>({detail:response.statusText})); throw new Error(typeof data.detail==='string'?data.detail:(data.detail||[]).map((e:{msg:string})=>e.msg).join(' '));}
    return response.json();
  },[]);
  const queue=useQueue(api,!!config);
  function restoreJob(saved:SavedJob){
    if(saved.kind==='video'){setVideoRestore({job:saved,key:Date.now()});setVideoModel(saved.request.model_id);setTab('video');window.history.replaceState(null,'','#video');}
    else if(saved.kind==='reference-video'){setReferenceRestore({job:saved,key:Date.now()});setTab('reference');window.history.replaceState(null,'','#reference');}
    else if(saved.kind==='upscale'){setUpscaleRestore({job:saved,key:Date.now()});setTab('upscale');window.history.replaceState(null,'','#upscale');}
    else{setParams({...defaults,...saved.request,seed:saved.seed,randomize_seed:false});setInputs((saved.request.images||[]).map(inputImage));if(saved.status==='succeeded')setResult(saved as Job);setTab('image');window.history.replaceState(null,'','#images');setNotice('Original prompt, input images, seed and settings loaded.');setError('');}
  }
  function viewJob(saved:SavedJob){setGalleryView({job:saved,key:Date.now()});setTab('gallery');window.history.replaceState(null,'','#gallery');}
  useEffect(()=>{let alive=true; (async()=>{try{
    const [cfg,ex,old]=await Promise.all([api('/api/config'),api('/api/examples'),api('/api/history')]);
    if(!alive)return; token.current=cfg.token; setConfig(cfg);setModelState(cfg.catalog);setEnhancement(cfg.enhancement); setExamples(ex); setHistory(old);setResult(old[0]||null);
    setParams(p=>({...p,output_dir:p.output_dir||cfg.output_dir}));
    if(tab==='image'&&!cfg.catalog.models.some((model:{kind:string;ready:boolean})=>model.kind==='image'&&model.ready))setShowSettings(true);
    if(cfg.active_job){
      const active:Job=await api(`/api/jobs/${cfg.active_job}`);if(!alive)return;
      if((active as Job&{kind?:string}).kind&&((active as Job&{kind?:string}).kind!=='image'))return;
      setJob(active);setPollId(active.id);
    }
  }catch(e){if(alive)setError(String(e));}})(); return()=>{alive=false;};},[]);
  useEffect(()=>{localStorage.setItem('image-studio-draft-v1',JSON.stringify({...params,images:inputs.map(i=>i.id)}));},[params,inputs]);
  useEffect(()=>{document.documentElement.lang=lang;document.documentElement.dir='ltr'; localStorage.setItem('image-studio-language',lang);},[lang]);
  useEffect(()=>{if(!pollId)return; let alive=true;let timeout:number;
    async function poll(){try{const current:Job=await api(`/api/jobs/${pollId}`);if(!alive)return;setJob(current);setNotice('');
      if(['succeeded','failed','cancelled'].includes(current.status)){setPollId(null);
        if(current.status==='succeeded'){setResult(current);setHistory(await api('/api/history'));setError('');}
        else if(current.status==='failed')setError(current.message); return;
      }
    }catch{if(alive)setNotice(t('offline'));} if(alive)timeout=window.setTimeout(poll,1200);}
    poll();return()=>{alive=false;window.clearTimeout(timeout);};
  },[pollId,lang]);
  useEffect(()=>{if(showSettings)return;let alive=true;let timer:number;
    async function refresh(){try{const [next,connection]=await Promise.all([api('/api/models'),api('/api/oauth/antigravity/status')]);if(alive){setModelState(next);setEnhancement(connection);}}catch{}if(alive)timer=window.setTimeout(refresh,modelState?.transfer.status==='running'?1500:5000);}
    timer=window.setTimeout(refresh,1500);return()=>{alive=false;clearTimeout(timer);};
  },[showSettings,modelState?.transfer.status,api]);
  useEffect(()=>{if(!busy)return;const timer=setInterval(()=>setClock(Date.now()),1000);return()=>clearInterval(timer);},[busy]);
  useEffect(()=>{if(lightbox||activeExample||showHelp||showSettings){if(!modal.current?.open)modal.current?.showModal();}else modal.current?.close();},[lightbox,activeExample,showHelp,showSettings]);
  async function upload(files:File[]){
    if(!files.length||locked)return;if(files.length+inputs.length>10){setError(t('maxImages'));return;}
    setUploading(true);setError('');try{const data=new FormData();files.forEach(f=>data.append('files',f));const uploaded=await api('/api/uploads',{method:'POST',body:data});setInputs(previous=>[...previous,...uploaded]);}
    catch(e){setError((e as Error).message);}finally{setUploading(false);if(uploadRef.current)uploadRef.current.value='';}
  }
  async function generate(){
    if(!validSteps){setError('Enter a whole number of steps from 1 to 80.');return;}
    if(locked)return;if(!imageReady){setError(t('modelsMissing'));return;}if(params.enhance&&!enhancement?.connected){setShowSettings(true);return;}if(!params.prompt.trim()){setError(t('enterPrompt'));promptRef.current?.focus();return;}
    setSubmitting(true);setError('');setNotice('');setJob(null);
    try{const created=await api('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...params,images:inputs.map(i=>i.id)})});setJob(created);setPollId(created.id);setNotice('Added to queue. You can keep editing and send another request.');queue.refresh();}
    catch(e){setError((e as Error).message);}finally{setSubmitting(false);}
  }
  function reorder(index:number,delta:number){setInputs(old=>{const next=[...old];[next[index],next[index+delta]]=[next[index+delta],next[index]];return next;});}
  function applyExample(example:Example){setParams(p=>({...p,prompt:example.prompt,enhance:example.prompt_extend}));setInputs(example.inputs.map((name)=>({id:`example:${name}`,name,url:`/examples/${name}`})));setActiveExample(null);setJob(null);setNotice(t('promptLoaded'));promptRef.current?.focus();}
  function closeModal(){setLightbox(null);setActiveExample(null);setShowHelp(false);setShowSettings(false);}
  async function copyPrompt(text:string){try{await navigator.clipboard.writeText(text);setCopied(true);setTimeout(()=>setCopied(false),1600);}catch(e){setError(String(e));}}
  const stageElapsed=job?.stage_started_at?Math.max(0,Math.floor((clock-job.stage_started_at*1000)/1000)):0;
  const elapsed=job?.started_at ? Math.max(0,Math.round(((job.finished_at?job.finished_at*1000:clock)-job.started_at*1000)/1000)):0;
  const featured=[examples.find(e=>e.title_en==='Transparent anime bride'),examples.find(e=>e.inputs.length>0),examples.find(e=>e.title_en==='Product photography')].filter(Boolean) as Example[];
  const displayed=allExamples?examples:(featured.length===3?featured:examples.slice(0,3));
  return <>
    <header className="header"><div className="brand"><img src="/branding/icon-64.png" alt=""/><strong>Qwen</strong><span className="brand-divider"/><nav className="app-tabs" aria-label="Generation type"><button aria-current={tab==='image'?'page':undefined} onClick={()=>{setTab('image');window.history.replaceState(null,'','#images');}}><ImageIcon size={17}/>Images</button><button aria-current={tab==='video'?'page':undefined} onClick={()=>{setTab('video');window.history.replaceState(null,'','#video');}}><span className="video-tab-icon" aria-hidden="true">▷</span>Video{videoBusy?<Loader2 className="spin" size={13}/>:null}</button><button aria-current={tab==='reference'?'page':undefined} onClick={()=>{setTab('reference');window.history.replaceState(null,'','#reference');}}>Video to video</button><button aria-current={tab==='upscale'?'page':undefined} onClick={()=>{setTab('upscale');window.history.replaceState(null,'','#upscale');}}>Upscale</button><button aria-current={tab==='gallery'?'page':undefined} onClick={()=>{setTab('gallery');window.history.replaceState(null,'','#gallery');}}>Gallery</button></nav><span className="brand-divider model-divider"/>{tab==='image'?<button className="model-name model-choice" dir="ltr" onClick={()=>setShowSettings(true)} aria-label={t('chooseModel')}>Qwen-Image-2.1 <span>{chosenModel?.title.split(' · ').at(-1)||'Q8_0'}</span><ChevronDown size={13}/></button>:tab==='video'?<select className="video-model-choice" aria-label="Video model" value={videoModel} disabled={submitting} onChange={e=>setVideoModel(e.target.value)}>{modelState?.models.filter(m=>m.kind==='video').map(m=><option key={m.id} value={m.id}>{m.title} · {m.modes?.includes('image')?'Text + images':'Text only'}{m.ready?'':' · Download needed'}</option>)}</select>:null}</div>
      <div className="header-tools"><button className="queue-header-button" onClick={()=>setQueueOpen(true)} aria-label={`Open queue (${queue.count})`}>Queue <span>{queue.count}</span></button><button className="icon-button" onClick={()=>setShowSettings(true)} title={t('settings')} aria-label={t('settings')}><Settings size={19}/></button><button className="icon-button" onClick={()=>setShowHelp(true)} title={t('help')} aria-label={t('help')}><CircleHelp size={19}/></button></div>
    </header>
    <JobQueue api={api} state={queue.state} error={queue.error} refresh={queue.refresh} open={queueOpen} onOpen={setQueueOpen} onRestore={restoreJob} onView={viewJob}/>
    <main hidden={tab!=='image'}>

      {config&&!imageReady?<div className="message warning"><span>{t('modelsMissing')}<small className="missing-models" dir="auto">{missingModels.map(m=>m!.title).join(' · ')}</small></span><button className="text-button" onClick={()=>setShowSettings(true)}>{t('settings')}</button></div>:null}
      <div className="image-panels">
        <section className="panel input-panel"><div className="section-heading"><h2>{t('input')}</h2>{inputs.length>0?<><span className="count">{inputs.length}/10</span><button className="text-button" disabled={locked} onClick={()=>setInputs([])}>{t('clear')}</button></>:null}</div>
          <div className={`upload-area ${dragging?'dragging':''} ${inputs.length?'has-images':''}`} onDragOver={e=>{e.preventDefault();if(!locked)setDragging(true);}} onDragLeave={()=>setDragging(false)} onDrop={e=>{e.preventDefault();setDragging(false);upload(Array.from(e.dataTransfer.files));}}>
            <input ref={uploadRef} type="file" accept="image/png,image/jpeg,image/webp" multiple hidden onChange={e=>upload(Array.from(e.target.files||[]))}/>
            {inputs.length===0?<button className="upload-empty" disabled={locked} onClick={()=>uploadRef.current?.click()}><UploadCloud size={42} strokeWidth={1.5}/><strong>{uploading?t('uploadBusy'):t('drop')}</strong><span>{t('uploadHint')}</span><span className="select-link">{t('upload')}</span></button>:<div className="input-grid">{inputs.map((input,index)=><div className="input-thumb" key={input.id}><button className="thumb-preview checker" onClick={()=>setLightbox(input.url)} aria-label={`${t('preview')} ${index+1}`}><img src={input.url} alt={`${index+1}. ${input.name}`}/></button><span className="image-number">{index+1}</span><button disabled={locked} className="remove-image" onClick={()=>setInputs(old=>old.filter((_,i)=>i!==index))} aria-label={`${t('remove')} ${index+1}`}><X size={13}/></button><div className="reorder"><button disabled={locked||index===0} onClick={()=>reorder(index,-1)} aria-label={`${t('left')} ${index+1}`}><ChevronLeft size={14}/></button><button disabled={locked||index===inputs.length-1} onClick={()=>reorder(index,1)} aria-label={`${t('right')} ${index+1}`}><ChevronRight size={14}/></button></div></div>)}{inputs.length<10?<button className="add-image" disabled={locked} onClick={()=>uploadRef.current?.click()} aria-label={t('upload')}><Plus size={25}/></button>:null}</div>}
          </div>
        </section>
        <section className="panel result-panel"><div className="section-heading"><h2>{t('result')}</h2>{result?<span className="result-size" dir="ltr">{result.width} × {result.height} · PNG</span>:null}{result?<button className="icon-button" onClick={()=>setLightbox(result.image_url!)} aria-label={t('preview')}><Expand size={17}/></button>:null}</div>
          <div className="result-canvas checker">{result?<button className="result-image-button" onClick={()=>setLightbox(result.image_url!)} aria-label={t('preview')}><img className="result-image" src={result.image_url} alt={result.request.prompt}/></button>:<div className="result-empty"><ImageIcon size={42} strokeWidth={1.4}/><span>{t('empty')}</span></div>}
            {busy?<div className="generation-overlay" aria-live="polite"><Loader2 className="spin" size={28}/><strong>{job?.status==='queued'?'Waiting in queue':job?job.message||t(job.stage):t('starting')}</strong>{job?.stage==='generating'?<><progress value={job.step} max={job.total_steps}/><span>{job.step} / {job.total_steps} · {elapsed} {t('seconds')}</span></>:<span>{elapsed>0?`${stageElapsed}s this step ? ${elapsed}s total`:''}{job?.stage==='rewriting'&&job.tokens?` · ${job.tokens} ${t('tokens')}`:''}</span>}</div>:null}
          </div>
          {result?<div className="result-actions"><a className="button secondary" href={`${result.image_url}?download=true`}><Download size={15}/>{t('download')}</a><button disabled={locked} className="text-button" onClick={async()=>{try{setInputs([await api(`/api/jobs/${result.id}/reuse`,{method:'POST'})]);promptRef.current?.focus();}catch(e){setError(String(e));}}}><ArrowUpRight size={16}/>{t('editResult')}</button><a className="icon-button" href={`/api/jobs/${result.id}/metadata`} aria-label={t('metadata')} title={t('metadata')}><Download size={15}/></a></div>:null}{result?<GenerationTiming job={result}/>:null}
        </section>
      </div>
      <section className="panel controls-panel">
        <label htmlFor="prompt" className="prompt-label">{t('prompt')}<span className="mode-label">{inputs.length?t('editMode'):t('createMode')}</span></label>
        <textarea ref={promptRef} id="prompt" dir="auto" className="prompt-input" placeholder={t('placeholder')} value={params.prompt} disabled={locked} onChange={e=>change('prompt',e.target.value)} onPaste={e=>{const files=Array.from(e.clipboardData.files);if(files.length){e.preventDefault();upload(files);}}} onKeyDown={e=>{if((e.ctrlKey||e.metaKey)&&e.key==='Enter'){e.preventDefault();generate();}}}/>
        <ImageSettings value={params} model={chosenModel} disabled={locked} onChange={values=>setParams(p=>({...p,...values}))}/>
        <div className="generate-row"><label className="toggle-label"><input type="checkbox" role="switch" checked={params.enhance} disabled={locked} onChange={e=>{change('enhance',e.target.checked);setJob(null);}}/><span className="switch"/><span>{t('enhance')}</span></label><div className="generate-controls"><span className="keyboard-hint">{t('keyboard')}</span>{busy?<button className="button secondary cancel" onClick={async()=>{try{if(pollId)await api(`/api/jobs/${pollId}/cancel`,{method:'POST'});}catch(e){setError(String(e));}}}>{t('cancel')}</button>:null}<button className="button primary" disabled={locked||!ready} onClick={generate}>{submitting?<Loader2 size={20} className="spin"/>:<Sparkles size={20}/>}<span>{submitting?'Adding\u2026':queue.count||queue.state.paused?'Add image to queue':t('generate')}</span></button></div></div>
        {params.enhance&&!enhancement?.connected?<div className="connection-hint"><span>{promptSettingsCopy[lang].description}</span><button className="text-button" onClick={()=>setShowSettings(true)}>{promptSettingsCopy[lang].connect}<ArrowUpRight size={15}/></button></div>:null}
        {error?<div className="message error" role="alert"><span>{error}</span><button className="icon-button" aria-label={t('close')} onClick={()=>setError('')}><X size={16}/></button></div>:null}
        {notice?<div className="notice" role="status">{notice}</div>:null}
        {job?.status==='cancelled'?<div className="notice">{t('cancelled')}</div>:null}
        <details className="disclosure" key={`rewrite-${Boolean(job?.rewritten_prompt)}`} open={!!job?.rewritten_prompt}><summary><ChevronRight size={15}/>{t('rewritten')}</summary><div className="details-body"><p className="rewritten-text" dir="auto">{job?.rewritten_prompt||t('rewriteEmpty')}</p>{job?.rewritten_prompt?<button className="text-button" onClick={()=>copyPrompt(job.rewritten_prompt)}>{copied?<Check size={15}/>:<Copy size={15}/>} {copied?t('copied'):t('copy')}</button>:null}</div></details>
        <details className="disclosure"><summary><ChevronRight size={15}/>{t('advanced')}<span className="summary-hint">{params.randomize_seed?'Random seed':`${t('seed')}: ${params.seed}`} · {params.custom_size?`${params.width} × ${params.height}`:`Auto ${params.auto_resolution===1024?'1K':'2K'}`} · {params.steps??chosenModel?.generation_defaults?.steps??'…'} steps</span></summary>
          <fieldset className="details-body settings" disabled={locked}>
            <div className="field"><label htmlFor="seed">{t('seed')}</label><input id="seed" type="number" min="0" max="2147483647" step="1" value={params.seed} disabled={params.randomize_seed} onChange={e=>change('seed',Number(e.target.value))}/></div>
            <label className="checkbox-label align-field"><input type="checkbox" checked={params.randomize_seed} onChange={e=>change('randomize_seed',e.target.checked)}/>{t('randomize')}</label>
            <div className="field span-2"><label htmlFor="negative">{t('negative')}</label><textarea id="negative" dir="auto" rows={2} placeholder={t('negativeHint')} value={params.negative_prompt} onChange={e=>change('negative_prompt',e.target.value)}/>{params.negative_prompt.trim()?<small>{t('negativeHelp')}</small>:null}</div>
            <div className="field span-2"><label htmlFor="folder">{t('folder')}</label><input id="folder" dir="ltr" value={params.output_dir} onChange={e=>change('output_dir',e.target.value)}/></div>
          </fieldset>
        </details>
        <section className="examples"><div className="section-heading"><h3>{t('examples')}</h3><button className="text-button" onClick={()=>setAllExamples(!allExamples)}>{allExamples?t('fewerExamples'):t('allExamples')}<ChevronDown className={allExamples?'rotate':''} size={15}/></button></div><div className="examples-grid">{displayed.map(example=><button key={example.id} className="example-card" onClick={()=>setActiveExample(example)}><div className="example-thumbnail checker"><img loading="lazy" src={`/examples/${example.outputs[0]||example.inputs[0]}`} alt=""/></div><span>{example.title_en}<small>{example.inputs.length?`${example.inputs.length} ${t('sourceCount')}`:t('createMode')}</small></span><ChevronRight size={16}/></button>)}</div></section>
        <details className="history disclosure"><summary><ChevronRight size={15}/>{t('history')}<span className="summary-hint">{history.length}</span><button className="text-button folder-button" onClick={e=>{e.preventDefault();api('/api/open-output',{method:'POST'}).catch(err=>setError(String(err)));}}><FolderOpen size={15}/>{t('openFolder')}</button></summary><div className="details-body"><div className="history-grid">{history.map(item=><button className={`history-item ${result?.id===item.id?'selected':''}`} key={item.id} onClick={()=>{setResult(item);if(!busy)setJob(item);}}><img src={item.image_url} alt={item.request.prompt} loading="lazy"/><span>{item.request.prompt}</span></button>)}</div>{!history.length?<p>{t('noHistory')}</p>:null}{result?<button disabled={locked} className="text-button reuse-settings" onClick={()=>{setParams({...defaults,...result.request,randomize_seed:false});setInputs(result.request.images.map(id=>({id,name:id,url:id.startsWith('example:')?`/examples/${id.slice(8)}`:`/api/uploads/${id}`})));setNotice(t('reuseSettings'));}}><RotateCcw size={15}/>{t('reuseSettings')}</button>:null}</div></details>
      </section>
      <footer><span className="local-status"><i/>{modelState?.remote?'Runs on RunPod GPU':t('local')}</span><span>{modelState?.remote?'Results saved on this computer':config?.device||''}</span></footer>
    </main>
    <VideoTab visible={tab==='video'} api={api} models={modelState} enhancement={enhancement} modelId={videoModel} onModel={setVideoModel} onBusy={setVideoBusy} queueActive={queue.count>0||queue.state.paused} onQueued={queue.refresh} restoreRequest={videoRestore} openSettings={()=>setShowSettings(true)}/>
    <ReferenceTab visible={tab==='reference'} api={api} token={token.current} models={modelState} enhancement={enhancement} restoreRequest={referenceRestore} onQueued={queue.refresh} openSettings={()=>setShowSettings(true)}/>
    <UpscaleTab visible={tab==='upscale'} api={api} token={token.current} models={modelState} restoreRequest={upscaleRestore} onQueued={queue.refresh} openSettings={()=>setShowSettings(true)}/>
    <Gallery api={api} visible={tab==='gallery'} revision={queue.revision} models={modelState} onRestore={restoreJob} onQueued={queue.refresh} view={galleryView}/>
    <dialog ref={modal} aria-label={showSettings?t('settings'):showHelp?t('help'):activeExample?.title_en||t('preview')} onCancel={closeModal} onClick={e=>{if(e.target===modal.current)closeModal();}} className={lightbox?'lightbox':'content-dialog'}><button className="dialog-close icon-button" onClick={closeModal} aria-label={t('close')}><X size={21}/></button>
      {showSettings?<ModelSettings enhancement={enhancement} onEnhancement={setEnhancement} lang={lang} api={api} state={modelState} onState={setModelState} selected={params.model_id} onSelect={id=>{setParams(p=>({...p,model_id:id,steps:p.model_id===id?p.steps:null}));setError('');}} videoSelected={videoModel} onVideoSelect={setVideoModel} encoder={params.text_encoder} onEncoder={id=>change('text_encoder',id)} busy={queue.count>0}/>:null}
      {lightbox?<img className="lightbox-image checker" src={lightbox} alt={t('preview')}/>:null}
      {showHelp&&tab==='video'?<div className="help-content"><h2>Video help</h2><p>Describe the scene, movement, camera and sounds. FastH3 creates video from text. Choose MiniMax H3 to animate an image or move between a first and last frame.</p><p>Use Standard for more detail or Turbo for faster results. Duration follows the supported model frame grid at 24 fps. You can change the seed, size and output folder in Advanced settings.</p><p>Download models in Settings. Required encoders and VAEs are selected automatically. Enhance prompt uses your Antigravity connection.</p><a href="https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2" target="_blank" rel="noreferrer">FastH3 model details</a><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3" target="_blank" rel="noreferrer">MiniMax H3 model details</a></div>:null}
      {showHelp&&tab==='image'?<div className="help-content"><h2>{t('help')}</h2><p>{t('helpText')}</p><blockquote dir="auto">{t('transparent')}</blockquote><button className="button primary" disabled={locked} onClick={()=>{change('prompt',t('transparent'));setShowHelp(false);promptRef.current?.focus();}}>{t('applyTransparent')}</button><a href="https://huggingface.co/spaces/Qwen/Qwen-Image-2.1" target="_blank" rel="noreferrer">Qwen-Image-2.1 · Hugging Face</a></div>:null}
      {activeExample?<div className="example-dialog"><h2>{activeExample.title_en}</h2><p className="muted">{t('exampleSource')}</p>{activeExample.inputs.length?<><h3>{t('input')}</h3><div className="example-inputs">{activeExample.inputs.map((name,i)=><figure key={name}><img src={`/examples/${name}`} alt={`${i+1}`}/><figcaption>{i+1}</figcaption></figure>)}</div></>:null}<h3>{t('reference')}</h3><div className="example-outputs">{activeExample.outputs.map(name=><img key={name} src={`/examples/${name}`} alt={t('reference')} className="checker"/>)}</div><h3>{t('fullPrompt')}</h3><p className="example-prompt" dir="auto">{activeExample.prompt}</p><button className="button primary" disabled={locked} onClick={()=>applyExample(activeExample)}>{t('useExample')}</button></div>:null}
    </dialog>
  </>;
}
createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
