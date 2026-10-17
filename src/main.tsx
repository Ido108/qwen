import {DownloadPill} from './DownloadPill';
import {PromptHistory,rememberPrompt} from './PromptHistory';
import {ModelDownloadProgress} from './ModelDownloadProgress';
import React, {useCallback, useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {UploadCloud, Image as ImageIcon, Download, X, ChevronDown, ChevronLeft, ChevronRight, Sparkles, Expand, Copy, Check, CircleHelp, SquareDashed, Compass, Loader2, Plus, ArrowUpRight, Settings, Images, LayoutGrid} from 'lucide-react';
import {ModelSettings,type ModelState} from './ModelSettings';
import {VideoTab} from './VideoTab';
import {MultiReferenceTab} from './MultiReferenceTab';
import {ReferenceTab} from './ReferenceTab';
import {UpscaleTab} from './UpscaleTab';
import {ImageSettings,toolPrompt,toolStrength,type ImageOptions,type ToolState} from './ImageSettings';
import {Gallery,inputImage} from './Gallery';
import {InputLibrary,type Person} from './InputLibrary';
import {CompareView} from './CompareView';
import {BoxEditor} from './BoxEditor';
import {ZoomView} from './ZoomView';
import {CivitaiPage,type InstalledItem} from './CivitaiPage';
import {TrainPage} from './TrainPage';
import {LoraStack,type LoraChoice} from './LoraStack';
import {LibraryPopover} from './LibraryPopover';
import {ResultsRail} from './ResultsRail';
import {JobQueue,useQueue,type SavedJob} from './JobQueue';
import {promptSettingsCopy,type EnhancementState} from './PromptSettings';
import type {SendVideo,VideoInput} from './SendVideoTo';
import type {UploadedVideo} from './mediaUpload';
import {GenerationTiming,type Timing} from './GenerationTiming';
import {en,type TextKey} from './i18n';
import './style.css';
import './workspace.css';
import './reference.css';
import './reference-workbench.css';
import './studio.css';
import {apiRequest} from './session';
import {formatDuration} from './duration';

type InputImage = {id:string; name:string; url:string};
type Params = ImageOptions & ToolState & {loras:LoraChoice[];vae_id:string;vae_choice?:boolean;model_id:string;text_encoder:string;prompt:string; images:string[]; enhance:boolean; seed:number; randomize_seed:boolean; negative_prompt:string; output_dir:string; denoise?:number|null};
type Job = Timing & {download_completed_bytes?:number;download_total_bytes?:number;id:string; status:string; stage:TextKey; message:string; seed:number; step:number; total_steps:number; request:Params; rewritten_prompt:string; created_at:number; started_at?:number; stage_started_at?:number; finished_at?:number; elapsed_seconds?:number; image_url?:string; width?:number; height?:number; tokens?:number; has_transparency?:boolean};
type Example = {id:string; title:string; title_en:string; prompt:string; prompt_extend:boolean; inputs:string[]; outputs:string[]; category:string};
type Config = {token:string; models:Record<string,boolean>; output_dir:string; active_job:string|null; device:string};
type VideoSection = 'create'|'reference'|'multi-reference'|'upscale';
const defaults:Params = {loras:[],tool_id:null,tool_options:{},lora_strength:null,vae_id:'auto',model_id:'gguf-q8_0',text_encoder:'auto',prompt:'',images:[],enhance:true,custom_size:false,auto_resolution:2048,width:2688,height:1536,seed:0,randomize_seed:true,negative_prompt:'',output_dir:'',steps:null};
function loadDraft():Params {try {const saved=JSON.parse(localStorage.getItem('image-studio-draft-v1')||'{}');return {...defaults,...saved,randomize_seed:true,vae_id:saved.vae_choice?saved.vae_id:'auto',steps:saved.auto_resolution===undefined&&saved.steps===40?null:(saved.steps??null)};} catch {return defaults;}}
const clearedResult=()=>{try{return localStorage.getItem('qwen-result-cleared');}catch{return null;}};
function App(){
  const [tab,setTab]=useState<'image'|'video'|'gallery'|'civitai'|'train'>(()=>location.hash==='#civitai'?'civitai':location.hash==='#train'?'train':location.hash==='#gallery'?'gallery':['#video','#reference','#video-to-video','#multi-reference','#upscale'].includes(location.hash)?'video':'image');
  const [videoSection,setVideoSection]=useState<VideoSection>(()=>['#reference','#video-to-video'].includes(location.hash)?'reference':location.hash==='#upscale'?'upscale':location.hash==='#multi-reference'?'multi-reference':'create');
  const [queueOpen,setQueueOpen]=useState(false);
  const [videoRestore,setVideoRestore]=useState<{job:SavedJob;key:number}|null>(null);
  const [referenceRestore,setReferenceRestore]=useState<{job:SavedJob;key:number}|null>(null);
  const [multiRestore,setMultiRestore]=useState<{job:SavedJob;key:number}|null>(null);
  const [upscaleRestore,setUpscaleRestore]=useState<{job:SavedJob;key:number}|null>(null);
  const [galleryView,setGalleryView]=useState<{job:SavedJob;key:number}|null>(null);
  const [videoInput,setVideoInput]=useState<VideoInput|null>(null);
  const [videoBusy,setVideoBusy]=useState(false);
  const [videoModel,setVideoModel]=useState(()=>{try{return JSON.parse(localStorage.getItem('qwen-video-draft')||'{}').model_id||'fasth3-q4_k_m';}catch{return 'fasth3-q4_k_m';}});
  const lang='en';
  const t=(key:TextKey)=>en[key];
  const [params,setParams]=useState<Params>(loadDraft);
  const [inputs,setInputs]=useState<InputImage[]>(()=>loadDraft().images.map(inputImage));
  const [config,setConfig]=useState<Config|null>(null);
  const [modelState,setModelState]=useState<ModelState|null>(null);
  const [showSettings,setShowSettings]=useState(()=>window.location.hash==='#settings');const [settingsView,setSettingsView]=useState<'models'|'enhance'|'location'>('models');
  const openSettings=(view:'models'|'enhance'|'location'='models')=>{setSettingsView(view);setShowSettings(true);};
  const [enhancement,setEnhancement]=useState<EnhancementState|null>(null);
  const [examples,setExamples]=useState<Example[]>([]);
  const [history,setHistory]=useState<Job[]>([]);
  const [job,setJob]=useState<Job|null>(null);
  const [result,setResult]=useState<Job|null>(null);
  const [pollId,setPollId]=useState<string|null>(null);
  const [error,setError]=useState(''); const [notice,setNotice]=useState('');
  const [submitting,setSubmitting]=useState(false); const [uploading,setUploading]=useState(false);
  const [showExamples,setShowExamples]=useState(false); const [activeExample,setActiveExample]=useState<Example|null>(null);
  const [libraryOpen,setLibraryOpen]=useState(false); const [replaceAt,setReplaceAt]=useState<number|null>(null); const closeLibrary=useCallback(()=>{setLibraryOpen(false);setReplaceAt(null);},[]);
  // A click on an input image opens the library to replace that image; + and Library open it to add.
  function openLibrary(at:number|null){setReplaceAt(at);setLibraryOpen(true);} const inputPanel=useRef<HTMLElement>(null);
  const [lightbox,setLightbox]=useState<string|null>(null); const [showHelp,setShowHelp]=useState(false);
  const [railOpen,setRailOpen]=useState(()=>window.innerWidth>=1280&&localStorage.getItem('qwen-results-rail')!=='closed');const [libraryRevision,setLibraryRevision]=useState(0);const [pendingVideoLora,setPendingVideoLora]=useState<{id:string;key:number;target:'create'|'reference'}|null>(null);
  const [people,setPeople]=useState<Person[]>([]);const [compareResult,setCompareResult]=useState(false);const [boxOpen,setBoxOpen]=useState(false);
  const [copied,setCopied]=useState(false); const [dragging,setDragging]=useState(false); const [clock,setClock]=useState(Date.now());
  const uploadRef=useRef<HTMLInputElement>(null); const promptRef=useRef<HTMLTextAreaElement>(null);
  const token=useRef(''); const modal=useRef<HTMLDialogElement>(null);
  const busy=!!pollId; const locked=submitting||uploading;
  const chosenModel=modelState?.models.find(m=>m.id===params.model_id);
  const validSteps=params.steps===null||(Number.isInteger(params.steps)&&params.steps>=1&&params.steps<=80);
  const tools=modelState?.tools||[];
  const activeTool=tools.find(t=>t.id===params.tool_id&&t.families.includes(chosenModel?.family||'qwen-image-2.1'));
  const family=chosenModel?.family||'qwen-image-2.1';const krea=family.startsWith('krea-2');const zimage=family.startsWith('z-image');
  // Krea 2 and Z-Image have their own text encoders; Qwen models pick INT8 or BF16 by the model format.
  const chosenEncoder=krea||zimage?(chosenModel?.text_encoder||(krea?'krea2-encoder-bf16':'zimage-encoder-bf16')):params.text_encoder==='auto'?(chosenModel?.format==='GGUF'?'encoder-int8':'encoder-bf16'):params.text_encoder;
  const userLoras=(modelState?.models||[]).filter(m=>m.kind==='image-lora'&&(m.source==='civitai'||m.source==='trained'));
  const activeLoras=params.loras.filter(l=>userLoras.find(u=>u.id===l.id)?.families?.includes(family));
  const vaes=(modelState?.models||[]).filter(m=>m.kind==='vae'&&(m.families||['qwen-image-2.1','qwen-image-2.1-turbo']).includes(family));const vaeTitle=(id:string)=>id==='vae'?'Standard':id==='vae-texture-fix'?'Texture fix':vaes.find(v=>v.id===id)?.title||id;
  const autoVae=chosenModel?.recommended_vae||(krea?'krea2-vae':zimage?'zimage-vae':'vae-texture-fix');const resolvedVae=params.vae_id==='auto'||!vaes.some(v=>v.id===params.vae_id)?autoVae:params.vae_id;
  const requiredIds=[params.model_id,chosenEncoder,resolvedVae,...(krea&&inputs.length?['krea2-lora-style-reference']:[]),...(activeTool?[activeTool.lora]:[]),...activeLoras.map(l=>l.id)];
  const enhanceOn=params.enhance&&!activeTool;
  const imageReady=modelState?.remote?!!chosenModel?.ready:!!chosenModel;
  const ready=imageReady&&!!chosenModel?.generation_defaults&&validSteps&&(!enhanceOn||!!enhancement?.connected);
  const missingModels=modelState?.remote?[]:requiredIds.map(id=>modelState?.models.find(m=>m.id===id)).filter(m=>m&&!m.installed);
  const change=<K extends keyof Params>(key:K,value:Params[K])=>setParams(p=>({...p,[key]:value}));
  const downloading=modelState?.transfer.status==='running'||!!modelState?.downloads?.some(d=>d.status==='running');
  const api=useCallback(apiRequest,[]);
  const queue=useQueue(api,!!config);
  function showVideo(section:VideoSection){setTab('video');setVideoSection(section);window.history.replaceState(null,'',section==='create'?'#video':section==='reference'?'#video-to-video':section==='multi-reference'?'#multi-reference':'#upscale');}
  const sendVideo:SendVideo=async(jobId,target,part='final')=>{
    const video:UploadedVideo=await api(`/api/jobs/${encodeURIComponent(jobId)}/reuse-video?part=${part}`,{method:'POST'});
    setVideoInput({video,target,key:Date.now()});
    showVideo(target==='upscale'?'upscale':'reference');
    window.scrollTo({top:0,behavior:'instant'});
  };
  function restoreJob(saved:SavedJob){
    if(saved.kind==='training'){setTab('train');window.history.replaceState(null,'','#train');return;}
    if(saved.kind==='video'){setVideoRestore({job:saved,key:Date.now()});setVideoModel(saved.request.model_id);showVideo('create');}
    else if(saved.kind==='multi-reference'){setMultiRestore({job:saved,key:Date.now()});showVideo('multi-reference');}
    else if(saved.kind==='reference-video'){setReferenceRestore({job:saved,key:Date.now()});showVideo('reference');}
    else if(saved.kind==='upscale'){setUpscaleRestore({job:saved,key:Date.now()});showVideo('upscale');}
    else{setParams({...defaults,...saved.request,seed:saved.seed,randomize_seed:true});setInputs((saved.request.images||[]).map(inputImage));if(saved.status==='succeeded')setResult(saved as Job);setTab('image');window.history.replaceState(null,'','#images');setNotice('Original prompt, input images, seed and settings loaded.');setError('');}
  }
  function applyInstalled(item:InstalledItem){
      item={...item,kind:modelState?.models.find(m=>m.id===item.id)?.kind??item.kind};
      if(item.kind==='video-lora'||item.kind==='video'||item.kind==='reference-video'){if(item.kind==='video')setVideoModel(item.id);const ref2va=item.kind==='reference-video'||(item.kind==='video-lora'&&modelState?.models.find(m=>m.id===item.id)?.made_for==='Ref2VA');setPendingVideoLora(item.kind==='video-lora'?{id:item.id,key:Date.now(),target:ref2va?'reference':'create'}:null);showVideo(ref2va?'reference':'create');setNotice('');return;}
      // A LoRA for another architecture also selects a model it works with (the installed one first).
      const lora=item.kind==='image-lora'?modelState?.models.find(m=>m.id===item.id):undefined;const families=lora?.families||[];
      const fitting=lora&&!families.includes(family)?(modelState?.models.filter(m=>m.kind==='image'&&families.includes(m.family||'qwen-image-2.1'))||[]).sort((a,b)=>Number(b.ready)-Number(a.ready))[0]:undefined;
      if(item.kind==='image-lora')setParams(p=>({...p,...(fitting?{model_id:fitting.id,steps:null,cfg:null}:{}),loras:p.loras.some(l=>l.id===item.id)?p.loras:[...p.loras,{id:item.id,strength:lora?.default_strength??1}].slice(0,4)}));
      else setParams(p=>({...p,model_id:item.id,steps:null,cfg:null}));
      setTab('image');window.history.replaceState(null,'','#images');setNotice(item.kind==='image-lora'?`LoRA added${fitting?`, and the model changed to ${fitting.title}, which the LoRA works with`:''}. Its strength is under LoRAs.`:'Model selected.');}
  function viewJob(saved:SavedJob){if(saved.kind==='training'){restoreJob(saved);return;}setGalleryView({job:saved,key:Date.now()});setTab('gallery');window.history.replaceState(null,'','#gallery');}
  useEffect(()=>{let alive=true; (async()=>{try{
    const [cfg,ex,old]=await Promise.all([api('/api/config'),api('/api/examples'),api('/api/history')]);
    if(!alive)return; token.current=cfg.token; setConfig(cfg);setModelState(cfg.catalog);setEnhancement(cfg.enhancement); setExamples(ex); setHistory(old);setResult(old[0]&&old[0].id!==clearedResult()?old[0]:null);
    setParams(p=>({...p,output_dir:p.output_dir||cfg.output_dir}));
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
    async function refresh(){try{const [next,connection]=await Promise.all([api('/api/models'),api('/api/oauth/antigravity/status')]);if(alive){setModelState(next);setEnhancement(connection);}}catch{}if(alive)timer=window.setTimeout(refresh,downloading?1500:5000);}
    timer=window.setTimeout(refresh,1500);return()=>{alive=false;clearTimeout(timer);};
  },[showSettings,downloading,api]);
  useEffect(()=>{if(!config)return;api('/api/history').then(setHistory).catch(()=>{});},[queue.revision,config,api]);
  useEffect(()=>{localStorage.setItem('qwen-results-rail',railOpen?'open':'closed');},[railOpen]);
  const imageJobs=[queue.state.active,...queue.state.pending].filter((j):j is SavedJob=>!!j&&(j.kind||'image')==='image');
  // Each newly finished image becomes the result, also when it is not the last job sent.
  const newestResult=useRef<string|null>(null);
  // Clear empties Result until the next image finishes; the cleared image stays in Recent and the gallery.
  function clearResult(){const newest=history[0]?.id;if(newest)try{localStorage.setItem('qwen-result-cleared',newest);}catch{}setResult(null);setCompareResult(false);}
  useEffect(()=>{const newest=history[0];if(!newest)return;if(newestResult.current&&newestResult.current!==newest.id)setResult(newest as Job);newestResult.current=newest.id;},[history]);
  const loadPeople=useCallback(()=>{api('/api/people').then(setPeople).catch(()=>{});},[api]);
  useEffect(()=>{if(config)loadPeople();},[config,loadPeople]);
  // Ctrl+Enter generates from anywhere on the Images page, not only from the prompt box.
  useEffect(()=>{if(tab!=='image')return;function key(e:KeyboardEvent){if((e.ctrlKey||e.metaKey)&&e.key==='Enter'&&!e.repeat&&!document.querySelector('dialog[open]')){e.preventDefault();generate();}}
    document.addEventListener('keydown',key);return()=>document.removeEventListener('keydown',key);});
  // Ctrl+V anywhere on the Images page adds copied images as inputs.
  useEffect(()=>{if(tab!=='image')return;function paste(e:ClipboardEvent){if(e.defaultPrevented||modal.current?.open||document.querySelector('dialog[open]'))return;const files=Array.from(e.clipboardData?.files||[]).filter(f=>f.type.startsWith('image/'));if(!files.length)return;e.preventDefault();upload(files);}
    document.addEventListener('paste',paste);return()=>document.removeEventListener('paste',paste);});
  useEffect(()=>{if(!busy)return;const timer=setInterval(()=>setClock(Date.now()),1000);return()=>clearInterval(timer);},[busy]);
  useEffect(()=>{if(lightbox||activeExample||showExamples||showHelp||showSettings){if(!modal.current?.open)modal.current?.showModal();}else modal.current?.close();},[lightbox,activeExample,showExamples,showHelp,showSettings]);
  async function upload(files:File[]){
    if(!files.length||locked)return;if(files.length+inputs.length>10){setError(t('maxImages'));return;}
    setUploading(true);setError('');try{const data=new FormData();files.forEach(f=>data.append('files',f));const uploaded=await api('/api/uploads',{method:'POST',body:data});const swapAt=libraryOpen?replaceAt:null;setInputs(previous=>{if(swapAt!==null&&swapAt<previous.length&&uploaded.length){const next=[...previous];next[swapAt]=uploaded[0];return next;}return [...previous,...uploaded.filter((u:InputImage)=>!previous.some(p=>p.id===u.id))].slice(0,10);});if(swapAt!==null)closeLibrary();setLibraryRevision(r=>r+1);}
    catch(e){setError((e as Error).message);}finally{setUploading(false);if(uploadRef.current)uploadRef.current.value='';}
  }
  async function generate(override?:string){
    if(override!==undefined)setParams(p=>({...p,prompt:override}));const prompt=override??params.prompt;
    if(!validSteps){setError('Enter a whole number of steps from 1 to 80.');return;}
    if(locked)return;if(!imageReady){setError('Choose an available model first.');return;}if(activeTool&&(inputs.length<activeTool.inputs.min||inputs.length>activeTool.inputs.max)){setError(`${activeTool.title} needs ${activeTool.inputs.labels.map((l,i)=>`image ${i+1}: ${l}`).join(', ')}.`);return;}if(zimage&&inputs.length>1){setError('Z-Image redraws one input image. Keep only image 1, or choose a Qwen-Image-2.1 model to combine images.');return;}if(enhanceOn&&!enhancement?.connected){openSettings('enhance');return;}if(!prompt.trim()){setError(t('enterPrompt'));promptRef.current?.focus();return;}
    setSubmitting(true);setError('');setNotice('');setJob(null);
    try{const created=await api('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...params,prompt,tool_id:activeTool?.id??null,lora_strength:activeTool?(params.lora_strength??toolStrength(activeTool,params.tool_options)):null,enhance:enhanceOn,loras:activeLoras,images:inputs.map(i=>i.id),denoise:zimage&&inputs.length?params.denoise??0.7:null})});rememberPrompt('images',prompt);setJob(created);setPollId(created.id);setNotice('Added to queue. You can keep editing and send another request.');queue.refresh();}
    catch(e){setError((e as Error).message);}finally{setSubmitting(false);}
  }
  function reorder(index:number,delta:number){setInputs(old=>{const next=[...old];[next[index],next[index+delta]]=[next[index+delta],next[index]];return next;});}
  function applyExample(example:Example){setParams(p=>({...p,prompt:example.prompt,enhance:example.prompt_extend}));setInputs(example.inputs.map((name)=>({id:`example:${name}`,name,url:`/examples/${name}`})));setActiveExample(null);setJob(null);setNotice(t('promptLoaded'));promptRef.current?.focus();}
  // The preview steps through the list it was opened from: results (newest first, like the Recent rail) or the input images.
  const [previewList,setPreviewList]=useState<string[]>([]);
  function openResultPreview(url:string){const list=history.map(h=>h.image_url).filter((u):u is string=>!!u);setPreviewList(list.includes(url)?list:[url,...list]);setLightbox(url);}
  function openInputPreview(url:string){setPreviewList(inputs.map(i=>i.url));setLightbox(url);}
  const previewIndex=lightbox?previewList.indexOf(lightbox):-1;
  const stepPreview=(delta:number)=>{const next=previewList[previewIndex+delta];if(previewIndex>=0&&next)setLightbox(next);};
  useEffect(()=>{if(!lightbox)return;function key(e:KeyboardEvent){if(e.key==='ArrowLeft'){e.preventDefault();stepPreview(-1);}else if(e.key==='ArrowRight'){e.preventDefault();stepPreview(1);}}
    document.addEventListener('keydown',key);return()=>document.removeEventListener('keydown',key);});
  function closeModal(){setLightbox(null);setActiveExample(null);setShowExamples(false);setShowHelp(false);setShowSettings(false);}
  async function copyPrompt(text:string){try{await navigator.clipboard.writeText(text);setCopied(true);setTimeout(()=>setCopied(false),1600);}catch(e){setError(String(e));}}
  const stageElapsed=job?.stage_started_at?Math.max(0,Math.floor((clock-job.stage_started_at*1000)/1000)):0;
  const elapsed=job?.started_at ? Math.max(0,Math.round(((job.finished_at?job.finished_at*1000:clock)-job.started_at*1000)/1000)):0;
  const settingsProps={value:params,model:chosenModel,models:modelState,modelId:params.model_id,tools,tool:activeTool,toolState:{tool_id:activeTool?.id??null,tool_options:params.tool_options,lora_strength:params.lora_strength},disabled:locked,
    onChange:(values:Partial<ImageOptions>)=>setParams(p=>({...p,...values})),
    onModel:(id:string)=>{setParams(p=>({...p,model_id:id,steps:p.model_id===id?p.steps:null,cfg:p.model_id===id?p.cfg:null}));setError('');},
    onTool:(state:ToolState)=>{const next=tools.find(t=>t.id===state.tool_id);setParams(p=>({...p,...state,prompt:next?toolPrompt(next,state.tool_options):(p.tool_id?'':p.prompt)}));setError('');}};
  // The enhanced prompt is shown with the result only when enhancement really changed the prompt.
  const rewrittenPrompt=result?.rewritten_prompt?.trim()&&result.rewritten_prompt.trim()!==(result.request?.prompt||'').trim()?result.rewritten_prompt.trim():'';
  return <>
    <header className="header"><div className="brand"><img src="/branding/icon-64.png" alt=""/><strong>Qwen</strong><span className="brand-divider"/><nav className="app-tabs" aria-label="Generation type"><button aria-current={tab==='image'?'page':undefined} onClick={()=>{setTab('image');window.history.replaceState(null,'','#images');}}><ImageIcon size={17}/>Images</button><button aria-current={tab==='video'?'page':undefined} onClick={()=>showVideo(videoSection)}><span className="video-tab-icon" aria-hidden="true">▷</span>Video{videoBusy?<Loader2 className="spin" size={13}/>:null}</button><button aria-current={tab==='gallery'?'page':undefined} onClick={()=>{setTab('gallery');window.history.replaceState(null,'','#gallery');}}>Gallery</button><button aria-current={tab==='civitai'?'page':undefined} onClick={()=>{setTab('civitai');window.history.replaceState(null,'','#civitai');}}><Compass size={16}/>Civitai</button><button aria-current={tab==='train'?'page':undefined} onClick={()=>{setTab('train');window.history.replaceState(null,'','#train');}}>Train</button></nav></div>
      <div className="header-tools"><button className="queue-header-button" onClick={()=>setQueueOpen(true)} aria-label={`Open queue (${queue.count})`}>Queue <span>{queue.count}</span></button><button className="icon-button" onClick={()=>openSettings()} title={t('settings')} aria-label={t('settings')}><Settings size={19}/></button><button className="icon-button" onClick={()=>setShowHelp(true)} title={t('help')} aria-label={t('help')}><CircleHelp size={19}/></button></div>
    </header>
    {inputs.length?<BoxEditor src={inputs[0].url} open={boxOpen} onClose={()=>setBoxOpen(false)} onApply={async png=>{const data=new FormData();data.append('files',new File([png],'marked.png',{type:'image/png'}));const [marked]=await api('/api/uploads',{method:'POST',body:data});setInputs(old=>[marked,...old.slice(1)]);}}/>:null}
    <JobQueue api={api} state={queue.state} error={queue.error} refresh={queue.refresh} open={queueOpen} onOpen={setQueueOpen} onRestore={restoreJob} onView={viewJob}/>
    {tab==='video'?<nav className="video-workflow-nav" role="tablist" aria-label="Video tools">
      <button role="tab" aria-selected={videoSection==='create'} onClick={()=>showVideo('create')}>Create video</button>
      <button role="tab" aria-selected={videoSection==='reference'} onClick={()=>showVideo('reference')}>Video to video</button>
      <button role="tab" aria-selected={videoSection==='multi-reference'} onClick={()=>showVideo('multi-reference')}>Reference images</button>
      <button role="tab" aria-selected={videoSection==='upscale'} onClick={()=>showVideo('upscale')}>Upscale</button>
    </nav>:null}
    <main hidden={tab!=='image'} className={`image-main studio-main ${railOpen?'with-rail':imageJobs.length?'with-queue':''}`}>
      <div className="studio-grid">
      <div className="studio-controls">
      <div className="controls-scroll">
        <section className="panel model-panel">
          <ImageSettings part="model" {...settingsProps}/>
          <LoraStack compact value={params.loras} onChange={next=>change('loras',next)} loras={userLoras} family={family} prompt={params.prompt} onPrompt={next=>change('prompt',next)} disabled={locked}
            onBrowse={()=>{setTab('civitai');window.history.replaceState(null,'','#civitai');}}/>
          {config&&missingModels.length?<small className="model-missing">Missing model files download automatically when you generate.</small>:null}
        </section>
        <section ref={inputPanel} className="panel input-panel"><div className="section-heading"><h2>{t('input')}</h2>{inputs.length>0?<span className="count">{inputs.length}/10</span>:null}{activeTool?.needs_box&&inputs.length?<button className="text-button mark-object" disabled={locked} onClick={()=>setBoxOpen(true)}><SquareDashed size={15}/>Mark object</button>:null}{inputs.length>0?<button className="text-button" disabled={locked} onClick={()=>setInputs([])}>Clear</button>:null}<button type="button" className={`library-toggle ${libraryOpen?'open':''}`} aria-expanded={libraryOpen} aria-haspopup="dialog" onClick={()=>libraryOpen?closeLibrary():openLibrary(null)} title="Uploads, results and people"><Images size={14}/>Library<ChevronDown size={13}/></button></div>
          <div className={`upload-area ${dragging?'dragging':''} ${inputs.length?'has-images':''}`} onDragOver={e=>{e.preventDefault();if(!locked)setDragging(true);}} onDragLeave={()=>setDragging(false)} onDrop={e=>{e.preventDefault();setDragging(false);upload(Array.from(e.dataTransfer.files));}}>
            <input ref={uploadRef} type="file" accept="image/png,image/jpeg,image/webp" multiple hidden onChange={e=>upload(Array.from(e.target.files||[]))}/>
            {inputs.length===0?<div className="upload-empty-row"><button className="upload-empty compact" disabled={locked} onClick={()=>uploadRef.current?.click()}><UploadCloud size={20} strokeWidth={1.6}/><span><strong>{uploading?t('uploadBusy'):'Drop, paste (Ctrl + V) or upload'}</strong></span></button><button type="button" className="upload-library" disabled={locked} onClick={()=>openLibrary(null)}><Images size={18} strokeWidth={1.6}/><span>Pick from library</span></button></div>:<div className="input-grid">{inputs.map((input,index)=><div className={`input-thumb ${libraryOpen&&replaceAt===index?'target':''}`} key={input.id}><button className="thumb-preview checker" disabled={locked} title="Replace this image" onClick={()=>openLibrary(index)} aria-label={`Replace image ${index+1}`}><img src={input.url} alt={`${index+1}. ${input.name}`}/></button><button className="thumb-zoom" onClick={()=>openInputPreview(input.url)} aria-label={`${t('preview')} ${index+1}`} title={t('preview')}><Expand size={12}/></button><span className="image-number">{index+1}{activeTool?.inputs.labels[index]?` · ${activeTool.inputs.labels[index]}`:''}{people.find(p=>p.image_ids.includes(input.id))?` · ${people.find(p=>p.image_ids.includes(input.id))!.name}`:''}</span><button disabled={locked} className="remove-image" onClick={()=>setInputs(old=>old.filter((_,i)=>i!==index))} aria-label={`${t('remove')} ${index+1}`}><X size={13}/></button><div className="reorder"><button disabled={locked||index===0} onClick={()=>reorder(index,-1)} aria-label={`${t('left')} ${index+1}`}><ChevronLeft size={14}/></button><button disabled={locked||index===inputs.length-1} onClick={()=>reorder(index,1)} aria-label={`${t('right')} ${index+1}`}><ChevronRight size={14}/></button></div></div>)}{inputs.length<10?<button className="add-image" disabled={locked} onClick={()=>openLibrary(null)} aria-label="Add an image" title="Add an image"><Plus size={25}/></button>:null}</div>}
          </div>
          {krea&&inputs.length?<p className="input-note">Krea 2 uses input images only as a style reference (the official Style Reference LoRA): it copies colors, light and clothing style. It does not keep the person or edit the photo. To edit a photo or keep a face, choose a Qwen-Image-2.1 model.</p>:null}
          {zimage&&inputs.length?<div className="input-note zimage-redraw"><p>Z-Image has no edit model. It redraws image 1 (image to image): the result keeps the layout, pose and colors and follows the prompt for the rest. It does not keep a face exactly. To edit a photo or keep a face, choose a Qwen-Image-2.1 model.{inputs.length>1?' Z-Image uses only image 1: remove the other images.':''}</p>
            <label htmlFor="zimage-denoise">Change <b>{Math.round((params.denoise??0.7)*100)}%</b></label><input id="zimage-denoise" type="range" min="0.1" max="1" step="0.05" disabled={locked} value={params.denoise??0.7} onChange={e=>change('denoise',Number(e.target.value))}/><small>Low keeps more of image 1. 100% ignores it.</small></div>:null}
          <LibraryPopover open={libraryOpen} onClose={closeLibrary} anchor={inputPanel}>
            <InputLibrary api={api} inputs={inputs} onChange={setInputs} results={history} people={people} onPeople={loadPeople} replace={replaceAt!==null} target={replaceAt??0} revision={libraryRevision} disabled={locked} limit={10} onUpload={()=>uploadRef.current?.click()} onClose={closeLibrary}/>
          </LibraryPopover>
        </section>
        <section className="panel controls-panel">
        <div className="prompt-row"><label htmlFor="prompt" className="prompt-label">{t('prompt')}<span className="mode-label">{inputs.length?(zimage?'Image to image':krea?'Style reference':t('editMode')):t('createMode')}</span></label><button type="button" className="prompt-chip" disabled={!examples.length} onClick={()=>setShowExamples(true)} title="Example prompts"><LayoutGrid size={13}/>Examples</button><PromptHistory scope="images" disabled={locked} onLoad={text=>{change('prompt',text);promptRef.current?.focus();}} onRun={text=>generate(text)}/></div>
        <textarea ref={promptRef} id="prompt" dir="auto" className="prompt-input" placeholder={t('placeholder')} value={params.prompt} disabled={locked} onChange={e=>change('prompt',e.target.value)}/>
        <ImageSettings part="main" {...settingsProps}/>
        <details className="disclosure advanced-settings" onToggle={e=>{const el=e.currentTarget;if(el.open)requestAnimationFrame(()=>el.scrollIntoView({block:'nearest',behavior:'smooth'}));}}><summary><ChevronRight size={15}/>{t('advanced')}<span className="summary-hint">{params.randomize_seed?'Random seed':`${t('seed')}: ${params.seed}`} · {params.steps??chosenModel?.generation_defaults?.steps??'…'} steps · CFG {params.cfg??chosenModel?.generation_defaults?.cfg??1}</span></summary>
          <div className="details-body">
            <ImageSettings part="advanced" {...settingsProps}/>
            <fieldset className="advanced-grid" disabled={locked}>
              <div className="field"><label htmlFor="seed">{t('seed')}</label><div className="seed-row"><input id="seed" type="number" min="0" max="2147483647" step="1" value={params.seed} disabled={params.randomize_seed} onChange={e=>change('seed',Number(e.target.value))}/><label className="checkbox-label" title={t('randomize')}><input type="checkbox" checked={params.randomize_seed} onChange={e=>change('randomize_seed',e.target.checked)}/>Random</label></div></div>
              <div className="field"><label htmlFor="vae" title="Image decoder">Decoder (VAE)</label><select id="vae" value={params.vae_id} title="Auto uses the VAE the model page names. Otherwise Texture fix: the same decoder with fewer checkerboard artifacts." onChange={e=>setParams(p=>({...p,vae_id:e.target.value,vae_choice:true}))}><option value="auto">Auto · {vaeTitle(autoVae)}</option>{vaes.map(v=><option key={v.id} value={v.id}>{vaeTitle(v.id)}{v.ready?'':` · ${(v.bytes/1e9).toFixed(1)} GB download`}</option>)}</select></div>
              <div className="field span-2"><label htmlFor="negative" title={t('negativeHelp')}>{t('negative')}</label><textarea id="negative" dir="auto" rows={2} placeholder="What to exclude (adds a guidance pass, slower)" title={t('negativeHelp')} value={params.negative_prompt} onChange={e=>change('negative_prompt',e.target.value)}/></div>
              <div className="field span-2"><label htmlFor="folder">{t('folder')}</label><input id="folder" dir="ltr" value={params.output_dir} title={params.output_dir} onChange={e=>change('output_dir',e.target.value)}/></div>
            </fieldset>
          </div>
        </details>
        </section>
      </div>
      <div className="panel generate-dock">
        <div className="generate-row"><label className="toggle-label" title={activeTool?'Tools use their own prompt wording, so enhancement is off.':undefined}><input type="checkbox" role="switch" checked={enhanceOn} disabled={locked||!!activeTool} onChange={e=>{change('enhance',e.target.checked);setJob(null);}}/><span className="switch"/><span>{t('enhance')}</span></label><div className="generate-controls">{busy?<button className="button secondary cancel" title={job?.status==='queued'?'This job has not started. Remove it from the queue.':'Stop the image that is generating now.'} onClick={async()=>{try{if(pollId)await api(`/api/jobs/${pollId}/cancel`,{method:'POST'});}catch(e){setError(String(e));}}}>{job?.status==='queued'?'Remove from queue':'Stop this job'}</button>:null}<button className="button primary" disabled={locked||!ready} onClick={()=>generate()} title={t('keyboard')}>{submitting?<Loader2 size={20} className="spin"/>:<Sparkles size={20}/>}<span>{submitting?'Adding…':queue.count||queue.state.paused?'Add image to queue':t('generate')}</span></button></div></div>
        {enhanceOn&&!enhancement?.connected?<div className="connection-hint"><span>{promptSettingsCopy[lang].description}</span><button className="text-button" onClick={()=>openSettings('enhance')}>{promptSettingsCopy[lang].connect}<ArrowUpRight size={15}/></button></div>:null}
        {error?<div className="message error" role="alert"><span>{error}</span><button className="icon-button" aria-label={t('close')} onClick={()=>setError('')}><X size={16}/></button></div>:null}
        {notice?<div className="notice" role="status">{notice}</div>:null}
        {job?.status==='cancelled'?<div className="notice">{t('cancelled')}</div>:null}
      </div>
      </div>
        <section className="panel result-panel studio-result"><div className="section-heading"><h2>{t('result')}</h2>{result?<span className="result-size" dir="ltr">{result.width} × {result.height} · PNG</span>:null}{result?.request.images?.length?<label className="checkbox-label compare-toggle"><input type="checkbox" checked={compareResult} onChange={e=>setCompareResult(e.target.checked)}/>Compare</label>:null}{result?<button className="text-button" onClick={clearResult} title="Empty the Result view. The image stays in Recent and the gallery.">Clear</button>:null}{result?<button className="icon-button" onClick={()=>openResultPreview(result.image_url!)} aria-label={t('preview')}><Expand size={17}/></button>:null}</div>
          <div className="result-canvas checker">{result&&compareResult&&result.request.images?.length?<CompareView before={inputImage(result.request.images[0]).url} after={result.image_url!} alt={result.request.prompt}/>:result?<button className="result-image-button" onClick={()=>openResultPreview(result.image_url!)} aria-label={t('preview')}><img className="result-image" src={result.image_url} alt={result.request.prompt}/></button>:<div className="result-empty"><ImageIcon size={42} strokeWidth={1.4}/><span>{t('empty')}</span></div>}
            {busy&&!result?<div className="generation-overlay" aria-live="polite"><Loader2 className="spin" size={28}/><strong>{job?.status==='queued'?'Waiting in queue':job?job.message||t(job.stage):t('starting')}</strong>{job?.stage==='downloading_models'?<ModelDownloadProgress job={job}/>:job?.stage==='generating'?<><progress value={job.step} max={job.total_steps}/><span>Step {job.step} / {job.total_steps} · {formatDuration(stageElapsed)} this stage · {formatDuration(elapsed)} total</span></>:<span>{elapsed>0?`${formatDuration(stageElapsed)} this stage · ${formatDuration(elapsed)} total`:''}{job?.stage==='rewriting'&&job.tokens?` · ${job.tokens} ${t('tokens')}`:''}</span>}</div>:null}
          </div>
          {result?<GenerationTiming job={result} models={modelState}/>:null}{result?<div className="result-actions"><a className="button secondary" href={`${result.image_url}?download=true`}><Download size={15}/>{t('download')}</a><button disabled={locked} className="text-button" onClick={async()=>{try{setInputs([await api(`/api/jobs/${result.id}/reuse`,{method:'POST'})]);promptRef.current?.focus();}catch(e){setError(String(e));}}}><ArrowUpRight size={16}/>{t('editResult')}</button><a className="icon-button" href={`/api/jobs/${result.id}/metadata`} aria-label={t('metadata')} title={t('metadata')}><Download size={15}/></a></div>:null}
          {rewrittenPrompt?<details className="result-rewrite" key={result?.id}><summary title="The prompt the model actually used"><ChevronRight size={12}/><b>Enhanced prompt</b><span dir="auto">{rewrittenPrompt}</span></summary><p dir="auto">{rewrittenPrompt}</p><button className="text-button" onClick={()=>copyPrompt(rewrittenPrompt)}>{copied?<Check size={13}/>:<Copy size={13}/>} {copied?t('copied'):t('copy')}</button></details>:null}
        </section>
      <ResultsRail open={railOpen} onToggle={()=>setRailOpen(o=>!o)} running={imageJobs} results={history} selected={result?.id}
        onSelect={item=>{setResult(item as Job);if(!busy)setJob(item as Job);}} onUseAsInput={async item=>{if(inputs.length>=10){setError(t('maxImages'));return;}try{const added=await api(`/api/jobs/${item.id}/reuse`,{method:'POST'});setInputs(old=>old.some(i=>i.id===added.id)?old:[...old,added]);}catch(e){setError((e as Error).message);}}}
        onUseSettings={item=>restoreJob(item)} onOpenGallery={()=>{setTab('gallery');window.history.replaceState(null,'','#gallery');}} onOpenFolder={()=>api('/api/open-output',{method:'POST'}).catch(err=>setError(String(err)))} onOpenQueue={()=>setQueueOpen(true)}/>
      </div>
      <footer><span className="local-status"><i/>{modelState?.remote?'Runs on RunPod GPU':t('local')}</span><span>{modelState?.remote?'Results saved on this computer':config?.device||''}</span></footer>
    </main>
    <VideoTab addLora={pendingVideoLora?.target==='create'?pendingVideoLora:null} onSendVideo={sendVideo} visible={tab==='video'&&videoSection==='create'} api={api} models={modelState} enhancement={enhancement} modelId={videoModel} onModel={setVideoModel} onBusy={setVideoBusy} queueActive={queue.count>0||queue.state.paused} onQueued={queue.refresh} restoreRequest={videoRestore} openSettings={()=>setShowSettings(true)}/>
    <ReferenceTab addLora={pendingVideoLora?.target==='reference'?pendingVideoLora:null} onSendVideo={sendVideo} videoInput={videoInput?.target!=='upscale'?videoInput:null} visible={tab==='video'&&videoSection==='reference'} api={api} token={token.current} models={modelState} enhancement={enhancement} restoreRequest={referenceRestore} onQueued={queue.refresh} openSettings={()=>setShowSettings(true)}/>
    <MultiReferenceTab onSendVideo={sendVideo} visible={tab==='video'&&videoSection==='multi-reference'} api={api} models={modelState} queue={queue.state} onQueued={queue.refresh} restoreRequest={multiRestore} onView={job=>setGalleryView({job,key:Date.now()})}/>
    <UpscaleTab onSendVideo={sendVideo} videoInput={videoInput?.target==='upscale'?videoInput:null} visible={tab==='video'&&videoSection==='upscale'} api={api} token={token.current} models={modelState} restoreRequest={upscaleRestore} onQueued={queue.refresh} openSettings={()=>setShowSettings(true)}/>
    <CivitaiPage api={api} visible={tab==='civitai'} models={modelState} onUse={applyInstalled}/>
    <TrainPage api={api} visible={tab==='train'} models={modelState} results={history} people={people} onPeople={loadPeople} onUse={id=>applyInstalled({id,kind:'image-lora'})}/>
    <Gallery onSendVideo={sendVideo} api={api} visible={tab==='gallery'} revision={queue.revision} models={modelState} onRestore={restoreJob} onQueued={queue.refresh} view={galleryView}/>

    <dialog ref={modal} aria-label={showSettings?t('settings'):showHelp?t('help'):showExamples?t('examples'):activeExample?.title_en||t('preview')} onCancel={closeModal} onClick={e=>{if(e.target===modal.current)closeModal();}} className={lightbox?'lightbox':'content-dialog'}><button className="dialog-close icon-button" onClick={closeModal} aria-label={t('close')}><X size={21}/></button>
      {showSettings?<ModelSettings initialView={settingsView} enhancement={enhancement} onEnhancement={setEnhancement} lang={lang} api={api} state={modelState} onState={setModelState} selected={params.model_id} onSelect={id=>{setParams(p=>({...p,model_id:id,steps:p.model_id===id?p.steps:null}));setError('');}} videoSelected={videoModel} onVideoSelect={setVideoModel} encoder={params.text_encoder} onEncoder={id=>change('text_encoder',id)} busy={queue.count>0}/>:null}
      {lightbox?<ZoomView className="lightbox-image checker" src={lightbox} alt={t('preview')}/>:null}
      {lightbox&&previewIndex>=0&&previewList.length>1?<><button className="lightbox-nav prev" disabled={previewIndex===0} onClick={()=>stepPreview(-1)} aria-label="Previous image" title="Previous (Left arrow)"><ChevronLeft size={26}/></button><button className="lightbox-nav next" disabled={previewIndex===previewList.length-1} onClick={()=>stepPreview(1)} aria-label="Next image" title="Next (Right arrow)"><ChevronRight size={26}/></button><span className="lightbox-count">{previewIndex+1} / {previewList.length}</span></>:null}
      {showHelp&&tab==='video'&&videoSection==='create'?<div className="help-content"><h2>Video help</h2><p>Describe the scene, movement, camera and sounds. FastH3 creates video from text. Choose MiniMax H3 to animate an image or move between a first and last frame.</p><p>Base model mode uses adjustable steps. Turbo uses 8 steps. Video is generated at 24 frames per second. You can change the seed, size and output folder in Advanced settings.</p><p>Missing models download automatically when you add a job. Supporting model files are selected automatically. Enhance prompt uses your Antigravity connection.</p><a href="https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2" target="_blank" rel="noreferrer">FastH3 model details</a><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3" target="_blank" rel="noreferrer">MiniMax H3 model details</a></div>:null}
      {showHelp&&tab==='video'&&videoSection==='reference'?<div className="help-content"><h2>Video to video help</h2><p>Replace subjects uses ordered person images. Camera view changes the viewpoint without replacing people. Open the optional subject replacement section in Camera view to run both steps in order.</p><p>Choose the model and duration in this tab. Missing model files download automatically when you add a job.</p></div>:null}
      {showHelp&&tab==='video'&&videoSection==='multi-reference'?<div className="help-content"><h2>Reference images to video</h2><p>Add up to four subjects and an optional background. Describe the action using Image 1, Image 2 and the remaining image labels. Optional voices correspond to the first two images. This creates a new video; use Video to video to edit an existing clip.</p><p>The LoRA is an add-on that lets LTX use multiple reference images. It downloads with the selected model and required files. LTX requires access from your Hugging Face account. Double width and height multiplies each dimension by 2 and adds 3 processing steps.</p></div>:null}
      {showHelp&&tab==='video'&&videoSection==='upscale'?<div className="help-content"><h2>Upscale help</h2><p>Select a source video and model. Set the shorter side in pixels or multiply both dimensions by 2 or 4. The output dimensions appear below. Frames per batch and other model settings are in Advanced settings. Keep source audio is beside Add to queue.</p></div>:null}
      {showHelp&&tab==='image'?<div className="help-content"><h2>{t('help')}</h2><p>{t('helpText')}</p><blockquote dir="auto">{t('transparent')}</blockquote><button className="button primary" disabled={locked} onClick={()=>{change('prompt',t('transparent'));setShowHelp(false);promptRef.current?.focus();}}>{t('applyTransparent')}</button><a href="https://huggingface.co/spaces/Qwen/Qwen-Image-2.1" target="_blank" rel="noreferrer">Qwen-Image-2.1 · Hugging Face</a></div>:null}
      {showExamples?<div className="examples-dialog"><h2>{t('examples')}</h2><div className="examples-grid">{examples.map(example=><button key={example.id} className="example-card" onClick={()=>{setShowExamples(false);setActiveExample(example);}}><div className="example-thumbnail checker"><img loading="lazy" src={`/examples/${example.outputs[0]||example.inputs[0]}`} alt=""/></div><span>{example.title_en}<small>{example.inputs.length?`${example.inputs.length} ${t('sourceCount')}`:t('createMode')}</small></span><ChevronRight size={16}/></button>)}</div></div>:null}
      {activeExample?<div className="example-dialog"><h2>{activeExample.title_en}</h2><p className="muted">{t('exampleSource')}</p>{activeExample.inputs.length?<><h3>{t('input')}</h3><div className="example-inputs">{activeExample.inputs.map((name,i)=><figure key={name}><img src={`/examples/${name}`} alt={`${i+1}`}/><figcaption>{i+1}</figcaption></figure>)}</div></>:null}<h3>{t('reference')}</h3><div className="example-outputs">{activeExample.outputs.map(name=><img key={name} src={`/examples/${name}`} alt={t('reference')} className="checker"/>)}</div><h3>{t('fullPrompt')}</h3><p className="example-prompt" dir="auto">{activeExample.prompt}</p><button className="button primary" disabled={locked} onClick={()=>applyExample(activeExample)}>{t('useExample')}</button></div>:null}
    </dialog>
    <DownloadPill api={api} state={modelState}/>
  </>;
}
createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
