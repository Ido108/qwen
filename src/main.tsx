import {ModelDefaultsRoot,ModelDefaultsScope,bindSettings} from './ModelDefaults';
import {ImageDestinations,UseImageAs,useImageDestination} from './UseImageAs';
import {Nudge,SettingLabel} from './SettingLabel';
import {EnhancerPick} from './EnhancerPick';
import {videoTasks} from './videoModels';
import {resultCfg} from './resultCfg';
import {ImageLabel,subjectLabel} from './ImageLabel';
import {ConnectionNotice} from './ConnectionNotice';
import {MobileResults} from './MobileResults';
import {ProgressiveImage,useResultNeighbors} from './ProgressiveImage';
import {PersonPhotoCycle,inputPerson} from './PersonPhotoCycle';
import {FilePicker} from './FilePicker';
import {DownloadPill} from './DownloadPill';
import {PromptHistory,rememberPrompt} from './PromptHistory';
import {stageLabel,stepLabel} from './QueueActivity';
import React, {useCallback, useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {UploadCloud, Image as ImageIcon, Download, X, ChevronDown, ChevronLeft, ChevronRight, Expand, Copy, Check, CircleHelp, SquareDashed, Compass, Loader2, Plus, ArrowUpRight, Settings, Images, LayoutGrid, RotateCcw} from 'lucide-react';
import {ModelSettings,type ModelState} from './ModelSettings';
import {OrbitTab} from './OrbitTab';
import {VideoTab} from './VideoTab';
import {MultiReferenceTab} from './MultiReferenceTab';
import {ReferenceTab} from './ReferenceTab';
import {UpscaleTab} from './UpscaleTab';
import {ImageSettings,toolPrompt,toolStrength,type ImageOptions,type JobPlan,type ToolState} from './ImageSettings';
import {SwapLoras} from './SwapLoras';
import {AddToPerson} from './AddToPerson';
import {Gallery,inputImage} from './Gallery';
import {InputLibrary,type Person,type InputImage} from './InputLibrary';
import {CompareView} from './CompareView';
import {BoxEditor} from './BoxEditor';
import {ZoomView} from './ZoomView';
import {CivitaiPage,type InstalledItem} from './CivitaiPage';
import {TrainPage} from './TrainPage';
import {SwapPanel,swapDefaults,swapNegative,type SwapState} from './SwapPanel';
import {PromptInput} from './PromptInput';
import {displayImageMentions,imageMentionOptions} from './promptMentions';
import {KnownPeoplePicker} from './KnownPeoplePicker';
import {LoraStack,enabledLoras,type LoraChoice} from './LoraStack';
import {compatibleLoras,loraSampling,type AuthorSettings,type LoraMeta} from './loraChoices';
import {LibraryPopover} from './LibraryPopover';
import {ResultsRail} from './ResultsRail';
import {JobQueue,useQueue,type SavedJob} from './JobQueue';
import type {SendVideo,VideoInput} from './SendVideoTo';
import type {UploadedVideo} from './mediaUpload';
import {GenerationTiming,type Timing} from './GenerationTiming';
import {ImageUpscaleDialog,InputUpscaleStatus,ResultUpscales,UPSCALE_KEYS,type InputUpscale,type UpscaleSettings,type UpscaleSource} from './ImageUpscale';
import {en,type TextKey} from './i18n';
import './style.css';
import './workspace.css';
import './reference.css';
import './reference-workbench.css';
import './studio.css';
import './responsive-workspace.css';
import './mobile-workspace.css';
import './photo-workspace.css';
import './civitai-workspace.css';
import './lora-controls.css';
import './image-preview.css';
import './viewer.css';
import './compact-layout.css';
import {apiRequest} from './session';
import {useLastSeed} from './lastSeed';

type Params = ImageOptions & ToolState & {loras:LoraChoice[];vae_id:string;vae_choice?:boolean;model_id:string;text_encoder:string;prompt:string; images:string[];image_labels?:string[]; enhance:boolean; seed:number; randomize_seed:boolean; negative_prompt:string; output_dir:string; denoise?:number|null; krea_reference_mode?:'identity'|'style'; krea_person?:number; krea_scene?:number|null; krea_identity_strength?:number|null; speed_lora?:boolean|null; speed_lora_strength?:number|null};
type Job = Timing & {download_completed_bytes?:number;download_total_bytes?:number;id:string; status:string; stage:TextKey; message:string; seed:number; step:number; total_steps:number; request:Params; rewritten_prompt:string; created_at:number; started_at?:number; stage_started_at?:number; finished_at?:number; elapsed_seconds?:number; image_url?:string; width?:number; height?:number; tokens?:number; has_transparency?:boolean};
type Example = {id:string; title:string; title_en:string; prompt:string; prompt_extend:boolean; inputs:string[]; outputs:string[]; category:string};
type Config = {token:string; models:Record<string,boolean>; output_dir:string; active_job:string|null; device:string};
type VideoSection = 'orbit'|'create'|'reference'|'multi-reference'|'upscale';
const defaults:Params = {loras:[],tool_id:null,tool_options:{},lora_strength:null,vae_id:'auto',model_id:'uc-int8-convrot',text_encoder:'auto',prompt:'',images:[],enhance:false,custom_size:false,auto_resolution:2048,width:2688,height:1536,seed:0,randomize_seed:true,negative_prompt:'',output_dir:'',steps:null};
// The same weights as INT8 ConvRot: native INT8 matmul, about 3x faster per step than Q8_0 GGUF on an RTX 5090. Saved drafts move once.
const SAMPLING_KEYS=['steps','cfg','sampler','scheduler'] as const;
// A People name or a real file name; stored upload names (a 32-character hash) say nothing to the user.
const photoName=(input:{name?:string},i:number)=>`Photo ${i+1}${input.name&&!/^[0-9a-f]{32}\.\w+$/i.test(input.name)?`: ${input.name}`:''}`;
const FAST_FORMAT:Record<string,string>={'gguf-q8_0':'uc-int8-convrot','turbo-gguf-q8_0':'turbo-int8-convrot'};
function loadDraft():Params {try {const saved=JSON.parse(localStorage.getItem('image-studio-draft-v1')||'{}');if(!localStorage.getItem('qwen-image-defaults-v2')){localStorage.setItem('qwen-image-defaults-v2','1');if(FAST_FORMAT[saved.model_id])saved.model_id=FAST_FORMAT[saved.model_id];}if(!localStorage.getItem('qwen-enhance-off-v1')){localStorage.setItem('qwen-enhance-off-v1','1');saved.enhance=false;}return {...defaults,...saved,randomize_seed:saved.randomize_seed!==false,vae_id:saved.vae_choice?saved.vae_id:'auto',steps:saved.auto_resolution===undefined&&saved.steps===40?null:(saved.steps??null)};} catch {return defaults;}}
function loadInputs():InputImage[]{
  const images=loadDraft().images.map(inputImage);
  try{const saved:InputImage[]=JSON.parse(localStorage.getItem('qwen-input-people')||'[]');return images.map((image,i)=>saved[i]?.id===image.id?{...image,personId:saved[i].personId,name:saved[i].name,subjectLabel:saved[i].subjectLabel}:image);}catch{return images;}
}
const clearedResult=()=>{try{return localStorage.getItem('qwen-result-cleared');}catch{return null;}};
function App(){
  const [tab,setTab]=useState<'image'|'video'|'gallery'|'civitai'|'train'>(()=>location.hash==='#civitai'?'civitai':location.hash==='#train'?'train':location.hash==='#gallery'?'gallery':['#video','#keyframes','#reference','#video-to-video','#multi-reference','#upscale','#orbit'].includes(location.hash)?'video':'image');
  const [videoSection,setVideoSection]=useState<VideoSection>(()=>location.hash==='#orbit'?'orbit':['#reference','#video-to-video'].includes(location.hash)?'reference':location.hash==='#upscale'?'upscale':location.hash==='#keyframes'?'create':'multi-reference');
  const [queueOpen,setQueueOpen]=useState(false);
  const [orbitRestore,setOrbitRestore]=useState<{job:SavedJob;key:number}|null>(null);
  const [videoRestore,setVideoRestore]=useState<{job:SavedJob;key:number}|null>(null);
  const [referenceRestore,setReferenceRestore]=useState<{job:SavedJob;key:number}|null>(null);
  const [multiModelChoice,setMultiModelChoice]=useState<{id:string;key:number}|null>(null);
  const [multiRestore,setMultiRestore]=useState<{job:SavedJob;key:number}|null>(null);
  const [upscaleRestore,setUpscaleRestore]=useState<{job:SavedJob;key:number}|null>(null);
  const [galleryView,setGalleryView]=useState<{job:SavedJob;key:number}|null>(null);
  const [videoInput,setVideoInput]=useState<VideoInput|null>(null);
  const [videoBusy,setVideoBusy]=useState(false);
  const [videoModel,setVideoModel]=useState(()=>{try{return JSON.parse(localStorage.getItem('qwen-video-draft')||'{}').model_id||'fasth3-q4_k_m';}catch{return 'fasth3-q4_k_m';}});
  const lang='en';
  const t=(key:TextKey)=>en[key];
  const [params,setParams]=useState<Params>(loadDraft);
  const [inputs,setInputs]=useState<InputImage[]>(loadInputs);
  const [config,setConfig]=useState<Config|null>(null);
  const [modelState,setModelState]=useState<ModelState|null>(null);
  const [showSettings,setShowSettings]=useState(()=>window.location.hash==='#settings');const [settingsView,setSettingsView]=useState<'models'|'enhance'|'location'>('models');
  const openSettings=(view:'models'|'enhance'|'location'='models')=>{setSettingsView(view);setShowSettings(true);};
  const [examples,setExamples]=useState<Example[]>([]);
  const [history,setHistory]=useState<Job[]>([]);
  useLastSeed(history,seed=>setParams(p=>p.randomize_seed&&p.seed!==seed?{...p,seed}:p));
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
  const [upscaleTarget,setUpscaleTarget]=useState<{source:UpscaleSource;preset?:Partial<UpscaleSettings>;key:number}|null>(null);
  const [inputUpscales,setInputUpscales]=useState<InputUpscale[]>(()=>{try{return JSON.parse(localStorage.getItem('qwen-input-upscales')||'[]');}catch{return [];}});
  useEffect(()=>{try{localStorage.setItem('qwen-input-upscales',JSON.stringify(inputUpscales));}catch{/* offers only last this visit */}},[inputUpscales]);
  // The Result panel steps through the results in the Recent order (newest first), also with the arrow keys.
  const shownResults=history.filter(item=>!!item.image_url);
  const resultIndex=result?shownResults.findIndex(item=>item.id===result.id):-1;
  const stepResult=(delta:number)=>{if(liveFor){setLiveFor(null);if(resultIndex>=0&&delta)return;}const next=shownResults[resultIndex+delta];if(next){setResult(next);if(!busy)setJob(next);}};
  useEffect(()=>{
    const keys=(event:KeyboardEvent)=>{
      if(event.key!=='ArrowLeft'&&event.key!=='ArrowRight')return;
      const target=event.target as HTMLElement|null;
      if(tab!=='image'||document.querySelector('dialog[open]')||target?.closest('input,textarea,select,[contenteditable="true"],[role="slider"]'))return;
      stepResult(event.key==='ArrowLeft'?-1:1);
    };
    window.addEventListener('keydown',keys);return()=>window.removeEventListener('keydown',keys);
  });
  // Phones: Download opens the share sheet (Save Image, Save Video) instead of the browser's download page.
  // Without sharing (plain http) the file opens by itself, to keep with a long press.
  useEffect(()=>{
    const touch=window.matchMedia('(pointer:coarse)');
    async function save(event:MouseEvent){
      const link=(event.target as Element|null)?.closest?.('a[href*="download=true"]') as HTMLAnchorElement|null;
      if(!link||!touch.matches)return;
      event.preventDefault();
      const plain=link.href.replace(/[?&]download=true/,'');
      if(navigator.canShare){
        try{
          const response=await fetch(plain);const blob=await response.blob();
          const name=link.href.split('/').slice(-2,-1)[0]+(blob.type.includes('video')?'.mp4':'.png');
          const file=new File([blob],name,{type:blob.type});
          if(navigator.canShare({files:[file]})){await navigator.share({files:[file]});return;}
        }catch(e){if((e as Error).name==='AbortError')return;}
      }
      window.open(plain,'_blank');
    }
    document.addEventListener('click',save);
    return()=>document.removeEventListener('click',save);
  },[]);
  const upscaleInput=(input:InputImage,index:number)=>setUpscaleTarget({source:{image:input.id,inputId:input.id,preview:input.url,label:`Input image ${index+1}`},key:Date.now()});
  const upscaleResult=(saved:{id:string;width?:number;height?:number})=>setUpscaleTarget({source:{job:saved.id,preview:`/api/jobs/${saved.id}/thumbnail`,label:'Result image'},key:Date.now()});
  const [previewError,setPreviewError]=useState('');
  useEffect(()=>setPreviewError(''),[lightbox]);
  const [railOpen,setRailOpen]=useState(()=>window.innerWidth>=1280&&localStorage.getItem('qwen-results-rail')!=='closed');const [libraryRevision,setLibraryRevision]=useState(0);const [pendingVideoLora,setPendingVideoLora]=useState<{id:string;key:number;target:'create'|'reference'|'multi-reference'}|null>(null);
  const [people,setPeople]=useState<Person[]>([]);const [compareResult,setCompareResult]=useState(false);const [boxOpen,setBoxOpen]=useState(false);
  const [copied,setCopied]=useState(false); const [dragging,setDragging]=useState(false);
  const promptRef=useRef<HTMLTextAreaElement>(null);
  const token=useRef(''); const modal=useRef<HTMLDialogElement>(null);
  const busy=!!pollId; const locked=submitting||uploading;
  const chosenModel=modelState?.models.find(m=>m.id===params.model_id);
  const validSteps=params.steps===null||(Number.isInteger(params.steps)&&params.steps>=1&&params.steps<=80);
  const tools=modelState?.tools||[];
  // Swap page tools: the BFS Qwen swaps (face picker and prompt templates) and the Krea 2 swaps (the author's prompt).
  const swapTools=tools.filter(t=>t.swap||t.swap_page);
  const fitting=swapTools.filter(t=>t.families.includes(chosenModel?.family||''));
  const pageSwapTools=fitting.length?fitting:swapTools.filter(t=>t.swap);
  const [imageSection,setImageSection]=useState<'create'|'swap'>(()=>location.hash==='#swap'?'swap':'create');
  const [swap,setSwapState]=useState<SwapState>(()=>{try{const saved={...swapDefaults,...JSON.parse(localStorage.getItem('qwen-swap-v1')||'{}')};
    // Once: the Swap size follows each author again (Qwen 2 MP, Krea 1 MP) instead of the old 1 MP default.
    if(!localStorage.getItem('qwen-swap-size-v2')){localStorage.setItem('qwen-swap-size-v2','1');delete saved.auto_resolution;}
    return saved;}catch{return swapDefaults;}});
  const setSwap=(next:SwapState)=>{setSwapState(next);try{localStorage.setItem('qwen-swap-v1',JSON.stringify(next));}catch{}};
  const activeTool=tools.find(t=>t.id===params.tool_id&&t.families.includes(chosenModel?.family||'qwen-image-2.1'));
  const family=chosenModel?.family||'qwen-image-2.1';const krea=family.startsWith('krea-2');const zimage=family.startsWith('z-image');const flux3=family.startsWith('flux3-klein');const moss=family.startsWith('moss-image');
  // Krea 2 and Z-Image have their own text encoders; Qwen models pick INT8 or BF16 by the model format.
  const chosenEncoder=flux3?'flux3-encoder-int8':moss?(chosenModel?.text_encoder||'moss-encoder-int8'):krea||zimage?(chosenModel?.text_encoder||(krea?'krea2-encoder-bf16':'zimage-encoder-bf16')):params.text_encoder==='auto'?'encoder-int8':params.text_encoder;
  // Every add-on LoRA, from any source (Civitai, Hugging Face, ModelScope, D:\models, the catalog); tool LoRAs and
  // Krea 2's reference adapters run inside their own modes.
  // Qwen tool LoRAs (swap, angles, outfit and the others) can also be added as plain LoRAs with your own prompt; they take
  // the families of their tool. The one the active tool already applies is left out. Krea 2 tool LoRAs need Krea's edit
  // nodes, so they stay in their tools only.
  const toolFor=new Map(tools.filter(t=>!t.lora.startsWith('krea2-lora-')).map(t=>[t.lora,t]));
  const userLoras=(modelState?.models||[]).filter(m=>m.kind==='image-lora'&&!m.id.startsWith('krea2-lora-')&&m.id!==activeTool?.lora&&(!!m.families?.length||toolFor.has(m.id)))
    .map(m=>{const tool=toolFor.get(m.id);return !tool?m:m.families?.length?{...m,internal:true}:{...m,internal:true,families:tool.families,title:`${m.title} (${tool.title} tool)`};});
  const activeLoras=enabledLoras(params.loras,userLoras,family);
  // People the model (or an active LoRA) knows by trigger, offered in the @ menu (MOSS Serenity).
  const knownFrom=chosenModel?.known_people?.length?chosenModel:userLoras.find(l=>l.known_people?.length&&activeLoras.some(a=>a.id===l.id));
  // Recommended sampling follows the model, the LoRAs and the size. A value the user typed stays until the next model change.
  const pixels=params.custom_size?params.width*params.height:params.auto_resolution**2;
  const loraAdvice=loraSampling(activeLoras,userLoras as (LoraMeta&{title?:string})[],family,pixels);
  const manualSampling=useRef(new Set<string>()),autoSampling=useRef<string[]>([]);
  const adviceKey=JSON.stringify(loraAdvice?.values||null);
  useEffect(()=>{setParams(p=>{const next:Record<string,unknown>={...p};const applied:string[]=[];
    for(const key of SAMPLING_KEYS){if(manualSampling.current.has(key))continue;const value=(loraAdvice?.values as Record<string,unknown>|undefined)?.[key];
      if(value!==undefined&&value!==null){next[key]=value;applied.push(key);}else if(autoSampling.current.includes(key))next[key]=null;}
    autoSampling.current=applied;return SAMPLING_KEYS.some(key=>next[key]!==p[key])?next as Params:p;});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[adviceKey,params.model_id]);
  const adviceShown=!!loraAdvice&&Object.keys(loraAdvice.values).some(key=>!manualSampling.current.has(key));
  const vaes=(modelState?.models||[]).filter(m=>m.kind==='vae'&&(m.families||['qwen-image-2.1','qwen-image-2.1-turbo']).includes(family));const vaeTitle=(id:string)=>id==='vae'?'Standard':id==='vae-texture-fix'?'Texture fix':vaes.find(v=>v.id===id)?.title||id;
  // A LoRA made for this model whose author decodes with a named VAE (Standard keeps the grain that Texture fix smooths).
  const loraVae=activeLoras.map(l=>(userLoras.find(m=>m.id===l.id) as LoraMeta|undefined)?.trained_on).find(t=>t?.family===family&&t.settings?.vae_id)?.settings?.vae_id;
  const autoVae=(loraVae&&(modelState?.models||[]).some(m=>m.id===loraVae&&m.kind==='vae')?loraVae:null)||chosenModel?.recommended_vae||(krea?'krea2-vae':zimage?'zimage-vae':moss?'moss-vae-small':'vae-texture-fix');
  const authorModel=(target:string)=>target===family?chosenModel:(modelState?.models||[]).filter(m=>m.kind==='image'&&m.family===target&&m.installed).sort((a,b)=>Number(a.source==='civitai')-Number(b.source==='civitai'))[0];
  const applyAuthorSettings=(s:AuthorSettings&{family:string})=>{const target=authorModel(s.family);if(!target)return;manualSampling.current.clear();markManual(s);setParams(p=>({...p,model_id:target.id,...(s.steps?{steps:s.steps}:{}),...(s.cfg?{cfg:s.cfg}:{}),...(s.sampler?{sampler:s.sampler}:{}),...(s.scheduler?{scheduler:s.scheduler}:{}),...(s.vae_id?{vae_id:s.vae_id,vae_choice:true}:{})}));setNotice(`Using the LoRA author's settings${target.id!==params.model_id?` with ${target.title}`:''}.`);};const resolvedVae=params.vae_id==='auto'||!vaes.some(v=>v.id===params.vae_id)?autoVae:params.vae_id;
  const requiredIds=[params.model_id,chosenEncoder,resolvedVae,...(flux3?['flux3-prompt']:[]),...(krea&&inputs.length&&!activeTool?[params.krea_reference_mode==='style'?'krea2-lora-style-reference':'krea2-lora-identity-edit']:[]),...(activeTool?[activeTool.lora]:[]),...activeLoras.map(l=>l.id)];
  const imageReady=modelState?.remote?!!chosenModel?.ready:!!chosenModel;
  const ready=imageReady&&!!chosenModel?.generation_defaults&&validSteps;
  const missingModels=modelState?.remote?[]:requiredIds.map(id=>modelState?.models.find(m=>m.id===id)).filter(m=>m&&!m.installed);
  const change=<K extends keyof Params>(key:K,value:Params[K])=>setParams(p=>({...p,[key]:value}));
  // A sampling value typed by the user stays; resetting it to the model default lets recommendations apply again.
  const markManual=(values:Record<string,unknown>)=>{for(const key of SAMPLING_KEYS)if(key in values){if(values[key]===null||values[key]===undefined)manualSampling.current.delete(key);else manualSampling.current.add(key);}};
  const downloading=modelState?.transfer.status==='running'||!!modelState?.downloads?.some(d=>d.status==='running');
  const api=useCallback((path:string,init:RequestInit={})=>apiRequest(path,init),[]);
  const queue=useQueue(api,!!config);
  useEffect(()=>{function navigate(){const hash=window.location.hash;if(hash==='#civitai')setTab('civitai');else if(hash==='#train')setTab('train');else if(hash==='#gallery')setTab('gallery');else if(['#video','#keyframes','#reference','#video-to-video','#multi-reference','#upscale','#orbit'].includes(hash)){setTab('video');setVideoSection(hash==='#orbit'?'orbit':hash==='#upscale'?'upscale':hash==='#keyframes'?'create':['#reference','#video-to-video'].includes(hash)?'reference':'multi-reference');}else if(hash==='#swap'||hash==='#images'){setTab('image');setImageSection(hash==='#swap'?'swap':'create');}}window.addEventListener('hashchange',navigate);return()=>window.removeEventListener('hashchange',navigate);},[]);
  function showVideo(section:VideoSection){setTab('video');setVideoSection(section);window.history.replaceState(null,'',section==='orbit'?'#orbit':section==='create'?'#keyframes':section==='reference'?'#video-to-video':section==='multi-reference'?'#multi-reference':'#upscale');}
  const sendVideo:SendVideo=async(jobId,target,part='final')=>{
    const video:UploadedVideo=await api(`/api/jobs/${encodeURIComponent(jobId)}/reuse-video?part=${part}`,{method:'POST'});
    setVideoInput({video,target,key:Date.now()});
    showVideo(target==='upscale'?'upscale':'reference');
    window.scrollTo({top:0,behavior:'instant'});
  };
  useImageDestination('edit',image=>{
    if(!inputs.some(i=>i.id===image.id)&&inputs.length>=10)throw new Error('Image editing has ten inputs. Remove one before adding another.');
    setInputs(old=>old.some(i=>i.id===image.id)?old:[...old,image]);setTab('image');showImageSection('create');
  });
  const sendSwap=(index:0|1,image:InputImage)=>{const slots=[...swap.slots] as SwapState['slots'];slots[index]=image;setSwap({...swap,slots,selection:null});setTab('image');showImageSection('swap');};
  useImageDestination('swap-scene',image=>sendSwap(0,image));
  useImageDestination('swap-person',image=>sendSwap(1,image));
  useEffect(()=>{const used=()=>{closeModal();setError('');};window.addEventListener('qwen-image-used',used);return()=>window.removeEventListener('qwen-image-used',used);},[]);
  const [imageRestoreKey,setImageRestoreKey]=useState(0);
  function restoreJob(saved:SavedJob){
    if(saved.kind==='image-upscale'){const r=saved.request;setUpscaleTarget({source:{image:r.source_image,preview:inputImage(r.source_image).url,label:'Source image'},preset:Object.fromEntries(UPSCALE_KEYS.filter(key=>r[key]!==undefined).map(key=>[key,r[key]])),key:Date.now()});return;}
    setImageRestoreKey(key=>key+1);
    const restoredImage={...defaults,...saved.request,seed:saved.seed??saved.request.seed??0,randomize_seed:false,steps:saved.actual_steps??(saved.total_steps>0?saved.total_steps:saved.request.steps??null),cfg:resultCfg({cfg:saved.cfg,true_cfg_scale:saved.true_cfg_scale,guidance:saved.guidance})??saved.request.cfg??null};
    // A restored run keeps its exact sampling: the automatic LoRA advice does not overwrite it.
    manualSampling.current=new Set(SAMPLING_KEYS.filter(key=>restoredImage[key]!==null&&restoredImage[key]!==undefined));
    if(['training','captioning'].includes(saved.kind||'')){setTab('train');window.history.replaceState(null,'','#train');return;}
    if(saved.kind==='video'&&saved.request.workflow==='orbit'){setOrbitRestore({job:saved,key:Date.now()});showVideo('orbit');}
    else if(saved.kind==='video'){setVideoRestore({job:saved,key:Date.now()});setVideoModel(saved.request.model_id);showVideo('create');}
    else if(saved.kind==='multi-reference'){setMultiRestore({job:saved,key:Date.now()});showVideo('multi-reference');}
    else if(saved.kind==='reference-video'){setReferenceRestore({job:saved,key:Date.now()});showVideo('reference');}
    else if(saved.kind==='upscale'){setUpscaleRestore({job:saved,key:Date.now()});showVideo('upscale');}
    else if(swapTools.some(t=>t.id===saved.request.tool_id)){
      // A swap opens in Swap with its two pictures, strength and extra instructions.
      const tool=swapTools.find(t=>t.id===saved.request.tool_id)!;const images=(saved.request.images||[]).map((id:string,index:number)=>({...inputImage(id),subjectLabel:saved.request.image_labels?.[index]}));const prompt=String(saved.request.prompt||'');
      setParams({...restoredImage,tool_id:null,tool_options:{},lora_strength:null,prompt:''});
      setSwap({tool:tool.id,slots:[images[0]??null,images[1]??null],strength:saved.request.lora_strength??tool.strength,extra:saved.request.swap_extra??(prompt.startsWith(tool.prompt)?prompt.slice(tool.prompt.length).trim():''),selection:saved.request.swap_selection??null,poseSource:saved.request.swap_pose_source??1,styleSource:saved.request.swap_style_source??1,promptTemplate:saved.request.swap_prompt_template??undefined,promptRevision:saved.request.swap_prompt_revision??undefined,steps:saved.request.steps??null,cfg:saved.request.cfg??null,sampler:saved.request.sampler??null,scheduler:saved.request.scheduler??null,custom_size:!!saved.request.custom_size,auto_resolution:saved.request.auto_resolution===2048?2048:1024,width:saved.request.width,height:saved.request.height,styleMode:saved.request.swap_style_mode??'realistic',...(saved.request.swap_style_mode==='non-realistic'?{nonRealisticNegative:saved.request.source_negative_prompt??saved.request.negative_prompt??''}:{negativePrompt:saved.request.source_negative_prompt??saved.request.negative_prompt??''})});
      if(saved.status==='succeeded')setResult(saved as Job);setTab('image');setImageSection('swap');window.history.replaceState(null,'','#swap');setNotice('The swap pictures, strength and settings are loaded.');setError('');}
    else{setImageSection('create');setParams({...restoredImage,prompt:saved.request.source_prompt??saved.request.prompt,negative_prompt:saved.request.source_negative_prompt??saved.request.negative_prompt??''});setInputs((saved.request.images||[]).map((id:string,index:number)=>({...inputImage(id),subjectLabel:saved.request.image_labels?.[index]})));if(saved.status==='succeeded')setResult(saved as Job);setTab('image');window.history.replaceState(null,'','#images');setNotice('Original prompt, input images, seed and settings loaded.');setError('');}
  }
  function useMobileSettings(saved:SavedJob){
    if(locked)return;
    closeModal();restoreJob(saved);
    requestAnimationFrame(()=>document.querySelector('.studio-controls')?.scrollIntoView({block:'start',behavior:'instant'}));
  }
  function applyInstalled(item:InstalledItem){
      item={...item,kind:modelState?.models.find(m=>m.id===item.id)?.kind??item.kind};
      if(item.id==='h3-360-orbit'){showVideo('orbit');return;}
      if(item.kind==='video-lora'||item.kind==='video'||item.kind==='reference-video'){
        const entry=modelState?.models.find(m=>m.id===item.id);
        const references=entry&&(item.kind==='video-lora'?entry.families?.some(f=>['minimax-h3-ref2va','ltx25','ltx25-msr'].includes(f)):videoTasks(entry).includes('references'));
        const target=references?'multi-reference':'create';
        if(item.kind==='video-lora')setPendingVideoLora({id:item.id,key:Date.now(),target});
        else if(references)setMultiModelChoice({id:item.id,key:Date.now()});else setVideoModel(item.id);
        showVideo(target);setNotice('');return;}
      // A LoRA for another architecture also selects a model it works with (the installed one first).
      const lora=item.kind==='image-lora'?modelState?.models.find(m=>m.id===item.id):undefined;const families=lora?.families||[];
      const fitting=lora&&!families.includes(family)?(modelState?.models.filter(m=>m.kind==='image'&&families.includes(m.family||'qwen-image-2.1'))||[]).sort((a,b)=>Number(b.ready)-Number(a.ready))[0]:undefined;
      if(item.kind==='image-lora'&&fitting)manualSampling.current.clear();
      if(item.kind==='image-lora')setParams(p=>{const selected=compatibleLoras(p.loras,modelState?.models||[],fitting?.family||family);return {...p,...(fitting?{model_id:fitting.id,steps:null,cfg:null,sampler:null,scheduler:null}:{}),loras:selected.some(l=>l.id===item.id)?selected:[...selected,{id:item.id,strength:lora?.default_strength??1}]};});
      else{manualSampling.current.clear();setParams(p=>({...p,model_id:item.id,steps:null,cfg:null,sampler:null,scheduler:null}));}
      setTab('image');window.history.replaceState(null,'','#images');setNotice(item.kind==='image-lora'?`LoRA added${fitting?`, and the model changed to ${fitting.title}, which the LoRA works with`:''}. Its strength is under LoRAs.`:'Model selected.');}
  function viewJob(saved:SavedJob){if(['training','captioning'].includes(saved.kind||'')){restoreJob(saved);return;}setGalleryView({job:saved,key:Date.now()});setTab('gallery');window.history.replaceState(null,'','#gallery');}
  useEffect(()=>{let alive=true,timer:number; async function connect(){try{
    const [cfg,ex,old]=await Promise.all([api('/api/config'),api('/api/examples'),api('/api/history')]);
    if(!alive)return; token.current=cfg.token; setConfig(cfg);setError('');setModelState(cfg.catalog); setExamples(ex); setHistory(old);setResult(old[0]&&old[0].id!==clearedResult()?old[0]:null);
    setParams(p=>({...p,output_dir:p.output_dir||cfg.output_dir}));
    if(cfg.active_job){
      const active:Job=await api(`/api/jobs/${cfg.active_job}`);if(!alive)return;
      if((active as Job&{kind?:string}).kind&&((active as Job&{kind?:string}).kind!=='image'))return;
      setJob(active);setPollId(active.id);
    }
  }catch(e){if(alive){setError('Connecting to Qwen. Your saved settings stay here.');timer=window.setTimeout(connect,4000);}}}connect();return()=>{alive=false;clearTimeout(timer);};},[]);
  useEffect(()=>{localStorage.setItem('image-studio-draft-v1',JSON.stringify({...params,images:inputs.map(i=>i.id)}));},[params,inputs]);
  useEffect(()=>{try{localStorage.setItem('qwen-input-people',JSON.stringify(inputs.map(({id,name,personId,subjectLabel})=>({id,name,personId,subjectLabel}))));}catch{}},[inputs]);
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
    async function refresh(){try{const next=await api('/api/models');if(alive)setModelState(next);}catch{}if(alive)timer=window.setTimeout(refresh,downloading?1500:5000);}
    timer=window.setTimeout(refresh,1500);return()=>{alive=false;clearTimeout(timer);};
  },[showSettings,downloading,api]);
  useEffect(()=>{if(!config)return;api('/api/history').then(setHistory).catch(()=>{});},[queue.revision,config,api]);
  // Results deleted in the Library leave Recent and Result at once.
  useEffect(()=>{const reload=()=>{api('/api/history').then(setHistory).catch(()=>{});};window.addEventListener('qwen-results-changed',reload);return()=>window.removeEventListener('qwen-results-changed',reload);},[api]);
  useEffect(()=>{localStorage.setItem('qwen-results-rail',railOpen?'open':'closed');},[railOpen]);
  const submittedJob=queue.state.active?.id===pollId?queue.state.active:queue.state.pending.find(item=>item.id===pollId)||(job as SavedJob|null);
  const imageJobs=[queue.state.active,...queue.state.pending].filter((j):j is SavedJob=>!!j&&['image','image-upscale'].includes(j.kind||'image'));
  // The running image job develops on the big Result picture, step by step (each step's estimate, as in ComfyUI).
  const runningJob=queue.state.active&&(queue.state.active.kind||'image')==='image'&&queue.state.active.status==='running'?queue.state.active as SavedJob&{preview_url?:string;preview_updated_at?:number}:null;
  // Shown when a job starts or its Recent tile is clicked; stepping to or choosing a result shows that result instead.
  const [liveFor,setLiveFor]=useState<string|null>(null);
  useEffect(()=>{if(runningJob?.id)setLiveFor(runningJob.id);},[runningJob?.id]);
  const liveJob=runningJob&&liveFor===runningJob.id?runningJob:null;
  // Each newly finished image becomes the result, also when it is not the last job sent.
  const newestResult=useRef<string|null>(null);
  // Clear empties Result until the next image finishes; the cleared image stays in Recent and the gallery.
  function clearResult(){const newest=history[0]?.id;if(newest)try{localStorage.setItem('qwen-result-cleared',newest);}catch{}setResult(null);setCompareResult(false);}
  useEffect(()=>{const newest=history[0];if(!newest)return;if(newestResult.current&&newestResult.current!==newest.id)setResult(newest as Job);newestResult.current=newest.id;},[history]);
  const loadPeople=useCallback(()=>{api('/api/people').then(setPeople).catch(()=>{});},[api]);
  useEffect(()=>{if(config)loadPeople();},[config,loadPeople]);
  // Ctrl+Enter generates from anywhere on the Images page, not only from the prompt box.
  useEffect(()=>{if(tab!=='image')return;function key(e:KeyboardEvent){if((e.ctrlKey||e.metaKey)&&e.key==='Enter'&&!e.repeat&&!document.querySelector('dialog[open], [popover]:popover-open')){e.preventDefault();if(imageSection==='swap')generateSwap();else generate();}}
    document.addEventListener('keydown',key);return()=>document.removeEventListener('keydown',key);});
  // Ctrl+V anywhere on the Images page adds copied images as inputs.
  useEffect(()=>{if(tab!=='image')return;function paste(e:ClipboardEvent){if(e.defaultPrevented||modal.current?.open||document.querySelector('dialog[open]'))return;const files=Array.from(e.clipboardData?.files||[]).filter(f=>f.type.startsWith('image/'));if(!files.length)return;e.preventDefault();upload(files);}
    document.addEventListener('paste',paste);return()=>document.removeEventListener('paste',paste);});
  useEffect(()=>{if(lightbox||activeExample||showExamples||showHelp||showSettings){if(!modal.current?.open){modal.current?.showModal();if(lightbox)modal.current?.focus();}}else modal.current?.close();},[lightbox,activeExample,showExamples,showHelp,showSettings]);
  async function upload(files:File[]){
    if(!files.length||locked)return;if(files.length+inputs.length-(libraryOpen&&replaceAt!==null?1:0)>10){setError(t('maxImages'));return;}
    setUploading(true);setError('');try{const data=new FormData();files.forEach(f=>data.append('files',f));const uploaded=await api('/api/uploads',{method:'POST',body:data});const swapAt=libraryOpen?replaceAt:null;setInputs(previous=>{if(swapAt!==null&&swapAt<previous.length&&uploaded.length){const next=[...previous];next[swapAt]=uploaded[0];return next;}return [...previous,...uploaded.filter((u:InputImage)=>!previous.some(p=>p.id===u.id))].slice(0,10);});closeLibrary();setLibraryRevision(r=>r+1);}
    catch(e){setError((e as Error).message);}finally{setUploading(false);}
  }
  async function generate(override?:string){
    if(override!==undefined)setParams(p=>({...p,prompt:override}));const prompt=override??params.prompt;
    if(!validSteps){setError('Enter a whole number of steps from 1 to 80.');return;}
    if(locked)return;if(!imageReady){setError('Choose an available model first.');return;}if(activeTool&&(inputs.length<activeTool.inputs.min||inputs.length>activeTool.inputs.max)){setError(`${activeTool.title} needs ${activeTool.inputs.labels.map((l,i)=>`image ${i+1}: ${l}`).join(', ')}.`);return;}if(zimage&&inputs.length>1){setError('Z-Image redraws one input image. Keep only image 1, or choose a Qwen-Image-2.1 model to combine images.');return;}if(!prompt.trim()&&!inputs.length){setError(t('enterPrompt'));promptRef.current?.focus();return;}
    setSubmitting(true);setError('');setNotice('');setJob(null);
    try{const created=await api('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...params,prompt,tool_id:activeTool?.id??null,swap_selection:null,swap_extra:'',lora_strength:activeTool?(params.lora_strength??toolStrength(activeTool,params.tool_options)):null,enhance:!!params.enhance&&!flux3&&!activeTool,vae_id:params.vae_id==='auto'&&loraVae?resolvedVae:params.vae_id,loras:activeLoras,images:inputs.map(i=>i.id),image_labels:inputs.map(labelFor),denoise:(zimage||moss)&&inputs.length?params.denoise??0.7:null})});rememberPrompt('images',prompt);setJob(created);setPollId(created.id);setNotice('Added to queue.');queue.refresh();}
    catch(e){setError((e as Error).message);}finally{setSubmitting(false);}
  }
  // Images > Swap: the chosen swap tool with picture 1 (scene) and picture 2 (head or person), on a Qwen-Image-2.1 model.
  async function generateSwap(){
    const family=chosenModel?.family||'';
    const tool=swapTools.find(t=>t.id===swap.tool&&t.families.includes(family))??swapTools.find(t=>t.families.includes(family))??swapTools[0];
    if(locked||!tool)return;if(!validSteps){setError('Enter a whole number of steps from 1 to 80.');return;}
    if(!params.randomize_seed&&(!Number.isInteger(params.seed)||params.seed<0||params.seed>2147483647)){setError('Seed must be a whole number from 0 to 2147483647.');return;}
    if(!imageReady||!tool.families.includes(family)){setError('Swap runs on a Qwen-Image-2.1 or Krea 2 Turbo model. Choose one above.');return;}
    if(!swap.slots[0]||!swap.slots[1]){setError(`Add both pictures: 1 ${tool.inputs.labels[0]}, 2 ${tool.inputs.labels[1]}.`);return;}
    // The scene, the head and, for a Qwen swap with two people, the second head (Krea's swap takes one head).
    const sent=swap.slots.slice(0,tool.swap_page?2:tool.inputs.max).filter((i):i is InputImage=>!!i);
    try{localStorage.setItem('qwen-swap-model',params.model_id);}catch{}
    if(tool.swap_page){
      // Krea 2 swap: the author's workflow, picture 1 the base and picture 2 the reference, with the author's prompt.
      const prompt=tool.prompt+(swap.extra.trim()?' '+swap.extra.trim():'');
      setSubmitting(true);setError('');setNotice('');setJob(null);
      try{const created=await api('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...params,...swapSampling,...swapSpeed,...swapSize,prompt,negative_prompt:swapNegative(swap),tool_id:tool.id,tool_options:{},lora_strength:swap.strength,swap_selection:null,swap_prompt_template:null,swap_prompt_revision:null,swap_extra:'',enhance:false,loras:activeLoras,images:sent.map(i=>i.id),image_labels:sent.map(labelFor),denoise:null})});
        setJob(created);setPollId(created.id);setNotice('Added to queue.');queue.refresh();}
      catch(e){setError((e as Error).message);}finally{setSubmitting(false);}
      return;
    }
    if(!swap.selection||!Object.values(swap.selection.assignments).some(Boolean)){setError('Choose a replacement for at least one face in picture 1.');return;}
    const prompt=tool.prompt+(swap.extra.trim()?' '+swap.extra.trim():'');
    setSubmitting(true);setError('');setNotice('');setJob(null);
    try{const created=await api('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...params,...swapSampling,...swapSpeed,...swapSize,prompt,negative_prompt:swapNegative(swap),swap_style_mode:swap.styleMode??'realistic',tool_id:tool.id,tool_options:{},swap_selection:swap.selection,swap_extra:swap.extra,swap_pose_source:swap.poseSource??1,swap_style_source:swap.styleSource??1,swap_prompt_template:swap.promptTemplate??null,swap_prompt_revision:swap.promptRevision??null,lora_strength:swap.strength,enhance:false,loras:activeLoras,images:sent.map(i=>i.id),image_labels:sent.map(labelFor),denoise:null})});
      setJob(created);setPollId(created.id);setNotice('Added to queue.');queue.refresh();}
    catch(e){setError((e as Error).message);}finally{setSubmitting(false);}
  }
  function showImageSection(next:'create'|'swap'){setImageSection(next);window.history.replaceState(null,'',next==='swap'?'#swap':'#');setError('');
    // Swap runs on Qwen-Image-2.1: switch from Krea 2 or Z-Image to a ready Qwen model, Turbo first (the author's 8-step setup).
    if(next==='swap'&&!swapTools.some(t=>t.families.includes(chosenModel?.family||''))){const swapFamilies=new Set(swapTools.flatMap(t=>t.families));const qwen=(modelState?.models||[]).filter(m=>m.kind==='image'&&swapFamilies.has(m.family||''));
      // The model last used on Swap first, then a ready Turbo.
      let last='';try{last=localStorage.getItem('qwen-swap-model')||'';}catch{}
      const pick=qwen.find(m=>m.id===last&&m.installed)??qwen.find(m=>m.family==='qwen-image-2.1-turbo'&&m.installed)??qwen.find(m=>m.installed)??qwen[0];if(pick){manualSampling.current.clear();}if(pick)setParams(p=>({...p,model_id:pick.id,steps:null,cfg:null,sampler:null,scheduler:null}));}}
  function reorder(index:number,delta:number){setInputs(old=>{const next=[...old];[next[index],next[index+delta]]=[next[index+delta],next[index]];return next;});}
  function applyExample(example:Example){setParams(p=>({...p,prompt:example.prompt,enhance:example.prompt_extend}));setInputs(example.inputs.map((name)=>({id:`example:${name}`,name,url:`/examples/${name}`})));setActiveExample(null);setJob(null);setNotice(t('promptLoaded'));promptRef.current?.focus();}
  // The preview steps through the list it was opened from: results (newest first, like the Recent rail) or the input images.
  const [previewList,setPreviewList]=useState<string[]>([]);
  function openResultPreview(url:string){const list=history.map(h=>h.image_url).filter((u):u is string=>!!u);setPreviewList(list.includes(url)?list:[url,...list]);setLightbox(url);}
  function openInputPreview(url:string){setPreviewList(inputs.map(i=>i.url));setLightbox(url);}
  const previewIndex=lightbox?previewList.indexOf(lightbox):-1;
  const previewJob=history.find(item=>item.image_url===lightbox);
  const previewInput=previewJob?-1:inputs.findIndex(item=>item.url===lightbox);
  // Full-screen Compare: an upscale against its source, an edit or swap against its first input picture.
  const [previewCompare,setPreviewCompare]=useState(false);
  const previewRequest=previewJob?.request as {source_image?:string;source_job?:string;images?:string[]}|undefined;
  const previewBefore=!previewJob?null:(previewJob as {kind?:string}).kind==='image-upscale'?(previewRequest?.source_image?`/api/uploads/${previewRequest.source_image}`:previewRequest?.source_job?`/api/jobs/${previewRequest.source_job}/image`:null)
    :previewRequest?.images?.[0]?`/api/uploads/${previewRequest.images[0]}`:null;
  useResultNeighbors(history.map(item=>item.image_url),result?.image_url,tab==='image'&&!lightbox);
  useResultNeighbors(previewList,lightbox,!!lightbox);
  const stepPreview=(delta:number)=>{const next=previewList[previewIndex+delta];if(previewIndex>=0&&next)setLightbox(next);};
  useEffect(()=>{if(!lightbox)return;function key(e:KeyboardEvent){if(e.key==='ArrowLeft'){e.preventDefault();stepPreview(-1);}else if(e.key==='ArrowRight'){e.preventDefault();stepPreview(1);}}
    document.addEventListener('keydown',key);return()=>document.removeEventListener('keydown',key);});
  function closeModal(){setLightbox(null);setActiveExample(null);setShowExamples(false);setShowHelp(false);setShowSettings(false);}
  async function copyPrompt(text:string){try{await navigator.clipboard.writeText(text);setCopied(true);setTimeout(()=>setCopied(false),1600);}catch(e){setError(String(e));}}
  // Swap keeps its own steps, CFG, sampler and schedule (Swap often needs a higher CFG than Create).
  // The label a mention reads: the one chosen, else a People photo's person (Person, woman or man), else none.
  const labelFor=(image:InputImage)=>subjectLabel(image.subjectLabel)||(inputPerson(image,people)?inputPerson(image,people)!.label||'person':'');
  async function useAsFirstInput(jobId:string){
    try{const image:InputImage=await api(`/api/jobs/${jobId}/reuse`,{method:'POST'});
      if(imageSection==='swap'){const slots=[...swap.slots] as SwapState['slots'];slots[0]=image;setSwap({...swap,slots,selection:null});setNotice('The result is picture 1 now.');}
      else{setInputs(old=>[image,...old.filter(i=>i.id!==image.id)].slice(0,10));setNotice('The result is input 1 now.');}
      setLightbox(null);setError('');}
    catch(e){setError((e as Error).message);}}
  const swapSampling={steps:swap.steps??null,cfg:swap.cfg??null,sampler:swap.sampler??null,scheduler:swap.scheduler??null};
  const swapSpeed={speed_lora:swap.speedLora??null,speed_lora_strength:swap.speedStrength??null,krea_identity_strength:swap.headBoost??null};
  const swapTool=pageSwapTools.find(t=>t.id===swap.tool)??pageSwapTools[0];
  // Auto size: each swap author's box, Qwen 2 MP (his note: 2K keeps more detail), Krea 2 1 MP (his 1024 box).
  const swapAuto=chosenModel?.family?.startsWith('krea-2')?1024:2048;
  const swapSize={custom_size:swap.custom_size??false,auto_resolution:swap.auto_resolution??swapAuto,width:swap.width??params.width,height:swap.height??params.height};
  const setSwapSampling=(values:Partial<ImageOptions>)=>setSwap({...swap,...Object.fromEntries(Object.entries(values).filter(([key])=>key in swapSampling))});
  const settingsProps={value:params,model:chosenModel,models:modelState,modelId:params.model_id,tools:tools.filter(t=>!t.swap),tool:activeTool,toolState:{tool_id:activeTool?.id??null,tool_options:params.tool_options,lora_strength:params.lora_strength},disabled:locked,
    onChange:(values:Partial<ImageOptions>)=>{markManual(values);setParams(p=>({...p,...values}));},
    onModel:(id:string)=>{if(id!==params.model_id){manualSampling.current.clear();
      // Swap's steps, CFG, sampler and schedule follow the new model too (its default or the saved Swap default).
      const nextKrea=!!modelState?.models.find(m=>m.id===id)?.family?.startsWith('krea-2'),wasKrea=!!chosenModel?.family?.startsWith('krea-2');
      if(swap.steps!=null||swap.cfg!=null||swap.sampler||swap.scheduler||(nextKrea!==wasKrea&&swap.auto_resolution!=null))setSwap({...swap,steps:null,cfg:null,sampler:null,scheduler:null,...(nextKrea!==wasKrea?{auto_resolution:undefined,strength:swapTools.find(t=>t.families.includes(modelState?.models.find(m=>m.id===id)?.family||''))?.strength??swap.strength}:{})});}setParams(p=>({...p,model_id:id,steps:p.model_id===id?p.steps:null,cfg:p.model_id===id?p.cfg:null,sampler:p.model_id===id?p.sampler:null,scheduler:p.model_id===id?p.scheduler:null}));setError('');},
    onTool:(state:ToolState)=>{const next=tools.find(t=>t.id===state.tool_id);setParams(p=>({...p,...state,prompt:next?toolPrompt(next,state.tool_options):(p.tool_id?'':p.prompt)}));setError('');}};
  // What the swap will really run, from the server's own plan (the job is filled from the same code).
  const [swapPlan,setSwapPlan]=useState<JobPlan|null>(null);
  const swapNegativeText=swapNegative(swap);
  useEffect(()=>{if(imageSection!=='swap'||!swapTool||!swapTool.families.includes(chosenModel?.family||'')){setSwapPlan(null);return;}let live=true;
    api('/api/job-plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model_id:params.model_id,tool_id:swapTool.id,lora_strength:swap.strength,...swapSpeed,negative_prompt:swapNegativeText,loras:activeLoras})})
      .then((plan:JobPlan)=>{if(live)setSwapPlan(plan);}).catch(()=>{if(live)setSwapPlan(null);});return()=>{live=false;};
  },[imageSection,params.model_id,swapTool?.id,swap.strength,swap.speedLora,swap.speedStrength,swapNegativeText,JSON.stringify(activeLoras)]);
  // The Create page's automatic LoRAs (a tool's, Krea 2 identity or style with photos, a Turbo model's speed LoRA).
  const [createPlan,setCreatePlan]=useState<JobPlan|null>(null);
  useEffect(()=>{if(imageSection!=='create'){setCreatePlan(null);return;}let live=true;
    api('/api/job-plan',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model_id:params.model_id,tool_id:activeTool?.id??null,lora_strength:params.lora_strength,image_count:inputs.length,krea_reference_mode:params.krea_reference_mode??'identity',speed_lora:params.speed_lora??null,speed_lora_strength:params.speed_lora_strength??null,negative_prompt:params.negative_prompt,loras:activeLoras})})
      .then((plan:JobPlan)=>{if(live)setCreatePlan(plan);}).catch(()=>{if(live)setCreatePlan(null);});return()=>{live=false;};
  },[imageSection,params.model_id,activeTool?.id,params.lora_strength,inputs.length,params.krea_reference_mode,params.speed_lora,params.speed_lora_strength,params.negative_prompt,JSON.stringify(activeLoras)]);
  // The enhanced prompt is shown with the result only when enhancement really changed the prompt.
  const rewrittenPrompt=result?.rewritten_prompt?.trim()&&result.rewritten_prompt.trim()!==(result.request?.prompt||'').trim()?result.rewritten_prompt.trim():'';
  const imageDefaultBindings=bindSettings({...params,steps:params.steps??chosenModel?.generation_defaults?.steps??40,cfg:params.cfg??chosenModel?.generation_defaults?.cfg??(params.negative_prompt?.trim()?2:1),sampler:params.sampler??chosenModel?.generation_defaults?.sampler??'euler',scheduler:params.scheduler??null},patch=>{markManual(patch);setParams(p=>({...p,...patch}));},{
    Steps:'steps',CFG:'cfg',Sampler:'sampler',Schedule:'scheduler','Output size':['custom_size','auto_resolution','width','height'],Width:'width',Height:'height','Identity strength':'krea_identity_strength',Seed:['seed','randomize_seed'],Decoder:['vae_id','vae_choice'],'Negative prompt':'negative_prompt','Output folder':'output_dir',Change:'denoise',
  });
  return <>
    <header className="header"><div className="brand"><img src="/branding/icon-64.png" alt=""/><strong>Qwen</strong><span className="brand-divider"/><nav className="app-tabs" aria-label="Generation type"><button aria-current={tab==='image'?'page':undefined} onClick={()=>{setTab('image');window.history.replaceState(null,'','#images');}}><ImageIcon size={17}/>Images</button><button aria-current={tab==='video'?'page':undefined} onClick={()=>showVideo(videoSection)}><span className="video-tab-icon" aria-hidden="true">▷</span>Video{videoBusy?<Loader2 className="spin" size={13}/>:null}</button><button aria-current={tab==='gallery'?'page':undefined} onClick={()=>{setTab('gallery');window.history.replaceState(null,'','#gallery');}}>Gallery</button><button aria-current={tab==='civitai'?'page':undefined} onClick={()=>{setTab('civitai');window.history.replaceState(null,'','#civitai');}}><Compass size={16}/>Civitai</button><button aria-current={tab==='train'?'page':undefined} onClick={()=>{setTab('train');window.history.replaceState(null,'','#train');}}>Train</button></nav></div>
      <div className="header-tools"><button className="queue-header-button" onClick={()=>setQueueOpen(true)} aria-label={`Open queue (${queue.count})`}>Queue <span>{queue.count}</span></button><button className="icon-button" onClick={()=>openSettings()} title={t('settings')} aria-label={t('settings')}><Settings size={19}/></button><button className="icon-button" onClick={()=>setShowHelp(true)} title={t('help')} aria-label={t('help')}><CircleHelp size={19}/></button></div>
    </header>
    {inputs.length?<BoxEditor src={inputs[0].url} open={boxOpen} {...(activeTool?.id==='move-object'?{title:'Mark the object and where it goes',hint:'Draw two boxes: one around the object, one where it should go.'}:{})} onClose={()=>setBoxOpen(false)} onApply={async png=>{const data=new FormData();data.append('files',new File([png],'marked.png',{type:'image/png'}));const [marked]=await api('/api/uploads',{method:'POST',body:data});setInputs(old=>[marked,...old.slice(1)]);}}/>:null}
    <ConnectionNotice/>
    <JobQueue api={api} state={queue.state} error={queue.error} refresh={queue.refresh} open={queueOpen} onOpen={setQueueOpen} onRestore={restoreJob} onView={viewJob} modelTitle={id=>modelState?.models.find(m=>m.id===id)?.title||id}/>
    {tab==='image'&&swapTools.length?<nav className="video-workflow-nav" role="tablist" aria-label="Image tools">
      <button role="tab" aria-selected={imageSection==='create'} onClick={()=>showImageSection('create')}>Create and edit</button>
      <button role="tab" aria-selected={imageSection==='swap'} onClick={()=>showImageSection('swap')}>Swap</button>
    </nav>:null}
    {tab==='video'?<nav className="video-workflow-nav" role="tablist" aria-label="Video tools">
      <button role="tab" aria-selected={videoSection==='multi-reference'} onClick={()=>showVideo('multi-reference')}>Image to video</button>
      <button role="tab" aria-selected={videoSection==='create'} onClick={()=>showVideo('create')}>Text or keyframes</button>
      <button role="tab" aria-selected={videoSection==='reference'} onClick={()=>showVideo('reference')}>Video to video</button>
      <button role="tab" aria-selected={videoSection==='orbit'} onClick={()=>showVideo('orbit')}>360 Orbit</button>
      <button role="tab" aria-selected={videoSection==='upscale'} onClick={()=>showVideo('upscale')}>Upscale</button>
    </nav>:null}
    {tab==='video'?<div className="mobile-tool-picker"><label htmlFor="mobile-video-tool">Video tool</label><select id="mobile-video-tool" value={videoSection} onChange={e=>{showVideo(e.target.value as VideoSection);}}><option value="multi-reference">Image to video</option><option value="create">Text or keyframes</option><option value="reference">Video to video</option><option value="orbit">360 Orbit</option><option value="upscale">Upscale</option></select></div>:null}
    <ModelDefaultsScope model={params.model_id} scope="image" bindings={imageDefaultBindings} draftKey="image-studio-draft-v1" restoreKey={imageRestoreKey}><main hidden={tab!=='image'} className={`image-main studio-main ${railOpen?'with-rail':imageJobs.length?'with-queue':'rail-closed'}`}>
      <div className="studio-grid">
      <div className="studio-controls">
      <div className="controls-scroll">
        <section className="panel model-panel">
          <ImageSettings part="model" {...settingsProps} plan={imageSection==='swap'?swapPlan:createPlan?.speed_lora?createPlan:null} planLabel={imageSection==='swap'?'Swap':'Runs'} swapFamilies={imageSection==='swap'?[...new Set(swapTools.flatMap(t=>t.families))]:undefined}/>
          <LoraStack compact value={params.loras} onChange={next=>change('loras',next)} loras={userLoras} family={family} pixels={pixels} prompt={params.prompt} onPrompt={next=>change('prompt',next)} disabled={locked}
            model={chosenModel} onAuthorSettings={applyAuthorSettings} hasModel={f=>!!authorModel(f)}
            onBrowse={()=>{setTab('civitai');window.history.replaceState(null,'','#civitai');}}/>
          {imageSection==='create'&&createPlan?<SwapLoras plan={createPlan} speedStrength={params.speed_lora_strength??1} disabled={locked} onSpeed={(on,strength)=>setParams(p=>({...p,speed_lora:on,speed_lora_strength:strength}))} strength={activeTool?{label:'Effect strength',value:params.lora_strength??toolStrength(activeTool,params.tool_options),min:0.2,max:1.5,fallback:toolStrength(activeTool,params.tool_options),onChange:value=>setParams(p=>({...p,lora_strength:value}))}:undefined}/>:null}
          {imageSection==='swap'&&swapPlan&&swapTool?<ModelDefaultsScope model={params.model_id} scope={'swap-'+swapTool.id} draftKey="qwen-swap-v1" restoreKey={imageRestoreKey} bindings={bindSettings(swap,patch=>setSwap({...swap,...patch}),{'Effect strength':'strength'})}><SwapLoras plan={swapPlan} speedStrength={swap.speedStrength??1} disabled={locked} onSpeed={(on,strength)=>setSwap({...swap,speedLora:on,speedStrength:strength})}
            strength={{label:swapPlan.loras.some(l=>l.id.startsWith('krea2'))?'Swap strength':'Keep original face',value:swap.strength,min:0.1,max:2,fallback:swapTool?.strength??1,hint:swapPlan.loras.some(l=>l.id.startsWith('krea2'))?'The author uses 1.0.':"Higher: more of picture 1's face. Lower: more of picture 2's face.",onChange:value=>setSwap({...swap,strength:value??swapTool?.strength??1})}}/></ModelDefaultsScope>:null}
          {adviceShown?<small className="lora-auto-settings" role="status">{loraAdvice!.note}</small>:null}
          {config&&missingModels.length?<small className="model-missing">Missing model files download automatically when you generate.</small>:null}
        </section>
        {imageSection==='swap'?<SwapPanel modelId={params.model_id} restoreKey={imageRestoreKey} effectiveCfg={swap.cfg??swapPlan?.cfg??chosenModel?.generation_defaults?.cfg??(swapNegative(swap).trim()?2:1)} modelFamily={chosenModel?.family} options={{...params,...swapSize}} onOptions={values=>{const size=Object.fromEntries(Object.entries(values).filter(([key])=>key in swapSize));const rest=Object.fromEntries(Object.entries(values).filter(([key])=>!(key in swapSize)));if(Object.keys(size).length)setSwap({...swap,...size});if(Object.keys(rest).length)setParams(p=>({...p,...rest}));}} steps={<ImageSettings idPrefix="swap" part="steps" {...settingsProps} plan={swapPlan} tool={swapTool} value={{...params,...swapSampling,negative_prompt:swapNegative(swap)}} onChange={setSwapSampling}/>} advanced={<ImageSettings idPrefix="swap" part="advanced" {...settingsProps} plan={swapPlan} tool={swapTool} value={{...params,...swapSampling,negative_prompt:swapNegative(swap)}} onChange={setSwapSampling}/>} api={api} tools={pageSwapTools} value={swap} onChange={setSwap} disabled={locked} results={history} people={people} onPeople={loadPeople} revision={libraryRevision}/>:null}
        {imageSection==='swap'?<MobileResults results={history.filter(item=>!!item.image_url)} onOpen={openResultPreview} onUseSettings={useMobileSettings} disabled={locked} onGallery={()=>{setTab('gallery');window.history.replaceState(null,'','#gallery');}}/>:null}
        <section ref={inputPanel} className="panel input-panel" hidden={imageSection==='swap'}><div className="section-heading"><h2>{t('input')}</h2>{inputs.length>0?<span className="count">{inputs.length}/10</span>:null}{activeTool?.needs_box&&inputs.length?<button className="text-button mark-object" disabled={locked} onClick={()=>setBoxOpen(true)}><SquareDashed size={15}/>Mark object</button>:null}{inputs.length>0?<button className="text-button" disabled={locked} onClick={()=>setInputs([])}>Clear</button>:null}<button type="button" className={`library-toggle ${libraryOpen?'open':''}`} aria-expanded={libraryOpen} aria-haspopup="dialog" onClick={()=>libraryOpen?closeLibrary():openLibrary(null)} title="Uploads, results and people"><Images size={14}/>Library<ChevronDown size={13}/></button></div>
          <div className={`upload-area ${dragging?'dragging':''} ${inputs.length?'has-images':''}`} onDragOver={e=>{e.preventDefault();if(!locked)setDragging(true);}} onDragLeave={()=>setDragging(false)} onDrop={e=>{e.preventDefault();setDragging(false);upload(Array.from(e.dataTransfer.files));}}>
            {inputs.length===0?<div className="upload-empty-row"><FilePicker className="upload-empty compact" disabled={locked} multiple onFiles={upload}><UploadCloud size={20} strokeWidth={1.6}/><span><strong>{uploading?t('uploadBusy'):'Upload images'}</strong></span></FilePicker><button type="button" className="upload-library" disabled={locked} onClick={()=>openLibrary(null)}><Images size={18} strokeWidth={1.6}/><span>Pick from library</span></button></div>:<div className="input-grid">{inputs.map((input,index)=><div className={`input-thumb ${libraryOpen&&replaceAt===index?'target':''}`} key={index}><button className="thumb-preview checker" disabled={locked} title="Replace this image" onClick={()=>openLibrary(index)} aria-label={`Replace image ${index+1}`}><img src={input.url} alt={`${index+1}. ${input.name}`}/></button><button className="thumb-zoom" onClick={()=>openInputPreview(input.url)} aria-label={`${t('preview')} ${index+1}`} title={t('preview')}><Expand size={12}/></button><span className="image-number">{index+1}{activeTool?.inputs.labels[index]?` · ${activeTool.inputs.labels[index]}`:''}{inputPerson(input,people)?` · ${inputPerson(input,people)!.name}`:''}</span><button disabled={locked} className="remove-image" onClick={()=>setInputs(old=>old.filter((_,i)=>i!==index))} aria-label={`${t('remove')} ${index+1}`}><X size={13}/></button><PersonPhotoCycle image={input} people={people} disabled={locked} onChange={next=>setInputs(old=>old.map((item,i)=>i===index?next:item))}/><AddToPerson api={api} image={input} people={people} onPeople={loadPeople} disabled={locked} onLinked={next=>setInputs(old=>old.map((item,i)=>i===index?next:item))}/><ImageLabel number={index+1} person={!!inputPerson(input,people)} value={input.subjectLabel||inputPerson(input,people)?.label||undefined} disabled={locked} onChange={subjectLabel=>setInputs(old=>old.map((item,i)=>i===index?{...item,subjectLabel}:item))}/><div className="reorder"><button disabled={locked||index===0} onClick={()=>reorder(index,-1)} aria-label={`${t('left')} ${index+1}`}><ChevronLeft size={14}/></button><button type="button" className="thumb-upscale" disabled={locked||input.id.startsWith('example:')} onClick={()=>upscaleInput(input,index)} aria-label={`Upscale image ${index+1}`}>Upscale</button><button disabled={locked||index===inputs.length-1} onClick={()=>reorder(index,1)} aria-label={`${t('right')} ${index+1}`}><ChevronRight size={14}/></button></div></div>)}{inputs.length<10?<button className="add-image" disabled={locked} onClick={()=>openLibrary(null)} aria-label="Add an image" title="Add an image"><Plus size={25}/></button>:null}</div>}
          </div>
          {inputUpscales.map(track=><InputUpscaleStatus key={track.jobId} api={api} track={track} inputs={inputs}
            onReplace={(inputId,image)=>setInputs(old=>old.map(item=>item.id===inputId?{...item,id:image.id,url:image.url}:item))}
            onAdd={image=>setInputs(old=>old.some(i=>i.id===image.id)||old.length>=10?old:[...old,image])}
            onDismiss={()=>setInputUpscales(old=>old.filter(item=>item.jobId!==track.jobId))}/>)}
          {flux3?<p className="input-note" title="It prepares a detailed image description locally before generation.">FLUX.3 uses up to 10 reference images.</p>:null}
          {krea&&inputs.length&&!activeTool?<div className="input-note krea-mode" role="group" aria-label="Krea 2 input photos">{([['identity','Keep the person'],['style','Style reference']] as const).map(([mode,label])=><button key={mode} type="button" aria-pressed={(params.krea_reference_mode??'identity')===mode} disabled={locked} onClick={()=>change('krea_reference_mode',mode)}>{label}</button>)}{(params.krea_reference_mode??'identity')==='identity'?<><p title="Add the person's photo. A scene photo is optional. Without a framing in the prompt, Krea copies the photo's crop.">Say the framing in the prompt (full body, close-up), or Krea copies the photo's crop.</p>{(()=>{const card=inputs.length>1&&params.krea_scene!=null?2.5:2,value=params.krea_identity_strength??card;return <div className="krea-strength" title="How closely the result follows the person photo (the card's ref_boost). Higher keeps the face closer; too high copies the photo's pose and crop."><SettingLabel htmlFor="krea-strength" label="Identity strength" value={String(card)} onReset={()=>change('krea_identity_strength',null)}>Identity strength <b>{value.toFixed(1)}</b><Nudge label="identity strength" value={value} step={0.1} min={0.5} max={4} disabled={locked} onChange={next=>change('krea_identity_strength',next)}/></SettingLabel><input id="krea-strength" type="range" min="0.5" max="4" step="0.1" disabled={locked} value={value} onChange={e=>change('krea_identity_strength',Number(e.target.value))}/><small>Higher: closer to the photo's face, but also its pose and crop. Lower: freer pose, less likeness.</small></div>;})()}{inputs.length>1?<div className="krea-roles"><label>Person<select value={Math.min(params.krea_person??0,inputs.length-1)} disabled={locked} onChange={e=>{const person=Number(e.target.value);setParams(p=>({...p,krea_person:person,krea_scene:p.krea_scene===person?null:p.krea_scene}));}}>{inputs.map((input,i)=><option key={input.id+i} value={i}>{photoName(input,i)}</option>)}</select></label><label>Scene (optional)<select value={params.krea_scene!=null&&params.krea_scene<inputs.length?params.krea_scene:''} disabled={locked} onChange={e=>change('krea_scene',e.target.value===''?null:Number(e.target.value))}><option value="">None</option>{inputs.map((input,i)=>i===Math.min(params.krea_person??0,inputs.length-1)?null:<option key={input.id+i} value={i}>{photoName(input,i)}</option>)}</select></label></div>:null}</>:<p>Colors, light and texture from 1 to 3 photos. The person is not kept.</p>}</div>:null}
          {(zimage||moss)&&inputs.length?<div className="input-note zimage-redraw"><p title={`It does not keep a person's identity: faces and details can change. To keep a person, choose Qwen-Image-2.1, FLUX.3 Klein or Krea 2${moss?', or train a MOSS LoRA of the person':''}.`}>Redraws image 1 from your prompt. Faces can change.{inputs.length>1?' Uses one input image.':''}</p>
            <SettingLabel htmlFor="zimage-denoise" label="Change" value="70%" onReset={()=>change('denoise',null)}>Change <b>{Math.round((params.denoise??0.7)*100)}%</b></SettingLabel><input id="zimage-denoise" type="range" min="0.1" max="1" step="0.05" disabled={locked} value={params.denoise??0.7} onChange={e=>change('denoise',Number(e.target.value))}/><small>Low keeps more of image 1. 100% ignores it.</small></div>:null}
          <LibraryPopover open={libraryOpen} onClose={closeLibrary} anchor={inputPanel}>
            <InputLibrary api={api} inputs={inputs} onChange={setInputs} results={history} people={people} onPeople={loadPeople} replace={replaceAt!==null} target={replaceAt??0} revision={libraryRevision} disabled={locked} limit={10} onUpload={upload} onClose={closeLibrary}/>
          </LibraryPopover>
        </section>
        <section className="panel controls-panel" hidden={imageSection==='swap'}>
        <div className="prompt-row"><label htmlFor="prompt" className="prompt-label">{t('prompt')}{inputs.length?<small>Optional</small>:null}<span className="mode-label">{inputs.length?(zimage?'Image to image':krea?((params.krea_reference_mode??'identity')==='style'?'Style reference':'Keep the person'):t('editMode')):t('createMode')}</span></label><button type="button" className="prompt-chip" disabled={!examples.length} onClick={()=>setShowExamples(true)} title="Example prompts"><LayoutGrid size={13}/>Examples</button><PromptHistory scope="images" disabled={locked} onLoad={text=>{change('prompt',text);promptRef.current?.focus();}} onRun={text=>generate(text)}/></div>
        <PromptInput ref={promptRef} id="prompt" dir="auto" className="prompt-input" placeholder={t('placeholder')} value={displayImageMentions(params.prompt)} disabled={locked} onValueChange={text=>change('prompt',text)} tools={knownFrom?<KnownPeoplePicker people={knownFrom.known_people!} source={knownFrom.title} prompt={params.prompt} disabled={locked} onInsert={text=>{
          // At the cursor, with a separating space; a later edit keeps the person in the prompt text.
          const el=promptRef.current,current=params.prompt,at=el?el.selectionStart??current.length:current.length,end=el?el.selectionEnd??at:at;
          const before=current.slice(0,at),pad=before&&!/\s$/.test(before)?(/,\s*$/.test(before)?' ':', '):'';
          change('prompt',before+pad+text+current.slice(end));}}/>:null} mentions={[...imageMentionOptions(inputs,people,image=>setInputs(old=>[...old,image]),zimage?1:activeTool?.inputs.max??10)]}/>
        {flux3||activeTool?null:<label className="checkbox-label enhance-toggle" title="Each model's official prompt rewriter runs on this computer and writes a detailed English prompt from yours, in any language. Text to show in the picture stays as you wrote it. About 10 seconds; a repeat is instant. Off by default."><input type="checkbox" checked={!!params.enhance} disabled={locked} onChange={e=>change('enhance',e.target.checked)}/>Enhance prompt</label>}
        {flux3||activeTool||!params.enhance?null:<EnhancerPick api={api} family={family} disabled={locked}/>}
        <MobileResults results={history.filter(item=>!!item.image_url)} onOpen={openResultPreview} onUseSettings={useMobileSettings} disabled={locked} onGallery={()=>{setTab('gallery');window.history.replaceState(null,'','#gallery');}}/>
        <ImageSettings part="main" {...settingsProps}/>
        <ImageSettings part="steps" {...settingsProps} plan={createPlan}/>
        <details className="disclosure advanced-settings" onToggle={e=>{const el=e.currentTarget;if(el.open)requestAnimationFrame(()=>el.scrollIntoView({block:'nearest',behavior:'smooth'}));}}><summary><ChevronRight size={15}/>{t('advanced')}<span className="summary-hint">{params.randomize_seed?'Random seed':`${t('seed')}: ${params.seed}`} · {params.sampler||chosenModel?.generation_defaults?.sampler||'euler'}{params.negative_prompt?.trim()?' · Negative prompt':''}</span></summary>
          <div className="details-body">
            <ImageSettings part="advanced" {...settingsProps} plan={createPlan}/>
            <fieldset className="advanced-grid" disabled={locked}>
              <div className="field"><SettingLabel htmlFor="seed" label="Seed" value={'Random'} onReset={()=>setParams(p=>({...p,seed:0,randomize_seed:true}))}>{t('seed')}</SettingLabel><div className="seed-row"><input id="seed" type="number" min="0" max="2147483647" step="1" value={params.seed} disabled={params.randomize_seed} onChange={e=>change('seed',Number(e.target.value))}/><label className="checkbox-label" title={t('randomize')}><input type="checkbox" checked={params.randomize_seed} onChange={e=>change('randomize_seed',e.target.checked)}/>Random</label></div></div>
              <div className="field"><SettingLabel htmlFor="vae" label="Decoder" value={'Auto'} onReset={()=>setParams(p=>({...p,vae_id:"auto",vae_choice:false}))}><span title="Decoder (VAE)">Decoder</span></SettingLabel><select id="vae" value={params.vae_id} title="Auto uses the VAE the model page names. Otherwise Texture fix: the same decoder with fewer checkerboard artifacts." onChange={e=>setParams(p=>({...p,vae_id:e.target.value,vae_choice:true}))}><option value="auto">Auto · {vaeTitle(autoVae)}</option>{vaes.map(v=><option key={v.id} value={v.id}>{vaeTitle(v.id)}{v.ready?'':` · ${(v.bytes/1e9).toFixed(1)} GB download`}</option>)}</select></div>
              <div className="field span-2"><SettingLabel htmlFor="negative" label="Negative prompt" value={'Empty'} onReset={()=>change('negative_prompt','')}>{t('negative')}</SettingLabel><PromptInput id="negative" dir="auto" rows={2} placeholder="What to exclude (adds a guidance pass, slower)" title={t('negativeHelp')} value={displayImageMentions(params.negative_prompt)} onValueChange={text=>change('negative_prompt',text)} mentions={imageMentionOptions(inputs,people)}/></div>
              <div className="field span-2"><SettingLabel htmlFor="folder" label="Output folder" onReset={()=>change('output_dir',config?.output_dir||'')}>{t('folder')}</SettingLabel><input id="folder" dir="ltr" value={params.output_dir} title={params.output_dir} onChange={e=>change('output_dir',e.target.value)}/></div>
            </fieldset>
          </div>
        </details>
        </section>
      </div>
      <div className="panel generate-dock">
        {busy&&submittedJob?<div className="submission-status"><span role="status"><Loader2 size={14} className={submittedJob.status==='queued'?'':'spin'}/>{submittedJob.status==='queued'?`${submittedJob.position?`Queued · #${submittedJob.position}`:'Added to queue'}${queue.state.paused?' · Paused':''}`:`${stageLabel(submittedJob)}${stepLabel(submittedJob)?' · '+stepLabel(submittedJob):''}`}</span><button className="text-button" onClick={async()=>{try{if(pollId){await api(`/api/jobs/${pollId}/cancel`,{method:'POST'});await queue.refresh();}}catch(e){setError(String(e));}}}>{submittedJob.status==='queued'?'Remove':'Stop'}</button></div>:null}
        <div className="generate-row"><div className="generate-controls"><button className="button primary" disabled={locked||!ready} onClick={()=>imageSection==='swap'?generateSwap():generate()} title={t('keyboard')}>{submitting?<span className="train-spinner" aria-hidden="true"/>:null}<span>{submitting?'Adding…':queue.count||queue.state.paused?'Add image to queue':t('generate')}</span></button></div></div>
        {error?<div className="message error" role="alert"><span>{error}</span><button className="icon-button" aria-label={t('close')} onClick={()=>setError('')}><X size={16}/></button></div>:null}
        {notice?<div className="notice" role="status">{notice}</div>:null}
        {job?.status==='cancelled'?<div className="notice">{t('cancelled')}</div>:null}
      </div>
      </div>
        <section className="panel result-panel studio-result"><div className="section-heading"><h2>{t('result')}</h2>{result?<span className="result-size" dir="ltr">{result.width} × {result.height} · PNG</span>:null}{result?.request.images?.length?<label className="checkbox-label compare-toggle"><input type="checkbox" checked={compareResult} onChange={e=>setCompareResult(e.target.checked)}/>Compare</label>:null}{result?<button className="text-button" onClick={clearResult} title="Empty the Result view. The image stays in Recent and the gallery.">Clear</button>:null}{result?<button className="icon-button" onClick={()=>openResultPreview(result.image_url!)} aria-label={t('preview')}><Expand size={17}/></button>:null}</div>
          <ResultUpscales jobs={imageJobs.filter(item=>item.kind==='image-upscale')}/>
          <div className="result-canvas checker">{liveJob?<div className="result-live" role="status" aria-label="Live preview">{liveJob.preview_url?<img src={`${liveJob.preview_url}?v=${liveJob.preview_updated_at}`} alt="Live preview of the image being made"/>:null}<span>{liveJob.stage==='generating'&&liveJob.total_steps?`Step ${liveJob.step} of ${liveJob.total_steps}`:stageLabel(liveJob)}</span>{liveJob.stage==='generating'&&liveJob.total_steps?<i style={{width:`${Math.round(100*liveJob.step/liveJob.total_steps)}%`}}/>:null}</div>:null}{resultIndex>=0&&shownResults.length>1?<><button type="button" className="result-step prev" disabled={resultIndex<=0} onClick={()=>stepResult(-1)} aria-label="Newer result" title="Newer (←)">‹</button><button type="button" className="result-step next" disabled={resultIndex>=shownResults.length-1} onClick={()=>stepResult(1)} aria-label="Older result" title="Older (→)">›</button></>:null}{result&&compareResult&&result.request.images?.length?<CompareView before={inputImage(result.request.images[0]).url} after={result.image_url!} alt={result.request.prompt}/>:result?<ProgressiveImage className="result-image" src={result.image_url!} alt={result.request.prompt} onOpen={()=>openResultPreview(result.image_url!)}/>:<div className="result-empty"><ImageIcon size={42} strokeWidth={1.4}/><span>{t('empty')}</span></div>}
          </div>
          {liveJob||result?<GenerationTiming job={(liveJob??result)!} models={modelState}/>:null}{(()=>{
            // The pictures used: the running job's while it develops, else the result's. A swap names them (Scene, Head)
            // and the person who got the new head.
            const shown=(liveJob??result) as Job|null;if(!shown?.request?.images?.length)return null;
            const used=tools.find(t=>t.id===shown.request.tool_id);
            const chosen=Object.entries((shown.request as {swap_selection?:{assignments?:Record<string,string|null>}}).swap_selection?.assignments||{}).find(([,v])=>v)?.[0];
            return <div className="result-inputs" aria-label="Pictures used">{shown.request.images.map((id:string,i:number)=><figure key={id+i}><button type="button" onClick={()=>{setPreviewList(shown.request.images.map((x:string)=>`/api/uploads/${x}`));setLightbox(`/api/uploads/${id}`);}} aria-label={`View picture ${i+1}`}><img src={`/api/uploads/${id}`} alt=""/></button><figcaption>{used?.inputs.labels[i]??`Image ${i+1}`}{i===0&&chosen?` · ${chosen.replace('face-','person ')}`:''}</figcaption></figure>)}</div>;})()}
          {result&&!liveJob?<div className="result-actions"><button type="button" className="button secondary" disabled={locked} onClick={()=>void useAsFirstInput(result.id)} title={imageSection==='swap'?'Make this result picture 1 (the scene)':'Make this result input 1'}>Use as input 1</button><button type="button" className="button secondary" disabled={locked} onClick={()=>useMobileSettings(result as unknown as SavedJob)}>Use settings</button><button type="button" className="button secondary" disabled={locked} onClick={async()=>{try{await api(`/api/jobs/${result.id}/retry`,{method:'POST'});setNotice('Added to queue with the same inputs, seed and settings.');queue.refresh();}catch(e){setError((e as Error).message);}}}>Run again</button><button type="button" className="button secondary" onClick={()=>upscaleResult(result)}>Upscale</button><a className="button secondary" href={`${result.image_url}?download=true`}><Download size={15}/>{t('download')}</a><UseImageAs jobId={result.id} disabled={locked}/><a className="icon-button" href={`/api/jobs/${result.id}/metadata`} aria-label={t('metadata')} title={t('metadata')}><Download size={15}/></a></div>:null}
          {rewrittenPrompt?<details className="result-rewrite" key={result?.id}><summary title="The prompt the model actually used"><ChevronRight size={12}/><b>Enhanced prompt</b><span dir="auto">{rewrittenPrompt}</span></summary><p dir="auto">{rewrittenPrompt}</p><button className="text-button" onClick={()=>copyPrompt(rewrittenPrompt)}>{copied?<Check size={13}/>:<Copy size={13}/>} {copied?t('copied'):t('copy')}</button></details>:null}
        </section>
      <ResultsRail open={railOpen} onToggle={()=>setRailOpen(o=>!o)} running={imageJobs} results={history} selected={result?.id}
        onSelect={item=>{setLiveFor(null);setResult(item as Job);if(!busy)setJob(item as Job);}} onShowLive={id=>setLiveFor(id)} onUseAsInput={async item=>{if(inputs.length>=10){setError(t('maxImages'));return;}try{const added=await api(`/api/jobs/${item.id}/reuse`,{method:'POST'});setInputs(old=>old.some(i=>i.id===added.id)?old:[...old,added]);}catch(e){setError((e as Error).message);}}}
        onUseSettings={item=>restoreJob(item)} onOpenGallery={()=>{setTab('gallery');window.history.replaceState(null,'','#gallery');}} onOpenFolder={()=>api('/api/open-output',{method:'POST'}).catch(err=>setError(String(err)))} onOpenQueue={()=>setQueueOpen(true)}/>
      </div>
      <footer><span className="local-status"><i/>{modelState?.remote?'Runs on RunPod GPU':t('local')}</span><span>{modelState?.remote?'Results saved on this computer':config?.device||''}</span></footer>
    </main></ModelDefaultsScope>
    <OrbitTab api={api} visible={tab==='video'&&videoSection==='orbit'} models={modelState} queue={queue.state} onQueued={queue.refresh} restoreRequest={orbitRestore} people={people} onPeople={loadPeople} results={history} onSendVideo={sendVideo}/>
    <VideoTab library={{people,onPeople:loadPeople,results:history}} addLora={pendingVideoLora?.target==='create'?pendingVideoLora:null} onSendVideo={sendVideo} visible={tab==='video'&&videoSection==='create'} api={api} models={modelState} modelId={videoModel} onModel={setVideoModel} onBusy={setVideoBusy} queueActive={queue.count>0||queue.state.paused} onQueued={queue.refresh} restoreRequest={videoRestore} openSettings={()=>setShowSettings(true)}/>
    <ReferenceTab library={{people,onPeople:loadPeople,results:history}} addLora={pendingVideoLora?.target==='reference'?pendingVideoLora:null} onSendVideo={sendVideo} videoInput={videoInput?.target!=='upscale'?videoInput:null} visible={tab==='video'&&videoSection==='reference'} api={api} token={token.current} models={modelState} restoreRequest={referenceRestore} onQueued={queue.refresh} openSettings={()=>setShowSettings(true)}/>
    <MultiReferenceTab modelChoice={multiModelChoice} addLora={pendingVideoLora?.target==='multi-reference'?pendingVideoLora:null} library={{people,onPeople:loadPeople,results:history}} onSendVideo={sendVideo} visible={tab==='video'&&videoSection==='multi-reference'} api={api} models={modelState} queue={queue.state} onQueued={queue.refresh} restoreRequest={multiRestore} onView={job=>setGalleryView({job,key:Date.now()})}/>
    <UpscaleTab onSendVideo={sendVideo} videoInput={videoInput?.target==='upscale'?videoInput:null} visible={tab==='video'&&videoSection==='upscale'} api={api} token={token.current} models={modelState} restoreRequest={upscaleRestore} onQueued={queue.refresh} openSettings={()=>setShowSettings(true)}/>
    <CivitaiPage api={api} visible={tab==='civitai'} models={modelState} onUse={applyInstalled}/>
    <TrainPage api={api} visible={tab==='train'} models={modelState} results={history} people={people} onPeople={loadPeople} onUse={id=>applyInstalled({id,kind:'image-lora'})}/>
    <Gallery onSendVideo={sendVideo} api={api} visible={tab==='gallery'} revision={queue.revision} models={modelState} onRestore={restoreJob} onQueued={queue.refresh} view={galleryView} onUpscale={upscaleResult}/>

    <dialog ref={modal} aria-label={showSettings?t('settings'):showHelp?t('help'):showExamples?t('examples'):activeExample?.title_en||t('preview')} onCancel={closeModal} onClick={e=>{if(e.target===modal.current)closeModal();}} className={lightbox?'lightbox':showSettings?'content-dialog settings-dialog':'content-dialog'} tabIndex={-1}>{!lightbox?<button className="dialog-close icon-button" onClick={closeModal} aria-label={t('close')}><X size={21}/></button>:null}
      {showSettings?<ModelSettings initialView={settingsView} lang={lang} api={api} state={modelState} onState={setModelState} selected={params.model_id} onSelect={id=>{if(id!==params.model_id)manualSampling.current.clear();setParams(p=>({...p,model_id:id,steps:p.model_id===id?p.steps:null,cfg:p.model_id===id?p.cfg:null,sampler:p.model_id===id?p.sampler:null,scheduler:p.model_id===id?p.scheduler:null}));setError('');}} videoSelected={videoModel} onVideoSelect={setVideoModel} encoder={params.text_encoder} onEncoder={id=>change('text_encoder',id)} busy={queue.count>0}/>:null}
      {lightbox?<header className="preview-topbar">
        <div className="preview-navigation">{previewIndex>=0&&previewList.length>1?<><button type="button" className="icon-button" disabled={previewIndex===0} onClick={()=>stepPreview(-1)} aria-label="Previous image"><ChevronLeft size={20}/></button><span>{previewIndex+1} / {previewList.length}</span><button type="button" className="icon-button" disabled={previewIndex===previewList.length-1} onClick={()=>stepPreview(1)} aria-label="Next image"><ChevronRight size={20}/></button></>:<span>{t('preview')}</span>}</div>
        {previewJob?<div className="preview-actions"><button type="button" className="button secondary" disabled={locked} onClick={()=>void useAsFirstInput(previewJob.id)}>Use as input 1</button><button type="button" className="button secondary" aria-label="Use settings" disabled={locked} onClick={()=>useMobileSettings(previewJob)}><RotateCcw size={17}/><span className="label-full">Use settings</span><span className="label-short">Settings</span></button><a className="button secondary" href={`${previewJob.image_url}?download=true`}><Download size={17}/>Download</a><UseImageAs jobId={previewJob.id} disabled={locked}/><button type="button" className="button secondary" onClick={()=>upscaleResult(previewJob)}>Upscale</button></div>:previewInput>=0&&!inputs[previewInput].id.startsWith('example:')?<div className="preview-actions"><button type="button" className="button secondary" disabled={locked} onClick={()=>upscaleInput(inputs[previewInput],previewInput)}>Upscale</button></div>:null}
        {previewBefore?<button type="button" className={`button secondary preview-compare ${previewCompare?'active':''}`} aria-pressed={previewCompare} onClick={()=>setPreviewCompare(v=>!v)} title={(previewJob as {kind?:string}|undefined)?.kind==='image-upscale'?'Before and after the upscale':'Input and result'}>Compare</button>:null}
        <button type="button" className="icon-button preview-close" onClick={closeModal} aria-label={t('close')}><X size={21}/></button>
      </header>:null}
      {lightbox?<ZoomView className="lightbox-image checker" src={lightbox} alt={t('preview')} onSwipe={stepPreview} separateControls before={previewCompare&&previewBefore?previewBefore:null}/>:null}
      {lightbox&&previewError?<p className="preview-action-error" role="alert">{previewError}</p>:null}
      {lightbox&&previewJob?<div className="preview-run-details"><GenerationTiming job={previewJob} models={modelState}/></div>:null}
      {showHelp&&tab==='video'&&videoSection==='create'?<div className="help-content"><h2>Video help</h2><p>Describe the scene, movement, camera and sounds. FastH3 creates video from text. Use first and last frames to control the start and end of a video. Use Image to video to place a referenced person or object in a new scene.</p><p>Steps follow the selected model and generation mode. Video is generated at 24 frames per second. You can change the seed, size and output folder in Advanced settings.</p><p>Missing models download automatically when you add a job. Supporting model files are selected automatically.</p><a href="https://huggingface.co/FastVideo/FastVideo-FastH3-8-Step-V2" target="_blank" rel="noreferrer">FastH3 model details</a><a href="https://huggingface.co/MiniMaxAI/MiniMax-H3" target="_blank" rel="noreferrer">MiniMax H3 model details</a></div>:null}
      {showHelp&&tab==='video'&&videoSection==='reference'?<div className="help-content"><h2>Video to video help</h2><p>Replace subjects uses ordered person images. Camera view changes the viewpoint without replacing people. Open the optional subject replacement section in Camera view to run both steps in order.</p><p>Choose the model and duration in this tab. Missing model files download automatically when you add a job.</p></div>:null}
      {showHelp&&tab==='video'&&videoSection==='multi-reference'?<div className="help-content"><h2>Reference images to video</h2><p>Add up to four subjects and an optional background. Describe the action using Image 1, Image 2 and the remaining image labels. Optional voices correspond to the first two images. This creates a new video; use Video to video to edit an existing clip.</p><p>The LoRA is an add-on that lets LTX use multiple reference images. It downloads with the selected model and required files. LTX requires access from your Hugging Face account. Double width and height multiplies each dimension by 2 and adds 3 processing steps.</p></div>:null}
      {showHelp&&tab==='video'&&videoSection==='upscale'?<div className="help-content"><h2>Upscale help</h2><p>Select a source video and model. Set the shorter side in pixels or multiply both dimensions by 2 or 4. The output dimensions appear below. Frames per batch and other model settings are in Advanced settings. Keep source audio is beside Add to queue.</p></div>:null}
      {showHelp&&tab==='image'?<div className="help-content"><h2>{t('help')}</h2><p>{t('helpText')}</p><blockquote dir="auto">{t('transparent')}</blockquote><button className="button primary" disabled={locked} onClick={()=>{change('prompt',t('transparent'));setShowHelp(false);promptRef.current?.focus();}}>{t('applyTransparent')}</button><a href="https://huggingface.co/spaces/Qwen/Qwen-Image-2.1" target="_blank" rel="noreferrer">Qwen-Image-2.1 · Hugging Face</a></div>:null}
      {showExamples?<div className="examples-dialog"><h2>{t('examples')}</h2><div className="examples-grid">{examples.map(example=><button key={example.id} className="example-card" onClick={()=>{setShowExamples(false);setActiveExample(example);}}><div className="example-thumbnail checker"><img loading="lazy" src={`/examples/${example.outputs[0]||example.inputs[0]}`} alt=""/></div><span>{example.title_en}<small>{example.inputs.length?`${example.inputs.length} ${t('sourceCount')}`:t('createMode')}</small></span><ChevronRight size={16}/></button>)}</div></div>:null}
      {activeExample?<div className="example-dialog"><h2>{activeExample.title_en}</h2><p className="muted">{t('exampleSource')}</p>{activeExample.inputs.length?<><h3>{t('input')}</h3><div className="example-inputs">{activeExample.inputs.map((name,i)=><figure key={name}><img src={`/examples/${name}`} alt={`${i+1}`}/><figcaption>{i+1}</figcaption></figure>)}</div></>:null}<h3>{t('reference')}</h3><div className="example-outputs">{activeExample.outputs.map(name=><img key={name} src={`/examples/${name}`} alt={t('reference')} className="checker"/>)}</div><h3>{t('fullPrompt')}</h3><p className="example-prompt" dir="auto">{activeExample.prompt}</p><button className="button primary" disabled={locked} onClick={()=>applyExample(activeExample)}>{t('useExample')}</button></div>:null}
    </dialog>
    {upscaleTarget?<ImageUpscaleDialog key={upscaleTarget.key} source={upscaleTarget.source} preset={upscaleTarget.preset} api={api} models={modelState} onClose={()=>setUpscaleTarget(null)} onView={viewJob}
      onQueued={created=>{queue.refresh();const inputId=upscaleTarget.source.inputId;if(inputId)setInputUpscales(old=>[...old.filter(item=>item.inputId!==inputId),{jobId:created.id,inputId}]);}}/>:null}
    <DownloadPill api={api} state={modelState}/>
  </>;
}
createRoot(document.getElementById('root')!).render(<React.StrictMode><ModelDefaultsRoot><ImageDestinations><App/></ImageDestinations></ModelDefaultsRoot></React.StrictMode>);
