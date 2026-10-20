import {useEffect,useLayoutEffect,useRef,useState} from 'react';
import {Check,Download,ExternalLink,Heart,Link2,Loader2,LogOut,Search,X} from 'lucide-react';
import type {Api,ModelState,SamplingProfile} from './ModelSettings';

type Preview={url:string;type:string;nsfw_level:number};
type Card={id:number;name:string;type:string;creator:string;downloads:number;likes:number;preview:Preview|null;versions:{id:number;name:string;base_model?:string;variant?:'standard'|'turbo'|'unknown'}[];installed:boolean};
type FileInfo={id:number;name:string;type:string;bytes:number;format:string;precision:string;primary:boolean;family?:string;modes?:string[];generation_defaults?:SamplingProfile};
type Version={id:number;name:string;base_model:string;problem?:string|null;ltx_variant?:'dev'|'distilled'|null;generation_defaults?:SamplingProfile;area:'image'|'video'|null;supported:boolean;trained_words:string[];description:string;files:FileInfo[];default_file:number|null;suggested_strength:number|null;turbo_hint:boolean;turbo_known?:boolean;turbo_source?:'tag'|'name'|'page'|'model name'|null;author_settings?:{steps:number;cfg?:number;sampler?:string;scheduler?:string;quote:string}|null;vae_hint?:string|null;images:Preview[]};
type Arch='qwen'|'krea'|'zimage'|'flux3';
/** The image architecture of a Civitai base model, its name, and its word for the non-Turbo type. */
const archOf=(base?:string):Arch=>base?.startsWith('Flux.3 Klein')?'flux3':base?.startsWith('Krea 2')?'krea':base?.startsWith('ZImage')?'zimage':'qwen';
const archName={flux3:'FLUX.3 Klein',qwen:'Qwen-Image-2.1',krea:'Krea 2',zimage:'Z-Image'} as const;
const standardWord={flux3:'Base',qwen:'Standard',krea:'Raw',zimage:'Base'} as const;
/** The architecture a Civitai base model tag names, for every architecture the app runs (images and video). */
const baseName=(base?:string)=>base?.startsWith('Flux.3 Klein')?'FLUX.3 Klein':base?.startsWith('LTX')?'LTX 2.5':base==='MiniMax H3'?'MiniMax H3':base?.startsWith('Krea')?'Krea 2':base?.startsWith('ZImage')?'Z-Image':base?.startsWith('Qwen')?'Qwen-Image-2.1':base||'Unknown model';
/** Architecture and type of an image checkpoint version, as one label: "Krea 2 Turbo", "Z-Image Base". */
const typeLabel=(base:string|undefined,variant:'standard'|'turbo'|'unknown'|undefined)=>`${archName[archOf(base)]} ${variant==='turbo'?(archOf(base)==='flux3'?'Distilled':'Turbo'):variant==='unknown'?'(type not stated)':standardWord[archOf(base)]}`;
/** Why a version counts as Turbo or not, when the Civitai tag does not say it. */
const typeSource=(v:Version)=>v.turbo_source==='page'?`${v.turbo_hint?'Turbo':standardWord[archOf(v.base_model)]} by the page settings; the Civitai tag says only "${v.base_model}"`:v.turbo_source==='name'?`${v.turbo_hint?'Turbo':standardWord[archOf(v.base_model)]} by the version name; the Civitai tag says only "${v.base_model}"`:v.turbo_source==='model name'?`${v.turbo_hint?'Turbo':standardWord[archOf(v.base_model)]} by the model name; the Civitai tag says only "${v.base_model}"`:'';
/** Steps, CFG and sampler for a version: the page's values, then the defaults for what the page does not say. */
function settingsText(a:Version['author_settings'],arch:Arch,turbo:boolean){const name=(s:string)=>s.replace(/_/g,' ');const page=[a?.steps?`${a.steps} steps`:'',a?.cfg?`CFG ${a.cfg}`:'',a?.sampler?name(a.sampler)+(a.scheduler?` / ${name(a.scheduler)}`:''):a?.scheduler?name(a.scheduler):''].filter(Boolean);const own=arch!=='qwen';const preset=[a?.steps?'':`${arch==='flux3'?(turbo?16:50):turbo?8:arch==='krea'?52:arch==='zimage'?30:25} steps`,a?.cfg?'':own||turbo?`CFG ${turbo?1:arch==='flux3'?3.5:arch==='krea'?4.5:4}`:'CFG 2 with a negative prompt, else 1',a?.sampler?'':arch==='zimage'?'res multistep':'euler'].filter(Boolean);return [page.length?`${page.join(', ')} (from the page)`:'',preset.length?`${preset.join(', ')} (${own?archName[arch]:'app'} default)`:''].filter(Boolean).join('; ');}
function videoSettingsText(p:SamplingProfile|undefined,ltx?:string|null){
  if(ltx||p?.variant)return `8 steps, CFG 1, Euler ancestral. References, text or first frame: optional 2× refinement adds 3 steps. First and last frame: 8 steps, eta 0. Duration follows 8n+1 frames at 24 fps.`;
  if(!p)return 'Settings are not stated. Check the model page before selecting a file.';
  const range=p.steps_range?.join('–')||p.step_options?.join(' or ');const origin=p.field_sources?.steps;
  return `${p.steps} steps${range?` (author: ${range})`:''}, CFG ${p.cfg??1}, ${p.sampler?.replaceAll('_',' ')} / ${p.scheduler}. Shift ${p.video_shift} video / ${p.audio_shift} audio. ${p.scheduler_video_shift!=null&&p.scheduler_video_shift!==p.video_shift?`Sampling schedule shift ${p.scheduler_video_shift}/${p.scheduler_audio_shift}. `:''}${p.embedded_turbo?'Turbo included. ':''}${origin==='version'||origin==='model'?'Steps from the author.':origin==='attached_workflow'?'Steps from the author workflow.':origin==='examples'?'Steps from examples.':'Uses default steps; author does not state them.'} Unstated values use model defaults. 17n+5 frames at 24 fps; up to 1344 × 768 in this workflow.`;
}
type Details=Omit<Card,'versions'>&{description:string;url:string;versions:Version[];installed_versions:number[]};
type Account={connected:boolean;username:string;method:string};
export type InstalledItem={id:string;kind:string};

const gb=(bytes:number)=>bytes>=1e9?`${(bytes/1e9).toFixed(1)} GB`:`${Math.round(bytes/1e6)} MB`;
const count=(n:number)=>n>=1000?`${(n/1000).toFixed(n>=10000?0:1)}k`:String(n);

function Media({item,className,active=false}:{item?:Preview;className?:string;active?:boolean}){
  const [failed,setFailed]=useState(false);
  useEffect(()=>setFailed(false),[item?.url]);
  if(!item||failed)return <span className={`civitai-no-preview ${className||''}`}>Preview unavailable</span>;
  return item.type==='video'?<video key={item.url} className={className} src={item.url} muted loop autoPlay={active} controls={active} preload="metadata" playsInline onError={()=>setFailed(true)}/>:<img className={className} src={item.url} alt="" loading={active?'eager':'lazy'} decoding="async" onError={()=>setFailed(true)}/>;
}

/** Browse Civitai (civitai.red) for Qwen-Image-2.1, Krea 2, Z-Image and video LoRAs and models and add them to the app. */
export function CivitaiPage({api,visible,models,onUse}:{api:Api;visible:boolean;models:ModelState|null;onUse:(item:InstalledItem)=>void}){
  const [account,setAccount]=useState<Account|null>(null);const [login,setLogin]=useState<{state:string;user_code:string;url:string}|null>(null);
  const [keyMode,setKeyMode]=useState(false);const [key,setKey]=useState('');
  const [kind,setKind]=useState<'LORA'|'Checkpoint'>('LORA');const [area,setArea]=useState<'image'|'video'>('image');const [sort,setSort]=useState('Most Downloaded');const [variant,setVariant]=useState<'all'|'standard'|'turbo'>('all');const [arch,setArch]=useState('all');const [total,setTotal]=useState<number|null>(null);const [indexing,setIndexing]=useState<number|null>(null);
  const [query,setQuery]=useState('');const [search,setSearch]=useState('');
  const [cards,setCards]=useState<Card[]>([]);const [cursor,setCursor]=useState<string|null>(null);
  const [loading,setLoading]=useState(false);const [error,setError]=useState('');
  const [open,setOpen]=useState<Details|null>(null);const [opening,setOpening]=useState(0);
  const latestDetails=useRef(0);
  useEffect(()=>{if(!visible){latestDetails.current++;setOpen(null);setOpening(0);}return()=>{latestDetails.current++;};},[visible]);
  // The account state is asked again every minute while the page is open, so an ended sign-in shows Connect.
  useEffect(()=>{if(!visible)return;const check=()=>api('/api/civitai/status').then(setAccount).catch(()=>{});check();const timer=setInterval(check,60000);return()=>clearInterval(timer);},[visible,api]);
  useEffect(()=>{if(!visible)return;const timer=setTimeout(()=>setSearch(query),350);return()=>clearTimeout(timer);},[query,visible]);
  useEffect(()=>{if(!visible)return;if(/civitai\.(red|com)\/models\/\d+/.test(search)){openLink(search);return;}load(true);},[visible,kind,sort,search,area,variant,arch]);
  useEffect(()=>{if(!visible||indexing===null)return;const timer=setTimeout(()=>load(true),2500);return()=>clearTimeout(timer);},[indexing,visible]);
  useEffect(()=>{if(!login)return;let alive=true;const timer=setInterval(async()=>{try{const next=await api(`/api/civitai/login/${login.state}`);if(!alive)return;
    if(next.status==='connected'){setAccount(next);setLogin(null);}else if(next.status!=='waiting'){setError(next.status==='denied'?'Sign-in was declined.':'Sign-in expired. Try again.');setLogin(null);}}catch{}},2000);
    return()=>{alive=false;clearInterval(timer);};},[login,api]);
  // Only the answer to the newest request fills the grid: an older, slower answer must not replace it.
  const latest=useRef(0);
  async function load(reset:boolean){
    const request=++latest.current;setLoading(true);setError('');
    try{const params=new URLSearchParams({kind,sort,query:search,area,variant,arch});if(!reset&&cursor)params.set('cursor',cursor);
      const page=await api(`/api/civitai/search?${params}`);if(request!==latest.current)return;setCards(old=>reset?page.items:[...old,...page.items.filter((c:Card)=>!old.some(o=>o.id===c.id))]);setCursor(page.next||null);setTotal(page.total??null);setIndexing(page.indexing?page.indexed:null);}
    catch(e){if(request===latest.current)setError((e as Error).message);}finally{if(request===latest.current)setLoading(false);}
  }
  async function show(id:number){const request=++latestDetails.current;setOpening(id);setError('');try{const details=await api(`/api/civitai/models/${id}`);if(request===latestDetails.current)setOpen(details);}catch(e){if(request===latestDetails.current)setError((e as Error).message);}finally{if(request===latestDetails.current)setOpening(0);}}
  async function openLink(link:string){const request=++latestDetails.current;latest.current++;setError('');try{const target=await api('/api/civitai/resolve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({link})});if(request!==latestDetails.current)return;setOpening(target.model_id);const details=await api(`/api/civitai/models/${target.model_id}`);if(request!==latestDetails.current)return;setOpen(details);setQuery('');}catch(e){if(request===latestDetails.current)setError((e as Error).message);}finally{if(request===latestDetails.current){setOpening(0);setLoading(false);}}}
  function closeDetails(){latestDetails.current++;setOpen(null);setOpening(0);}
  async function connect(){setError('');try{const flow=await api('/api/civitai/login',{method:'POST'});setLogin(flow);window.open(flow.url,'_blank','noopener,width=520,height=720');}catch(e){setError((e as Error).message);}}
  async function saveKey(){setError('');try{setAccount(await api('/api/civitai/key',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key})}));setKey('');setKeyMode(false);}catch(e){setError((e as Error).message);}}
  async function disconnect(){setAccount(await api('/api/civitai/disconnect',{method:'POST'}));}
  return <main className="civitai-page" hidden={!visible}>
    <div className="civitai-top">
      <div><h1>Civitai</h1><p>Find models and LoRAs for your images and videos.</p></div>
      <div className="civitai-account">{account?.connected?<><span className="connection-status connected"><Check size={13}/>{account.username||'Connected'}</span><button className="text-button" onClick={disconnect}><LogOut size={14}/>Disconnect</button></>
        :login?<div className="civitai-login"><Loader2 className="spin" size={16}/><span>Approve in the Civitai window. Code <b>{login.user_code}</b></span><a className="text-button" href={login.url} target="_blank" rel="noreferrer">Open again</a><button className="text-button" onClick={()=>setLogin(null)}>Cancel</button></div>
        :keyMode?<div className="civitai-login"><input type="password" placeholder="Civitai API key" value={key} onChange={e=>setKey(e.target.value)}/><button className="button primary" disabled={key.length<10} onClick={saveKey}>Save</button><button className="text-button" onClick={()=>setKeyMode(false)}>Cancel</button></div>
        :<><button className="button primary" onClick={connect}>Connect Civitai account</button><button className="text-button" onClick={()=>setKeyMode(true)}>Use an API key</button></>}</div>
    </div>
    <div className="civitai-toolbar">
      <div className="segmented">{([['image','Images'],['video','Video']] as const).map(([id,label])=><button key={id} aria-selected={area===id} role="tab" onClick={()=>{setArea(id);setArch('all');}}>{label}</button>)}</div>
      <div className="segmented">{([['LORA','LoRAs'],['Checkpoint','Models']] as const).map(([id,label])=><button key={id} aria-selected={kind===id} role="tab" onClick={()=>setKind(id)}>{label}</button>)}</div>
      <select aria-label="Model architecture" title="Model architecture" value={arch} onChange={e=>setArch(e.target.value)}>{(area==='image'?[['all','All architectures'],['qwen-image-2.1','Qwen-Image 2.1'],['krea-2','Krea 2'],['z-image','Z-Image'],['flux3-klein','FLUX.3 Klein']]:[['all','All architectures'],['minimax-h3','MiniMax H3'],['ltx25','LTX 2.5']]).map(([id,label])=><option key={id} value={id}>{label}</option>)}</select>
      <label className="gallery-search civitai-search"><Search size={17}/><input aria-label="Search Civitai or paste a link" value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search, or paste a Civitai link"/>{query?<button className="icon-button" aria-label="Clear" onClick={()=>setQuery('')}><X size={15}/></button>:<Link2 size={15} className="muted-icon"/>}</label>
      {kind==='Checkpoint'?<div className="segmented">{([['all','All'],['standard',arch==='krea-2'?'Raw':(arch==='z-image'||arch==='flux3-klein')?'Base':arch==='qwen-image-2.1'||area==='video'?'Standard':'Standard / Raw / Base'],['turbo',arch==='flux3-klein'?'Distilled':'Turbo']] as const).map(([id,label])=><button key={id} aria-selected={variant===id} role="tab" onClick={()=>setVariant(id)}>{label}</button>)}</div>:null}
      {indexing!==null?<span className="civitai-total" role="status"><Loader2 className="spin" size={13}/> Finding more models…</span>:total!==null?<span className="civitai-total">{total} found</span>:null}
      <select aria-label="Sort" value={sort} onChange={e=>setSort(e.target.value)}><option value="Most Downloaded">Downloads</option><option value="Highest Rated">Top rated</option><option>Newest</option></select>
    </div>
    {error?<div className="message error" role="alert"><span>{error}</span><button className="icon-button" onClick={()=>setError('')} aria-label="Close"><X size={15}/></button></div>:null}
    <div className="civitai-grid">{cards.map(card=><button key={card.id} className="civitai-card" onClick={()=>show(card.id)} disabled={opening===card.id}>
      <div className="civitai-thumb">{card.preview?<Media item={card.preview}/>:null}{card.installed?<span className="civitai-badge"><Check size={12}/>Added</span>:null}{opening===card.id?<span className="civitai-opening"><Loader2 className="spin" size={22}/></span>:null}</div>
      <strong>{card.name}</strong><span className="civitai-variants">{[...new Set(card.versions.map(v=>kind==='Checkpoint'&&area==='image'?typeLabel(v.base_model,v.variant):baseName(v.base_model)))].map(v=><em key={v} className={v.endsWith('Turbo')?'turbo':'standard'}>{v}</em>)}</span><small><span className="civitai-creator">{card.creator}</span><span className="civitai-count" title="Downloads"><Download size={11}/>{count(card.downloads)}</span><span className="civitai-count" title="Likes"><Heart size={11}/>{count(card.likes)}</span></small>
    </button>)}</div>
    {!cards.length&&!loading?<p className="civitai-empty">No {area==='image'?'Qwen-Image-2.1, Krea 2, Z-Image or FLUX.3 Klein':'MiniMax H3 or LTX 2.5'} {kind==='LORA'?'LoRAs':'models'} found.</p>:null}
    {loading?<div className="civitai-empty"><Loader2 className="spin" size={22}/></div>:cursor?<button className="button secondary civitai-more" onClick={()=>load(false)}>Load more</button>:null}
    {open?<CivitaiDetails key={open.id} api={api} model={open} models={models} prefer={arch} area={area} onClose={closeDetails} onAdded={()=>{setCards(old=>old.map(c=>c.id===open.id?{...c,installed:true}:c));}} onUse={item=>{closeDetails();onUse(item);}}/>:null}
  </main>;
}

function CivitaiDetails({api,model,models,prefer,area,onClose,onAdded,onUse}:{api:Api;model:Details;models:ModelState|null;prefer:string;area:'image'|'video';onClose:()=>void;onAdded:()=>void;onUse:(item:InstalledItem)=>void}){
  const dialog=useRef<HTMLDialogElement>(null);
  const supported=model.versions.filter(v=>v.supported);
  // The dialog opens on the first version of the area being browsed (an image search never opens on a video LoRA),
  // and with an architecture filter on, on the first version of that architecture.
  const wanted:Record<string,string>={'qwen-image-2.1':'Qwen-Image-2.1','krea-2':'Krea 2','z-image':'Z-Image','minimax-h3':'MiniMax H3','ltx25':'LTX 2.5','flux3-klein':'FLUX.3 Klein'};const opening=supported.find(v=>baseName(v.base_model)===wanted[prefer])??supported.find(v=>v.area===area)??supported[0];
  const [versionId,setVersionId]=useState(opening?.id??model.versions[0]?.id);
  const versionList=useRef<HTMLDivElement>(null);
  useEffect(()=>{const frame=requestAnimationFrame(()=>{const list=versionList.current;const chosen=list?.querySelector<HTMLElement>('[aria-checked="true"]');if(list&&chosen)list.scrollTop+=chosen.getBoundingClientRect().top-list.getBoundingClientRect().top-6;});return ()=>cancelAnimationFrame(frame);},[]);
  const version=model.versions.find(v=>v.id===versionId)!;
  const [fileId,setFileId]=useState<number|null>(version?.default_file??null);
  const [typeChoice,setTypeChoice]=useState<boolean|null>(null);
  const needsType=!!version&&version.area==='image'&&version.supported&&!version.turbo_known&&model.type!=='LORA';
  const turbo=needsType?!!typeChoice:!!version?.turbo_hint;
  const [image,setImage]=useState({versionId,index:0});const [error,setError]=useState('');const [adding,setAdding]=useState(false);
  const lora=model.type==='LORA';
  const entryId=`civitai-${lora?'lora':'model'}-${versionId}`;const video=version?.area==='video';const arch=archOf(version?.base_model);const krea=arch==='krea';
  const installedEntry=models?.models.find(m=>m.civitai?.version_id===versionId&&m.civitai?.file_id===fileId&&m.installed);const installed=!!installedEntry;
  const transfer=models?.downloads?.find(d=>d.requested_id===`civitai:${versionId}`);const mine=!!transfer;
  const running=adding||(mine&&transfer?.status==='running');
  useLayoutEffect(()=>{const node=dialog.current;const previous=document.activeElement as HTMLElement|null;const overflow=document.body.style.overflow;document.body.style.overflow='hidden';node?.showModal();node?.focus({preventScroll:true});return()=>{node?.close();document.body.style.overflow=overflow;previous?.focus({preventScroll:true});};},[]);
  function selectVersion(next:Version){setVersionId(next.id);setFileId(next.default_file??null);setImage({versionId:next.id,index:0});setTypeChoice(null);setError('');}
  useEffect(()=>{if(!adding||!mine)return;if(transfer?.status==='succeeded'){setAdding(false);onAdded();}else if(transfer?.status==='failed'||transfer?.status==='cancelled'){setAdding(false);setError(transfer.message||'The download did not finish.');}},[transfer?.status,adding,mine]);
  async function add(){setError('');setAdding(true);try{await api('/api/civitai/install',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({version_id:versionId,file_id:fileId,turbo,title:model.versions.length>1?`${model.name} · ${version?.name}`:model.name})});}catch(e){setAdding(false);setError((e as Error).message);}}
  const files=version?.files.filter(f=>lora?f.type==='Model':['Model','Diffusion Model','UNet'].includes(f.type))||[];
  const selectedFile=files.find(f=>f.id===fileId);
  const sampling=selectedFile?.generation_defaults||version?.generation_defaults;
  const ltxVariant=sampling?.variant??version?.ltx_variant;
  const progress=mine&&transfer?.total_bytes?Math.round(100*(transfer.completed_bytes||0)/transfer.total_bytes):0;
  const samples=useRef<{t:number;b:number}[]>([]);
  const done=mine?transfer?.completed_bytes||0:0;
  useEffect(()=>{if(!running){samples.current=[];return;}samples.current=[...samples.current.filter(x=>Date.now()-x.t<8000),{t:Date.now(),b:done}];},[done,running]);
  const span=samples.current.length>1?samples.current[samples.current.length-1]:null;const first=samples.current[0];
  const speed=span&&first&&span.t>first.t?(span.b-first.b)/((span.t-first.t)/1000):0;
  const left=speed>0&&transfer?.total_bytes?Math.round((transfer.total_bytes-done)/speed):0;
  const selectedImage=Math.min(image.versionId===versionId?image.index:0,Math.max(0,(version?.images.length||0)-1));
  return <dialog ref={dialog} className="civitai-details" aria-label={model.name} tabIndex={-1} onCancel={e=>{e.preventDefault();onClose();}} onClick={e=>{if(e.target!==dialog.current)return;const r=dialog.current.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)onClose();}}>
    <header className="civitai-details-heading"><div><h2>{model.name}</h2><p>{lora?'LoRA':'Model'} by {model.creator} · <a href={model.url} target="_blank" rel="noreferrer">Open on Civitai<ExternalLink size={12}/></a></p></div><button className="icon-button" onClick={onClose} aria-label="Close"><X size={20}/></button></header>
    <div className="civitai-details-body">
    <div className="civitai-media"><div className="civitai-preview"><Media key={`${versionId}:${selectedImage}`} item={version?.images[selectedImage]} className="civitai-hero" active/></div>{(version?.images.length||0)>1?<div className="civitai-strip" aria-label="Preview images">{version.images.map((item,i)=><button key={`${versionId}:${i}`} aria-label={`Preview ${i+1}`} aria-pressed={i===selectedImage} onClick={()=>setImage({versionId,index:i})}><Media item={item}/></button>)}</div>:null}</div>
    <section className="civitai-info">
      <div className="field"><span className="field-label">Versions on this page · {model.versions.length}</span><div className="civitai-versions" role="radiogroup" aria-label="Version" ref={versionList}>
        {model.versions.map(v=>{const file=v.files.find(f=>f.id===v.default_file);const size=file?.bytes;const vArch=archOf(v.base_model);const added=model.installed_versions.includes(v.id)||models?.models.some(m=>m.id===`civitai-${lora?'lora':'model'}-${v.id}`&&m.installed);
          return <button key={v.id} type="button" role="radio" aria-checked={v.id===versionId} className="civitai-version" onClick={()=>selectVersion(v)}>
            <b>{v.name}</b>
            <span>{!v.supported?(v.problem?v.problem:`${v.base_model}, not supported`):lora?`${baseName(v.base_model)} LoRA`:v.area==='video'?`${baseName(v.base_model)} video model`:`${archName[vArch]} ${!v.turbo_known?'(type not stated)':v.turbo_hint?(vArch==='flux3'?'Distilled':'Turbo'):standardWord[vArch]}${file?` · ${(file.format==='GGUF'?'GGUF':file.precision||file.format||'').toUpperCase()}`:''}`}{size?` · ${gb(size)}`:''}</span>
            {v.supported&&lora&&v.trained_words.length?<small className="civitai-trigger">Trigger: {v.trained_words.join(', ')}</small>:null}
            {v.supported&&!lora&&v.area==='video'?<small>{videoSettingsText(v.generation_defaults,v.ltx_variant)}</small>:null}
            {v.supported&&!lora&&v.area==='image'?<small>{v.turbo_known?settingsText(v.author_settings,vArch,v.turbo_hint)+(typeSource(v)?` (${typeSource(v)})`:''):`The page does not say Turbo or ${standardWord[vArch]}: you choose it below`}</small>:null}
            {added?<em><Check size={11}/>Added</em>:null}
          </button>;})}</div></div>
      {!version?.supported?<div className="message warning"><span>{version?.problem||`This version is for ${version?.base_model}, which this app cannot run yet.`}</span></div>:null}
      {!lora&&files.length?<div className="field"><label htmlFor="civitai-file">File</label><select id="civitai-file" value={fileId??''} onChange={e=>setFileId(Number(e.target.value))}>{files.map(f=><option key={f.id} value={f.id}>{f.name} · {f.precision||f.format} · {gb(f.bytes)}</option>)}</select></div>:null}
      <div className="civitai-usage">{lora?<>
        <div><b>Works with</b><span>{video?(version?.base_model?.startsWith('LTX')?'LTX 2.5 video models (Create video and Reference images)':'MiniMax H3 video models (Create video, Video to video, Reference images)'):arch==='flux3'?'FLUX.3 Klein models (Base and Distilled)':krea?'Krea 2 models (Raw and Turbo)':arch==='zimage'?'Z-Image models (Base and Turbo)':'All Qwen-Image-2.1 models (Standard and Turbo)'}</span></div>
        <div><b>Trigger words</b><span>{version?.trained_words.length?version.trained_words.join(', '):'None. Describe the effect in your prompt.'}</span></div>
        <div><b>Strength</b><span>{version?.suggested_strength?`${version.suggested_strength} (from the description)`:'1.0 (no value in the description)'}</span></div>
      </>:video?<><div><b>Loads as</b><span>{ltxVariant==='dev'?`LTX 2.5 dev with the official distilled LoRA at strength ${sampling?.distill_lora_strength??0.5}`:ltxVariant==='distilled'?'LTX 2.5 distilled. Uses its built-in eight-step model.':selectedFile?.family==='minimax-h3-ref2va'?'MiniMax H3 Ref2VA: reference images, video and audio'+(selectedFile.modes?.includes('text')?', or text only':''):'MiniMax H3 FL2VA: text, first frame and last frame'}</span></div><div><b>Generation settings</b><span>{videoSettingsText(sampling,ltxVariant)}</span></div></>:<><div><b>Loads as</b><span>{`${archName[arch]} ${turbo?(arch==='flux3'?'Distilled':'Turbo'):standardWord[arch]}: ${settingsText(version?.author_settings,arch,turbo)}`}{!needsType&&version&&typeSource(version)?<small className="civitai-source">{typeSource(version)}.</small>:null}</span></div><div><b>VAE</b><span>{arch==='flux3'?'FLUX.3 image decoder (FLUX.2 VAE)':krea?'Krea 2 VAE (Qwen-Image VAE)':arch==='zimage'?'Z-Image VAE (Flux ae)':version?.vae_hint==='vae-texture-fix'?'Texture fix (named by the page)':version?.vae_hint==='vae'?'Standard Qwen-Image-2.1 VAE (shipped with the model)':version?.vae_hint==='file'?'The VAE file that ships with this version (downloaded too)':'Auto: Texture fix'}</span></div></>}</div>
      {needsType?<div className="civitai-type-choice"><p>The page does not say whether this is a Turbo or a {standardWord[arch]} model. Choose one. Turbo: 8 steps, CFG 1. {krea?'Raw: 52 steps, CFG 4.5':arch==='zimage'?'Base: 30 steps, CFG 4':'Standard: 25 steps, CFG 2 with a negative prompt'}. The example images on Civitai often show it.</p><div className="segmented" role="radiogroup" aria-label="Model type"><button type="button" role="radio" aria-checked={typeChoice===true} aria-selected={typeChoice===true} onClick={()=>setTypeChoice(true)}>Turbo</button><button type="button" role="radio" aria-checked={typeChoice===false} aria-selected={typeChoice===false} onClick={()=>setTypeChoice(false)}>{standardWord[arch]}</button></div></div>:null}
      {version?.description||model.description?<details className="disclosure"><summary>Description</summary><p className="civitai-description">{version?.description||model.description}</p></details>:null}
    </section>
    </div>
    <div className="civitai-details-footer">
      {error?<div className="message error" role="alert"><span>{error}</span></div>:null}
      <div className="civitai-actions">{installed?<button className="button primary" onClick={()=>onUse({id:installedEntry?.id||entryId,kind:video?(lora?'video-lora':'video'):(lora?'image-lora':'image')})}><Check size={17}/>{lora?'Use this LoRA':'Use this model'}</button>
        :<button className="button primary" disabled={!version?.supported||fileId===null||running||(needsType&&typeChoice===null)} onClick={add}>{running?<><Loader2 className="spin" size={17}/>{transfer?.stage==='verifying'?'Checking file…':`Adding… ${progress}%`}</>:<><Download size={17}/>Add to Qwen{files.find(f=>f.id===fileId)?` · ${gb(files.find(f=>f.id===fileId)!.bytes)}`:''}</>}</button>}
        {running?<button className="text-button" onClick={()=>api(`/api/civitai/install/${versionId}/cancel`,{method:'POST'})}>Cancel</button>:null}</div>
      {running&&mine&&transfer?.total_bytes?<div className="civitai-progress"><progress value={done} max={transfer.total_bytes}/><span>{gb(done)} of {gb(transfer.total_bytes)}{speed>0?` · ${(speed/1e6).toFixed(0)} MB/s · ${left>=60?`${Math.floor(left/60)}m ${String(left%60).padStart(2,'0')}s`:`${left}s`} left`:''}</span></div>:null}
      {!installed&&models?.models.some(m=>m.installed&&m.civitai?.version_id===versionId)?<small>A different file of this version is installed. This file is added separately.</small>:null}
    </div>
  </dialog>;
}
