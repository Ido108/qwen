import {useEffect,useRef,useState} from 'react';
import {Check,Download,ExternalLink,Heart,Link2,Loader2,LogOut,Search,X} from 'lucide-react';
import type {Api,ModelState} from './ModelSettings';

type Preview={url:string;type:string;nsfw_level:number};
type Card={id:number;name:string;type:string;creator:string;downloads:number;likes:number;preview:Preview|null;versions:{id:number;name:string;base_model?:string;variant?:'standard'|'turbo'|'unknown'}[];installed:boolean};
type FileInfo={id:number;name:string;type:string;bytes:number;format:string;precision:string;primary:boolean};
type Version={id:number;name:string;base_model:string;problem?:string|null;ltx_variant?:'dev'|'distilled'|null;area:'image'|'video'|null;supported:boolean;trained_words:string[];description:string;files:FileInfo[];default_file:number|null;suggested_strength:number|null;turbo_hint:boolean;turbo_known?:boolean;author_settings?:{steps:number;cfg?:number;sampler?:string;scheduler?:string;quote:string}|null;vae_hint?:string|null;images:Preview[]};
/** Steps, CFG and sampler for a version: the page's values, then the defaults for what the page does not say. */
function settingsText(a:Version['author_settings'],krea:boolean,turbo:boolean){const name=(s:string)=>s.replace(/_/g,' ');const page=[a?.steps?`${a.steps} steps`:'',a?.cfg?`CFG ${a.cfg}`:'',a?.sampler?name(a.sampler)+(a.scheduler?` / ${name(a.scheduler)}`:''):a?.scheduler?name(a.scheduler):''].filter(Boolean);const preset=[a?.steps?'':`${krea?(turbo?8:52):turbo?8:25} steps`,a?.cfg?'':krea||turbo?`CFG ${krea&&!turbo?4.5:1}`:'CFG 2 with a negative prompt, else 1',a?.sampler?'':'euler'].filter(Boolean);return [page.length?`${page.join(', ')} (from the page)`:'',preset.length?`${preset.join(', ')} (${krea?'Krea 2':'app'} default)`:''].filter(Boolean).join('; ');}
type Details=Omit<Card,'versions'>&{description:string;url:string;versions:Version[];installed_versions:number[]};
type Account={connected:boolean;username:string;method:string};
export type InstalledItem={id:string;kind:string};

const gb=(bytes:number)=>bytes>=1e9?`${(bytes/1e9).toFixed(1)} GB`:`${Math.round(bytes/1e6)} MB`;
const count=(n:number)=>n>=1000?`${(n/1000).toFixed(n>=10000?0:1)}k`:String(n);

function Media({item,className}:{item:Preview;className?:string}){
  return item.type==='video'?<video className={className} src={item.url} muted loop autoPlay playsInline/>:<img className={className} src={item.url} alt="" loading="lazy"/>;
}

/** Browse Civitai (civitai.red) for Qwen-Image-2.1 LoRAs and models and add them to the app. */
export function CivitaiPage({api,visible,models,onUse}:{api:Api;visible:boolean;models:ModelState|null;onUse:(item:InstalledItem)=>void}){
  const [account,setAccount]=useState<Account|null>(null);const [login,setLogin]=useState<{state:string;user_code:string;url:string}|null>(null);
  const [keyMode,setKeyMode]=useState(false);const [key,setKey]=useState('');
  const [kind,setKind]=useState<'LORA'|'Checkpoint'>('LORA');const [area,setArea]=useState<'image'|'video'>('image');const [sort,setSort]=useState('Most Downloaded');const [variant,setVariant]=useState<'all'|'standard'|'turbo'>('all');const [arch,setArch]=useState('all');const [total,setTotal]=useState<number|null>(null);const [indexing,setIndexing]=useState<number|null>(null);
  const [query,setQuery]=useState('');const [search,setSearch]=useState('');
  const [cards,setCards]=useState<Card[]>([]);const [cursor,setCursor]=useState<string|null>(null);
  const [loading,setLoading]=useState(false);const [error,setError]=useState('');
  const [open,setOpen]=useState<Details|null>(null);const [opening,setOpening]=useState(0);
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
  async function show(id:number){setOpening(id);setError('');try{setOpen(await api(`/api/civitai/models/${id}`));}catch(e){setError((e as Error).message);}finally{setOpening(0);}}
  async function openLink(link:string){try{const target=await api('/api/civitai/resolve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({link})});setQuery('');await show(target.model_id);}catch(e){setError((e as Error).message);}}
  async function connect(){setError('');try{const flow=await api('/api/civitai/login',{method:'POST'});setLogin(flow);window.open(flow.url,'_blank','noopener,width=520,height=720');}catch(e){setError((e as Error).message);}}
  async function saveKey(){setError('');try{setAccount(await api('/api/civitai/key',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({key})}));setKey('');setKeyMode(false);}catch(e){setError((e as Error).message);}}
  async function disconnect(){setAccount(await api('/api/civitai/disconnect',{method:'POST'}));}
  return <main className="civitai-page" hidden={!visible}>
    <div className="civitai-top">
      <div><h1>Civitai</h1><p>{area==='image'?'Qwen-Image-2.1 and Krea 2':'MiniMax H3 and LTX 2.5'} LoRAs and models from civitai.red. Only files that run in this app are shown.</p></div>
      <div className="civitai-account">{account?.connected?<><span className="connection-status connected"><Check size={13}/>{account.username||'Connected'}</span><button className="text-button" onClick={disconnect}><LogOut size={14}/>Disconnect</button></>
        :login?<div className="civitai-login"><Loader2 className="spin" size={16}/><span>Approve in the Civitai window. Code <b>{login.user_code}</b></span><a className="text-button" href={login.url} target="_blank" rel="noreferrer">Open again</a><button className="text-button" onClick={()=>setLogin(null)}>Cancel</button></div>
        :keyMode?<div className="civitai-login"><input type="password" placeholder="Civitai API key" value={key} onChange={e=>setKey(e.target.value)}/><button className="button primary" disabled={key.length<10} onClick={saveKey}>Save</button><button className="text-button" onClick={()=>setKeyMode(false)}>Cancel</button></div>
        :<><button className="button primary" onClick={connect}>Connect Civitai account</button><button className="text-button" onClick={()=>setKeyMode(true)}>Use an API key</button></>}</div>
    </div>
    <div className="civitai-toolbar">
      <div className="segmented">{([['image','Images'],['video','Video']] as const).map(([id,label])=><button key={id} aria-selected={area===id} role="tab" onClick={()=>{setArea(id);setArch('all');}}>{label}</button>)}</div>
      <div className="segmented">{([['LORA','LoRAs'],['Checkpoint','Models']] as const).map(([id,label])=><button key={id} aria-selected={kind===id} role="tab" onClick={()=>setKind(id)}>{label}</button>)}</div>
      <select aria-label="Model architecture" title="Model architecture" value={arch} onChange={e=>setArch(e.target.value)}>{(area==='image'?[['all','All architectures'],['qwen-image-2.1','Qwen-Image 2.1'],['krea-2','Krea 2']]:[['all','All architectures'],['minimax-h3','MiniMax H3'],['ltx25','LTX 2.5']]).map(([id,label])=><option key={id} value={id}>{label}</option>)}</select>
      <label className="gallery-search civitai-search"><Search size={17}/><input aria-label="Search Civitai or paste a link" value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search, or paste a Civitai link"/>{query?<button className="icon-button" aria-label="Clear" onClick={()=>setQuery('')}><X size={15}/></button>:<Link2 size={15} className="muted-icon"/>}</label>
      {kind==='Checkpoint'&&area==='image'?<div className="segmented">{([['all','All'],['standard',arch==='krea-2'?'Raw':arch==='qwen-image-2.1'?'Standard':'Standard / Raw'],['turbo','Turbo']] as const).map(([id,label])=><button key={id} aria-selected={variant===id} role="tab" onClick={()=>setVariant(id)}>{label}</button>)}</div>:null}
      {indexing!==null?<span className="civitai-total"><Loader2 className="spin" size={13}/> Indexing Civitai: {indexing} models so far</span>:total!==null?<span className="civitai-total">{total} found</span>:null}
      <select aria-label="Sort" value={sort} onChange={e=>setSort(e.target.value)}><option>Most Downloaded</option><option>Highest Rated</option><option>Newest</option></select>
    </div>
    {error?<div className="message error" role="alert"><span>{error}</span><button className="icon-button" onClick={()=>setError('')} aria-label="Close"><X size={15}/></button></div>:null}
    <div className="civitai-grid">{cards.map(card=><button key={card.id} className="civitai-card" onClick={()=>show(card.id)} disabled={opening===card.id}>
      <div className="civitai-thumb">{card.preview?<Media item={card.preview}/>:null}{card.installed?<span className="civitai-badge"><Check size={12}/>Added</span>:null}{opening===card.id?<span className="civitai-opening"><Loader2 className="spin" size={22}/></span>:null}</div>
      <strong>{card.name}</strong>{kind==='Checkpoint'&&area==='image'?<span className="civitai-variants">{[...new Set(card.versions.map(v=>v.variant==='turbo'?'Turbo':v.variant==='unknown'?'Type not stated':v.base_model?.startsWith('Krea 2')?'Raw':'Standard'))].map(v=><em key={v} className={v==='Turbo'?'turbo':'standard'}>{v}</em>)}</span>:null}<small>{card.creator} · <Download size={11}/>{count(card.downloads)} · <Heart size={11}/>{count(card.likes)}</small>
    </button>)}</div>
    {!cards.length&&!loading?<p className="civitai-empty">No {area==='image'?'Qwen-Image-2.1 or Krea 2':'MiniMax H3 or LTX 2.5'} {kind==='LORA'?'LoRAs':'models'} found.</p>:null}
    {loading?<div className="civitai-empty"><Loader2 className="spin" size={22}/></div>:cursor?<button className="button secondary civitai-more" onClick={()=>load(false)}>Load more</button>:null}
    {open?<CivitaiDetails api={api} model={open} models={models} onClose={()=>setOpen(null)} onAdded={()=>{setCards(old=>old.map(c=>c.id===open.id?{...c,installed:true}:c));}} onUse={item=>{setOpen(null);onUse(item);}}/>:null}
  </main>;
}

function CivitaiDetails({api,model,models,onClose,onAdded,onUse}:{api:Api;model:Details;models:ModelState|null;onClose:()=>void;onAdded:()=>void;onUse:(item:InstalledItem)=>void}){
  const dialog=useRef<HTMLDialogElement>(null);
  const supported=model.versions.filter(v=>v.supported);
  const [versionId,setVersionId]=useState(supported[0]?.id??model.versions[0]?.id);
  const version=model.versions.find(v=>v.id===versionId)!;
  const [fileId,setFileId]=useState<number|null>(version?.default_file??null);
  const [typeChoice,setTypeChoice]=useState<boolean|null>(null);useEffect(()=>setTypeChoice(null),[versionId]);
  const needsType=!!version&&version.area==='image'&&version.supported&&!version.turbo_known&&model.type!=='LORA';
  const turbo=needsType?!!typeChoice:!!version?.turbo_hint;
  const [image,setImage]=useState(0);const [error,setError]=useState('');const [adding,setAdding]=useState(false);
  const lora=model.type==='LORA';
  const entryId=`civitai-${lora?'lora':'model'}-${versionId}`;const video=version?.area==='video';const krea=!!version?.base_model?.startsWith('Krea 2');
  const installed=models?.models.find(m=>m.id===entryId)?.installed||model.installed_versions.includes(versionId);
  const transfer=models?.downloads?.find(d=>d.requested_id===`civitai:${versionId}`);const mine=!!transfer;
  const running=adding||(mine&&transfer?.status==='running');
  useEffect(()=>{dialog.current?.showModal();},[]);
  useEffect(()=>{setFileId(version?.default_file??null);setImage(0);},[versionId]);
  useEffect(()=>{if(!adding||!mine)return;if(transfer?.status==='succeeded'){setAdding(false);onAdded();}else if(transfer?.status==='failed'||transfer?.status==='cancelled'){setAdding(false);setError(transfer.message||'The download did not finish.');}},[transfer?.status,adding,mine]);
  async function add(){setError('');setAdding(true);try{await api('/api/civitai/install',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({version_id:versionId,file_id:fileId,turbo,title:model.versions.length>1?`${model.name} · ${version?.name}`:model.name})});}catch(e){setAdding(false);setError((e as Error).message);}}
  const files=version?.files.filter(f=>lora?f.type==='Model':['Model','Diffusion Model'].includes(f.type))||[];
  const progress=mine&&transfer?.total_bytes?Math.round(100*(transfer.completed_bytes||0)/transfer.total_bytes):0;
  const samples=useRef<{t:number;b:number}[]>([]);
  const done=mine?transfer?.completed_bytes||0:0;
  useEffect(()=>{if(!running){samples.current=[];return;}samples.current=[...samples.current.filter(x=>Date.now()-x.t<8000),{t:Date.now(),b:done}];},[done,running]);
  const span=samples.current.length>1?samples.current[samples.current.length-1]:null;const first=samples.current[0];
  const speed=span&&first&&span.t>first.t?(span.b-first.b)/((span.t-first.t)/1000):0;
  const left=speed>0&&transfer?.total_bytes?Math.round((transfer.total_bytes-done)/speed):0;
  return <dialog ref={dialog} className="civitai-details" aria-label={model.name} onCancel={onClose} onClick={e=>{if(e.target===dialog.current)onClose();}}>
    <button className="dialog-close icon-button" onClick={onClose} aria-label="Close"><X size={20}/></button>
    <div className="civitai-media">{version?.images.length?<><Media item={version.images[image]} className="civitai-hero"/>{version.images.length>1?<div className="civitai-strip">{version.images.map((item,i)=><button key={i} aria-pressed={i===image} onClick={()=>setImage(i)}><Media item={item}/></button>)}</div>:null}</>:<div className="civitai-hero"/>}</div>
    <section className="civitai-info">
      <h2>{model.name}</h2><p className="muted">{lora?'LoRA':'Model'} by {model.creator} · <a href={model.url} target="_blank" rel="noreferrer">Open on Civitai<ExternalLink size={12}/></a></p>
      <div className="field"><span className="field-label">Versions on this page · {model.versions.length}</span><div className="civitai-versions" role="radiogroup" aria-label="Version">
        {model.versions.map(v=>{const file=v.files.find(f=>f.id===v.default_file);const size=file?.bytes;const vKrea=!!v.base_model?.startsWith('Krea 2');const added=model.installed_versions.includes(v.id)||models?.models.some(m=>m.id===`civitai-${lora?'lora':'model'}-${v.id}`&&m.installed);
          return <button key={v.id} type="button" role="radio" aria-checked={v.id===versionId} className="civitai-version" onClick={()=>setVersionId(v.id)}>
            <b>{v.name}</b>
            <span>{!v.supported?(v.problem?'Not supported: '+(v.problem.includes('dev')?'LTX 2.5 dev checkpoint':'not marked as distilled'):`${v.base_model}, not supported`):lora?(v.trained_words.length?`Trigger: ${v.trained_words.slice(0,2).join(', ')}`:'LoRA'):v.area==='video'?'MiniMax H3 video model':`${vKrea?'Krea 2 ':'Qwen-Image-2.1 '}${!v.turbo_known?'(type not stated)':v.turbo_hint?'Turbo':vKrea?'Raw':'Standard'}${file?` · ${(file.format==='GGUF'?'GGUF':file.precision||file.format||'').toUpperCase()}`:''}`}{size?` · ${gb(size)}`:''}</span>
            {v.supported&&!lora&&v.area==='image'?<small>{v.turbo_known?settingsText(v.author_settings,vKrea,v.turbo_hint):`The page does not say Turbo or ${vKrea?'Raw':'Standard'}: you choose it below`}</small>:null}
            {added?<em><Check size={11}/>Added</em>:null}
          </button>;})}</div></div>
      {!version?.supported?<div className="message warning"><span>{version?.problem||`This version is for ${version?.base_model}, which this app cannot run yet.`}</span></div>:null}
      {!lora&&files.length?<div className="field"><label htmlFor="civitai-file">File</label><select id="civitai-file" value={fileId??''} onChange={e=>setFileId(Number(e.target.value))}>{files.map(f=><option key={f.id} value={f.id}>{f.name} · {f.precision||f.format} · {gb(f.bytes)}</option>)}</select></div>:null}
      <div className="civitai-usage">{lora?<>
        <div><b>Works with</b><span>{video?(version?.base_model?.startsWith('LTX')?'LTX 2.5 video models (Create video and Reference images)':'MiniMax H3 video models (Create video, Video to video, Reference images)'):krea?'Krea 2 models (Raw and Turbo)':'All Qwen-Image-2.1 models (Standard and Turbo)'}</span></div>
        <div><b>Trigger words</b><span>{version?.trained_words.length?version.trained_words.join(', '):'None. Describe the effect in your prompt.'}</span></div>
        <div><b>Strength</b><span>{version?.suggested_strength?`${version.suggested_strength} (from the description)`:'1.0 (no value in the description)'}</span></div>
      </>:video?<><div><b>Loads as</b><span>{version?.ltx_variant==='dev'?'LTX 2.5 dev checkpoint: the app adds the official distilled LoRA (8.9 GB, once) at 0.5 and runs the 8-step schedule':version?.ltx_variant==='distilled'?'LTX 2.5 distilled checkpoint: 8 steps, then 3 refine steps':'MiniMax H3 video model; uses the installed H3 text encoder and video and audio VAEs'}</span></div></>:<><div><b>Loads as</b><span>{`${krea?'Krea 2 ':''}${turbo?'Turbo':krea?'Raw':'Standard'}: ${settingsText(version?.author_settings,krea,turbo)}`}</span></div><div><b>VAE</b><span>{krea?'Krea 2 VAE (Qwen-Image VAE)':version?.vae_hint==='vae-texture-fix'?'Texture fix (named by the page)':version?.vae_hint==='vae'?'Standard Qwen-Image-2.1 VAE (shipped with the model)':version?.vae_hint==='file'?'The VAE file that ships with this version (downloaded too)':'Auto: Texture fix'}</span></div></>}</div>
      {needsType?<div className="civitai-type-choice"><p>The page does not say whether this is a Turbo or a {krea?'Raw':'Standard'} model. Choose one. Turbo: 8 steps, CFG 1. {krea?'Raw: 52 steps, CFG 4.5':'Standard: 25 steps, CFG 2 with a negative prompt'}. The example images on Civitai often show it.</p><div className="segmented" role="radiogroup" aria-label="Model type"><button type="button" role="radio" aria-checked={typeChoice===true} aria-selected={typeChoice===true} onClick={()=>setTypeChoice(true)}>Turbo</button><button type="button" role="radio" aria-checked={typeChoice===false} aria-selected={typeChoice===false} onClick={()=>setTypeChoice(false)}>{krea?'Raw':'Standard'}</button></div></div>:null}
      {error?<div className="message error"><span>{error}</span></div>:null}
      <div className="civitai-actions">{installed?<button className="button primary" onClick={()=>onUse({id:entryId,kind:video?(lora?'video-lora':'video'):(lora?'image-lora':'image')})}><Check size={17}/>{lora?'Use this LoRA':'Use this model'}</button>
        :<button className="button primary" disabled={!version?.supported||running||(needsType&&typeChoice===null)} onClick={add}>{running?<><Loader2 className="spin" size={17}/>{transfer?.stage==='verifying'?'Checking file…':`Adding… ${progress}%`}</>:<><Download size={17}/>Add to Qwen{files.find(f=>f.id===fileId)?` · ${gb(files.find(f=>f.id===fileId)!.bytes)}`:''}</>}</button>}
        {running?<button className="text-button" onClick={()=>api(`/api/civitai/install/${versionId}/cancel`,{method:'POST'})}>Cancel</button>:null}</div>
      {running&&mine&&transfer?.total_bytes?<div className="civitai-progress"><progress value={done} max={transfer.total_bytes}/><span>{gb(done)} of {gb(transfer.total_bytes)}{speed>0?` · ${(speed/1e6).toFixed(0)} MB/s · ${left>=60?`${Math.floor(left/60)}m ${String(left%60).padStart(2,'0')}s`:`${left}s`} left`:''}</span></div>:null}
      {version?.description||model.description?<details className="disclosure"><summary>Description</summary><p className="civitai-description">{version?.description||model.description}</p></details>:null}
    </section>
  </dialog>;
}
