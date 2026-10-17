import {useEffect,useState} from 'react';
import {Check,Download,Images,Loader2,Pencil,Search,Star,Trash2,Upload,UserPlus,X} from 'lucide-react';
import type {Api} from './ModelSettings';

export type InputImage={id:string;name:string;url:string};
type Upload={id:string;name:string;url:string;thumb_url:string};
type Result={id:string;request:{prompt:string}};
export type LibraryTab='uploads'|'results'|'people';
export type Person={id:string;name:string;image_id:string;image_ids:string[];thumb_url:string;url:string};

/** Every image you have used, with a checkmark: checked images are the inputs, in the order you check them. */
export function InputLibrary({api,inputs,onChange,results,people,onPeople,revision,disabled,onUpload,limit,replace,target,onClose}:{api:Api;inputs:InputImage[];onChange:(next:InputImage[])=>void;
  results:Result[];people:Person[];onPeople:()=>void;revision:number;disabled:boolean;onUpload:()=>void;limit:number;
  replace:boolean;target:number;onClose?:()=>void}){
  const swap=replace&&inputs.length>0;
  function add(image:InputImage){if(swap){const next=[...inputs];next[Math.min(target,inputs.length-1)]=image;onChange(next);onClose?.();}else if(inputs.length<limit)onChange([...inputs,image]);}
  const [filter,setFilter]=useState('');const [importing,setImporting]=useState<{state:string;message:string;done?:number;total?:number}|null>(null);const [message,setMessage]=useState('');
  const [tab,setTab]=useState<LibraryTab>(()=>(localStorage.getItem('qwen-input-tab') as LibraryTab)||'uploads');
  const [uploads,setUploads]=useState<Upload[]>([]);const [total,setTotal]=useState(0);
  const [resultInputs,setResultInputs]=useState<Record<string,string>>({});const [busy,setBusy]=useState('');
  useEffect(()=>{api('/api/uploads?limit=200').then(page=>{setUploads(page.items);setTotal(page.total);}).catch(()=>{});},[api,revision]);
  useEffect(()=>{try{localStorage.setItem('qwen-input-tab',tab);}catch{}},[tab]);
  const index=(id:string)=>inputs.findIndex(i=>i.id===id);
  useEffect(()=>{if(importing?.state!=='running')return;const timer=setInterval(async()=>{const next=await api('/api/people/import/status');setImporting(next);onPeople();if(next.state!=='running'){setMessage(next.message);clearInterval(timer);}},1500);return()=>clearInterval(timer);},[importing?.state]);
  async function importMoise(){setMessage('');try{await api('/api/people/import/moise',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});setImporting({state:'running',message:'Connecting to Moise…'});}catch(e){setMessage((e as Error).message);}}
  async function newPerson(){const name=window.prompt('Name for the checked photos');if(!name)return;try{await api('/api/people',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name,image_ids:inputs.map(i=>i.id)})});onPeople();setTab('people');}catch(e){setMessage((e as Error).message);}}
  async function rename(person:Person){const name=window.prompt('New name',person.name);if(!name||name===person.name)return;try{await api(`/api/people/${person.id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({name})});onPeople();}catch(e){setMessage((e as Error).message);}}
  async function remove(person:Person){if(!window.confirm(`Remove ${person.name} from People? The photos stay in Uploads.`))return;await api(`/api/people/${person.id}`,{method:'DELETE'});onPeople();}
  const shown=people.filter(p=>p.name.toLowerCase().includes(filter.trim().toLowerCase()));
  const [drawer,setDrawer]=useState<string|null>(null);const opened=people.find(p=>p.id===drawer);
  // Which of the person's photos is in the inputs now (any of them, not only the main photo).
  const personAt=(p:Person)=>inputs.findIndex(i=>p.image_ids.includes(i.id));
  function togglePerson(p:Person){const at=personAt(p);if(at>=0)onChange(inputs.filter((_,i)=>i!==at));else add({id:p.image_id,name:p.name,url:p.url});}
  function pickPhoto(p:Person,id:string){
    const image={id,name:p.name,url:`/api/uploads/${id}`};const at=personAt(p);
    if(inputs[at]?.id===id){onChange(inputs.filter((_,i)=>i!==at));return;}
    if(at>=0){const next=[...inputs];next[at]=image;onChange(next);}else add(image);
  }
  async function makeMain(p:Person,id:string){try{await api(`/api/people/${p.id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_ids:[id,...p.image_ids.filter(x=>x!==id)]})});onPeople();}catch(e){setMessage((e as Error).message);}}
  function toggle(image:InputImage){
    const at=index(image.id);
    if(at>=0)onChange(inputs.filter((_,i)=>i!==at));
    else add(image);
  }
  async function toggleResult(result:Result){
    const existing=resultInputs[result.id];
    if(existing&&index(existing)>=0){onChange(inputs.filter(i=>i.id!==existing));return;}
    if(inputs.length>=limit&&!swap)return;
    setBusy(result.id);
    try{const added:InputImage=await api(`/api/jobs/${result.id}/reuse`,{method:'POST'});setResultInputs(m=>({...m,[result.id]:added.id}));if(index(added.id)<0)add(added);}
    finally{setBusy('');}
  }
  const tile=(key:string,thumb:string,title:string,at:number,onClick:()=>void,loading=false,label?:string)=>
    <button key={key} type="button" className={`library-tile checker ${at>=0?'checked':''}`} disabled={disabled||(at<0&&inputs.length>=limit&&!swap)} onClick={onClick} title={title} aria-pressed={at>=0}>
      <img src={thumb} alt="" loading="lazy"/>
      <span className="library-check">{loading?<Loader2 className="spin" size={13}/>:at>=0?at+1:<Check size={13}/>}</span>
      {label?<span className="library-label">{label}</span>:null}
    </button>;
  return <div className="input-library">
    <div className="library-tabs" role="tablist">
      <button role="tab" aria-selected={tab==='uploads'} onClick={()=>setTab('uploads')}>Uploads <span>{total}</span></button>
      <button role="tab" aria-selected={tab==='results'} onClick={()=>setTab('results')}>Results <span>{results.length}</span></button>
      <button role="tab" aria-selected={tab==='people'} onClick={()=>setTab('people')}>People <span>{people.length}</span></button>
      {swap?<span className="library-replace">Replacing image {Math.min(target,inputs.length-1)+1}</span>:null}
      <button type="button" className="text-button library-upload" disabled={disabled} onClick={onUpload}><Upload size={14}/>Upload</button>
      {onClose?<button type="button" className="icon-button library-close" onClick={onClose} aria-label="Close library" title="Close (Esc)"><X size={15}/></button>:null}
    </div>
    {tab==='people'?<div className="people-tools">
      <label className="people-search"><Search size={13}/><input value={filter} onChange={e=>setFilter(e.target.value)} placeholder="Find a person"/></label>
      <button type="button" className="text-button" disabled={disabled||!inputs.length} onClick={newPerson} title="Make a person from the checked photos"><UserPlus size={14}/>New</button>
      <button type="button" className="text-button" disabled={importing?.state==='running'} onClick={importMoise} title="Import contacts and photos from Moise">{importing?.state==='running'?<Loader2 className="spin" size={14}/>:<Download size={14}/>}Moise</button>
    </div>:null}
    {importing?.state==='running'?<div className="people-progress"><progress value={importing.done||0} max={importing.total||1}/><span>{importing.message}{importing.total?` · ${importing.done}/${importing.total}`:''}</span></div>:message?<p className="people-message">{message}</p>:null}
    {tab==='people'?<div className="people-grid">{shown.map(p=>{const at=personAt(p);return <div key={p.id} className={`person ${at>=0?'checked':''}`}>
      <button type="button" className="person-face" disabled={disabled||(at<0&&inputs.length>=limit&&!swap)} onClick={()=>togglePerson(p)} title={`${p.name} · ${p.image_ids.length} photo${p.image_ids.length>1?'s':''}`} aria-pressed={at>=0}>
        <img src={p.thumb_url} alt="" loading="lazy"/><span className="library-check">{at>=0?at+1:<Check size={12}/>}</span></button>
      <span className="person-name" dir="auto">{p.name}</span>
      {p.image_ids.length>1?<button type="button" className={`person-photos ${drawer===p.id?'open':''}`} onClick={()=>setDrawer(d=>d===p.id?null:p.id)} title={`Choose one of ${p.image_ids.length} photos`} aria-expanded={drawer===p.id}><Images size={10}/>{p.image_ids.length}</button>:null}
      <span className="person-actions"><button type="button" onClick={()=>rename(p)} aria-label={`Rename ${p.name}`}><Pencil size={11}/></button><button type="button" onClick={()=>remove(p)} aria-label={`Remove ${p.name}`}><Trash2 size={11}/></button></span>
    </div>;})}{!shown.length?<p className="library-empty">{people.length?'No match.':'No people yet. Import your contacts from Moise, or check photos and click New.'}</p>:null}</div>:null}
    {tab==='people'&&opened?<div className="person-drawer">
      <div className="person-drawer-head"><b dir="auto">{opened.name}</b><span>Choose the photo to use · {opened.image_ids.length} photos</span><button type="button" className="text-button" onClick={()=>setDrawer(null)}>Close</button></div>
      <div className="person-drawer-grid">{opened.image_ids.map((id,n)=>{const used=inputs.findIndex(i=>i.id===id);return <div key={id} className={`person-photo ${used>=0?'checked':''}`}>
        <button type="button" className={`library-tile checker ${used>=0?"checked":""}`} disabled={disabled} onClick={()=>pickPhoto(opened,id)} aria-pressed={used>=0} title={used>=0?'Remove from inputs':personAt(opened)>=0?'Use this photo instead':'Use this photo'}>
          <img src={`/api/uploads/${id}/thumb`} alt="" loading="lazy"/><span className="library-check">{used>=0?used+1:<Check size={13}/>}</span></button>
        {n===0?<span className="person-main">Main</span>:<button type="button" className="person-main-set" onClick={()=>makeMain(opened,id)} title="Make this the main photo and avatar"><Star size={10}/>Main</button>}
      </div>;})}</div>
    </div>:null}
    <div className="library-grid" hidden={tab==='people'}>
      {tab==='uploads'?uploads.map(u=>tile(u.id,u.thumb_url,u.name,index(u.id),()=>toggle({id:u.id,name:u.name,url:u.url}))):null}
      {tab==='results'?results.map(r=>tile(r.id,`/api/jobs/${r.id}/thumbnail`,r.request.prompt,resultInputs[r.id]?index(resultInputs[r.id]):-1,()=>toggleResult(r),busy===r.id)):null}
      {(tab==='uploads'&&!uploads.length)||(tab==='results'&&!results.length)||false?<p className="library-empty">{tab==='uploads'?'Images you upload or paste appear here.':tab==='results'?'Your finished images appear here.':'No people yet.'}</p>:null}
    </div>
  </div>;
}
