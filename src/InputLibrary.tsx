import {useEffect,useState} from 'react';
import {Check,Download,Images,Loader2,Pencil,Search,Star,Upload,UserPlus,X} from 'lucide-react';
import type {Api} from './ModelSettings';
import {FilePicker} from './FilePicker';
import {WebSearch} from './WebSearch';
import {PersonEditor} from './PersonEditor';

export type InputImage={id:string;name:string;url:string;personId?:string;subjectLabel?:string};
type Upload={id:string;name:string;url:string;thumb_url:string};
type Result={id:string;request:{prompt:string}};
export type LibraryTab='uploads'|'results'|'people'|'web';
export type Person={id:string;name:string;image_id:string;image_ids:string[];thumb_url:string;url:string;label?:string|null};

/** Every image you have used, with a checkmark: checked images are the inputs, in the order you check them. */
export function InputLibrary({api,inputs,onChange,results,people,onPeople,revision,disabled,onUpload,limit,replace,target,onClose,multiple=false}:{api:Api;inputs:InputImage[];onChange:(next:InputImage[])=>void;
  results:Result[];people:Person[];onPeople:()=>void;revision:number;disabled:boolean;onUpload:(files:File[])=>void;limit:number;
  replace:boolean;target:number;onClose?:()=>void;multiple?:boolean}){
  const swap=replace&&inputs.length>0;
  function add(image:InputImage){
    if(disabled)return;
    if(swap){const next=[...inputs];next[Math.min(target,inputs.length-1)]=image;onChange(next);}
    else if(inputs.length<limit)onChange([...inputs,image]);else return;
    if(!multiple){setDrawer(null);onClose?.();}
  }
  const [filter,setFilter]=useState('');const [importing,setImporting]=useState<{state:string;message:string;done?:number;total?:number}|null>(null);const [message,setMessage]=useState('');
  // The Library always opens on People.
  const [tab,setTab]=useState<LibraryTab>('people');
  const [uploads,setUploads]=useState<Upload[]>([]);const [total,setTotal]=useState(0);
  const [resultInputs,setResultInputs]=useState<Record<string,string>>({});const [busy,setBusy]=useState('');
  // Select mode on Uploads: a tap marks a picture for deletion instead of using it.
  const [picked,setPicked]=useState<string[]>([]);const [allResults,setAllResults]=useState<number|null>(null);const selecting=picked.length>0||allResults!==null;
  async function selectAll(){setConfirming(false);setMessage('');
    try{if(tab==='uploads'){const ids:string[]=[];for(let offset=0;offset<total;offset+=200){const page=await api(`/api/uploads?limit=200&offset=${offset}`);ids.push(...page.items.map((u:Upload)=>u.id));}setPicked(ids);}
      else{const data=await api('/api/library?kind=image&limit=1');setPicked(shownResults.map(r=>r.id));setAllResults(data.total);}}
    catch(e){setMessage((e as Error).message);}}const [confirming,setConfirming]=useState(false);
  const stopSelecting=()=>{setPicked([]);setAllResults(null);setConfirming(false);};
  const togglePicked=(id:string)=>{setConfirming(false);setAllResults(null);setPicked(old=>old.includes(id)?old.filter(x=>x!==id):[...old,id]);};
  const [removedResults,setRemovedResults]=useState<string[]>([]);
  const shownResults=results.filter(r=>!removedResults.includes(r.id));
  async function deletePicked(){
    setBusy('delete');setMessage('');
    if(tab==='results'){
      // Finished results: the picture file, its settings file and the job; running or waiting jobs are never touched.
      try{const result=await api('/api/library/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(allResults!==null?{all:true,kind:'image'}:{ids:picked})});
        setRemovedResults(old=>[...old,...picked]);window.dispatchEvent(new Event('qwen-results-changed'));
        setMessage(result.failed?.length?`${result.failed.length} could not be deleted.`:'');stopSelecting();}
      catch(e){setMessage((e as Error).message);}
      finally{setBusy('');}
      return;
    }
    try{const result=await api('/api/uploads/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ids:picked})});
      const gone=new Set<string>(result.deleted_ids||[]);
      setUploads(old=>old.filter(u=>!gone.has(u.id)));setTotal(old=>Math.max(0,old-picked.filter(id=>gone.has(id)).length));
      if(inputs.some(i=>gone.has(i.id)))onChange(inputs.filter(i=>!gone.has(i.id)));
      setMessage(result.kept?`${result.kept} picture${result.kept>1?'s were':' was'} kept: People photos and pictures that a waiting job uses stay.`:'');
      stopSelecting();}
    catch(e){setMessage((e as Error).message);}
    finally{setBusy('');}
  }
  useEffect(()=>{api('/api/uploads?limit=200').then(page=>{setUploads(page.items);setTotal(page.total);}).catch(()=>{});},[api,revision]);
  useEffect(()=>{stopSelecting();},[tab]);
  const index=(id:string)=>inputs.findIndex(i=>i.id===id);
  useEffect(()=>{if(importing?.state!=='running')return;const timer=setInterval(async()=>{const next=await api('/api/people/import/status');setImporting(next);onPeople();if(next.state!=='running'){setMessage(next.message);clearInterval(timer);}},1500);return()=>clearInterval(timer);},[importing?.state]);
  async function importMoise(){setMessage('');try{await api('/api/people/import/moise',{method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});setImporting({state:'running',message:'Connecting to Moise…'});}catch(e){setMessage((e as Error).message);}}
  const shown=people.filter(p=>p.name.toLowerCase().includes(filter.trim().toLowerCase()));
  const [drawer,setDrawer]=useState<string|null>(null);const opened=people.find(p=>p.id===drawer);
  // The person editor: new (also from a web picture) or one person's name and photos.
  const [editing,setEditing]=useState<{person?:Person;start?:{id:string;url:string}[];name?:string}|null>(null);
  // Which of the person's photos is in the inputs now (any of them, not only the main photo).
  const personAt=(p:Person)=>inputs.findIndex(i=>p.image_ids.includes(i.id));
  function togglePerson(p:Person){const at=personAt(p);if(multiple&&at>=0)onChange(inputs.filter((_,i)=>i!==at));else add({id:p.image_id,name:p.name,url:p.url,personId:p.id});}
  function pickPhoto(p:Person,id:string){
    const image={id,name:p.name,url:`/api/uploads/${id}`,personId:p.id};const at=personAt(p);
    if(!multiple){add(image);return;}
    if(inputs[at]?.id===id){onChange(inputs.filter((_,i)=>i!==at));return;}
    if(at>=0){const next=[...inputs];next[at]=image;onChange(next);}else add(image);
  }
  async function makeMain(p:Person,id:string){try{await api(`/api/people/${p.id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_ids:[id,...p.image_ids.filter(x=>x!==id)]})});onPeople();}catch(e){setMessage((e as Error).message);}}
  function toggle(image:InputImage){
    const at=index(image.id);
    if(multiple&&at>=0)onChange(inputs.filter((_,i)=>i!==at));
    else add(image);
  }
  async function toggleResult(result:Result){
    const existing=resultInputs[result.id];
    if(multiple&&existing&&index(existing)>=0){onChange(inputs.filter(i=>i.id!==existing));return;}
    if(inputs.length>=limit&&!swap)return;
    setBusy(result.id);
    try{const added:InputImage=await api(`/api/jobs/${result.id}/reuse`,{method:'POST'});setResultInputs(m=>({...m,[result.id]:added.id}));add(added);}
    catch(e){setMessage((e as Error).message);}
    finally{setBusy('');}
  }
  const tile=(key:string,thumb:string,title:string,at:number,onClick:()=>void,loading=false,label?:string,mark?:string)=>{
    const button=<button key={mark?undefined:key} type="button" className={`library-tile checker ${at>=0?'checked':''} ${mark&&picked.includes(mark)?'picked':''}`} disabled={disabled||loading||(inputs.length>=limit&&!swap&&(!multiple||at<0))} onClick={onClick} title={title} aria-pressed={at>=0}>
      <img src={thumb} alt="" loading="lazy"/>
      {mark?(loading||at>=0?<span className="library-order">{loading?<Loader2 className="spin" size={13}/>:at+1}</span>:null)
        :<span className="library-check">{loading?<Loader2 className="spin" size={13}/>:at>=0?at+1:<Check size={13}/>}</span>}
      {label?<span className="library-label">{label}</span>:null}
    </button>;
    if(!mark)return button;
    // Uploads and Results: the box marks the picture for Select all / Delete; a click on the picture uses it.
    const on=picked.includes(mark);
    return <div key={key} className="library-cell">{button}
      <button type="button" className={`library-select ${on?'on':''}`} role="checkbox" aria-checked={on} aria-label={`${on?'Unmark':'Mark'} ${title||'picture'} for delete`} title="Mark for delete" onClick={e=>{e.stopPropagation();togglePicked(mark);}}>{on?'✓':''}</button></div>;
  };
  return <div className="input-library">
    <div className="library-tabs" role="tablist">
      <button role="tab" aria-selected={tab==='people'} onClick={()=>{setTab('people');setDrawer(null);}}>People <span>{people.length}</span></button>
      <button role="tab" aria-selected={tab==='uploads'} onClick={()=>{setTab('uploads');}}>Uploads <span>{total}</span></button>
      <button role="tab" aria-selected={tab==='results'} onClick={()=>setTab('results')}>Results <span>{shownResults.length}</span></button>
      <button role="tab" aria-selected={tab==='web'} onClick={()=>setTab('web')}>Web</button>
      {swap?<span className="library-replace">Replacing image {Math.min(target,inputs.length-1)+1}</span>:null}
      <FilePicker className="text-button library-upload" disabled={disabled} multiple={!swap} onFiles={onUpload}><Upload size={14}/>Upload</FilePicker>
      {onClose?<button type="button" className="icon-button library-close" onClick={()=>{setDrawer(null);onClose();}} aria-label="Close library" title="Close (Esc)"><X size={15}/></button>:null}
    </div>
    {(()=>{const pool=tab==='uploads'?uploads.map(u=>u.id):tab==='results'?shownResults.map(r=>r.id):[];if(!pool.length)return null;const noun=tab==='results'?'result':'uploaded picture';
      return <><div className="library-select-bar">
      {selecting?<span>{allResults??picked.length} marked{allResults!==null?' (all image results)':''}</span>:<span className="library-select-hint">The box on a picture marks it.</span>}
      {selecting?<button type="button" className="text-button" onClick={stopSelecting}>Select none</button>:<button type="button" className="text-button" onClick={()=>void selectAll()}>Select all</button>}
      {selecting?<><button type="button" className="text-button library-delete" disabled={busy==='delete'} onClick={()=>setConfirming(true)}>Delete {allResults??picked.length}</button>
</>:null}
    </div>
    {confirming?<div className="library-confirm" role="alert"><span>Delete {allResults??picked.length} {noun}{(allResults??picked.length)>1?'s':''} from this computer? {tab==='results'?'The pictures and their settings are removed.':'Results made from them stay.'}</span>
      <button type="button" className="button danger" disabled={busy==='delete'} onClick={()=>void deletePicked()}>{busy==='delete'?'Deleting…':'Delete'}</button>
      <button type="button" className="text-button" onClick={()=>setConfirming(false)}>Keep</button></div>:null}</>;})()}
    {tab==='people'&&!editing?<div className="people-tools">
      <label className="people-search"><Search size={13}/><input value={filter} onChange={e=>setFilter(e.target.value)} placeholder="Find a person"/></label>
      <button type="button" className="text-button" disabled={disabled} onClick={()=>{setDrawer(null);setEditing({start:inputs.map(i=>({id:i.id,url:i.url}))});}} title="New person: a name and photos from uploads, the checked inputs or the web"><UserPlus size={14}/>New</button>
      <button type="button" className="text-button" disabled={importing?.state==='running'} onClick={importMoise} title="Import contacts and photos from Moise">{importing?.state==='running'?<Loader2 className="spin" size={14}/>:<Download size={14}/>}Moise</button>
    </div>:null}
    {importing?.state==='running'?<div className="people-progress"><progress value={importing.done||0} max={importing.total||1}/><span>{importing.message}{importing.total?` · ${importing.done}/${importing.total}`:''}</span></div>:message?<p className="people-message">{message}</p>:null}
    {tab==='people'&&editing?<PersonEditor key={editing.person?.id??'new'} api={api} person={editing.person} start={editing.start} startName={editing.name} checked={inputs} onPeople={onPeople}
      onDone={made=>{setEditing(null);if(made&&!editing.person&&!multiple&&inputs.length<limit)add({id:made.image_id,name:made.name,url:made.url,personId:made.id});}}/>:null}
    {tab==='people'&&!opened&&!editing?<div className="people-grid">{shown.map(p=>{const at=personAt(p);return <div key={p.id} className={`person ${at>=0?'checked':''}`}>
      <button type="button" className="person-face" disabled={disabled||(inputs.length>=limit&&!swap&&!(multiple&&at>=0))} onClick={()=>togglePerson(p)} aria-label={`Use ${p.name}`} title={`${p.name} · ${p.image_ids.length} photo${p.image_ids.length>1?'s':''}`} aria-pressed={at>=0}>
        <img src={p.thumb_url} alt="" loading="lazy"/><span className="library-check">{at>=0?at+1:<Check size={12}/>}</span></button>
      <button type="button" className="person-name" dir="auto" disabled={disabled||(inputs.length>=limit&&!swap&&!(multiple&&at>=0))} onClick={()=>togglePerson(p)}>{p.name}</button>
      {p.image_ids.length>1?<button type="button" className={`person-photos ${drawer===p.id?'open':''}`} onClick={()=>setDrawer(p.id)} aria-label={`Choose photo of ${p.name}`} title={`Choose one of ${p.image_ids.length} photos`} aria-expanded={drawer===p.id}><Images size={14}/>{p.image_ids.length} photos</button>:null}
      <span className="person-actions"><button type="button" onClick={()=>{setDrawer(null);setEditing({person:p});}} aria-label={`Edit ${p.name}`} title="Name and photos"><Pencil size={11}/></button></span>
    </div>;})}{!shown.length?<p className="library-empty">{people.length?'No match.':'No people yet. Import your contacts from Moise, or check photos and click New.'}</p>:null}</div>:null}
    {tab==='people'&&opened&&!editing?<div className="person-drawer">
      <div className="person-drawer-head"><button type="button" className="text-button" onClick={()=>setDrawer(null)}>Back to people</button><b dir="auto">{opened.name}</b><span>{opened.image_ids.length} photos</span></div>
      <div className="person-drawer-grid">{opened.image_ids.map((id,n)=>{const used=inputs.findIndex(i=>i.id===id);return <div key={id} className={`person-photo ${used>=0?'checked':''}`}>
        <button type="button" className={`library-tile checker ${used>=0?"checked":""}`} disabled={disabled} onClick={()=>pickPhoto(opened,id)} aria-label={`Use photo ${n+1} of ${opened.name}`} aria-pressed={used>=0} title={multiple&&used>=0?'Remove from inputs':'Use this photo'}>
          <img src={`/api/uploads/${id}/thumb`} alt="" loading="lazy"/><span className="library-check">{used>=0?used+1:<Check size={13}/>}</span></button>
        {n===0?<span className="person-main">Main</span>:<button type="button" className="person-main-set" onClick={()=>makeMain(opened,id)} title="Make this the main photo and avatar"><Star size={10}/>Main</button>}
      </div>;})}</div>
    </div>:null}
    {tab==='web'?<WebSearch api={api} disabled={disabled} onUse={add} onPeople={onPeople} onPerson={(image,query)=>{setTab('people');setDrawer(null);setEditing({start:[{id:image.id,url:image.url}],name:query});}}/>:null}
    <div className="library-grid" hidden={tab==='people'||tab==='web'}>
      {tab==='uploads'?uploads.map(u=>tile(u.id,u.thumb_url,u.name,index(u.id),()=>toggle({id:u.id,name:u.name,url:u.url}),false,undefined,u.id)):null}
      {tab==='results'?shownResults.map(r=>tile(r.id,`/api/jobs/${r.id}/thumbnail`,r.request.prompt,resultInputs[r.id]?index(resultInputs[r.id]):-1,()=>toggleResult(r),busy===r.id,undefined,r.id)):null}
      {(tab==='uploads'&&!uploads.length)||(tab==='results'&&!shownResults.length)||false?<p className="library-empty">{tab==='uploads'?'Images you upload or paste appear here.':tab==='results'?'Your finished images appear here.':'No people yet.'}</p>:null}
    </div>
  </div>;
}
