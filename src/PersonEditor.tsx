import {useState} from 'react';
import type {Api} from './ModelSettings';
import type {InputImage,Person} from './InputLibrary';
import {FilePicker} from './FilePicker';
import {WebSearch} from './WebSearch';

type Photo={id:string;url:string};

/** New person or edit one, inside the Library (as Image Studio's avatar manager): a name, the photos, and three ways
 *  to add photos: upload, the checked inputs, or a web search. No browser dialog. */
export function PersonEditor({api,person,start=[],startName='',checked=[],onDone,onPeople}:{api:Api;person?:Person;start?:Photo[];startName?:string;checked:InputImage[];onDone:(made?:Person)=>void;onPeople:()=>void}){
  const [name,setName]=useState(person?.name??startName);
  const [photos,setPhotos]=useState<Photo[]>(person?person.image_ids.map(id=>({id,url:`/api/uploads/${id}`})):start);
  const [busy,setBusy]=useState('');const [message,setMessage]=useState('');const [confirmRemove,setConfirmRemove]=useState(false);
  const addPhotos=(next:Photo[])=>setPhotos(old=>[...old,...next.filter(n=>!old.some(o=>o.id===n.id))]);
  async function upload(files:File[]){if(!files.length)return;setBusy('upload');setMessage('');
    try{const body=new FormData();files.slice(0,10).forEach(f=>body.append('files',f));const images:Photo[]=await api('/api/uploads',{method:'POST',body});addPhotos(images.map(i=>({id:i.id,url:i.url})));}
    catch(e){setMessage((e as Error).message);}finally{setBusy('');}}
  async function save(){setBusy('save');setMessage('');
    try{const body=JSON.stringify({name:name.trim(),image_ids:photos.map(p=>p.id)});
      const made:Person=person?await api(`/api/people/${person.id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body})
        :await api('/api/people',{method:'POST',headers:{'Content-Type':'application/json'},body});
      onPeople();onDone(made);}
    catch(e){setMessage((e as Error).message);}finally{setBusy('');}}
  async function remove(){if(!person)return;setBusy('remove');try{await api(`/api/people/${person.id}`,{method:'DELETE'});onPeople();onDone();}catch(e){setMessage((e as Error).message);setBusy('');}}
  const extra=checked.filter(c=>!photos.some(p=>p.id===c.id));
  return <section className="person-editor" aria-label={person?`Edit ${person.name}`:'New person'}>
    <div className="person-editor-head"><button type="button" className="text-button" onClick={()=>onDone()}>Back to people</button><b>{person?'Edit person':'New person'}</b></div>
    <input className="person-editor-name" value={name} onChange={e=>setName(e.target.value)} placeholder="Name" aria-label="Name" maxLength={80} dir="auto" autoFocus/>
    <div className="person-editor-photos">{photos.map((p,i)=><figure key={p.id}><img src={`/api/uploads/${p.id}/thumb`} alt=""/>{i===0?<span className="person-main">Main</span>:null}
      <button type="button" aria-label={`Remove photo ${i+1}`} onClick={()=>setPhotos(old=>old.filter(o=>o.id!==p.id))}>×</button></figure>)}
      {!photos.length?<p className="library-empty">No photos yet. Add them below.</p>:null}</div>
    <div className="person-editor-add"><span>Add photos:</span>
      <FilePicker className="text-button" multiple accept="image/*,.heic,.heif" disabled={!!busy} onFiles={upload}>{busy==='upload'?'Uploading…':'Upload'}</FilePicker>
      {extra.length?<button type="button" className="text-button" onClick={()=>addPhotos(extra.map(c=>({id:c.id,url:c.url})))}>The {extra.length} checked</button>:null}</div>
    <WebSearch api={api} disabled={!!busy} compact onUse={image=>addPhotos([{id:image.id,url:image.url}])} onPeople={()=>{}}/>
    {message?<p className="people-message" role="alert">{message}</p>:null}
    <div className="person-editor-actions">
      <button type="button" className="button primary" disabled={!!busy||!name.trim()||!photos.length} onClick={()=>void save()}>{busy==='save'?'Saving…':person?'Save':'Create person'}</button>
      <button type="button" className="text-button" onClick={()=>onDone()}>Cancel</button>
      {person?(confirmRemove?<span className="person-editor-remove"><span>Remove {person.name}? The photos stay in Uploads.</span><button type="button" className="button danger" disabled={!!busy} onClick={()=>void remove()}>Remove</button><button type="button" className="text-button" onClick={()=>setConfirmRemove(false)}>Keep</button></span>
        :<button type="button" className="text-button library-delete" onClick={()=>setConfirmRemove(true)}>Remove person</button>):null}
    </div>
  </section>;
}
