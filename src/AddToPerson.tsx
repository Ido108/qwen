import {useEffect,useRef,useState} from 'react';
import {createPortal} from 'react-dom';
import type {Api} from './ModelSettings';
import type {InputImage,Person} from './InputLibrary';
import {inputPerson} from './PersonPhotoCycle';

/** "+ Person" on a picture that belongs to nobody in People: add it to an existing person, or make a new person
 *  from it. A People photo then shows under People, no longer under Uploads. */
export function AddToPerson({api,image,people,onPeople,onLinked,disabled=false}:{api:Api;image:InputImage;people:Person[];onPeople:()=>void;onLinked?:(image:InputImage)=>void;disabled?:boolean}){
  // Drawn above the page at the button's place: a picture frame crops anything inside it.
  const [open,setOpen]=useState<{top:number;left:number}|null>(null);const menu=useRef<HTMLDivElement>(null);const [filter,setFilter]=useState('');const [name,setName]=useState('');
  const [busy,setBusy]=useState(false);const [message,setMessage]=useState('');
  const box=useRef<HTMLSpanElement>(null);
  useEffect(()=>{if(!open)return;const close=(e:PointerEvent)=>{if(!box.current?.contains(e.target as Node)&&!menu.current?.contains(e.target as Node))setOpen(null);};
    const away=()=>setOpen(null);document.addEventListener('pointerdown',close);window.addEventListener('resize',away);
    return()=>{document.removeEventListener('pointerdown',close);window.removeEventListener('resize',away);};},[open]);
  // An example picture is not a saved file; a People photo already has its person.
  if(image.id.startsWith('example:')||inputPerson(image,people)||people.some(p=>p.image_ids.includes(image.id)))return null;
  async function link(person?:Person){setBusy(true);setMessage('');
    try{const made:Person=person
        ?await api(`/api/people/${person.id}`,{method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({image_ids:[...person.image_ids,image.id]})})
        :await api('/api/people',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name.trim(),image_ids:[image.id]})});
      onPeople();onLinked?.({...image,personId:made.id,name:made.name});setOpen(null);setName('');setFilter('');}
    catch(e){setMessage((e as Error).message);}finally{setBusy(false);}}
  const shown=people.filter(p=>p.name.toLowerCase().includes(filter.trim().toLowerCase())).slice(0,40);
  return <span className="add-to-person" ref={box}>
    <button type="button" className="add-to-person-open" disabled={disabled} aria-expanded={!!open} onClick={e=>{e.stopPropagation();const r=e.currentTarget.getBoundingClientRect();
      setOpen(o=>o?null:{top:Math.min(r.bottom+4,window.innerHeight-330),left:Math.max(8,Math.min(r.left,window.innerWidth-262))});}} title="Add this picture to People">+ Person</button>
    {open?createPortal(<div ref={menu} className="add-to-person-menu" role="dialog" aria-label="Add to People" style={{top:open.top,left:open.left}} onClick={e=>e.stopPropagation()}>
      <form className="add-to-person-new" onSubmit={e=>{e.preventDefault();if(name.trim())void link();}}>
        <input value={name} onChange={e=>setName(e.target.value)} placeholder="New person: name" aria-label="Name of the new person" dir="auto" autoFocus/>
        <button type="submit" className="button secondary" disabled={busy||!name.trim()}>Create</button></form>
      <input className="add-to-person-filter" value={filter} onChange={e=>setFilter(e.target.value)} placeholder="Or add to: find a person" aria-label="Find a person"/>
      <div className="add-to-person-list">{shown.map(p=><button type="button" key={p.id} disabled={busy} onClick={()=>void link(p)}><img src={p.thumb_url} alt=""/><span dir="auto">{p.name}</span></button>)}</div>
      {message?<small role="alert">{message}</small>:null}
    </div>,document.body):null}
  </span>;
}
