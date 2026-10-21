import {useEffect,useRef,useState} from 'react';
import {knownPeopleOptions,type KnownPerson} from './promptMentions';
import './known-people.css';

/** People a model knows by trigger (MOSS Serenity), kept apart from the image @ mentions: a search box, a Women/Men
 *  filter, and the model README's prompt form inserted at the cursor. */
export function KnownPeoplePicker({people,source,prompt,disabled,onInsert}:{people:KnownPerson[];source:string;prompt:string;disabled?:boolean;onInsert:(text:string)=>void}){
  const [open,setOpen]=useState(false),[query,setQuery]=useState(''),[kind,setKind]=useState<'all'|'woman'|'man'>('all');
  const panel=useRef<HTMLDivElement>(null),search=useRef<HTMLInputElement>(null);
  useEffect(()=>{if(!open)return;search.current?.focus();
    const close=(e:PointerEvent)=>{if(!panel.current?.contains(e.target as Node))setOpen(false);};
    document.addEventListener('pointerdown',close);return()=>document.removeEventListener('pointerdown',close);},[open]);
  const words=query.trim().toLocaleLowerCase();
  const options=knownPeopleOptions(people,source,prompt).map((option,i)=>({option,person:people[i]}))
    .filter(({person})=>(kind==='all'||person.class===kind)&&(!words||`${person.label} ${person.trigger}`.toLocaleLowerCase().includes(words)));
  return <div className="known-people" ref={panel}>
    <button type="button" className="known-people-toggle" disabled={disabled} aria-expanded={open} onClick={()=>setOpen(!open)}>People in this model ({people.length})</button>
    {open?<div className="known-people-panel" role="dialog" aria-label={`People ${source} knows`} onKeyDown={e=>{if(e.key==='Escape'){e.stopPropagation();setOpen(false);}}}>
      <input ref={search} type="search" value={query} placeholder="Search by name" aria-label="Search people" onChange={e=>setQuery(e.target.value)}/>
      <div className="known-people-filter" role="group" aria-label="Filter">{([['all','All'],['woman','Women'],['man','Men']] as const).map(([id,label])=>
        <button type="button" key={id} aria-pressed={kind===id} onClick={()=>setKind(id)}>{label}</button>)}</div>
      <div className="known-people-list">{options.length?options.map(({option,person})=><button type="button" key={person.trigger} title={`Inserts: ${option.tag}`}
        onClick={()=>{onInsert(option.tag);setOpen(false);setQuery('');}}><strong>{person.label}</strong><small>{person.trigger} · {person.class}</small></button>)
        :<p>No one matches.</p>}</div>
      <small className="known-people-note">Inserts the model's prompt form, for example "photo of keanureeves, a man". Faces work best large in the frame.</small>
    </div>:null}
  </div>;
}
