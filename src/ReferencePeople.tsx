import {useRef,useState} from 'react';
import {ImagePlus,Plus,X} from 'lucide-react';
import {inputImage} from './Gallery';
import type {ReferencePerson} from './personReferences';
import './reference-people.css';

export function ReferencePeople({people,busy,onAdd,onChange}:{people:ReferencePerson[];busy:boolean;onAdd:(files:FileList|null,personId:string|null)=>void;onChange:(people:ReferencePerson[])=>void}){
  const input=useRef<HTMLInputElement>(null),target=useRef<string|null>(null);
  const [open,setOpen]=useState<string|null>(null);
  const total=people.reduce((count,person)=>count+person.images.length,0);
  function add(id:string|null){target.current=id;input.current?.click();}
  return <section className="reference-people" aria-label="Replacement people">
    <div className="reference-people-heading"><strong>People to use</strong><small>{total}/9 images</small></div>
    <input ref={input} hidden type="file" accept="image/png,image/jpeg,image/webp" multiple onChange={e=>{onAdd(e.target.files,target.current);e.target.value='';}}/>
    <div className="reference-people-cards">{people.map((person,index)=><div className="reference-person" key={person.id}>
      <button type="button" className="reference-person-cover" onClick={()=>setOpen(open===person.id?null:person.id)} aria-expanded={open===person.id} aria-label={`View images for Person ${index+1}`}>
        <img src={inputImage(person.images[0]).url} alt={`Person ${index+1}`}/><span>Person {index+1}</span><small>{person.images.length} {person.images.length===1?'image':'images'}</small>
      </button>
      <button type="button" className="reference-person-add" disabled={busy||total>=9} onClick={()=>add(person.id)} aria-label={`Add images to Person ${index+1}`} title="Add another view"><Plus size={14}/></button>
      <button type="button" className="reference-person-remove" disabled={busy} aria-label={`Remove Person ${index+1}`} onClick={()=>{onChange(people.filter(p=>p.id!==person.id));if(open===person.id)setOpen(null);}}><X size={12}/></button>
    </div>)}<button type="button" className="reference-person-new" disabled={busy||total>=9} onClick={()=>add(null)}><ImagePlus size={20}/><span>Add person</span></button></div>
    {people.filter(person=>person.id===open).map(person=><div className="reference-person-images" key={person.id} aria-label={`Images for Person ${people.indexOf(person)+1}`}>
      <span>Person {people.indexOf(person)+1} · Different views of the same person</span>
      <div>{person.images.map((image,index)=><figure key={image}><img src={inputImage(image).url} alt={`View ${index+1}`}/><button type="button" disabled={busy} aria-label={`Remove view ${index+1}`} onClick={()=>onChange(people.map(p=>p.id===person.id?{...p,images:p.images.filter(id=>id!==image)}:p).filter(p=>p.images.length))}><X size={12}/></button></figure>)}<button type="button" className="reference-person-new" disabled={busy||total>=9} onClick={()=>add(person.id)}><Plus size={18}/><span>Add view</span></button></div>
    </div>)}
  </section>;
}
