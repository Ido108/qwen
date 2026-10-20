import {useState} from 'react';
import {ImagePlus,Plus,X} from 'lucide-react';
import {inputImage} from './Gallery';
import type {ReferencePerson} from './personReferences';
import './reference-people.css';
import {VideoImageLibrary,type VideoLibraryData} from './VideoImageLibrary';
import type {Api} from './ModelSettings';
import type {InputImage} from './InputLibrary';

export function ReferencePeople({api,library,people,busy,onAdd,onChange}:{api:Api;library:VideoLibraryData;people:ReferencePerson[];busy:boolean;onAdd:(files:File[],personId:string|null,index?:number)=>void;onChange:(people:ReferencePerson[])=>void}){
  const [open,setOpen]=useState<string|null>(null);
  const total=people.reduce((count,person)=>count+person.images.length,0);
  function pick(image:InputImage,personId:string|null,index?:number){
    if(index===undefined&&total>=9)return;
    onChange(personId?people.map(p=>p.id===personId?{...p,images:index===undefined?[...p.images,image.id]:p.images.map((id,i)=>i===index?image.id:id)}:p):[...people,{id:'ref-person-'+crypto.randomUUID(),images:[image.id]}]);
  }
  return <section className="reference-people" aria-label="Replacement people">
    <div className="reference-people-heading"><strong>People to use</strong><small>{total}/9 images</small></div>
    <div className="reference-people-cards">{people.map((person,index)=><div className="reference-person" key={person.id}>
      <button type="button" className="reference-person-cover" onClick={()=>setOpen(open===person.id?null:person.id)} aria-expanded={open===person.id} aria-label={`View images for Person ${index+1}`}>
        <img src={inputImage(person.images[0]).url} alt={`Person ${index+1}`}/><span>Person {index+1}</span><small>{person.images.length} {person.images.length===1?'image':'images'}</small>
      </button>
      <VideoImageLibrary api={api} library={library} value={null} className="reference-person-add" label={`Add images to Person ${index+1}`} disabled={busy||total>=9} onChoose={image=>pick(image,person.id)} onUpload={files=>onAdd(files,person.id)}><Plus size={14}/></VideoImageLibrary>
      <button type="button" className="reference-person-remove" disabled={busy} aria-label={`Remove Person ${index+1}`} onClick={()=>{onChange(people.filter(p=>p.id!==person.id));if(open===person.id)setOpen(null);}}><X size={12}/></button>
      <VideoImageLibrary api={api} library={library} value={inputImage(person.images[0])} label={`Replace photo for Person ${index+1}`} disabled={busy} onChoose={image=>pick(image,person.id,0)} onUpload={files=>onAdd(files,person.id,0)}/>
    </div>)}<VideoImageLibrary api={api} library={library} value={null} className="reference-person-new" label="Add person from Library or People" disabled={busy||total>=9} onChoose={image=>pick(image,null)} onUpload={files=>onAdd(files,null)}><ImagePlus size={20}/><span>Add person</span></VideoImageLibrary></div>
    {people.filter(person=>person.id===open).map(person=><div className="reference-person-images" key={person.id} aria-label={`Images for Person ${people.indexOf(person)+1}`}>
      <span>Person {people.indexOf(person)+1} · Different views of the same person</span>
      <div>{person.images.map((image,index)=><figure key={index}><img src={inputImage(image).url} alt={`View ${index+1}`}/><button type="button" disabled={busy} aria-label={`Remove view ${index+1}`} onClick={()=>onChange(people.map(p=>p.id===person.id?{...p,images:p.images.filter((_,i)=>i!==index)}:p).filter(p=>p.images.length))}><X size={12}/></button><VideoImageLibrary api={api} library={library} value={inputImage(image)} label={`Replace view ${index+1}`} disabled={busy} onChoose={next=>pick(next,person.id,index)} onUpload={files=>onAdd(files,person.id,index)}/></figure>)}<VideoImageLibrary api={api} library={library} value={null} className="reference-person-new" label="Add view from Library or People" disabled={busy||total>=9} onChoose={next=>pick(next,person.id)} onUpload={files=>onAdd(files,person.id)}><Plus size={18}/><span>Add view</span></VideoImageLibrary></div>
    </div>)}
  </section>;
}
