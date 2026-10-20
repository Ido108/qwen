import {ChevronLeft,ChevronRight} from 'lucide-react';
import type {InputImage,Person} from './InputLibrary';

export function inputPerson(image:InputImage,people:Person[]){
  if(image.personId)return people.find(p=>p.id===image.personId&&p.image_ids.includes(image.id));
  const owners=people.filter(p=>p.image_ids.includes(image.id));
  return owners.length===1?owners[0]:undefined;
}

/** Change only this input. Shared photos never imply a person without an explicit choice. */
export function PersonPhotoCycle({image,people,onChange,disabled=false}:{image:InputImage;people:Person[];onChange:(image:InputImage)=>void;disabled?:boolean}){
  const person=inputPerson(image,people);
  if(!person||person.image_ids.length<2)return null;
  const index=person.image_ids.indexOf(image.id);
  function move(delta:number){
    const id=person!.image_ids[(index+delta+person!.image_ids.length)%person!.image_ids.length];
    onChange({...image,id,name:person!.name,url:`/api/uploads/${id}`,personId:person!.id});
  }
  return <div className="person-photo-cycle" aria-label={`Photos of ${person.name}`}>
    <button type="button" disabled={disabled} aria-label={`Previous photo of ${person.name}`} onClick={()=>move(-1)}><ChevronLeft size={16}/></button>
    <span aria-live="polite">{index+1}/{person.image_ids.length}</span>
    <button type="button" disabled={disabled} aria-label={`Next photo of ${person.name}`} onClick={()=>move(1)}><ChevronRight size={16}/></button>
  </div>;
}
