import {useRef,useState,type ReactNode} from 'react';
import {Images} from 'lucide-react';
import {InputLibrary,type InputImage,type Person} from './InputLibrary';
import {LibraryPopover} from './LibraryPopover';
import {PersonPhotoCycle} from './PersonPhotoCycle';
import type {Api} from './ModelSettings';

export type VideoLibraryData={people:Person[];onPeople:()=>void;results:{id:string;request:{prompt:string}}[]};

/** One picker owns one destination. A person tap chooses a photo and closes. */
export function VideoImageLibrary({api,library,value,onChoose,onUpload,disabled=false,label='Library or People',className='text-button',children,cycle=true}:{
  api:Api;library:VideoLibraryData;value:InputImage|null;onChoose:(image:InputImage)=>void;onUpload:(files:File[])=>void|Promise<void>;
  disabled?:boolean;label?:string;className?:string;children?:ReactNode;cycle?:boolean;
}){
  const anchor=useRef<HTMLButtonElement>(null);
  const [open,setOpen]=useState(false),[choice,setChoice]=useState<InputImage|null>(null);
  const current=choice?.id===value?.id?choice:value;
  function choose(image:InputImage){setChoice(image);onChoose(image);setOpen(false);}
  return <>
    <button ref={anchor} type="button" className={className} disabled={disabled} aria-label={label} aria-haspopup="dialog" aria-expanded={open} onClick={()=>setOpen(true)}>{children||<><Images size={16}/>Library or People</>}</button>
    {cycle&&current?<PersonPhotoCycle image={current} people={library.people} disabled={disabled} onChange={choose}/>:null}
    <LibraryPopover open={open} onClose={()=>setOpen(false)} anchor={anchor}>{open?<InputLibrary api={api} inputs={current?[current]:[]} onChange={next=>{if(next[0])choose(next[0]);}} results={library.results} people={library.people} onPeople={library.onPeople} revision={0} disabled={disabled} onUpload={files=>{onUpload(files);setOpen(false);}} limit={1} replace={!!current} target={0} onClose={()=>setOpen(false)}/>:null}</LibraryPopover>
  </>;
}
