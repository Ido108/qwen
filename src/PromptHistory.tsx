import {useEffect,useRef,useState} from 'react';
import {History,Play,X} from 'lucide-react';

const LIMIT=60;
const key=(scope:string)=>`prompt-history:${scope}`;

function read(scope:string):string[]{
  try{const value=JSON.parse(localStorage.getItem(key(scope))||'[]');return Array.isArray(value)?value.filter(v=>typeof v==='string'):[];}catch{return [];}
}

function write(scope:string,list:string[]){
  try{localStorage.setItem(key(scope),JSON.stringify(list));}catch{}
  window.dispatchEvent(new CustomEvent('prompt-history',{detail:scope}));
}

/** Call when a job is submitted. The newest prompt goes first, without duplicates. */
export function rememberPrompt(scope:string,prompt:string){
  const text=prompt.trim();if(!text)return;
  write(scope,[text,...read(scope).filter(p=>p!==text)].slice(0,LIMIT));
}

/** A small dropdown of earlier prompts: click a line to load it, or run it again. */
export function PromptHistory({scope,onLoad,onRun,disabled}:{scope:string;onLoad:(prompt:string)=>void;onRun?:(prompt:string)=>void;disabled?:boolean}){
  const [open,setOpen]=useState(false);const [list,setList]=useState<string[]>(()=>read(scope));const box=useRef<HTMLDivElement>(null);
  useEffect(()=>{const update=(e:Event)=>{if((e as CustomEvent).detail===scope)setList(read(scope));};window.addEventListener('prompt-history',update);return()=>window.removeEventListener('prompt-history',update);},[scope]);
  useEffect(()=>{if(!open)return;const close=(e:MouseEvent)=>{if(!box.current?.contains(e.target as Node))setOpen(false);};const escape=(e:KeyboardEvent)=>{if(e.key==='Escape')setOpen(false);};
    document.addEventListener('mousedown',close);document.addEventListener('keydown',escape);return()=>{document.removeEventListener('mousedown',close);document.removeEventListener('keydown',escape);};},[open]);
  return <div className="prompt-history" ref={box}>
    <button type="button" className="prompt-history-toggle" aria-expanded={open} disabled={!list.length} onClick={()=>setOpen(o=>!o)} title={list.length?'Earlier prompts':'No earlier prompts yet'}><History size={14}/><span>{list.length}</span></button>
    {open?<div className="prompt-history-menu" role="menu">
      {list.map(prompt=><div key={prompt} className="prompt-history-row">
        <button type="button" role="menuitem" className="prompt-history-text" dir="auto" title={prompt} onClick={()=>{onLoad(prompt);setOpen(false);}}>{prompt}</button>
        {onRun?<button type="button" className="prompt-history-run" disabled={disabled} title="Load and run again" aria-label="Run again" onClick={()=>{onRun(prompt);setOpen(false);}}><Play size={12}/></button>:null}
        <button type="button" className="prompt-history-remove" title="Remove from history" aria-label="Remove from history" onClick={()=>{const next=list.filter(p=>p!==prompt);write(scope,next);if(!next.length)setOpen(false);}}><X size={12}/></button>
      </div>)}
    </div>:null}
  </div>;
}
