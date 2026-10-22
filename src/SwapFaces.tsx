import {useEffect,useRef,useState} from 'react';
import {ArrowRight,Loader2,RefreshCw} from 'lucide-react';
import {ReferenceChoice} from './ReferenceChoice';
import type {Api} from './ModelSettings';
import './swap-faces.css';

type Face={id:string;box:number[];preview:string};
type Analysis={faces:Face[];revision:string;width:number;height:number};
/** Scene face id -> the picture number of its new head ("2" or "3"), or null to keep that person. */
export type SwapSelection={scene_image:string;reference_image:string;scene_revision:string;reference_revision?:string|null;assignments:Record<string,string|null>};

/** Only picture 1 (the scene) is scanned: each head picture is one person. One face and one head need no choice. */
export function SwapFaces({api,scene,heads,value,onChange,onFaces,disabled}:{api:Api;scene:string|null;heads:{id:string;url:string}[];value:SwapSelection|null;onChange:(value:SwapSelection|null)=>void;onFaces?:(count:number)=>void;disabled:boolean}){
  const [scan,setScan]=useState<Analysis|null>(null);
  const [status,setStatus]=useState<'idle'|'loading'|'ready'|'error'>('idle');
  const [error,setError]=useState('');const [retry,setRetry]=useState(0);
  const current=useRef({value,onChange,onFaces});current.current={value,onChange,onFaces};
  const headKey=heads.map(h=>h.id).join('|');
  useEffect(()=>{
    setScan(null);setError('');
    if(!scene){setStatus('idle');current.current.onFaces?.(0);return;}
    let alive=true;const controller=new AbortController();setStatus('loading');
    api(`/api/image-people?image_id=${encodeURIComponent(scene)}`,{signal:controller.signal}).then((result:Analysis)=>{
      if(!alive)return;
      setScan(result);setStatus('ready');current.current.onFaces?.(result.faces.length);
    }).catch(issue=>{if(alive){setStatus('error');setError(issue.message);}});
    return()=>{alive=false;controller.abort();};
  },[api,scene,retry]);
  // Keep the choices valid for the current heads; one face with one head is chosen automatically.
  useEffect(()=>{
    if(!scan||!scene||!heads.length)return;
    const saved=current.current.value;
    const same=saved?.scene_image===scene&&saved.scene_revision===scan.revision;
    const pictures=heads.map((_,index)=>String(index+2));
    const assignments=Object.fromEntries(scan.faces.map(face=>{
      const previous=same?saved!.assignments[face.id]:undefined;
      const kept=previous&&pictures.includes(previous)?previous:previous&&!/^\d+$/.test(previous)?'2':null;
      return [face.id,scan.faces.length===1&&heads.length===1?'2':kept??null];
    }));
    const next={scene_image:scene,reference_image:heads[0].id,scene_revision:scan.revision,assignments};
    if(JSON.stringify(next)!==JSON.stringify(saved&&{scene_image:saved.scene_image,reference_image:saved.reference_image,scene_revision:saved.scene_revision,assignments:saved.assignments}))current.current.onChange(next);
  },[scan,scene,headKey]);
  if(!scene||!heads.length)return <p className="swap-face-hint">Add the scene and a head picture.</p>;
  if(status==='loading')return <p className="swap-face-status" role="status"><Loader2 size={16} className="spin"/>Finding faces in picture 1…</p>;
  if(status==='error')return <p className="message error" role="alert">Could not find faces: {error} <button type="button" className="text-button" onClick={()=>setRetry(n=>n+1)}>Scan again</button></p>;
  if(!scan)return null;
  if(!scan.faces.length)return <p className="swap-face-hint" role="status">No face found in picture 1. Use a picture with a clear face, or <button type="button" className="text-button" onClick={()=>setRetry(n=>n+1)}>scan again</button>.</p>;
  if(scan.faces.length===1&&heads.length===1)return null;
  const count=Object.values(value?.assignments||{}).filter(Boolean).length;
  return <section className="swap-face-picker" aria-label="Face replacements">
    <div className="swap-face-heading"><h3>{scan.faces.length} people in picture 1: choose a head for each</h3><button type="button" className="text-button" disabled={disabled} onClick={()=>setRetry(n=>n+1)}><RefreshCw size={14}/>Scan again</button></div>
    <div className="swap-face-rows">{scan.faces.map((face,index)=><div className={`swap-face-row ${value?.assignments[face.id]?'is-replaced':''}`} key={face.id}>
      <div className="swap-face-source"><img src={face.preview} alt={`Face ${index+1} in picture 1`}/><div><strong>Person {index+1}</strong><small>Picture 1</small></div></div>
      <ArrowRight size={16} aria-hidden="true"/>
      <ReferenceChoice personNumber={index+1} sourceLabel="Person" optionLabel="Head" disabled={disabled||!value} references={heads.map((head,i)=>({id:String(i+2),url:head.url}))} value={value?.assignments[face.id]||null} onChange={chosen=>{if(value)onChange({...value,assignments:{...value.assignments,[face.id]:chosen}});}}/>
    </div>)}</div>
    <p className="swap-face-status" role="status">{count?`${count} ${count===1?'person':'people'} will get a new head.`:'Choose a head for at least one person.'}</p>
  </section>;
}
