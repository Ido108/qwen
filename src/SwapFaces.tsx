import {useEffect,useRef,useState} from 'react';
import {Loader2,RefreshCw} from 'lucide-react';
import type {Api} from './ModelSettings';
import './swap-faces.css';

type Face={id:string;box:number[];preview:string;gender?:'female'|'male'|null};
type Analysis={faces:Face[];revision:string;width:number;height:number};
/** Scene face id -> "2" for the one person who gets the head of picture 2, null for everyone else. */
export type SwapSelection={scene_image:string;reference_image:string;scene_revision:string;reference_revision?:string|null;assignments:Record<string,string|null>};

const area=(face:Face)=>face.box[2]*face.box[3];
/** The default person: the one whose gender matches the head picture (the largest such face), else the largest face. */
export function defaultFace(faces:Face[],headGender?:string|null){
  const pool=headGender?faces.filter(face=>face.gender===headGender):[];
  return [...(pool.length?pool:faces)].sort((a,b)=>area(b)-area(a))[0];
}

/** Picture 1 (the scene) is scanned; picture 2 is the one head. With one face nothing is asked; with more, one is chosen. */
export function SwapFaces({api,scene,head,value,onChange,disabled}:{api:Api;scene:string|null;head:string|null;value:SwapSelection|null;onChange:(value:SwapSelection|null)=>void;disabled:boolean}){
  const [scan,setScan]=useState<Analysis|null>(null);const [headGender,setHeadGender]=useState<string|null>(null);
  const [status,setStatus]=useState<'idle'|'loading'|'ready'|'error'>('idle');
  const [error,setError]=useState('');const [retry,setRetry]=useState(0);
  const current=useRef({value,onChange});current.current={value,onChange};
  // A person you clicked stays chosen; until then the default follows the head picture's gender.
  const manual=useRef<string|null>(null);
  useEffect(()=>{
    setScan(null);setError('');
    if(!scene){setStatus('idle');return;}
    let alive=true;const controller=new AbortController();setStatus('loading');
    api(`/api/image-people?image_id=${encodeURIComponent(scene)}`,{signal:controller.signal}).then((result:Analysis)=>{if(alive){setScan(result);setStatus('ready');}})
      .catch(issue=>{if(alive){setStatus('error');setError(issue.message);}});
    return()=>{alive=false;controller.abort();};
  },[api,scene,retry]);
  // The head picture is one person: scanned only to know whether it is a woman or a man.
  useEffect(()=>{
    setHeadGender(null);if(!head)return;let alive=true;
    api(`/api/image-people?image_id=${encodeURIComponent(head)}`).then((result:Analysis)=>{if(alive)setHeadGender([...result.faces].sort((a,b)=>area(b)-area(a))[0]?.gender??null);}).catch(()=>{});
    return()=>{alive=false;};
  },[api,head]);
  useEffect(()=>{
    if(!scan||!scene||!scan.faces.length)return;
    const saved=current.current.value;
    const kept=manual.current===scene&&saved?.scene_image===scene&&saved.scene_revision===scan.revision?Object.entries(saved.assignments).find(([id,v])=>v&&scan.faces.some(f=>f.id===id))?.[0]:undefined;
    const chosen=kept??defaultFace(scan.faces,headGender).id;
    const next={scene_image:scene,reference_image:head??'',scene_revision:scan.revision,assignments:Object.fromEntries(scan.faces.map(face=>[face.id,face.id===chosen?'2':null]))};
    if(JSON.stringify(next)!==JSON.stringify(saved&&{scene_image:saved.scene_image,reference_image:saved.reference_image,scene_revision:saved.scene_revision,assignments:saved.assignments}))current.current.onChange(next);
  },[scan,scene,head,headGender]);
  if(!scene)return null;
  if(status==='loading')return <p className="swap-face-status" role="status"><Loader2 size={16} className="spin"/>Finding faces in picture 1…</p>;
  if(status==='error')return <p className="message error" role="alert">Could not find faces: {error} <button type="button" className="text-button" onClick={()=>setRetry(n=>n+1)}>Scan again</button></p>;
  if(!scan)return null;
  if(!scan.faces.length)return <p className="swap-face-hint" role="status">No face found in picture 1. Use a picture with a clear face, or <button type="button" className="text-button" onClick={()=>setRetry(n=>n+1)}>scan again</button>.</p>;
  if(scan.faces.length===1)return null;
  const chosen=Object.entries(value?.assignments||{}).find(([,v])=>v)?.[0];
  return <section className="swap-face-picker" aria-label="Person to replace">
    <div className="swap-face-heading"><h3>Which person gets the new head?</h3><button type="button" className="text-button" disabled={disabled} onClick={()=>setRetry(n=>n+1)}><RefreshCw size={14}/>Scan again</button></div>
    <div className="swap-face-pick" role="radiogroup" aria-label="Person in picture 1">{scan.faces.map((face,index)=>
      <button type="button" role="radio" aria-checked={face.id===chosen} key={face.id} className={face.id===chosen?'chosen':''} disabled={disabled||!value}
        onClick={()=>{manual.current=scene;if(value)onChange({...value,assignments:Object.fromEntries(scan.faces.map(f=>[f.id,f.id===face.id?'2':null]))});}}>
        <img src={face.preview} alt=""/><span>{face.gender==='female'?'Woman':face.gender==='male'?'Man':'Person'} {index+1}</span>
      </button>)}</div>
  </section>;
}
