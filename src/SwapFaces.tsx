import {useEffect,useRef,useState} from 'react';
import {ArrowRight,Loader2,RefreshCw} from 'lucide-react';
import {ReferenceChoice} from './ReferenceChoice';
import type {Api} from './ModelSettings';
import './swap-faces.css';

type Face={id:string;box:number[];preview:string};
type Analysis={faces:Face[];revision:string;width:number;height:number};
export type SwapSelection={scene_image:string;reference_image:string;scene_revision:string;reference_revision:string;assignments:Record<string,string|null>};

export function SwapFaces({api,images,value,onChange,disabled}:{api:Api;images:[string|null,string|null];value:SwapSelection|null;onChange:(value:SwapSelection|null)=>void;disabled:boolean}){
  const [scans,setScans]=useState<[Analysis,Analysis]|null>(null);
  const [status,setStatus]=useState<'idle'|'loading'|'ready'|'error'>('idle');
  const [error,setError]=useState('');const [retry,setRetry]=useState(0);
  const current=useRef({value,onChange});current.current={value,onChange};
  const previousChoice=useRef(value);if(value)previousChoice.current=value;
  useEffect(()=>{
    setScans(null);setError('');
    if(!images[0]||!images[1]){setStatus('idle');return;}
    const ids=images as [string,string];const saved=current.current.value||previousChoice.current;
    let alive=true;const controller=new AbortController();setStatus('loading');
    Promise.all(ids.map(id=>api(`/api/image-people?image_id=${encodeURIComponent(id)}`,{signal:controller.signal}))).then(results=>{
      if(!alive)return;
      const [scene,reference]=results as [Analysis,Analysis];setScans([scene,reference]);setStatus('ready');
      const same=saved?.scene_image===ids[0]&&saved?.reference_image===ids[1]&&saved.scene_revision===scene.revision&&saved.reference_revision===reference.revision;
      const assignments=Object.fromEntries(scene.faces.map(face=>[face.id,
        same&&reference.faces.some(candidate=>candidate.id===saved.assignments[face.id])?saved.assignments[face.id]:
        !same&&scene.faces.length===1&&reference.faces.length===1?reference.faces[0].id:null]));
      current.current.onChange({scene_image:ids[0],reference_image:ids[1],scene_revision:scene.revision,reference_revision:reference.revision,assignments});
    }).catch(issue=>{if(alive){setStatus('error');setError(issue.message);}});
    return()=>{alive=false;controller.abort();};
  },[api,images[0],images[1],retry]);
  if(!images[0]||!images[1])return <p className="swap-face-hint">Add both pictures to choose which people to replace.</p>;
  const count=Object.values(value?.assignments||{}).filter(Boolean).length;
  return <section className="swap-face-picker" aria-label="Face replacements" aria-busy={status==='loading'}>
    <div className="swap-face-heading"><h3>Choose people to replace</h3><button type="button" className="text-button" disabled={disabled||status==='loading'} onClick={()=>setRetry(n=>n+1)}><RefreshCw size={14}/>Scan again</button></div>
    {status==='loading'?<p className="swap-face-status" role="status"><Loader2 size={16} className="spin"/>Finding faces in both pictures…</p>:null}
    {status==='error'?<p className="message error" role="alert">Could not find faces: {error}</p>:null}
    {scans&&status==='ready'?<>
      <div className="swap-face-counts"><span>Picture 1: {scans[0].faces.length} {scans[0].faces.length===1?'face':'faces'}</span><span>Picture 2: {scans[1].faces.length} {scans[1].faces.length===1?'face':'faces'}</span></div>
      {!scans[0].faces.length||!scans[1].faces.length?<p className="swap-face-hint" role="status">No faces found in picture {!scans[0].faces.length?'1':'2'}. Use a picture with a clear face, or scan again.</p>:<>
        <p className="swap-face-hint">For each person in picture 1, choose a person from picture 2. Leave other people unchanged.</p>
        <div className="swap-face-rows">{scans[0].faces.map((face,index)=><div className={`swap-face-row ${value?.assignments[face.id]?'is-replaced':''}`} key={face.id}>
          <div className="swap-face-source"><img src={face.preview} alt={`Face ${index+1} in picture 1`}/><div><strong>Person {index+1}</strong><small>Picture 1</small></div></div>
          <ArrowRight size={16} aria-hidden="true"/>
          <ReferenceChoice personNumber={index+1} sourceLabel="Person" disabled={disabled||!value} references={scans[1].faces.map(item=>({id:item.id,url:item.preview}))} value={value?.assignments[face.id]||null} onChange={chosen=>{if(value)onChange({...value,assignments:{...value.assignments,[face.id]:chosen}});}}/>
        </div>)}</div>
        <p className="swap-face-status" role="status">{count?`${count} ${count===1?'person':'people'} selected for replacement.`:'Choose a replacement for at least one person.'}</p>
      </>}
    </>:null}
  </section>;
}
