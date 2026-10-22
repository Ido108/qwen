import {useEffect,useState} from 'react';
import type {Api} from './ModelSettings';

type Option={id:string;label:string;official:boolean};
type Arch={arch:string;title:string;choice:string;options:Option[];fixed:boolean};
const archOf=(family:string)=>['krea-2','z-image','moss-image','flux3-klein'].find(prefix=>family.startsWith(prefix))??'qwen-image-2.1';

/** Which text model writes the enhanced prompt, beside Enhance prompt; the same choice as Settings > Prompt enhancement. */
export function EnhancerPick({api,family,disabled}:{api:Api;family:string;disabled:boolean}){
  const [rows,setRows]=useState<Arch[]>([]);const [error,setError]=useState('');
  useEffect(()=>{api('/api/enhancer-settings').then(r=>setRows(r.architectures)).catch(()=>{});},[api]);
  const row=rows.find(r=>r.arch===archOf(family));
  if(!row||row.fixed||row.options.length<2)return null;
  async function choose(choice:string){
    setError('');
    try{const r=await api(`/api/enhancer-settings/${row!.arch}`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({choice})});setRows(r.architectures);}
    catch(e){setError((e as Error).message);}
  }
  return <span className="enhancer-pick">
    <select aria-label={`Text model for ${row.title}`} title="The text model that writes the enhanced prompt (also in Settings)" disabled={disabled} value={row.choice} onChange={e=>void choose(e.target.value)}>
      {row.options.map(option=><option key={option.id} value={option.id}>{option.label}</option>)}
    </select>{error?<small role="alert">{error}</small>:null}
  </span>;
}
