import {useState} from 'react';
import './image-label.css';

const labels=['person','dog','cat','product','object','clothing','logo','style','background'];
export const subjectLabel=(value?:string)=>value?.trim()||'person';
/** A subject label describes what a mention selects. It is not the file or person's name. */
export function ImageLabel({value,onChange,number,disabled=false}:{value?:string;onChange:(label:string)=>void;number:number;disabled?:boolean}){
  const label=subjectLabel(value),[custom,setCustom]=useState(false);
  const other=custom||!labels.includes(label);
  return <div className="image-label">
    <select aria-label={`Label for image ${number}`} title="Image label" value={other?'custom':label} disabled={disabled} onChange={e=>{setCustom(e.target.value==='custom');onChange(e.target.value==='custom'?'subject':e.target.value);}}>
      {labels.map(item=><option key={item} value={item}>{item[0].toUpperCase()+item.slice(1)}</option>)}<option value="custom">Other…</option>
    </select>
    {other?<input aria-label={`Custom label for image ${number}`} placeholder="For example: car" maxLength={48} value={value??''} disabled={disabled} onChange={e=>onChange(e.target.value)}/>:null}
  </div>;
}
