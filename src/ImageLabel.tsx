import {useState} from 'react';
import './image-label.css';

const labels=['person','dog','cat','product','object','clothing','logo','style','background'];
export const subjectLabel=(value?:string)=>value?.trim()||'person';
/** A subject label describes what a mention selects. It is not the file or person's name. */
export function ImageLabel({value,onChange,number,disabled=false}:{value?:string;onChange:(label:string)=>void;number:number;disabled?:boolean}){
  const label=subjectLabel(value),[custom,setCustom]=useState(false);
  // A person is "Person", then female or male (saved as woman or man, the words the prompt uses).
  const person=label==='person'||label==='woman'||label==='man';
  const other=custom||(!labels.includes(label)&&!person);
  return <div className="image-label">
    <select aria-label={`Label for image ${number}`} title="Image label" value={other?'custom':person?'person':label} disabled={disabled} onChange={e=>{setCustom(e.target.value==='custom');onChange(e.target.value==='custom'?'subject':e.target.value);}}>
      {labels.map(item=><option key={item} value={item}>{item[0].toUpperCase()+item.slice(1)}</option>)}<option value="custom">Other…</option>
    </select>
    {person&&!other?<div className="image-label-gender" role="radiogroup" aria-label={`Image ${number}: female or male`}>{(['woman','man'] as const).map(g=><button type="button" role="radio" key={g} aria-checked={label===g} disabled={disabled} onClick={()=>onChange(label===g?'person':g)}>{g==='woman'?'Female':'Male'}</button>)}</div>:null}
    {other?<input aria-label={`Custom label for image ${number}`} placeholder="For example: car" maxLength={48} value={value??''} disabled={disabled} onChange={e=>onChange(e.target.value)}/>:null}
  </div>;
}
