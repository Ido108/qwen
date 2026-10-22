import {useState} from 'react';
import './image-label.css';

const labels=['person','dog','cat','product','object','clothing','logo','style','background'];
/** No label is allowed: a mention of an unlabelled picture is just that picture. */
export const subjectLabel=(value?:string)=>value?.trim()||'';
/** A subject label describes what a mention selects. It is not the file or person's name. */
/** person: a People photo; it shows Person with Female / Male. Any other picture has no label unless one is chosen. */
export function ImageLabel({value,onChange,number,disabled=false,person=false}:{value?:string;onChange:(label:string)=>void;number:number;disabled?:boolean;person?:boolean}){
  const label=subjectLabel(value)||(person?'person':''),[custom,setCustom]=useState(false);
  // A person is "Person", then female or male (saved as woman or man, the words the prompt uses).
  const isPerson=label==='person'||label==='woman'||label==='man';
  const other=custom||(!!label&&!labels.includes(label)&&!isPerson);
  return <div className="image-label">
    <select aria-label={`Label for image ${number}`} title="Image label" value={other?'custom':isPerson?'person':label} disabled={disabled} onChange={e=>{setCustom(e.target.value==='custom');onChange(e.target.value==='custom'?'':e.target.value);}}>
      <option value="">No label</option>{labels.map(item=><option key={item} value={item}>{item[0].toUpperCase()+item.slice(1)}</option>)}<option value="custom">Other…</option>
    </select>
    {isPerson&&!other?<div className="image-label-gender" role="radiogroup" aria-label={`Image ${number}: female or male`}>{(['woman','man'] as const).map(g=><button type="button" role="radio" key={g} aria-checked={label===g} disabled={disabled} onClick={()=>onChange(label===g?'person':g)}>{g==='woman'?'Female':'Male'}</button>)}</div>:null}
    {other?<input aria-label={`Custom label for image ${number}`} placeholder="For example: car" maxLength={48} value={value??''} disabled={disabled} onChange={e=>onChange(e.target.value)}/>:null}
  </div>;
}
