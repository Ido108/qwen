import {useRef,useState} from 'react';
import type {ImageTool} from './ImageSettings';
import {InputLibrary,type InputImage,type Person} from './InputLibrary';
import {LibraryPopover} from './LibraryPopover';
import type {Api} from './ModelSettings';

export type SwapState={tool:string;slots:[InputImage|null,InputImage|null];strength:number;extra:string};
export const swapDefaults:SwapState={tool:'head-swap',slots:[null,null],strength:1,extra:''};

// What each picture gives, per swap type (the author's docs: image 1 is the scene, image 2 the head or the person).
const ROLES:Record<string,[string,string]>={
  'head-swap':['Keeps the body, pose, expression, light and background','Gives the head: face, hair, eye color'],
  'head-swap-expression':['Keeps the body, pose, expression, light and background','Gives the head: face, hair, eye color'],
  'body-swap':['Keeps the pose, framing, light and background','Gives the person: face, clothing, body shape. Best full body, facing the camera, plain light background'],
};

type Result={id:string;request:{prompt:string}};

/** Images > Swap: head or body swap with the BFS LoRAs for Qwen-Image-2.1, from two pictures. */
export function SwapPanel({api,tools,value,onChange,disabled,results,people,onPeople,revision}:{api:Api;tools:ImageTool[];value:SwapState;onChange:(next:SwapState)=>void;
  disabled:boolean;results:Result[];people:Person[];onPeople:()=>void;revision:number}){
  const [uploading,setUploading]=useState<number|null>(null);const [error,setError]=useState('');const [picking,setPicking]=useState<number|null>(null);
  const files=[useRef<HTMLInputElement>(null),useRef<HTMLInputElement>(null)];const panel=useRef<HTMLElement>(null);
  const tool=tools.find(t=>t.id===value.tool)??tools[0];
  const setSlot=(index:number,image:InputImage|null)=>{const slots=[...value.slots] as SwapState['slots'];slots[index]=image;onChange({...value,slots});};
  async function upload(index:number,file?:File){if(!file)return;setUploading(index);setError('');
    try{const body=new FormData();body.append('files',file);const [image]=await api('/api/uploads',{method:'POST',body});setSlot(index,image);}
    catch(e){setError((e as Error).message);}finally{setUploading(null);const input=files[index].current;if(input)input.value='';}}
  if(!tool)return null;
  const roles=ROLES[tool.id]??['',''];
  return <section ref={panel} className="panel swap-panel">
    <div className="section-heading"><h2>Swap</h2></div>
    <div className="segmented swap-types" role="radiogroup" aria-label="Swap type">{tools.map(t=><button key={t.id} type="button" role="radio" aria-checked={t.id===tool.id} aria-selected={t.id===tool.id} disabled={disabled} onClick={()=>onChange({...value,tool:t.id})}>{t.title}</button>)}</div>
    <p className="swap-summary">{tool.summary}</p>
    <div className="swap-slots">{[0,1].map(index=>{const image=value.slots[index];return <div key={index} className="swap-slot">
      <div className="swap-slot-head"><b>{index+1}</b><strong>{tool.inputs.labels[index]}</strong></div>
      <small>{roles[index]}</small>
      {image?<div className="swap-image checker"><img src={image.url} alt={tool.inputs.labels[index]}/>
        <button type="button" className="swap-remove" aria-label={`Remove picture ${index+1}`} disabled={disabled} onClick={()=>setSlot(index,null)}>×</button></div>
        :<button type="button" className="swap-empty" disabled={disabled||uploading!==null} onClick={()=>files[index].current?.click()}><span>{uploading===index?'Uploading…':'Upload picture '+(index+1)}</span></button>}
      <div className="swap-slot-actions">{image?<button type="button" className="text-button" disabled={disabled||uploading!==null} onClick={()=>files[index].current?.click()}>{uploading===index?'Uploading…':'Replace'}</button>:null}
        <button type="button" className="text-button" disabled={disabled} onClick={()=>setPicking(index)}>Library or People</button></div>
      <input ref={files[index]} type="file" accept="image/png,image/jpeg,image/webp" hidden onChange={e=>upload(index,e.target.files?.[0])}/>
    </div>;})}</div>
    <LibraryPopover open={picking!==null} onClose={()=>setPicking(null)} anchor={panel}>
      <InputLibrary api={api} inputs={picking!==null&&value.slots[picking]?[value.slots[picking]!]:[]} onChange={next=>{if(picking!==null)setSlot(picking,next[0]??null);setPicking(null);}}
        results={results} people={people} onPeople={onPeople} replace target={0} revision={revision} disabled={disabled} limit={1}
        onUpload={()=>{if(picking!==null)files[picking].current?.click();setPicking(null);}} onClose={()=>setPicking(null)}/>
    </LibraryPopover>
    <div className="field swap-strength"><label htmlFor="swap-strength">Effect strength <b>{value.strength.toFixed(2)}</b></label>
      <input id="swap-strength" type="range" min="0.5" max="1.5" step="0.05" value={value.strength} disabled={disabled} onChange={e=>onChange({...value,strength:Number(e.target.value)})}/>
      <small>1.0 is the author's setting. Lower it when picture 1 changes too much; raise it when the new {tool.id==='body-swap'?'person':'head'} looks too little like picture 2.</small></div>
    <div className="field"><label htmlFor="swap-extra">Extra instructions <span className="muted">optional</span></label>
      <textarea id="swap-extra" dir="auto" rows={2} placeholder="For example: keep the earrings from picture 1" value={value.extra} disabled={disabled} onChange={e=>onChange({...value,extra:e.target.value})}/>
      <small>Added after the swap prompt the LoRA was trained with. The result has the shape of picture 1.</small></div>
    {error?<div className="message error" role="alert"><span>{error}</span></div>:null}
  </section>;
}
