import {useEffect,useRef,useState} from 'react';
import {Compass,Plus,X} from 'lucide-react';
import type {ModelEntry} from './ModelSettings';

export type LoraChoice={id:string;strength:number};

/** The user's LoRAs (for example from Civitai) on top of the selected model, with a strength each. */
export function LoraStack({value,onChange,loras,family,prompt,onPrompt,onBrowse,disabled,compact=false}:{value:LoraChoice[];onChange:(next:LoraChoice[])=>void;loras:ModelEntry[];family:string;
  prompt:string;onPrompt:(next:string)=>void;onBrowse:()=>void;disabled:boolean;compact?:boolean}){
  const [picking,setPicking]=useState(false);const dialog=useRef<HTMLDialogElement>(null);
  useEffect(()=>{if(picking&&!dialog.current?.open)dialog.current?.showModal();else if(!picking)dialog.current?.close();},[picking]);
  const usable=loras.filter(l=>l.installed&&(l.families||[]).includes(family));
  const active=value.map(choice=>({choice,lora:loras.find(l=>l.id===choice.id)})).filter(x=>x.lora);
  function add(lora:ModelEntry){
    if(value.length>=4||value.some(v=>v.id===lora.id))return;
    onChange([...value,{id:lora.id,strength:lora.default_strength??1}]);
    const words=(lora.trained_words||[]).filter(w=>!prompt.toLowerCase().includes(w.toLowerCase()));
    if(words.length)onPrompt(`${words.join(', ')}${prompt.trim()?', '+prompt.trim():''}`);
    setPicking(false);
  }
  const setStrength=(id:string,strength:number)=>onChange(value.map(v=>v.id===id?{...v,strength}:v));
  return <div className={`lora-stack ${compact?'compact':''}`}>
    {compact?<>
    <span className="lora-label">LoRAs</span>
    {active.map(({choice,lora})=>{const fits=(lora!.families||[]).includes(family);return <div className={`lora-pill ${fits?'':'unfit'}`} key={choice.id}
      title={!fits?'Not for this model. Remove it or change the model.':lora!.trained_words?.length?`${lora!.title} · Trigger: ${lora!.trained_words.join(', ')}`:lora!.title}>
      {lora!.preview?<img src={lora!.preview} alt=""/>:<span className="lora-noimg"/>}
      <strong>{lora!.title}</strong>
      <input type="number" min="0" max="1.5" step="0.05" value={choice.strength} disabled={disabled} aria-label={`${lora!.title} strength`} title="Strength, 0 to 1.5"
        onChange={e=>setStrength(choice.id,Math.max(0,Math.min(1.5,Number(e.target.value)||0)))}/>
      <button type="button" className="lora-remove" disabled={disabled} aria-label={`Remove ${lora!.title}`} onClick={()=>onChange(value.filter(v=>v.id!==choice.id))}><X size={12}/></button>
    </div>;})}
    <button type="button" className="lora-add" disabled={disabled||value.length>=4} onClick={()=>setPicking(true)} title={value.length>=4?'Up to 4 LoRAs':'Add a LoRA (up to 4)'}><Plus size={13}/>{active.length?'Add':'Add LoRA'}</button>
    </>:<>
    <div className="lora-head"><b>LoRAs</b><span>{active.length?`${active.length} of 4`:'Optional'}</span>
      <button type="button" className="text-button" disabled={disabled||value.length>=4} onClick={()=>setPicking(true)}><Plus size={14}/>Add LoRA</button></div>
    {active.map(({choice,lora})=><div className="lora-chip" key={choice.id}>
      {lora!.preview?<img src={lora!.preview} alt=""/>:<span className="lora-noimg"/>}
      <div className="lora-copy"><strong title={lora!.title}>{lora!.title}</strong>
        {!(lora!.families||[]).includes(family)?<em>Not for this model. Remove it or change the model.</em>:lora!.trained_words?.length?<small>Trigger: {lora!.trained_words.join(', ')}</small>:null}
        <input type="range" min="0" max="1.5" step="0.05" value={choice.strength} disabled={disabled} aria-label={`${lora!.title} strength`}
          onChange={e=>onChange(value.map(v=>v.id===choice.id?{...v,strength:Number(e.target.value)}:v))}/></div>
      <span className="lora-value">{choice.strength.toFixed(2)}</span>
      <button type="button" className="icon-button" disabled={disabled} aria-label={`Remove ${lora!.title}`} onClick={()=>onChange(value.filter(v=>v.id!==choice.id))}><X size={15}/></button>
    </div>)}
    </>}
    <dialog ref={dialog} className="lora-picker" aria-label="Add a LoRA" onCancel={()=>setPicking(false)} onClick={e=>{if(e.target===dialog.current)setPicking(false);}}>
      <div className="recent-inputs-head"><div><h2>Add a LoRA</h2><p>LoRAs that work with the selected model. Trigger words go into your prompt.</p></div><button className="icon-button" onClick={()=>setPicking(false)} aria-label="Close"><X size={20}/></button></div>
      <div className="lora-picker-grid">{usable.map(lora=><button key={lora.id} className="lora-option" disabled={value.some(v=>v.id===lora.id)} onClick={()=>add(lora)}>
        {lora.preview?<img src={lora.preview} alt=""/>:<span className="lora-noimg"/>}<strong>{lora.title}</strong>
        <small>{lora.trained_words?.length?`Trigger: ${lora.trained_words.join(', ')}`:'No trigger words'}{lora.civitai?.creator?` · ${lora.civitai.creator}`:''}</small></button>)}
        {!usable.length?<p className="recent-empty">You have no LoRAs for this model yet.</p>:null}</div>
      <div className="recent-inputs-actions"><span>Find more LoRAs for Qwen-Image-2.1</span><button className="button primary" onClick={()=>{setPicking(false);onBrowse();}}><Compass size={16}/>Browse Civitai</button></div>
    </dialog>
  </div>;
}
