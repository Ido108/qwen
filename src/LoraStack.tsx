import {useEffect,useRef,useState} from 'react';
import {Compass,Plus,X} from 'lucide-react';
import type {ModelEntry} from './ModelSettings';
import type {LoraChoice} from './loraChoices';
import {compatibleLoras} from './loraChoices';
import {LoraPromptEditor} from './LoraPromptEditor';
export type {LoraChoice} from './loraChoices';
export {enabledLoras} from './loraChoices';

/** The user's LoRAs (for example from Civitai) on top of the selected model, with a strength each. */
export function LoraStack({value,onChange,loras,family,onBrowse,disabled,compact=false}:{value:LoraChoice[];onChange:(next:LoraChoice[])=>void;loras:ModelEntry[];family:string;
  prompt:string;onPrompt:(next:string)=>void;onBrowse:()=>void;disabled:boolean;compact?:boolean}){
  const [picking,setPicking]=useState(false);const dialog=useRef<HTMLDialogElement>(null);
  useEffect(()=>{if(picking&&!dialog.current?.open)dialog.current?.showModal();else if(!picking)dialog.current?.close();},[picking]);
  const usable=loras.filter(l=>l.installed&&(l.families||[]).includes(family));
  useEffect(()=>{if(!loras.length||!family)return;const next=compatibleLoras(value,loras,family);if(next.length!==value.length)onChange(next);},[value,loras,family,onChange]);
  useEffect(()=>{const saved=(event:Event)=>{const change=(event as CustomEvent).detail;if(value.some(v=>v.id===change.id))onChange(value.map(v=>{if(v.id!==change.id)return v;const {prompt_addition,...choice}=v;return choice;}));};window.addEventListener('qwen-lora-settings',saved);return()=>window.removeEventListener('qwen-lora-settings',saved);},[value,onChange]);
  const active=value.map(choice=>({choice,lora:loras.find(l=>l.id===choice.id)})).filter(x=>x.lora);
  function add(lora:ModelEntry){
    if(value.length>=4||value.some(v=>v.id===lora.id))return;
    onChange([...value,{id:lora.id,strength:lora.default_strength??1}]);
    setPicking(false);
  }
  const setStrength=(id:string,strength:number)=>onChange(value.map(v=>v.id===id?{...v,strength}:v));
  return <div className={`lora-stack lora-controls ${compact?'compact':''}`}>
    <div className="lora-head"><b>LoRAs</b><span>{active.length?`${active.filter(({choice,lora})=>choice.enabled!==false&&lora!.families?.includes(family)).length} on · ${active.length} saved`:compact?'':'Optional'}</span>
      <button type="button" className="lora-add" disabled={disabled||value.length>=4} onClick={()=>setPicking(true)} title={value.length>=4?'Up to 4 LoRAs':'Add a LoRA (up to 4)'}><Plus size={14}/>Add LoRA</button></div>
    {active.map(({choice,lora})=>{const enabled=choice.enabled!==false;const fits=(lora!.families||[]).includes(family);return <div className={`lora-control ${enabled?'':'off'} ${fits?'':'unfit'}`} key={choice.id}>
      <div className="lora-control-heading">
        {lora!.preview?<img src={lora!.preview} alt=""/>:<span className="lora-noimg"/>}
        <strong title={lora!.title}>{lora!.title}</strong>
        <button type="button" className="lora-switch" role="switch" aria-checked={enabled} aria-label={`Use ${lora!.title}`} disabled={disabled}
          onClick={()=>onChange(value.map(v=>v.id===choice.id?{...v,enabled:!enabled}:v))}>
          <span className="lora-switch-track" aria-hidden="true"/><span>{enabled?'On':'Off'}</span>
        </button>
        <button type="button" className="lora-remove" disabled={disabled} aria-label={`Remove ${lora!.title}`} onClick={()=>onChange(value.filter(v=>v.id!==choice.id))}><X size={16}/></button>
      </div>
      {!fits?<p className="lora-compatibility">Not used with this model.</p>:!enabled?<p className="lora-bypass">Off for new jobs. Strength stays saved.</p>:!compact&&lora!.trained_words?.length?<p className="lora-trigger">Required words are added automatically.</p>:null}
      <div className="lora-strength-row"><span>Strength</span>
        <input type="range" min="0" max="2" step="0.05" value={choice.strength} disabled={disabled||!enabled} aria-label={`${lora!.title} strength`}
          onChange={e=>setStrength(choice.id,Number(e.target.value))}/>
        <input type="number" min="0" max="2" step="0.05" value={choice.strength} disabled={disabled||!enabled} aria-label={`${lora!.title} strength value`} title="Strength, 0 to 2"
          onChange={e=>setStrength(choice.id,Math.max(0,Math.min(2,Number(e.target.value)||0)))}/>
      </div>
      <LoraPromptEditor model={lora!}/>
    </div>;})}
    <dialog ref={dialog} className="lora-picker" aria-label="Add a LoRA" onCancel={()=>setPicking(false)} onClick={e=>{if(e.target===dialog.current)setPicking(false);}}>
      <div className="recent-inputs-head"><div><h2>Add a LoRA</h2><p>LoRAs for this model. Required words are added automatically.</p></div><button className="icon-button" onClick={()=>setPicking(false)} aria-label="Close"><X size={20}/></button></div>
      <div className="lora-picker-grid">{(picking?usable:[]).map(lora=><button key={lora.id} className="lora-option" disabled={value.some(v=>v.id===lora.id)} onClick={()=>add(lora)}>
        {lora.preview?<img src={lora.preview} alt=""/>:<span className="lora-noimg"/>}<strong>{lora.title}</strong>
        <small>{lora.trained_words?.length?'Required words are added automatically.':'No required words listed.'}{lora.civitai?.creator?` · ${lora.civitai.creator}`:''}</small></button>)}
        {!usable.length?<p className="recent-empty">You have no LoRAs for this model yet.</p>:null}</div>
      <div className="recent-inputs-actions"><span>Find more LoRAs for your model</span><button className="button primary" onClick={()=>{setPicking(false);onBrowse();}}><Compass size={16}/>Browse Civitai</button></div>
    </dialog>
  </div>;
}
