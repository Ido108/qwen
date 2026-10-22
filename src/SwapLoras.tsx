import {Nudge,ResetButton} from './SettingLabel';
import type {JobPlan} from './ImageSettings';

type Strength={label:string;value:number;min:number;max:number;fallback:number;hint?:string;onChange:(value:number|null)=>void};

/** The LoRAs the app adds by itself, listed under the user's LoRAs, each with its own control: a tool's LoRA with its
 *  strength, a tool author's speed LoRA with a switch and strength, and a model's own LoRA (Krea 2 identity or style
 *  with input photos, a Turbo model's speed LoRA). */
export function SwapLoras({plan,strength,speedStrength,disabled,onSpeed}:{plan:JobPlan;strength?:Strength;speedStrength:number;disabled:boolean;onSpeed:(on:boolean,strength:number)=>void}){
  const tool=plan.loras.find(l=>l.source==='tool');const own=plan.loras.filter(l=>l.source==='model');const speed=plan.loras.find(l=>l.source==='speed');
  const speedOn=!!plan.speed_lora?.on;
  if(!tool&&!own.length&&!plan.speed_lora)return null;
  return <div className="auto-loras" aria-label="LoRAs added automatically">
    {tool?<div className="auto-lora"><div className="auto-lora-head"><strong title={tool.title}>{tool.title}</strong><span className="auto-lora-tag">Tool</span></div>
      {strength?<div className="auto-lora-strength" title={strength.hint}><span>{strength.label} <b>{strength.value.toFixed(2)}</b></span>
        <Nudge label={strength.label} value={strength.value} step={0.05} min={strength.min} max={strength.max} disabled={disabled} onChange={value=>strength.onChange(value)}/>
        <input type="range" min={strength.min} max={strength.max} step="0.05" value={strength.value} disabled={disabled} aria-label={strength.label} onChange={e=>strength.onChange(Number(e.target.value))}/>
        {/* Set default saves your value for this model; Reset returns to it (or to the factory value). */}
        <ResetButton label="Effect strength" value={strength.fallback} disabled={disabled} onReset={()=>strength.onChange(null)}/></div>:null}
      {strength?.hint?<small className="auto-lora-hint">{strength.hint}</small>:null}</div>:null}
    {plan.speed_lora?<div className={`auto-lora ${speedOn?'':'off'}`}><div className="auto-lora-head"><strong title={`${speed?.title??'Pruna 8-step LoRA'}: Qwen Standard in 8 steps instead of 40, as in the swap author's setup`}>{speed?.title??'Pruna 8-step LoRA'}</strong><span className="auto-lora-tag">8 steps, not 40</span>
      <button type="button" className="lora-switch" role="switch" aria-checked={speedOn} aria-label="Use the speed LoRA" disabled={disabled} title="On: 8 steps, CFG 1, deis 2m (the swap author's setup). Off: the model's own 40 steps."
        onClick={()=>onSpeed(!speedOn,speedStrength)}><span className="lora-switch-track" aria-hidden="true"/><span>{speedOn?'On':'Off'}</span></button></div>
      {speedOn?<div className="auto-lora-strength"><span>Strength <b>{speedStrength.toFixed(2)}</b></span>
        <Nudge label="speed LoRA strength" value={speedStrength} step={0.05} min={0} max={2} disabled={disabled} onChange={value=>onSpeed(true,value)}/>
        <input type="range" min="0" max="2" step="0.05" value={speedStrength} disabled={disabled} aria-label="Speed LoRA strength" onChange={e=>onSpeed(true,Number(e.target.value))}/></div>:null}</div>:null}
    {own.map(l=><div className="auto-lora" key={l.id}><div className="auto-lora-head"><strong title={l.title}>{l.title}</strong><span className="auto-lora-tag">{l.id.startsWith('krea2-lora')?'For input photos':'Part of this model'}</span></div></div>)}
  </div>;
}
