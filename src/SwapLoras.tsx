import type {JobPlan} from './ImageSettings';

/** The LoRAs the app adds by itself, listed under the user's LoRAs: a tool's LoRA (its strength has its own control),
 *  a model's own LoRA (Krea 2 identity or style with input photos, a Turbo model's speed LoRA) and a tool author's
 *  speed LoRA with its own switch and strength. */
export function SwapLoras({plan,strengthNote,speedStrength,disabled,onSpeed}:{plan:JobPlan;strengthNote:string;speedStrength:number;disabled:boolean;onSpeed:(on:boolean,strength:number)=>void}){
  const added=plan.loras.filter(l=>l.source==='tool'||l.source==='model');const speed=plan.loras.find(l=>l.source==='speed');
  const speedOn=!!plan.speed_lora?.on;
  if(!added.length&&!plan.speed_lora)return null;
  const note=(l:JobPlan['loras'][number])=>l.source==='tool'?strengthNote
    :l.id==='krea2-lora-identity-edit'?'Added for the person photo. Its effect: Identity strength'
    :l.id==='krea2-lora-style-reference'?'Added for the style photos, strength 1'
    :`Part of this Turbo model, strength ${l.strength}`;
  return <div className="lora-stack lora-controls compact swap-loras" aria-label="LoRAs added automatically">
    {added.map(l=><div className="lora-control swap-lora" key={l.id}><div className="lora-control-heading"><strong>{l.title}</strong><span className="swap-lora-tag">{l.source==='tool'?'Tool':'Model'}</span></div>
      <div className="lora-strength-row"><span>{note(l)}</span></div></div>)}
    {plan.speed_lora?<div className={`lora-control swap-lora ${speedOn?'':'off'}`}><div className="lora-control-heading"><strong>{speed?.title??'Pruna 8-step LoRA'}</strong><span className="swap-lora-tag">Speed</span>
      <button type="button" className="lora-switch" role="switch" aria-checked={speedOn} aria-label="Use the speed LoRA" disabled={disabled} title="The tool author's 8-step setup: 8 steps, CFG 1, deis 2m. Off: the model's own steps and CFG."
        onClick={()=>onSpeed(!speedOn,speedStrength)}><span className="lora-switch-track" aria-hidden="true"/><span>{speedOn?'On':'Off'}</span></button></div>
      <p className="swap-lora-note">Speed LoRA: Qwen Standard makes the picture in 8 steps instead of 40 (about 5x faster), as in the swap author's setup. Off: the model's own 40 steps.</p>
      <div className="lora-strength-row"><span>Strength</span>
        <input type="range" min="0" max="2" step="0.05" value={speedStrength} disabled={disabled||!speedOn} aria-label="Speed LoRA strength" onChange={e=>onSpeed(true,Number(e.target.value))}/>
        <input type="number" min="0" max="2" step="0.05" value={speedStrength} disabled={disabled||!speedOn} aria-label="Speed LoRA strength value" title="Strength, 0 to 2. The author uses 1." onChange={e=>onSpeed(true,Math.max(0,Math.min(2,Number(e.target.value)||0)))}/>
      </div></div>:null}
  </div>;
}
