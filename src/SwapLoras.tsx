import type {JobPlan} from './ImageSettings';

/** The LoRAs Swap adds by itself, shown with the user's LoRAs: the swap LoRA (its strength is "Keep original face")
 *  and the author's speed LoRA with its own switch and strength. */
export function SwapLoras({plan,strength,speedStrength,disabled,onSpeed}:{plan:JobPlan;strength:number;speedStrength:number;disabled:boolean;onSpeed:(on:boolean,strength:number)=>void}){
  const tool=plan.loras.find(l=>l.source==='tool');const speed=plan.loras.find(l=>l.source==='speed');
  const speedOn=!!plan.speed_lora?.on;
  return <div className="lora-stack lora-controls compact swap-loras" aria-label="LoRAs added by Swap">
    {tool?<div className="lora-control swap-lora"><div className="lora-control-heading"><strong>{tool.title}</strong><span className="swap-lora-tag">Swap</span></div>
      <div className="lora-strength-row"><span>Strength {strength.toFixed(2)}: set with Keep original face below</span></div></div>:null}
    {plan.speed_lora?<div className={`lora-control swap-lora ${speedOn?'':'off'}`}><div className="lora-control-heading"><strong>{speed?.title??'Pruna 8-step LoRA'}</strong><span className="swap-lora-tag">Speed</span>
      <button type="button" className="lora-switch" role="switch" aria-checked={speedOn} aria-label="Use the speed LoRA" disabled={disabled} title="The swap author's 8-step setup: 8 steps, CFG 1, deis 2m. Off: the model's own steps and CFG."
        onClick={()=>onSpeed(!speedOn,speedStrength)}><span className="lora-switch-track" aria-hidden="true"/><span>{speedOn?'On':'Off'}</span></button></div>
      <div className="lora-strength-row"><span>Strength</span>
        <input type="range" min="0" max="2" step="0.05" value={speedStrength} disabled={disabled||!speedOn} aria-label="Speed LoRA strength" onChange={e=>onSpeed(true,Number(e.target.value))}/>
        <input type="number" min="0" max="2" step="0.05" value={speedStrength} disabled={disabled||!speedOn} aria-label="Speed LoRA strength value" title="Strength, 0 to 2. The author uses 1." onChange={e=>onSpeed(true,Math.max(0,Math.min(2,Number(e.target.value)||0)))}/>
      </div></div>:null}
  </div>;
}
