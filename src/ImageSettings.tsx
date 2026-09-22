import {useState} from 'react';
import type {ModelEntry} from './ModelSettings';

export type ImageOptions={custom_size:boolean;auto_resolution:1024|2048;width:number;height:number;steps:number|null};
const presets=[[2688,1536,'16:9'],[1536,2688,'9:16'],[2048,2048,'1:1'],[2368,1728,'4:3'],[1728,2368,'3:4'],[1344,768,'16:9'],[768,1344,'9:16'],[1184,864,'4:3'],[864,1184,'3:4'],[1024,1024,'1:1']] as const;

export function ImageSettings({value,onChange,model,disabled}:{value:ImageOptions;onChange:(values:Partial<ImageOptions>)=>void;model?:ModelEntry;disabled:boolean}){
  const [forceCustom,setForceCustom]=useState(false);
  const defaultSteps=model?.generation_defaults?.steps??40;
  const preset=!forceCustom&&presets.some(([w,h])=>w===value.width&&h===value.height)?`${value.width}x${value.height}`:'custom';
  const validSteps=value.steps===null||(Number.isInteger(value.steps)&&value.steps>=1&&value.steps<=80);
  return <fieldset className="image-generation-settings" disabled={disabled} aria-label="Image generation settings">
    <div className="field"><label htmlFor="image-size-mode">Output size</label>
      <select id="image-size-mode" value={value.custom_size?'custom':String(value.auto_resolution)} onChange={e=>onChange(e.target.value==='custom'?{custom_size:true}:{custom_size:false,auto_resolution:Number(e.target.value) as 1024|2048})}>
        <option value="1024">Auto 1K · Faster</option><option value="2048">Auto 2K · More detail</option><option value="custom">Custom size</option>
      </select>
      <small>{value.custom_size?'Choose exact dimensions below.':`Automatic aspect ratio · About ${value.auto_resolution===1024?'1':'4'} million pixels. Uses the prompt or input image ratio; otherwise square.`}</small>
    </div>
    <div className="field"><div className="steps-heading"><label htmlFor="image-steps">Steps</label>
      {value.steps===null?<span>Model default · {defaultSteps}</span>:<button type="button" className="text-button" onClick={()=>onChange({steps:null})}>Reset to model default ({defaultSteps})</button>}
    </div><input id="image-steps" type="number" min="1" max="80" step="1" value={value.steps??defaultSteps} aria-invalid={!validSteps} aria-describedby="image-steps-help" onChange={e=>onChange({steps:Number(e.target.value)})}/>
      <small id="image-steps-help">{!validSteps?'Enter a whole number from 1 to 80.':'Fewer steps run faster but can reduce detail. The model default follows its published settings.'}</small>
    </div>
    {value.custom_size?<div className="custom-image-settings">
      <div className="field span-2"><label htmlFor="preset">Resolution preset</label><select id="preset" value={preset} onChange={e=>{setForceCustom(e.target.value==='custom');if(e.target.value!=='custom'){const [width,height]=e.target.value.split('x').map(Number);onChange({width,height});}}}><option value="custom">Custom</option>{presets.map(([w,h,r],i)=><option key={`${w}x${h}`} value={`${w}x${h}`}>{i<5?'2K':'1K'} — {w} × {h} ({r})</option>)}</select></div>
      {(['width','height'] as const).map(key=><div className="field" key={key}><label htmlFor={key}>{key==='width'?'Width':'Height'}</label><div className="dimension"><input id={key} type="number" min="256" max="2688" step="8" value={value[key]} onChange={e=>onChange({[key]:Number(e.target.value)})}/><span>px</span></div><input aria-label={`${key==='width'?'Width':'Height'} slider`} type="range" min="256" max="2688" step="8" value={value[key]} onChange={e=>onChange({[key]:Number(e.target.value)})}/></div>)}
      {value.width%32!==0||value.height%32!==0?<small className="span-2">Sizes that are not multiples of 32 are rendered at the next compatible size, then resized to your exact dimensions.</small>:null}
    </div>:null}
  </fieldset>;
}
