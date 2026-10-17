import {useState} from 'react';
import type {ModelEntry,ModelState} from './ModelSettings';

export type ImageOptions={custom_size:boolean;auto_resolution:1024|2048;width:number;height:number;steps:number|null;cfg?:number|null};
export type ToolField={key:string;label:string;placeholder?:string;default:string;options?:{value:string;label:string;prompt?:string;strength?:number}[]};
export type ImageTool={id:string;title:string;summary:string;lora:string;strength:number;families:string[];tested_on:string[];swap?:boolean;inputs:{min:number;max:number;labels:string[]};
  fields:ToolField[];prompt:string;reference_resolution:number;prepare?:string;needs_box?:boolean;source:string};
export type ToolState={tool_id:string|null;tool_options:Record<string,string>;lora_strength:number|null};

const presets=[[2688,1536,'16:9'],[1536,2688,'9:16'],[2048,2048,'1:1'],[2368,1728,'4:3'],[1728,2368,'3:4'],[1344,768,'16:9'],[768,1344,'9:16'],[1184,864,'4:3'],[864,1184,'3:4'],[1024,1024,'1:1']] as const;
const familyLabel=(family?:string)=>family==='qwen-image-2.1-turbo'?'Turbo':'Standard';
const gb=(bytes:number)=>`${(bytes/1e9).toFixed(1)} GB`;

/** Fill a tool's prompt template from its options. */
export function toolPrompt(tool:ImageTool,options:Record<string,string>){
  return tool.prompt.replace(/\{(\w+)\}/g,(_,key:string)=>{
    if(key==='scene_suffix')return options.scene?.trim()?` Scene: ${options.scene.trim()}`:'';
    const field=tool.fields.find(f=>f.key===key);const value=options[key]??field?.default??'';
    return field?.options?.find(o=>o.value===value)?.prompt??(value.trim()||field?.default||'');
  });
}
export function toolStrength(tool:ImageTool,options:Record<string,string>){
  for(const field of tool.fields){const option=field.options?.find(o=>o.value===(options[field.key]??field.default));if(option?.strength!==undefined)return option.strength;}
  return tool.strength;
}
export function toolDefaults(tool:ImageTool){return Object.fromEntries(tool.fields.map(f=>[f.key,f.default]));}

export function ImageSettings({part='main',value,onChange,model,models,modelId,onModel,tools,tool,toolState,onTool,disabled}:{value:ImageOptions;onChange:(values:Partial<ImageOptions>)=>void;model?:ModelEntry;
  models:ModelState|null;modelId:string;onModel:(id:string)=>void;tools:ImageTool[];tool?:ImageTool;toolState:ToolState;onTool:(state:ToolState)=>void;disabled:boolean;part?:'model'|'main'|'advanced'}){
  const defaults=model?.generation_defaults;const defaultSteps=defaults?.steps;const turbo=!!defaults?.sigmas;
  const page=typeof defaults?.note==='string'&&defaults.note.startsWith('From the Civitai page')?defaults:null;
  const pageText=page?`Author settings: ${page.steps} steps${page.cfg?`, CFG ${page.cfg}`:''}${page.sampler?`, ${page.sampler}`:''}${page.scheduler?` / ${page.scheduler}`:''}.`:'';
  const validSteps=value.steps===null||(Number.isInteger(value.steps)&&value.steps>=1&&value.steps<=80);
  const defaultCfg=defaults?.cfg??1;const validCfg=value.cfg==null||(value.cfg>=1&&value.cfg<=15);
  const family=model?.family||'qwen-image-2.1';
  const imageModels=(models?.models||[]).filter(m=>m.kind==='image');
  const groups=[['qwen-image-2.1','Qwen 2.1 Standard · 25 steps'],['qwen-image-2.1-turbo','Qwen 2.1 Turbo · 8 steps, fast'],['krea-2-turbo','Krea 2 Turbo · 8 steps, fast'],['krea-2','Krea 2 Raw · 52 steps'],['z-image-turbo','Z-Image Turbo · 8 steps, fast'],['z-image','Z-Image Base · 30 steps']] as const;
  const order=['Q8_0','Q6_K','Q5_K_M','Q4_K_M','Q4_0','FP8','NVFP4','INT8','BF16'];
  const quant=(m:ModelEntry)=>m.title.split(' · ').at(-1)||m.title;
  const isPreset=presets.some(([w,h])=>w===value.width&&h===value.height);
  const [customMode,setCustomMode]=useState(()=>value.custom_size&&!isPreset);
  const sizeValue=value.custom_size?(customMode||!isPreset?'custom':`${value.width}x${value.height}`):`auto-${value.auto_resolution}`;
  const followsImage=tool?.reference_resolution===0;
  function setSize(choice:string){
    setCustomMode(choice==='custom');
    if(choice.startsWith('auto-'))onChange({custom_size:false,auto_resolution:Number(choice.slice(5)) as 1024|2048});
    else if(choice==='custom')onChange({custom_size:true});
    else{const [width,height]=choice.split('x').map(Number);onChange({custom_size:true,width,height});}
  }
  function chooseTool(id:string){
    const next=tools.find(t=>t.id===id);
    onTool(next?{tool_id:next.id,tool_options:toolDefaults(next),lora_strength:null}:{tool_id:null,tool_options:{},lora_strength:null});
  }
  const options=toolState.tool_options;
  const strength=toolState.lora_strength??(tool?toolStrength(tool,options):1);
  const hintSteps=defaults?.steps??(family.endsWith('turbo')?8:family==='krea-2'?52:family==='z-image'?30:25);const hintCfg:number|null=defaults?.cfg??(family==='krea-2'?4.5:family==='z-image'?4:family.endsWith('turbo')?1:null);
  const hintSampler=defaults?.sampler&&defaults.sampler!=='euler'?defaults.sampler.replace(/_/g,' '):'';const hintScheduler=defaults?.scheduler&&defaults.scheduler!=='simple'?defaults.scheduler.replace(/_/g,' '):'';
  // CFG 1 runs one pass without the negative branch, so a negative prompt changes nothing.
  const settingsHint=`${family==='qwen-image-2.1-turbo'?'Turbo':family==='krea-2-turbo'?'Krea 2 Turbo':family==='krea-2'?'Krea 2 Raw':family==='z-image-turbo'?'Z-Image Turbo':family==='z-image'?'Z-Image Base':'Standard'}, ${hintSteps} steps, ${hintCfg===null?'CFG 2 with a negative prompt, else 1':`CFG ${hintCfg}`}${hintSampler?`, ${hintSampler}`:''}${hintScheduler?` / ${hintScheduler}`:''}${hintCfg!==null&&hintCfg<=1?' (negative prompt has no effect at CFG 1)':''}`;
  const status=[model?.id==='gguf-q8_0'||model?.id==='turbo-gguf-q8_0'?'★ Recommended':'',model?.ready?'✓ Installed':model?'Downloads when you generate':'',
    settingsHint,pageText?'Author settings':''].filter(Boolean).join(' · ');
  const cls=`image-generation-settings studio-settings settings-${part}`;
  if(part==='model')return <fieldset className={cls} disabled={disabled} aria-label="Model">
    <div className="field model-field"><label htmlFor="image-model">Model</label>
      <select id="image-model" value={modelId} onChange={e=>onModel(e.target.value)} title={model?.title}>
        {groups.map(([id,label])=><optgroup key={id} label={label}>{imageModels.filter(m=>(m.family||'qwen-image-2.1')===id).sort((a,b)=>order.indexOf(quant(a))-order.indexOf(quant(b))).map(m=>
          <option key={m.id} value={m.id}>{m.title}{m.id==='gguf-q8_0'||m.id==='turbo-gguf-q8_0'?' ★':''}{m.ready?' ✓':` · ↓ ${gb(m.bytes)}`}</option>)}</optgroup>)}
      </select>
      <small className="model-status" title={[pageText,defaults?.note].filter(Boolean).join(' ')||undefined}>{status}</small>
    </div>
  </fieldset>;
  if(part==='advanced')return <fieldset className={cls} disabled={disabled} aria-label="Steps and CFG">
    <div className="field"><div className="steps-heading"><label htmlFor="image-steps">Steps</label>
      {value.steps===null?(defaults?.source?<a href={defaults.source} target="_blank" rel="noreferrer" title={`Default: ${defaultSteps} steps`}>Model default</a>:<span>Model default</span>):<button type="button" className="text-button" disabled={!defaultSteps} onClick={()=>onChange({steps:null})}>Reset ({defaultSteps??'…'})</button>}
    </div><input id="image-steps" type="number" min="1" max="80" step="1" disabled={!defaultSteps} value={value.steps??defaultSteps??''} aria-invalid={!validSteps} title={turbo?`Turbo is trained for ${defaultSteps} steps; other counts stretch its schedule.`:`Default: ${defaultSteps??'…'} steps`} onChange={e=>onChange({steps:Number(e.target.value)})}/>
      {!validSteps?<small>Enter a whole number from 1 to 80.</small>:turbo&&value.steps!==null&&value.steps!==defaultSteps?<small>Turbo is trained for {defaultSteps}.</small>:null}
    </div>
    <div className="field"><div className="steps-heading"><label htmlFor="image-cfg">CFG</label>
      {value.cfg==null?<span>Model default</span>:<button type="button" className="text-button" onClick={()=>onChange({cfg:null})}>Reset ({defaultCfg})</button>}
    </div><input id="image-cfg" type="number" min="1" max="15" step="0.5" disabled={!defaults} value={value.cfg??defaultCfg} aria-invalid={!validCfg} title={`${defaults?.cfg?'Default from the model page. ':turbo?'Turbo is trained for CFG 1. ':'1, or 2 with a negative prompt. '}Above 1 runs a second guidance pass, so each step takes about twice as long.`} onChange={e=>onChange({cfg:e.target.value===''?null:Number(e.target.value)})}/>
      {!validCfg?<small>Enter a number from 1 to 15.</small>:value.cfg!=null&&value.cfg>1?<small>About 2x slower per step.</small>:null}
    </div>
  </fieldset>;
  return <fieldset className={cls} disabled={disabled} aria-label="Tool and output size">
    <div className="field"><label htmlFor="image-tool">Tool</label>
      <select id="image-tool" value={toolState.tool_id||''} onChange={e=>chooseTool(e.target.value)} title={tool?tool.summary:'Optional add-ons for try-on, restoration, expansion and more.'}>
        <option value="">None · free prompt</option>
        {tools.map(t=>{const ok=t.families.includes(family);return <option key={t.id} value={t.id} disabled={!ok}>{t.title}{ok?'':` · ${familyLabel(t.families[0])} model only`}</option>;})}
      </select>
    </div>
    <div className="field"><label htmlFor="image-size">Output size</label>
      <select id="image-size" value={sizeValue} onChange={e=>setSize(e.target.value)} title={followsImage?'Follows the shape of image 1.':value.custom_size?'Exact size in pixels.':'Shape follows the prompt or input image.'}>
        <option value="auto-1024">Auto · 1K (about 1 MP)</option><option value="auto-2048">Auto · 2K (about {followsImage?'2':'4'} MP)</option>
        {followsImage?null:<><option disabled>──────────</option>{presets.map(([w,h,r])=><option key={`${w}x${h}`} value={`${w}x${h}`}>{w} × {h} · {r}</option>)}<option disabled>──────────</option><option value="custom">Custom…</option></>}
      </select>
    </div>
    {tool?<small className="tool-summary">{tool.summary}</small>:null}
    {sizeValue==='custom'&&!followsImage?<div className="custom-image-settings">
      {(['width','height'] as const).map(key=><div className="field" key={key}><label htmlFor={key}>{key==='width'?'Width':'Height'}</label><div className="dimension"><input id={key} type="number" min="256" max="2688" step="8" value={value[key]} onChange={e=>onChange({[key]:Number(e.target.value)})}/><span>px</span></div><input aria-label={`${key==='width'?'Width':'Height'} slider`} type="range" min="256" max="2688" step="8" value={value[key]} onChange={e=>onChange({[key]:Number(e.target.value)})}/></div>)}
      {value.width%32!==0||value.height%32!==0?<small className="span-2">Sizes that are not multiples of 32 are rendered at the next compatible size, then resized to your exact dimensions.</small>:null}
    </div>:null}
    {tool?<div className="tool-panel">
      <div className="tool-inputs">{tool.inputs.labels.map((label,i)=><span key={label}><b>{i+1}</b>{label}</span>)}{!tool.tested_on.includes(family)?<em>Not tested on {familyLabel(family)} by its author</em>:null}</div>
      {tool.fields.map(field=><div className="field" key={field.key}><label htmlFor={`tool-${field.key}`}>{field.label}</label>
        {field.options?<select id={`tool-${field.key}`} value={options[field.key]??field.default} onChange={e=>onTool({...toolState,tool_options:{...options,[field.key]:e.target.value},lora_strength:null})}>{field.options.map(o=><option key={o.value} value={o.value}>{o.label}</option>)}</select>
          :<input id={`tool-${field.key}`} value={options[field.key]??''} placeholder={field.placeholder} onChange={e=>onTool({...toolState,tool_options:{...options,[field.key]:e.target.value}})}/>}
      </div>)}
      <div className="field"><label htmlFor="tool-strength">Effect strength <span className="tool-strength-value">{strength.toFixed(2)}</span></label>
        <input id="tool-strength" type="range" min="0.2" max="1.5" step="0.05" value={strength} onChange={e=>onTool({...toolState,lora_strength:Number(e.target.value)})}/></div>
      <a className="tool-source" href={tool.source} target="_blank" rel="noreferrer">About this tool</a>
    </div>:null}
  </fieldset>;
}
