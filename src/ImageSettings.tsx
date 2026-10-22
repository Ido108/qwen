import {useState} from 'react';
import {Nudge,SettingLabel} from './SettingLabel';
import type {ModelEntry,ModelState} from './ModelSettings';

export type ImageOptions={negative_prompt?:string;custom_size:boolean;auto_resolution:1024|2048;width:number;height:number;steps:number|null;cfg?:number|null;sampler?:string|null;scheduler?:string|null};
export type ToolField={key:string;label:string;placeholder?:string;default:string;options?:{value:string;label:string;prompt?:string;strength?:number}[]};
export type ImageTool={id:string;title:string;summary:string;lora:string;strength:number;steps?:number;families:string[];tested_on:string[];swap?:boolean;swap_page?:boolean;sampling?:{steps?:number;sampler?:string;scheduler?:string;cfg?:number;speed_lora?:string;families?:string[]};inputs:{min:number;max:number;labels:string[]};
  fields:ToolField[];prompt:string;reference_resolution:number;prepare?:string;needs_box?:boolean;source:string};
/** What a job will really run (POST /api/job-plan): the server fills the job from the same plan. */
export type JobPlan={steps:number;cfg:number;sampler:string;scheduler:string;loras:{id:string;title:string;source:'tool'|'speed'|'model'|'yours';strength:number}[];speed_lora:{id:string;on:boolean}|null;recipe:boolean;speed_note?:string};
export type ToolState={tool_id:string|null;tool_options:Record<string,string>;lora_strength:number|null};

const presets=[[2688,1536,'16:9'],[1536,2688,'9:16'],[2048,2048,'1:1'],[2368,1728,'4:3'],[1728,2368,'3:4'],[1344,768,'16:9'],[768,1344,'9:16'],[1184,864,'4:3'],[864,1184,'3:4'],[1024,1024,'1:1']] as const;
/** The dropdown group: a Turbo entry built from a speed LoRA on a Base family (MOSS Serenity Turbo: the 8-step DMD
 *  LoRA) is listed with that architecture's Turbo models, never under a Base or Standard heading. */
const TURBO_GROUP:Record<string,string>={'moss-image-base':'moss-image-turbo','moss-image':'moss-image-turbo','qwen-image-2.1':'qwen-image-2.1-turbo','krea-2':'krea-2-turbo','z-image':'z-image-turbo'};
export function modelGroup(m:{family?:string;generation_defaults?:{speed_lora?:{id:string}}|null}){const family=m.family||'qwen-image-2.1';return m.generation_defaults?.speed_lora?TURBO_GROUP[family]||family:family;}
const familyLabel=(family?:string)=>family==='qwen-image-2.1-turbo'?'Turbo':family==='krea-2-turbo'?'Krea 2 Turbo':family==='krea-2'?'Krea 2 Raw':'Standard';
/** Why a tool is unavailable for the chosen model: Qwen-Image-2.1 tools on another architecture name the architecture. */
const toolOnly=(families:string[],family:string)=>families.every(f=>f.startsWith('qwen-image'))&&!family.startsWith('qwen-image')?'Qwen-Image-2.1 model only':`${familyLabel(families[0])} model only`;
const gb=(bytes:number)=>`${(bytes/1e9).toFixed(1)} GB`;
// Uncensored Q8_0 GGUF files, and the official INT8 files: native INT8 matmul, the fastest Qwen-Image-2.1 on RTX 40 and 50 GPUs.
const RECOMMENDED=new Set(['uc-int8-convrot','int8-convrot','turbo-int8-convrot']);

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

export function ImageSettings({part='main',idPrefix='image',value,onChange,model,models,modelId,onModel,tools,tool,toolState,onTool,disabled,plan,swapFamilies,planLabel='Swap'}:{plan?:JobPlan|null;swapFamilies?:string[];planLabel?:string;value:ImageOptions;onChange:(values:Partial<ImageOptions>)=>void;model?:ModelEntry;
  models:ModelState|null;modelId:string;onModel:(id:string)=>void;tools:ImageTool[];tool?:ImageTool;toolState:ToolState;onTool:(state:ToolState)=>void;disabled:boolean;part?:'model'|'main'|'steps'|'advanced';idPrefix?:string}){
  // A tool's own steps, sampler, schedule and CFG (its author's workflow) win over the model's, as on the server.
  const recipe=tool?.sampling&&(!tool.sampling.families||tool.sampling.families.includes(model?.family||''))?tool.sampling:null;
  const modelDefaults=model?.generation_defaults;const defaults=modelDefaults&&recipe?{...modelDefaults,...Object.fromEntries(Object.entries(recipe).filter(([key])=>key!=='families'&&key!=='speed_lora'))}:modelDefaults;
  // With a plan (Swap), every default shown is the one the job will run.
  const defaultSteps=plan?.steps??tool?.steps??defaults?.steps;
  const turbo=plan?plan.scheduler==='trained':!!defaults?.sigmas;
  const planSampler=plan?.sampler??defaults?.sampler??'euler';const planSchedule=plan?(plan.scheduler==='trained'?'trained schedule':plan.scheduler):turbo?'trained schedule':defaults?.scheduler||'simple';
  const page=typeof defaults?.note==='string'&&defaults.note.startsWith('From the Civitai page')?defaults:null;
  const pageText=page?`Author settings: ${page.steps} steps${page.cfg?`, CFG ${page.cfg}`:''}${page.sampler?`, ${page.sampler}`:''}${page.scheduler?` / ${page.scheduler}`:''}.`:'';
  const validSteps=value.steps===null||(Number.isInteger(value.steps)&&value.steps>=1&&value.steps<=80);
  const defaultCfg=plan?.cfg??defaults?.cfg??(value.negative_prompt?.trim()?2:1);const validCfg=value.cfg==null||(value.cfg>=1&&value.cfg<=15);
  const family=model?.family||'qwen-image-2.1';
  // Swap lists only the models a swap runs on.
  const imageModels=(models?.models||[]).filter(m=>m.kind==='image'&&(!swapFamilies||swapFamilies.includes(m.family||'qwen-image-2.1')));
  const groups=[['flux3-klein','FLUX.3 Image Klein'],['flux3-klein-base','FLUX.3 Image Klein Base'],['qwen-image-2.1','Qwen 2.1 Standard'],['qwen-image-2.1-turbo','Qwen 2.1 Turbo (fast)'],['krea-2-turbo','Krea 2 Turbo (fast)'],['krea-2','Krea 2 Raw'],['z-image-turbo','Z-Image Turbo (fast)'],['z-image','Z-Image Base'],['moss-image-turbo','MOSS Image Turbo (fast)'],['moss-image','MOSS Image SFT'],['moss-image-base','MOSS Image Base']] as const;
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
  const hintSteps=defaults?.steps??(family.endsWith('turbo')?8:family==='krea-2'?52:family==='z-image'?30:family==='flux3-klein'?16:family==='flux3-klein-base'||family==='moss-image'||family==='moss-image-base'?50:40);const hintCfg:number|null=defaults?.cfg??(family==='krea-2'?4.5:family==='moss-image'||family==='moss-image-base'?3:family==='z-image'?4:family==='flux3-klein-base'?4:family.endsWith('turbo')||family==='flux3-klein'?1:null);
  const hintSampler=defaults?.sampler&&defaults.sampler!=='euler'?defaults.sampler.replace(/_/g,' '):'';const hintScheduler=defaults?.scheduler&&defaults.scheduler!=='simple'?defaults.scheduler.replace(/_/g,' '):'';
  // CFG 1 runs one pass without the negative branch, so a negative prompt changes nothing.
  const settingsHint=`${family.startsWith('flux3-klein')?(family.endsWith('base')?'FLUX.3 Klein Base':'FLUX.3 Klein'):family==='qwen-image-2.1-turbo'?'Turbo':family==='krea-2-turbo'?'Krea 2 Turbo':family==='krea-2'?'Krea 2 Raw':family==='z-image-turbo'?'Z-Image Turbo':family==='z-image'?'Z-Image Base':family==='moss-image-turbo'?'MOSS Image Turbo':family==='moss-image'?'MOSS Image SFT':family==='moss-image-base'?(defaults?.speed_lora?'MOSS Image Base Turbo (8-step DMD LoRA)':'MOSS Image Base'):'Standard'}, ${hintSteps} steps, ${hintCfg===null?'CFG 2 with a negative prompt, else 1':`CFG ${hintCfg}`}${hintSampler?`, ${hintSampler}`:''}${hintScheduler?` / ${hintScheduler}`:''}${hintCfg!==null&&hintCfg<=1?' (negative prompt has no effect at CFG 1)':''}`;
  // On Swap the line shows what the swap runs (your values or its defaults), not the Create defaults.
  const swapHint=plan?`${planLabel}: ${value.steps??plan.steps} steps, CFG ${value.cfg??plan.cfg}, ${(value.sampler||plan.sampler).replace(/_/g,' ')} / ${value.scheduler||planSchedule}${plan.loras.some(l=>l.source==='speed')?', with the speed LoRA':''}`:'';
  const status=[RECOMMENDED.has(model?.id??'')?'★ Recommended':'',model?.ready?'✓ Installed':model?'Downloads when you generate':'',
    swapHint||settingsHint,pageText&&!plan?'Author settings':''].filter(Boolean).join(' · ');
  const cls=`image-generation-settings studio-settings settings-${part}`;
  if(part==='model')return <fieldset className={cls} disabled={disabled} aria-label="Model">
    <div className="field model-field"><label htmlFor="image-model">Model</label>
      <select id="image-model" value={modelId} onChange={e=>onModel(e.target.value)} title={model?.title}>
        {groups.map(([id,label])=><optgroup key={id} label={label}>{imageModels.filter(m=>modelGroup(m)===id).sort((a,b)=>order.indexOf(quant(a))-order.indexOf(quant(b))).map(m=>
          <option key={m.id} value={m.id}>{m.title}{RECOMMENDED.has(m.id)?' ★':''}{m.ready?' ✓':` · ↓ ${gb(m.bytes)}`}</option>)}</optgroup>)}
      </select>
      <small className="model-status" title={[pageText,defaults?.note].filter(Boolean).join(' ')||undefined}>{status}</small>
    </div>
  </fieldset>;
  if(part==='steps')return <fieldset className={cls} disabled={disabled} aria-label="Steps and CFG">
    <div className="field slider-field"><SettingLabel htmlFor={`${idPrefix}-steps`} label="Steps" value={defaultSteps} disabled={!defaultSteps} onReset={()=>onChange({steps:null})}>Steps <span className="tool-strength-value">{value.steps??defaultSteps??'…'}</span>{defaultSteps?<Nudge label="steps" value={value.steps??defaultSteps} step={1} min={1} max={80} onChange={steps=>onChange({steps})}/>:null}</SettingLabel><input id={`${idPrefix}-steps`} type="range" min="1" max="80" step="1" disabled={!defaultSteps} value={value.steps??defaultSteps??1} aria-valuetext={`${value.steps??defaultSteps} steps`} title={turbo?`Turbo is trained for ${defaultSteps} steps; other counts stretch its schedule.`:`Default: ${defaultSteps??'…'} steps`} onChange={e=>onChange({steps:Number(e.target.value)})}/>
      {!validSteps?<small>Choose a whole number from 1 to 80.</small>:turbo&&value.steps!==null&&value.steps!==defaultSteps?<small>Turbo is trained for {defaultSteps}.</small>:null}
    </div>
    <div className="field slider-field"><SettingLabel htmlFor={`${idPrefix}-cfg`} label="CFG" value={defaultCfg} onReset={()=>onChange({cfg:null})}>CFG <span className="tool-strength-value">{(value.cfg??defaultCfg).toFixed(1)}</span>{defaults?<Nudge label="CFG" value={value.cfg??defaultCfg} step={0.5} min={1} max={15} onChange={cfg=>onChange({cfg})}/>:null}</SettingLabel><input id={`${idPrefix}-cfg`} type="range" min="1" max="15" step="0.5" disabled={!defaults} value={value.cfg??defaultCfg} aria-valuetext={`CFG ${value.cfg??defaultCfg}`} title={`${defaults?.cfg?'Default from the model page. ':turbo?'Turbo is trained for CFG 1. ':'1, or 2 with a negative prompt. '}Above 1 runs a second guidance pass, so each step takes about twice as long.`} onChange={e=>onChange({cfg:Number(e.target.value)})}/>
      {!validCfg?<small>Choose a number from 1 to 15.</small>:(value.cfg??defaultCfg)>1?<small>Above 1: about 2x slower per step.</small>:null}
    </div>
  </fieldset>;
  if(part==='advanced')return <fieldset className={cls} disabled={disabled} aria-label="Sampler and schedule">
    <div className="field"><SettingLabel htmlFor={`${idPrefix}-sampler`} label="Sampler" value={planSampler} onReset={()=>onChange({sampler:null})}>Sampler</SettingLabel><select id={`${idPrefix}-sampler`} value={value.sampler||''} onChange={e=>onChange({sampler:e.target.value||null})}><option value="">{plan?.recipe?'Default':'Model default'} ({planSampler})</option>{['euler','res_multistep','dpmpp_2m','heun','res_2m','res_2s','res_2m_ode','res_2s_ode','deis_2m'].map(name=><option key={name} value={name}>{name}</option>)}</select></div>
    <div className="field"><SettingLabel htmlFor={`${idPrefix}-scheduler`} label="Schedule" value={planSchedule} onReset={()=>onChange({scheduler:null})}>Schedule</SettingLabel><select id={`${idPrefix}-scheduler`} value={value.scheduler||''} onChange={e=>onChange({scheduler:e.target.value||null})}><option value="">{plan?.recipe?'Default':'Model default'} ({planSchedule})</option>{['simple','normal','beta','karras','sgm_uniform'].map(name=><option key={name} value={name}>{name}</option>)}</select>{turbo&&value.scheduler?<small>Custom schedule. Reset restores this model’s trained schedule.</small>:null}</div>
  </fieldset>;
  return <fieldset className={cls} disabled={disabled} aria-label="Tool and output size">
    <div className="field"><SettingLabel htmlFor="image-tool" label="Tool" value={'None'} onReset={()=>chooseTool('')}>Tool</SettingLabel>
      <select id="image-tool" value={toolState.tool_id||''} onChange={e=>chooseTool(e.target.value)} title={tool?tool.summary:'Optional add-ons for try-on, restoration, expansion and more.'}>
        <option value="">None · free prompt</option>
        {tools.map(t=>{const ok=t.families.includes(family);return <option key={t.id} value={t.id} disabled={!ok}>{t.title}{ok?'':` · ${toolOnly(t.families,family)}`}</option>;})}
      </select>
    </div>
    <div className="field"><SettingLabel htmlFor="image-size" label="Output size" value="Auto · 2K" onReset={()=>setSize('auto-2048')}>Output size</SettingLabel>
      <select id="image-size" value={sizeValue} onChange={e=>setSize(e.target.value)} title={followsImage?'Follows the shape of image 1.':value.custom_size?'Exact size in pixels.':'Shape follows the prompt or input image.'}>
        <option value="auto-1024">Auto · 1K (about 1 MP)</option><option value="auto-2048">Auto · 2K (about {followsImage?'2':'4'} MP)</option>
        {followsImage?null:<><option disabled>──────────</option>{presets.map(([w,h,r])=><option key={`${w}x${h}`} value={`${w}x${h}`}>{w} × {h} · {r}</option>)}<option disabled>──────────</option><option value="custom">Custom…</option></>}
      </select>
    </div>
    {tool?<small className="tool-summary">{tool.summary}</small>:null}
    {sizeValue==='custom'&&!followsImage?<div className="custom-image-settings">
      {(['width','height'] as const).map(key=><div className="field" key={key}><SettingLabel htmlFor={key} label={key==='width'?'Width':'Height'} value={key==='width'?2688:1536} onReset={()=>onChange({[key]:key==='width'?2688:1536})}>{key==='width'?'Width':'Height'}</SettingLabel><div className="dimension"><input id={key} type="number" min="256" max="2688" step="8" value={value[key]} onChange={e=>onChange({[key]:Number(e.target.value)})}/><span>px</span></div><input aria-label={`${key==='width'?'Width':'Height'} slider`} type="range" min="256" max="2688" step="8" value={value[key]} onChange={e=>onChange({[key]:Number(e.target.value)})}/></div>)}
      {value.width%32!==0||value.height%32!==0?<small className="span-2">Sizes that are not multiples of 32 are rendered at the next compatible size, then resized to your exact dimensions.</small>:null}
    </div>:null}
    {tool?<div className="tool-panel">
      <div className="tool-inputs">{tool.inputs.labels.map((label,i)=><span key={label}><b>{i+1}</b>{label}</span>)}{!tool.tested_on.includes(family)?<em>Not tested on {familyLabel(family)} by its author</em>:null}</div>
      {tool.fields.map(field=><div className="field" key={field.key}><SettingLabel htmlFor={`tool-${field.key}`} label={field.label} value={field.default} onReset={()=>onTool({...toolState,tool_options:{...options,[field.key]:field.default},lora_strength:null})}>{field.label}</SettingLabel>
        {field.options?<select id={`tool-${field.key}`} value={options[field.key]??field.default} onChange={e=>onTool({...toolState,tool_options:{...options,[field.key]:e.target.value},lora_strength:null})}>{field.options.map(o=><option key={o.value} value={o.value}>{o.label}</option>)}</select>
          :<input id={`tool-${field.key}`} value={options[field.key]??''} placeholder={field.placeholder} onChange={e=>onTool({...toolState,tool_options:{...options,[field.key]:e.target.value}})}/>}
      </div>)}
      {/* The tool LoRA's Effect strength is on its row under LoRAs. */}
      <a className="tool-source" href={tool.source} target="_blank" rel="noreferrer">About this tool</a>
    </div>:null}
  </fieldset>;
}
