import {ModelDefaultsScope,bindSettings} from './ModelDefaults';
import {Nudge,SettingLabel} from './SettingLabel';
import {ImageLabel} from './ImageLabel';
import {FilePicker} from './FilePicker';
import {useEffect,useRef,useState} from 'react';
import type {ImageTool,ImageOptions} from './ImageSettings';
import {InputLibrary,type InputImage,type Person} from './InputLibrary';
import {LibraryPopover} from './LibraryPopover';
import {PersonPhotoCycle} from './PersonPhotoCycle';
import type {Api} from './ModelSettings';
import {SwapFaces,type SwapSelection} from './SwapFaces';
import {SwapPromptEditor} from './SwapPromptEditor';
import {PromptInput} from './PromptInput';
import {displayImageMentions,imageMentionOptions} from './promptMentions';

export type SwapState={tool:string;slots:[InputImage|null,InputImage|null];strength:number;extra:string;selection:SwapSelection|null;poseSource:1|2;styleSource:1|2;promptTemplate?:string;promptRevision?:string;styleMode?:'realistic'|'non-realistic';negativePrompt?:string;nonRealisticNegative?:string;
  // Swap's own sampling: its defaults are saved per model and swap type, apart from the Create page's.
  steps?:number|null;cfg?:number|null;sampler?:string|null;scheduler?:string|null};
export const nonRealisticNegative='photorealism, photographic face, realistic skin pores, photographic skin texture, individual skin microdetails, photographic hair strands, photographic highlights, mismatched rendering style, pasted-on face';
export function swapNegative(value:SwapState){return value.styleMode==='non-realistic'?(value.nonRealisticNegative??nonRealisticNegative):(value.negativePrompt??'');}
export const swapDefaults:SwapState={tool:'head-swap',slots:[null,null],strength:1,extra:'',selection:null,poseSource:1,styleSource:1,styleMode:'realistic',negativePrompt:''};

// What each picture gives, per swap type (the author's docs: image 1 is the scene, image 2 the head or the person).
const ROLES:Record<string,[string,string]>={
  'head-swap':['Provides the body, framing and background','Provides the replacement face, hair and eye color'],
  'head-swap-expression':['Provides the body, framing and background','Provides the replacement face, hair and eye color'],
  'body-swap':['Provides the body pose, framing and background','Provides the face, clothing and body shape. Use a full-body picture with a plain background'],
};

type Result={id:string;request:{prompt:string}};

/** Images > Swap: head or body swap with the BFS LoRAs for Qwen-Image-2.1, from two pictures. */
export function SwapPanel({modelId,restoreKey,api,tools,value,onChange,disabled,results,people,onPeople,revision,options,onOptions,advanced,steps,modelFamily,effectiveCfg}:{api:Api;tools:ImageTool[];value:SwapState;onChange:(next:SwapState)=>void;
  disabled:boolean;results:Result[];people:Person[];onPeople:()=>void;revision:number;options:ImageOptions&{seed:number;randomize_seed:boolean};onOptions:(next:Partial<ImageOptions&{seed:number;randomize_seed:boolean}>)=>void;advanced:React.ReactNode;steps:React.ReactNode;modelFamily?:string;effectiveCfg:number;modelId:string;restoreKey:number}){
  const [uploading,setUploading]=useState<number|null>(null);const [error,setError]=useState('');const [picking,setPicking]=useState<number|null>(null);
  const panel=useRef<HTMLElement>(null);
  const latest=useRef(value);latest.current=value;
  const tool=tools.find(t=>t.id===value.tool)??tools[0];
  const setSlot=(index:number,image:InputImage|null)=>{const slots=[...latest.current.slots] as SwapState['slots'];slots[index]=image;latest.current={...latest.current,slots,selection:null};onChange(latest.current);};
  async function upload(index:number,file?:File){if(!file)return;setUploading(index);setError('');
    try{const body=new FormData();body.append('files',file);const [image]=await api('/api/uploads',{method:'POST',body});setSlot(index,image);}
    catch(e){setError((e as Error).message);}finally{setUploading(null);}}
  // The current template of a Qwen swap: pose and style choices show only when it uses them (the author's prompt does not).
  const [template,setTemplate]=useState('');
  useEffect(()=>{if(!tool?.swap){setTemplate('');return;}let live=true;api(`/api/swap-prompts/${tool.id}`).then(r=>{if(live)setTemplate(r.text||'');}).catch(()=>{});return()=>{live=false;};},[tool?.id,tool?.swap,value.promptRevision,value.promptTemplate]);
  if(!tool)return null;
  const krea=!!tool.swap_page;
  const usesPose=(value.promptTemplate??template).includes('<pose_source>'),usesStyle=(value.promptTemplate??template).includes('<style_source>');
  const roles=ROLES[tool.id]??['',''];
  const bindings=bindSettings(value,patch=>onChange({...value,...patch}),{'Image style':['styleMode','styleSource'],'Head pose and expression':'poseSource','Color and style':'styleSource','Effect strength':'strength','Extra instructions':'extra',Steps:'steps',CFG:'cfg',Sampler:'sampler',Schedule:'scheduler','Negative prompt':value.styleMode==='non-realistic'?'nonRealisticNegative':'negativePrompt'});
  bindings['Negative prompt'].value={negativePrompt:value.negativePrompt??'',nonRealisticNegative:value.nonRealisticNegative??nonRealisticNegative};
  return <ModelDefaultsScope model={modelId} scope={'swap-'+tool.id} bindings={bindings} draftKey="qwen-swap-v1" restoreKey={restoreKey}><section ref={panel} className="panel swap-panel">
    <div className="section-heading"><h2>Swap</h2>{krea?<a className="text-button" href={tool.source} target="_blank" rel="noreferrer" title={tool.prompt}>Author's guide</a>:<SwapPromptEditor key={tool.id} api={api} tool={tool} value={value} onChange={onChange} disabled={disabled}/>}</div>
    <div className="segmented swap-types" role="radiogroup" aria-label="Swap type">{tools.map(t=><button key={t.id} type="button" role="radio" aria-checked={t.id===tool.id} aria-selected={t.id===tool.id} title={t.summary} disabled={disabled} onClick={()=>onChange({...value,tool:t.id,promptTemplate:undefined,promptRevision:undefined})}>{t.title}</button>)}</div>
    {modelFamily==='qwen-image-2.1-turbo'?<p className="swap-turbo-note">Made for Qwen Standard; Turbo can differ. <a href={tool.source} target="_blank" rel="noreferrer">Guide</a></p>:null}
    <div className="swap-slots">{[0,1].map(index=>{const image=value.slots[index];return <div key={index} className="swap-slot">
      <div className="swap-slot-head" title={roles[index]}><b>{index+1}</b><strong>{tool.inputs.labels[index]}</strong></div>
      {image?<div className="swap-image checker"><img src={image.url} alt={tool.inputs.labels[index]}/>
        <button type="button" className="swap-remove" aria-label={`Remove picture ${index+1}`} disabled={disabled} onClick={()=>setSlot(index,null)}>×</button></div>
        :<FilePicker className="swap-empty" title={roles[index]} disabled={disabled||uploading!==null} onFiles={files=>upload(index,files[0])}><span>{uploading===index?'Uploading…':'Upload picture '+(index+1)}</span></FilePicker>}
      {image?<PersonPhotoCycle image={image} people={people} disabled={disabled||uploading!==null} onChange={next=>setSlot(index,next)}/>:null}
      {image?<ImageLabel number={index+1} value={image.subjectLabel} disabled={disabled} onChange={subjectLabel=>{const slots=[...latest.current.slots] as SwapState['slots'];slots[index]={...image,subjectLabel};onChange({...latest.current,slots});}}/>:null}
      <div className="swap-slot-actions">{image?<FilePicker className="text-button" disabled={disabled||uploading!==null} onFiles={files=>upload(index,files[0])}>{uploading===index?'Uploading…':'Replace'}</FilePicker>:null}
        <button type="button" className="text-button" disabled={disabled} onClick={()=>setPicking(index)} title="Library or People">Library</button></div>
    </div>;})}</div>
    {krea?null:<SwapFaces api={api} images={[value.slots[0]?.id||null,value.slots[1]?.id||null]} value={value.selection} onChange={selection=>onChange({...latest.current,selection})} disabled={disabled}/>}
    <LibraryPopover open={picking!==null} onClose={()=>setPicking(null)} anchor={panel}>
      <InputLibrary api={api} inputs={picking!==null&&value.slots[picking]?[value.slots[picking]!]:[]} onChange={next=>{if(picking!==null)setSlot(picking,next[0]??null);setPicking(null);}}
        results={results} people={people} onPeople={onPeople} replace target={0} revision={revision} disabled={disabled} limit={1}
        onUpload={files=>{if(picking!==null)upload(picking,files[0]);setPicking(null);}} onClose={()=>setPicking(null)}/>
    </LibraryPopover>
    <div className="swap-row">
    {krea?null:<div className="field" title={value.styleMode==='non-realistic'?'Match the exact shapes, lines, colors, texture and shading of picture 1. Keep the replacement person recognizable in that style.':'Follow the selected picture’s color, light and texture.'}><SettingLabel disabled={disabled} label="Image style" value="Realistic" onReset={()=>onChange({...value,styleMode:'realistic'})}>Image style</SettingLabel><div className="segmented" role="radiogroup" aria-label="Image style">{(['realistic','non-realistic'] as const).map(mode=><button type="button" role="radio" aria-checked={(value.styleMode??'realistic')===mode} key={mode} disabled={disabled} onClick={()=>onChange({...value,styleMode:mode,...(mode==='non-realistic'?{styleSource:1 as const}:{})})}>{mode==='realistic'?'Realistic':'Non-realistic'}</button>)}</div></div>}
      <div className="field"><SettingLabel disabled={disabled} htmlFor="swap-size" label="Output size" value={'Auto · 2 MP'} onReset={()=>onOptions({custom_size:false,auto_resolution:2048})}>Output size</SettingLabel><select id="swap-size" title="Auto follows picture 1. Custom size changes the output shape." disabled={disabled} value={options.custom_size?'custom':String(options.auto_resolution)} onChange={e=>onOptions(e.target.value==='custom'?{custom_size:true}:{custom_size:false,auto_resolution:Number(e.target.value) as 1024|2048})}>
        <option value="1024">Auto · 1 MP</option><option value="2048">Auto · 2 MP</option><option value="custom">Custom size</option></select></div>
    </div>
    {options.custom_size?<fieldset className="swap-output" disabled={disabled}>
      <div className="swap-dimensions">{(['width','height'] as const).map(key=><div className="field" key={key}><SettingLabel disabled={disabled} htmlFor={'swap-'+key} label={key==='width'?'Width':'Height'} onReset={()=>onOptions({[key]:key==='width'?2688:1536})}>{key==='width'?'Width':'Height'} (px)</SettingLabel><input id={'swap-'+key} type="number" min="256" max="2688" step="8" value={options[key]} onChange={e=>onOptions({[key]:Number(e.target.value)})}/></div>)}</div>
    </fieldset>:null}
    {steps}
    <div className="swap-row">
    <div className="field swap-strength" title="The swap LoRA's strength. 1.0 is the author's value."><SettingLabel disabled={disabled} htmlFor="swap-strength" label="Effect strength" value={tool.strength} onReset={()=>onChange({...value,strength:tool.strength})}><span title="The swap LoRA's strength">Keep original face</span> <b>{value.strength.toFixed(2)}</b><Nudge label="strength" value={value.strength} step={0.05} min={0.1} max={2} disabled={disabled} onChange={strength=>onChange({...value,strength})}/></SettingLabel>
      <input id="swap-strength" type="range" min="0.1" max="2" step="0.05" value={value.strength} disabled={disabled} onChange={e=>onChange({...value,strength:Number(e.target.value)})}/>
      <small>Higher: more of picture 1's face. Lower: more of picture 2's face.</small>
</div>
    <div className="field swap-seed" title="Turn off Random seed to compare prompt or strength changes with the same seed."><SettingLabel disabled={disabled} htmlFor="swap-seed" label="Seed" value={'Random'} onReset={()=>onOptions({seed:0,randomize_seed:true})}>Seed</SettingLabel><div className="seed-row"><input id="swap-seed" type="number" min="0" max="2147483647" step="1" value={options.seed} disabled={disabled||options.randomize_seed} onChange={e=>onOptions({seed:Number(e.target.value)})}/><label className="checkbox-label"><input type="checkbox" checked={options.randomize_seed} disabled={disabled} onChange={e=>onOptions({randomize_seed:e.target.checked})}/>Random</label></div></div>
    </div>
    <div className="field"><SettingLabel disabled={disabled} htmlFor="swap-extra" label="Extra instructions" value={'Empty'} onReset={()=>onChange({...value,extra:''})}>Extra instructions <span className="muted">optional</span></SettingLabel>
      <PromptInput id="swap-extra" dir="auto" rows={2} title="Your face choices define the replacement. Auto aspect ratio keeps the shape of picture 1." placeholder="For example: keep the earrings from @Image1" value={displayImageMentions(value.extra)} disabled={disabled} onValueChange={extra=>onChange({...latest.current,extra})} mentions={imageMentionOptions(value.slots,people,image=>setSlot(value.slots[0]?1:0,image),2)}/>
</div>
    <div className="field"><SettingLabel disabled={disabled} htmlFor="swap-negative" label="Negative prompt" onReset={()=>onChange({...value,[value.styleMode==='non-realistic'?'nonRealisticNegative':'negativePrompt']:value.styleMode==='non-realistic'?nonRealisticNegative:''})}>Negative prompt <span className="muted">optional</span></SettingLabel>
      <PromptInput id="swap-negative" dir="auto" rows={2} maxLength={8000} placeholder="Features to exclude" value={displayImageMentions(swapNegative(value))} disabled={disabled} onValueChange={text=>onChange({...latest.current,[value.styleMode==='non-realistic'?'nonRealisticNegative':'negativePrompt']:text})} mentions={imageMentionOptions(value.slots,people)}/>
      {effectiveCfg<=1&&swapNegative(value).trim()?<small role="status" title="The style instructions still apply. Turbo models are usually tuned for CFG 1.">Ignored at CFG 1. Raise CFG in Advanced image settings to use it.</small>:null}
</div>
    <details className="disclosure swap-advanced"><summary>Advanced</summary>
      {!krea&&(usesPose||usesStyle)?<p className="swap-advanced-note">The face, hair and eye color always come from picture 2. For a plain face swap keep both on Picture 1.</p>:null}
  {krea||!(usesPose||usesStyle)?null:<div className="swap-dimensions" title="These choices guide the model. Check the result for changes outside the selected face.">{([['poseSource','Head pose and expression'],['styleSource','Color and style']] as const).map(([key,label])=><div className="field" key={key}><SettingLabel disabled={disabled} htmlFor={'swap-'+key} label={label} value="Picture 1" onReset={()=>onChange({...value,[key]:1})}><span title={label}>{key==='poseSource'?'Head pose':label}</span></SettingLabel><select id={'swap-'+key} value={key==='styleSource'&&value.styleMode==='non-realistic'?1:value[key]??1} disabled={disabled||(key==='styleSource'&&value.styleMode==='non-realistic')} onChange={e=>onChange({...value,[key]:Number(e.target.value) as 1|2})}><option value="1">Picture 1</option><option value="2">Picture 2</option></select></div>)}</div>}
      {advanced}</details>
    {error?<div className="message error" role="alert"><span>{error}</span></div>:null}
  </section></ModelDefaultsScope>;
}
