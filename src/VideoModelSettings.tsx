import type {ModelEntry,SamplingProfile} from './ModelSettings';
import {isLtx} from './videoModels';

export function createSampling(model?:ModelEntry,quality='standard'):SamplingProfile{
  const d=model?.generation_defaults;
  if(isLtx(model))return {steps:8,cfg:1,sampler:'euler_ancestral',scheduler:'fixed schedule',source:d?.source||''};
  if(model?.family==='fasth3')return {steps:8,cfg:1,sampler:'res_multistep',scheduler:'simple',video_shift:10,audio_shift:3,source:d?.source||''};
  if(model?.family==='minimax-h3-ref2va'){const profiles=d?.profiles;return profiles?.[quality as 'standard'|'turbo']||profiles?.[d?.quality||'standard']||{steps:d?.steps||20,source:d?.source||''};}
  if(quality==='turbo'&&!d?.embedded_turbo)return {steps:8,cfg:1,sampler:'res_multistep',scheduler:'simple',video_shift:12,audio_shift:3,lora_id:'h3-turbo',source:d?.source||''};
  return {steps:20,cfg:1,sampler:'res_multistep',scheduler:'simple',video_shift:12,audio_shift:3,source:'',...d};
}

export function VideoModelSettings({model,profile,steps,frames,width,height,lastFrame=false,workflow='create',fps=24,refine=false,customSteps=false}:{model?:ModelEntry;profile?:SamplingProfile;steps:number;frames:number;width:number;height:number;lastFrame?:boolean;workflow?:'create'|'references'|'video-to-video';fps?:number;refine?:boolean;customSteps?:boolean}){
  if(!model)return null;
  const p=profile||createSampling(model);const ltx=isLtx(model);
  const range=p.steps_range?.join('–')||p.step_options?.join(' or ');
  const origin=p.field_sources?.steps;
  return <div className="video-model-settings">
    <p><strong>{steps} steps · CFG {p.cfg??p.guidance??1}</strong> · {(p.sampler||'euler').replaceAll('_',' ')} / {p.scheduler||'simple'}{ltx&&lastFrame?' · eta 0':''}</p>
    {customSteps?<small>Custom steps: {steps}. Model default: {p.steps}.</small>:null}
    <p>{width} × {height} px · {frames} frames · {(frames/fps).toFixed(2)} seconds at {fps} fps{p.video_shift!=null?` · Shift ${p.video_shift} video / ${p.audio_shift} audio`:''}</p>
    {p.scheduler_video_shift!=null&&p.scheduler_video_shift!==p.video_shift?<small>Sampling schedule shift: {p.scheduler_video_shift} video / {p.scheduler_audio_shift} audio.</small>:null}
    {range?<small>Author steps: {range}. Selected: {steps}.</small>:null}
    {model.source==='civitai'&&!ltx?<small>{origin==='version'||origin==='model'?`Author default: ${p.steps} steps.`:origin==='attached_workflow'?`Author workflow: ${p.steps} steps.`:origin==='examples'?`Example default: ${p.steps} steps.`:`Steps not specified by the author. Default: ${p.steps}.`} Other values use the model defaults when the author does not specify them.</small>:null}
    {ltx?<small>{workflow==='references'?`Reference images: 8 steps${refine?', then 3 steps at twice the size':''}.`:lastFrame?'First and last frame: 8 steps at full size.':'Text or first frame: 8 steps, then 3 steps at twice the size.'} CFG stays at 1 for this distilled schedule.</small>:null}
    <details><summary>Size and duration rules</summary><p>{ltx?`LTX uses 8n+1 frames. ${lastFrame||workflow==='references'?'Dimensions use multiples of 32, up to 1 MP.':'Dimensions use multiples of 64, up to 1.57 MP.'}`:'H3 uses 17n+5 frames. Dimensions use multiples of 32, up to 1344 pixels per side and 1.03 MP.'} The duration is the frame count divided by {fps}. These are the limits of this workflow.</p>{p.source?<a href={p.workflow_source||p.source} target="_blank" rel="noreferrer">Model settings</a>:null}</details>
  </div>;
}

export function VideoStepInput({id,value,defaultSteps,onChange}:{id:string;value:number|null;defaultSteps:number;onChange:(value:number|null)=>void}){
  return <div className="field"><label htmlFor={id}>Steps</label><input id={id} type="number" min="1" max="60" step="1" aria-describedby={`${id}-help`} value={value??defaultSteps} onChange={e=>onChange(e.target.value===''?null:Number(e.target.value))}/><small id={`${id}-help`}>{value===null?'Uses model settings.':`Custom value. Model default: ${defaultSteps}.`}</small><button className="text-button" type="button" disabled={value===null} onClick={()=>onChange(null)}>Use model settings ({defaultSteps})</button></div>;
}
