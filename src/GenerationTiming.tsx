import {useEffect,useId,useRef,useState} from 'react';
import {createPortal} from 'react-dom';
import {Clock3,Copy,X} from 'lucide-react';
import type {ModelState} from './ModelSettings';
import './run-details.css';

export type Timing={stage_seconds?:Record<string,number>;encoder_load_seconds?:number;diffusion_load_seconds?:number;encoding_seconds?:number;model_reused?:boolean;conditioning_reused?:boolean};
type Run=Timing&{id:string;kind?:string;status:string;request:{model_id?:string;steps?:number|null;seed?:number;randomize_seed?:boolean;execution?:string;workflow?:string;replace_subjects_first?:boolean;refine?:boolean};seed?:number;seed_used?:boolean;elapsed_seconds?:number;started_at?:number;finished_at?:number;created_at?:number;width?:number;height?:number;actual_steps?:number;total_steps?:number;total_chunks?:number;fps?:number;frames?:number;duration_seconds?:number;peak_vram_gb?:number;sampler?:string;guidance?:number};
const valid=(value:unknown):value is number=>typeof value==='number'&&Number.isFinite(value)&&value>=0;
const positive=(value:unknown):value is number=>valid(value)&&value>0;
function time(seconds:number){
  if(seconds<.1)return '< 0.1s';
  if(seconds<60)return `${Number(seconds.toFixed(1))}s`;
  const whole=Math.round(seconds),hours=Math.floor(whole/3600),minutes=Math.floor(whole%3600/60),rest=whole%60;
  return hours?`${hours}h ${minutes}m ${rest}s`:`${minutes}m ${rest}s`;
}
const stageLabels:Record<string,string>={loading:'Preparing tools',loading_model:'Loading model',loading_encoder:'Loading text encoder',loading_video_vae:'Loading video and audio encoders',rewriting:'Prompt enhancement',encoding:'Processing inputs',encoding_text:'Encoding text',encoding_video:'Encoding video',encoding_images:'Encoding images',encoding_audio:'Encoding audio',encoding_references:'Encoding references',reading_references:'Preparing references',decoding_source:'Reading source video',generating:'Generation',upscaling:'Upscaling',upscaling_latent:'Resolution refinement',decoding:'Decoding output',saving:'Saving output',muxing_audio:'Adding source audio',downloading_models:'Model download'};

/** Every value belongs to the displayed result, never to the editable form. */
export function GenerationTiming({job,models}:{job:Run;models?:ModelState|null}){
  const dialog=useRef<HTMLDialogElement>(null),headingId=useId();
  const [copyState,setCopyState]=useState('Copy details');
  useEffect(()=>{dialog.current?.close();setCopyState('Copy details');},[job.id]);
  if(!['succeeded','failed','cancelled'].includes(job.status))return null;
  const elapsed=valid(job.elapsed_seconds)?job.elapsed_seconds:valid(job.started_at)&&valid(job.finished_at)&&job.finished_at>=job.started_at?job.finished_at-job.started_at:null;
  const model=models?.models.find(item=>item.id===job.request.model_id)?.title||job.request.model_id;
  const seed=job.seed_used===false?null:valid(job.seed)?job.seed:job.request.randomize_seed===false&&valid(job.request.seed)?job.request.seed:null;
  const steps=job.kind==='upscale'?job.total_chunks:[job.actual_steps,job.total_steps,job.request.steps].find(positive);
  const twoPasses=job.request.workflow==='crossview'&&job.request.replace_subjects_first;
  const countLabel=job.kind==='upscale'?'sections':twoPasses?'steps/pass':'steps';
  const size=positive(job.width)&&positive(job.height)?`${job.width} × ${job.height}`:null;
  const facts:[string,string][]=[];
  if(elapsed!==null)facts.push(['Run time',time(elapsed)]);
  if(model)facts.push(['Model',model]);
  if(steps)facts.push([job.kind==='upscale'?'Processed sections':twoPasses?'Steps per pass':'Steps',String(steps)]);
  if(twoPasses)facts.push(['Passes','2: subject replacement, then camera view']);
  if(job.request.refine&&job.kind==='multi-reference')facts.push(['Passes','8 generation steps + 3 refinement steps']);
  if(seed!==null)facts.push(['Seed',String(seed)]);
  if(size)facts.push(['Output size',size]);
  if(positive(job.duration_seconds))facts.push(['Video length',`${Number(job.duration_seconds.toFixed(2))}s`]);
  if(positive(job.fps))facts.push(['Frame rate',`${job.fps} fps`]);
  if(positive(job.frames))facts.push(['Frames',String(job.frames)]);
  if(job.sampler)facts.push(['Sampler',job.sampler]);
  if(valid(job.guidance))facts.push(['Guidance',String(job.guidance)]);
  if(positive(job.peak_vram_gb))facts.push(['Peak GPU memory',`${job.peak_vram_gb.toFixed(2)} GB`]);
  if(job.request.execution)facts.push(['Processing',job.request.execution==='runpod'?'RunPod':'Local']);
  if(job.model_reused!==undefined)facts.push(['Model reused',job.model_reused?'Yes':'No']);
  if(job.conditioning_reused!==undefined)facts.push(['Prompt and inputs reused',job.conditioning_reused?'Yes':'No']);
  if(valid(job.created_at)&&valid(job.finished_at)&&job.finished_at>=job.created_at)facts.push(['Request to result',time(job.finished_at-job.created_at)]);
  const stages=Object.entries(job.stage_seconds||{}).filter(([stage,value])=>!['queued','complete','failed','cancelled'].includes(stage)&&positive(value));
  const stageName=(name:string)=>stageLabels[name]||name.replaceAll('_',' ').replace(/^./,value=>value.toUpperCase());
  async function copy(){try{await navigator.clipboard.writeText([...facts,...stages.map(([label,value]):[string,string]=>[stageName(label),time(value)])].map(([label,value])=>`${label}: ${value}`).join('\n'));setCopyState('Copied');}catch{setCopyState('Could not copy');}}
  return <div className="run-details" data-run-id={job.id}>
    <div className="run-details-summary" aria-label="Completed run summary">
      <span className="run-time"><Clock3 size={12}/>{elapsed!==null?`Time ${time(elapsed)}`:'Time unavailable'}</span>
      {model?<span className="run-model" title={model}>{model}</span>:null}
      {steps?<span>{steps} {countLabel}</span>:null}{size?<span>{size}</span>:null}{seed!==null?<span>Seed {seed}</span>:null}
      <button type="button" className="run-details-button" onClick={()=>dialog.current?.showModal()} aria-label="View run details">Details</button>
    </div>
    {createPortal(<dialog className="run-details-dialog" ref={dialog} aria-labelledby={headingId} onCancel={e=>e.stopPropagation()} onClose={()=>setCopyState('Copy details')} onClick={e=>{if(e.target===e.currentTarget)dialog.current?.close();}}>
      <header><h2 id={headingId}>Run details</h2><button type="button" className="icon-button" aria-label="Close run details" onClick={()=>dialog.current?.close()}><X size={19}/></button></header>
      <div className="run-details-body"><dl>{facts.map(([label,value])=><div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
        {stages.length?<><h3>Time by stage</h3><dl>{stages.map(([label,value])=><div key={label}><dt>{stageName(label)}</dt><dd>{time(value)}</dd></div>)}</dl></>:null}
      </div><footer><button type="button" className="text-button" onClick={copy}><Copy size={13}/>{copyState}</button></footer>
    </dialog>,document.body)}
  </div>;
}
