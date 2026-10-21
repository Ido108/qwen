import {useEffect,useId,useRef,useState} from 'react';
import {createPortal} from 'react-dom';
import {Clock3,Copy,X} from 'lucide-react';
import type {ModelState} from './ModelSettings';
import './run-details.css';
import {resultCfg} from './resultCfg';
import {resultLoras} from './resultLoras';
import {ResultPrompt} from './ResultPrompt';
import {SentPrompts,type SentPrompt} from './SentPrompts';
import {resultSampling,scheduleLabel} from './resultSampling';

export type Timing={stage_seconds?:Record<string,number>;encoder_load_seconds?:number;diffusion_load_seconds?:number;encoding_seconds?:number;model_reused?:boolean;conditioning_reused?:boolean;input_count?:number;input_cache_hits?:number;input_read_seconds?:number;input_preparation_reused?:boolean;cfg?:number;true_cfg_scale?:number;applied_lora?:string|null;sampler?:string;scheduler?:string;sampling?:string};
type Run=Timing&{id:string;kind?:string;status:string;request:{prompt?:string;negative_prompt?:string;model_id?:string;steps?:number|null;seed?:number;randomize_seed?:boolean;execution?:string;workflow?:string;replace_subjects_first?:boolean;refine?:boolean};actual_prompt?:string;expanded_prompt?:string;rewritten_prompt?:string;seed?:number;seed_used?:boolean;elapsed_seconds?:number;started_at?:number;finished_at?:number;created_at?:number;width?:number;height?:number;actual_steps?:number;total_steps?:number;total_chunks?:number;fps?:number;frames?:number;duration_seconds?:number;peak_vram_gb?:number;sampler?:string;guidance?:number};
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
export function GenerationTiming({job,models,detailsOnly=false}:{job:Run;models?:ModelState|null;detailsOnly?:boolean}){
  const dialog=useRef<HTMLDialogElement>(null),headingId=useId();
  const [copyState,setCopyState]=useState('Copy details');
  useEffect(()=>{dialog.current?.close();setCopyState('Copy details');},[job.id]);
  if(!['succeeded','failed','cancelled'].includes(job.status))return null;
  const elapsed=valid(job.elapsed_seconds)?job.elapsed_seconds:valid(job.started_at)&&valid(job.finished_at)&&job.finished_at>=job.started_at?job.finished_at-job.started_at:null;
  const model=models?.models.find(item=>item.id===job.request.model_id)?.title||job.request.model_id;
  const seed=job.seed_used===false?null:valid(job.seed)?job.seed:job.request.randomize_seed===false&&valid(job.request.seed)?job.request.seed:null;
  const steps=job.kind==='upscale'?job.total_chunks:[job.actual_steps,job.total_steps,job.request.steps].find(positive);
  const cfg=resultCfg(job);
  const sampling=resultSampling(job);
  const showSampling=job.kind!=='upscale'&&job.kind!=='training'&&job.kind!=='captioning';
  const loras=resultLoras(job);
  const prompt=job.request.prompt;
  const usedPrompt=job.actual_prompt||job.expanded_prompt||job.rewritten_prompt;
  const twoPasses=job.request.workflow==='crossview'&&job.request.replace_subjects_first;
  const countLabel=job.kind==='upscale'?'sections':twoPasses?'steps/pass':'steps';
  const size=positive(job.width)&&positive(job.height)?`${job.width} × ${job.height}`:null;
  // Net inference: the model's own sampling, without loading, prompt work, decoding or saving.
  const stagesSeen=job.stage_seconds||{};
  const inference=[(job as {sampling_seconds?:number}).sampling_seconds,stagesSeen.generating,stagesSeen.upscaling].find(positive)??null;
  const facts:[string,string][]=[];
  if(elapsed!==null)facts.push(['Run time',time(elapsed)]);
  if(inference!==null)facts.push(['Inference (sampling only)',time(inference)]);
  if(model)facts.push(['Model',model]);
  if(loras!==null||!job.kind||job.kind==='image')facts.push(['LoRAs (strength)',loras??'Not saved for this result']);
  if(prompt?.trim())facts.push(['Prompt',prompt]);
  if(usedPrompt?.trim()&&usedPrompt.trim()!==prompt?.trim())facts.push(['Prompt used',usedPrompt]);
  if(job.request.negative_prompt?.trim())facts.push(['Negative prompt',job.request.negative_prompt]);
  if(steps)facts.push([job.kind==='upscale'?'Processed sections':twoPasses?'Steps per pass':'Steps',String(steps)]);
  if(twoPasses)facts.push(['Passes','2: subject replacement, then camera view']);
  if(job.request.refine&&job.kind==='multi-reference')facts.push(['Passes','8 generation steps + 3 refinement steps']);
  if(seed!==null)facts.push(['Seed',String(seed)]);
  if(size)facts.push(['Output size',size]);
  if(positive(job.duration_seconds))facts.push(['Video length',`${Number(job.duration_seconds.toFixed(2))}s`]);
  if(positive(job.fps))facts.push(['Frame rate',`${job.fps} fps`]);
  if(positive(job.frames))facts.push(['Frames',String(job.frames)]);
  if(showSampling){facts.push(['Sampler',sampling.sampler??'Not saved']);facts.push(['Schedule',scheduleLabel(sampling.scheduler)]);}
  if(cfg!==null)facts.push(['CFG',String(cfg)]);
  if(positive(job.peak_vram_gb))facts.push(['Peak GPU memory',`${job.peak_vram_gb.toFixed(2)} GB`]);
  if(job.request.execution)facts.push(['Processing',job.request.execution==='runpod'?'RunPod':'Local']);
  if(job.model_reused!==undefined)facts.push(['Model reused',job.model_reused?'Yes':'No']);
  if(job.conditioning_reused!==undefined)facts.push(['Prompt and inputs reused',job.conditioning_reused?'Yes':'No']);
  if(positive(job.input_count)&&valid(job.input_cache_hits))facts.push(['Input images reused',`${job.input_cache_hits} of ${job.input_count}`]);
  if(job.input_preparation_reused!==undefined)facts.push(['Image preparation reused',job.input_preparation_reused?'Yes':'No']);
  if(valid(job.input_read_seconds)&&positive(job.input_count))facts.push(['Read input images',time(job.input_read_seconds)]);
  if(valid(job.created_at)&&valid(job.finished_at)&&job.finished_at>=job.created_at)facts.push(['Request to result',time(job.finished_at-job.created_at)]);
  const stages=Object.entries(job.stage_seconds||{}).filter(([stage,value])=>!['queued','complete','failed','cancelled'].includes(stage)&&positive(value));
  const stageName=(name:string)=>stageLabels[name]||name.replaceAll('_',' ').replace(/^./,value=>value.toUpperCase());
  async function copy(){try{await navigator.clipboard.writeText([...facts,...stages.map(([label,value]):[string,string]=>[stageName(label),time(value)])].map(([label,value])=>`${label}: ${value}`).join('\n'));setCopyState('Copied');}catch{setCopyState('Could not copy');}}
  return <div className={`run-details${detailsOnly?' run-details-compact':''}`} data-run-id={job.id}>
    {detailsOnly?<button type="button" className="button secondary" onClick={()=>dialog.current?.showModal()} aria-label="View run details">Details</button>:<><div className="run-details-summary" aria-label="Completed run summary">
      <button type="button" className="run-time" title="Time by stage" onClick={()=>dialog.current?.showModal()}><Clock3 size={12}/>{elapsed!==null?`Time ${time(elapsed)}`:'Time unavailable'}{inference!==null?` · Inference ${time(inference)}`:''}</button>
      {model?<span className="run-model" title={model}>{model}</span>:null}
      {steps?<span>{steps} {countLabel}</span>:null}{cfg!==null?<span>CFG {cfg}</span>:null}{size?<span>{size}</span>:null}{seed!==null?<span>Seed {seed}</span>:null}
      {showSampling?<><span>Sampler {sampling.sampler??'Not saved'}</span><span>Schedule {scheduleLabel(sampling.scheduler)}</span></>:null}
      <button type="button" className="run-details-button" onClick={()=>dialog.current?.showModal()} aria-label="View run details">Details</button>
    </div>
    {loras!==null?<dl className="run-loras"><dt>LoRAs (strength)</dt><dd>{loras}</dd></dl>:null}
    <ResultPrompt key={job.id} text={prompt||usedPrompt}/></>}
    {createPortal(<dialog className="run-details-dialog" ref={dialog} aria-labelledby={headingId} onKeyDown={e=>e.stopPropagation()} onCancel={e=>e.stopPropagation()} onClose={()=>setCopyState('Copy details')} onClick={e=>{if(e.target===e.currentTarget)dialog.current?.close();}}>
      <header><h2 id={headingId}>Run details</h2><button type="button" className="icon-button" aria-label="Close run details" onClick={()=>dialog.current?.close()}><X size={19}/></button></header>
      <div className="run-details-body"><dl>{facts.map(([label,value])=><div key={label} className={label==='LoRAs (strength)'?'run-lora-fact':label.toLowerCase().includes('prompt')?'run-prompt-fact':undefined}><dt>{label}</dt><dd dir="auto">{value}</dd></div>)}</dl>
        {stages.length?<><h3>Time by stage</h3><dl>{stages.map(([label,value])=><div key={label}><dt>{stageName(label)}</dt><dd>{time(value)}</dd></div>)}</dl></>:null}
        <SentPrompts items={(job as {sent_prompts?:SentPrompt[]}).sent_prompts}/>
      </div><footer><button type="button" className="text-button" onClick={copy}><Copy size={13}/>{copyState}</button></footer>
    </dialog>,document.body)}
  </div>;
}
