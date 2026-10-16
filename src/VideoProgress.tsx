import {useEffect,useState} from 'react';
import {AlertTriangle,Clock,Loader2} from 'lucide-react';
import type {SavedJob} from './JobQueue';
import type {Api,ModelEntry} from './ModelSettings';
import {ModelDownloadProgress} from './ModelDownloadProgress';
import {formatDuration as time} from './duration';

type Step={label:string;stages:string[]};
const START=['queued','downloading_models','waiting_for_download','loading','uploading','waiting','processing'];
const SAVE=['saving','muxing_audio','downloading','complete'];

/** The user-facing stages of a job, in the order the backend workers run them. */
function stagePlan(job:SavedJob):Step[]{
  const r=job.request||{};const plan:Step[]=[];
  const add=(label:string,stages:string[],when=true)=>{if(when)plan.push({label,stages});};
  if(job.kind==='upscale'){
    add('Prepare',START);add('Load model',['loading_model']);add('Read frames',['encoding']);
    add('Upscale frames',['upscaling']);add('Decode frames',['decoding']);add('Save video',SAVE);
    return plan;
  }
  if(job.kind==='multi-reference'){
    add('Prepare inputs',[...START,'reading_references']);add('Load text encoder',['loading_encoder']);
    add('Encode prompt and voices',['encoding_text','loading_video_vae','encoding_audio']);
    add('Load model and references',['loading_model','encoding_references']);
    add(r.refine?'Generate and refine':'Generate',['generating','upscaling_latent']);
    add('Decode',['decoding']);add('Save video',SAVE);
    return plan;
  }
  const reference=job.kind==='reference-video';
  add('Prepare',START);
  add('Read source video',['decoding_source','reading_audio','reading_references'],reference);
  add('Enhance prompt',['rewriting'],!!r.enhance);
  add('Build camera guide',['camera_depth','camera_warp'],reference&&r.workflow==='crossview'&&!r.warp_video);
  add('Load text encoder',['initializing_video','loading_encoder']);
  add(reference?'Encode prompt and video':r.mode==='image'?'Encode prompt and images':'Encode prompt',
    ['loading_video_vae','preparing_conditioning','encoding_prompt','encoding_reference_image','loading_audio_vae','encoding_source_video','encoding_source_audio']);
  add('Load video model',['loading_model']);add('Generate',['generating']);add('Decode',['decoding']);add('Save video',SAVE);
  return plan;
}

export function VideoProgress({job,clock}:{job:SavedJob;clock:number}){
  const end=job.finished_at?job.finished_at*1000:clock;
  const total=job.started_at?Math.max(0,(end-job.started_at*1000)/1000):0;
  const stageTime=job.stage_started_at?Math.max(0,(end-job.stage_started_at*1000)/1000):0;
  if(job.status==='queued'||job.stage==='queued'){
    const waited=job.created_at?Math.max(0,(clock-job.created_at*1000)/1000):0;
    return <div className="vp-card" role="status" aria-live="polite">
      <div className="vp-head"><Clock size={16}/><strong>Waiting in queue</strong>{job.position?<span className="vp-badge">Position {job.position}</span>:null}</div>
      <p className="vp-message">This job starts when the jobs before it finish.</p>
      <div className="vp-times"><span>Waiting {time(waited)}</span></div>
    </div>;
  }
  const plan=stagePlan(job);
  const at=(stage:string)=>plan.findIndex(step=>step.stages.includes(stage));
  let index=at(job.stage);
  if(index<0)index=Math.max(0,...Object.keys(job.stage_seconds||{}).map(at));
  const current=plan[index];
  const next=plan[index+1];
  const upscale=job.kind==='upscale';
  const phase=typeof job.phase==='string'?job.phase:'';
  const pass=phase?(phase==='Changing camera view'?2:1):0;
  const raw=job.message||current.label;
  const message=phase&&raw.startsWith(`${phase} · `)?raw.slice(phase.length+3):raw;
  const step=job.step||0,steps=job.total_steps||0;
  let counter:{done:number;total:number;unit:string}|null=null;
  if(job.stage==='decoding_source'&&job.input_frames_total)counter={done:job.input_frames_done||0,total:job.input_frames_total,unit:'frames read'};
  else if(job.stage==='generating'&&steps>0)counter={done:step,total:steps,unit:'steps'};
  else if(upscale&&['encoding','upscaling','decoding','saving'].includes(job.stage)&&steps>1)counter={done:step,total:steps,unit:'batches'};
  // Steps finished before this stage started (second pass, or LTX refinement after the first 8 steps).
  const base=job.stage!=='generating'?0:pass===2?steps/2:job.kind==='multi-reference'&&job.request?.refine&&step>8?8:0;
  const measured=counter&&['generating','upscaling'].includes(job.stage)?counter.done-base:0;
  const perStep=measured>=1?stageTime/measured:0;
  const left=perStep&&counter&&counter.done<counter.total?perStep*(counter.total-counter.done):0;
  const offloaded=Number(job.model_offloaded_weight_gb)||0;
  const onGpu=Number(job.model_gpu_weight_gb)||0;
  const chunk=upscale&&job.total_chunks>1&&job.chunk?`Clip ${job.chunk} of ${job.total_chunks}`:'';
  return <div className="vp-card" role="status" aria-live="polite">
    <div className="vp-head"><Loader2 className="spin" size={16}/><span className="vp-badge">Stage {index+1} of {plan.length}</span><strong>{current.label}</strong>{pass?<span className="vp-badge vp-pass">Pass {pass} of 2</span>:null}{chunk?<span className="vp-badge vp-pass">{chunk}</span>:null}</div>
    <p className="vp-message" title={message}>{message}</p>
    {job.stage==='downloading_models'?<ModelDownloadProgress job={job}/>:<progress aria-label={`${current.label} progress`} max={counter?.total||1} value={counter?counter.done:undefined}/>}
    <div className="vp-times">
      {counter?<span className="vp-count">{counter.done} / {counter.total} {counter.unit} · {Math.floor(100*counter.done/counter.total)}%</span>:null}
      <span>{time(stageTime)} this stage</span><span>{time(total)} total</span>
      {left?<span className="vp-eta">About {time(left)} left · {time(perStep)} per {upscale?'batch':'step'}</span>:null}
    </div>
    {job.stage==='generating'&&!step&&!job.preview_unavailable?<small className="vp-note">The first step takes longest. The preview and time estimate appear after it finishes.</small>:null}
    {offloaded>=0.1?<p className="vp-warning"><AlertTriangle size={14}/><span>{offloaded.toFixed(1)} GB of the model streams from RAM, so each step is slower. A smaller model version fits fully on the GPU.{onGpu?` ${onGpu.toFixed(1)} GB is on the GPU. GPU use rises and falls while the rest loads.`:''}</span></p>:null}
    <ol className="vp-steps" aria-label="Stages">{plan.map((item,i)=><li key={item.label} className={i<index?'done':i===index?'current':''} title={`${i+1}. ${item.label}`}><span className="sr-only">{i+1}. {item.label}{i<index?' (done)':i===index?' (current)':''}</span></li>)}</ol>
    {next?<small className="vp-next">Next: {next.label}</small>:null}
  </div>;
}

export type GpuMemory={available:boolean;free_gb?:number;total_gb?:number};

/** Dedicated GPU memory, refreshed while the tab is visible on a local GPU. */
export function useGpuMemory(api:Api,enabled:boolean){
  const [memory,setMemory]=useState<GpuMemory|null>(null);
  useEffect(()=>{if(!enabled)return;let alive=true;const refresh=()=>api('/api/gpu-memory').then((snapshot:GpuMemory)=>{if(alive)setMemory(snapshot);}).catch(()=>{});refresh();const timer=window.setInterval(refresh,15000);return()=>{alive=false;window.clearInterval(timer);};},[api,enabled]);
  return memory;
}

const GIB=1024**3;
/** Model weights plus about 4 GB of working memory for activations. */
export const neededGb=(model:ModelEntry)=>model.bytes/GIB+4;
const firstWord=(value?:string)=>(value||'').split(/[\s·]/)[0];

/** The largest installed, smaller version of the same model family that fits in free GPU memory. */
export function smallerFit(models:ModelEntry[],selected:ModelEntry,freeGb:number){
  const options=models.filter(m=>m.id!==selected.id&&m.kind===selected.kind&&m.family===selected.family&&m.installed&&m.bytes<selected.bytes&&neededGb(m)<=freeGb);
  const sameGroup=options.filter(m=>(m.model_group||'')===(selected.model_group||''));
  const related=options.filter(m=>firstWord(m.model_group)===firstWord(selected.model_group));
  const pool=sameGroup.length?sameGroup:related;
  return pool.sort((a,b)=>b.bytes-a.bytes)[0]||null;
}

/** Warns before submitting when the chosen model will not fit in free GPU memory, and offers a smaller installed version. */
export function VramFitHint({models,selected,gpu,onUse}:{models:ModelEntry[];selected:ModelEntry|undefined;gpu:GpuMemory|null;onUse:(id:string)=>void}){
  if(!selected?.bytes||!gpu?.available||typeof gpu.free_gb!=='number')return null;
  const need=neededGb(selected);
  if(gpu.free_gb>=need)return null;
  const option=smallerFit(models,selected,gpu.free_gb);
  return <div className="vs-vram" role="status"><AlertTriangle size={15}/><div>
    <p>{selected.title} needs about {need.toFixed(0)} GB of GPU memory. {gpu.free_gb.toFixed(1)} of {gpu.total_gb?.toFixed(1)} GB is free, so part of the model streams from RAM and each step is slower.</p>
    {option?<button type="button" className="button secondary" onClick={()=>onUse(option.id)}>Use {option.title} ({(option.bytes/GIB).toFixed(1)} GB, fits)</button>:<small>Close other GPU workloads, or download a smaller version in Settings.</small>}
  </div></div>;
}
