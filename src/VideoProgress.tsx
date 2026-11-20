import {useEffect,useState} from 'react';
import {AlertTriangle,Clock,Loader2} from 'lucide-react';
import type {SavedJob} from './JobQueue';
import type {Api,ModelEntry} from './ModelSettings';
import {ModelDownloadProgress} from './ModelDownloadProgress';
import {formatDuration as time} from './duration';
import {formatClipTime} from './referenceTiming';
import {PORTRAIT_KINDS} from './portraitControls';

type Step={label:string;stages:string[]};
// Up to 2.9 GB of H3 weights in RAM did not change the step time on the RTX 5090 (10.13 s per step with 2.9 GB
// and with 1.0 GB, measured 2026-11-15), so smaller amounts show no warning.
const OFFLOAD_WARNING_GB=4;
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
    add('Decode',['decoding','decoding_audio']);add('Save video',SAVE);
    return plan;
  }
  if(PORTRAIT_KINDS.includes(job.kind||'')){
    add('Install tools',['installing'],job.stage==='installing'||!!job.stage_seconds?.installing);add('Prepare',START);
    add(job.kind==='animate'?'Animate':job.kind==='lipsync'?'Lip sync':job.kind==='face-restore'?'Restore faces':'Extract audio',['processing']);add('Save',SAVE);
    return plan;
  }
  const reference=job.kind==='reference-video';
  add('Prepare',START);
  add('Read source video',['decoding_source','reading_audio','reading_references'],reference);
  add('Enhance prompt',['rewriting'],!!r.enhance);
  add('Build camera guide',['camera_depth','camera_warp'],reference&&r.workflow==='crossview'&&!r.warp_video);
  add('Load text encoder',['initializing_video','loading_encoder','loading_encoder_gpu']);
  add(reference?'Encode prompt and video':r.mode==='image'?'Encode prompt and images':'Encode prompt',
    ['loading_video_vae','loading_vae_gpu','preparing_conditioning','encoding_prompt','encoding_reference_image','loading_audio_vae','loading_audio_encoder_gpu','encoding_source_video','encoding_source_audio']);
  add('Load video model',['loading_model','loading_model_gpu']);add('Generate',['generating']);add('Decode',['releasing_model','loading_video_decoder','loading_decoder_gpu','loading_audio_decoder','loading_audio_decoder_gpu','decoding','decoding_audio']);add('Save video',SAVE);
  return plan;
}

/** Frame edges of each part of a long replacement, as backend/long_faceswap.part_plan makes them. A short last part
 *  joins the one before it: under 60 frames now, under 17 in jobs made before; the job's part count tells which. */
function partEdges(total:number,parts:number){
  const plan=(shortest:number)=>{
    const edges=[0,Math.min(total,124)];
    while(edges[edges.length-1]<total)edges.push(Math.min(total,edges[edges.length-1]+119));
    if(edges.length>2&&edges[edges.length-1]-edges[edges.length-2]<shortest)edges.splice(edges.length-2,1);
    return edges;
  };
  const current=plan(60);
  return current.length-1===parts?current:plan(17);
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
  else if(job.stage==='processing'&&steps>0&&PORTRAIT_KINDS.includes(job.kind||''))counter={done:step,total:steps,unit:'frames'};
  // Steps finished before this stage started (second pass, or LTX refinement after the first 8 steps).
  const base=job.stage!=='generating'?0:pass===2?steps/2:job.kind==='multi-reference'&&job.request?.refine&&step>8?8:0;
  const measured=counter&&['generating','upscaling'].includes(job.stage)?counter.done-base:0;
  const perStep=job.stage==='generating'?(Number(job.sampling_step_seconds)||0):measured>=1?stageTime/measured:0;
  const left=perStep&&counter&&counter.done<counter.total?perStep*(counter.total-counter.done):0;
  const offloaded=Number(job.model_offloaded_weight_gb)||0;
  const onGpu=Number(job.model_gpu_weight_gb)||0;
  const chunk=upscale&&job.total_chunks>1&&job.chunk?`Clip ${job.chunk} of ${job.total_chunks}`:job.total_parts>1&&job.part?`Part ${job.part} of ${job.total_parts}${job.total_shots>1&&job.shot?` · Shot ${job.shot} of ${job.total_shots}`:''}`:job.total_shots>1&&job.shot?`Shot ${job.shot} of ${job.total_shots}`:'';
  if(job.kind==='reference-video'&&job.total_parts>1&&job.parts_phase==='prepare'){
    // Before the parts are generated, every part's source video and prompt are encoded, each in one pass.
    const parts=job.total_parts,at=Number(job.preparing_part)||0;
    const pass=['loading_encoder','loading_encoder_gpu','encoding_prompt'].includes(job.stage)?2:1;
    const done=pass===2?parts+at-1:at-1;
    return <div className="vp-card vp-compact" role="status" aria-live="polite" aria-label={`Preparing the ${parts} parts: pass ${pass} of 2, part ${at}`}>
      <div className="vp-row"><Loader2 className="spin" size={14}/><span className="vp-badge">Preparing {parts} parts</span><strong>{pass===1?'Encoding source video':'Encoding prompt and references'} · part {at} of {parts}</strong><span className="vp-muted vp-right">pass {pass} of 2 · {time(total)}</span></div>
      <progress aria-label="Preparing progress" max={2*parts} value={Math.max(0,done)}/>
    </div>;
  }
  if(job.kind==='reference-video'&&job.total_parts>1&&job.part){
    // A long video runs in parts; every part goes through all the stages, so the part leads and the stage is secondary.
    const parts=job.total_parts,part=job.part,frames=Number(job.request?.frames)||0;
    const edges=partEdges(frames,parts);
    const range=(n:number)=>`${formatClipTime((edges[n-1]??0)/24)} to ${formatClipTime((edges[n]??frames)/24)}`;
    const clipStart=Number(job.request?.source_start_seconds)||0;
    const from=job.part_start_seconds!=null?job.part_start_seconds-clipStart:(edges[part-1]??0)/24;
    const to=job.part_end_seconds!=null?job.part_end_seconds-clipStart:(edges[part]??frames)/24;
    const generate=plan.findIndex(item=>item.label==='Generate');
    const inPart=index>generate?1:job.stage==='generating'&&steps>0?step/steps:0;
    const done=part-1+inPart;
    const times:number[]=Array.isArray(job.part_seconds)?job.part_seconds:[];
    const perPart=times.length?times.reduce((sum,value)=>sum+value,0)/times.length:part>2?total/done:0;
    const wholeLeft=perPart?perPart*(parts-done):0;
    const generating=job.stage==='generating'&&steps>0;
    const partLeft=generating&&perStep?perStep*(steps-step):0;
    const saved=Number(job.parts_saved)||0;
    const percent=Math.floor(100*done/parts);
    return <div className="vp-card vp-compact" role="status" aria-live="polite" aria-label={`Part ${part} of ${parts}, ${percent}% of the whole video done`}>
      <div className="vp-row"><Loader2 className="spin" size={14}/><span className="vp-badge">Part {part} of {parts}</span><strong>Clip {formatClipTime(from)} to {formatClipTime(to)}</strong><span className="vp-muted">of {formatClipTime(frames/24)}</span>
        <span className="vp-right"><span className="vp-count">{percent}%</span> · {time(total)}{wholeLeft?<span className="vp-eta"> · about {time(wholeLeft)} left</span>:null}</span></div>
      <ol className="vp-parts" aria-label="Parts of the video">{Array.from({length:parts},(_,i)=>{const n=i+1;const state=n<part?'done':n===part?'current':'';const word=n<part?'done':n===part?'in progress':'waiting';return <li key={n} className={state} title={`Part ${n}: ${range(n)} (${word})`}><span className="sr-only">Part {n}, {range(n)}, {word}</span></li>;})}</ol>
      <div className="vp-row vp-now" title={message}><span className="vp-muted">Now:</span><strong>{current.label}</strong>
        {generating?<><span className="vp-count">step {step} of {steps}</span><progress className="vp-inline" aria-label={`Step ${step} of ${steps} in part ${part}`} max={steps} value={step}/>{partLeft?<span className="vp-muted">about {time(partLeft)} left in this part</span>:null}</>
          :<span className="vp-muted">stage {index+1} of {plan.length} · {time(stageTime)}</span>}
        {saved?<span className="vp-muted vp-right" title={job.parts_folder||''}>{saved} part{saved>1?'s':''} saved</span>:null}</div>
      {generating&&offloaded>=OFFLOAD_WARNING_GB?<p className="vp-warning"><AlertTriangle size={14}/><span>{offloaded.toFixed(1)} GB of model weights are in RAM.{onGpu?` ${onGpu.toFixed(1)} GB is on the GPU.`:''} Transfers can increase generation time.</span></p>:null}
    </div>;
  }
  return <div className="vp-card" role="status" aria-live="polite">
    <div className="vp-head"><Loader2 className="spin" size={16}/><span className="vp-badge">Stage {index+1} of {plan.length}</span><strong>{current.label}</strong>{pass?<span className="vp-badge vp-pass">Pass {pass} of 2</span>:null}{chunk?<span className="vp-badge vp-pass">{chunk}</span>:null}</div>
    <p className="vp-message" title={message}>{message}</p>
    {job.stage==='downloading_models'?<ModelDownloadProgress job={job}/>:<progress aria-label={`${current.label} progress`} max={counter?.total||1} value={counter?counter.done:job.stage==='processing'&&typeof job.engine_progress==='number'?job.engine_progress:undefined}/>}
    <div className="vp-times">
      {counter?<span className="vp-count">{counter.done} / {counter.total} {counter.unit} · {Math.floor(100*counter.done/counter.total)}%</span>:null}
      <span>{time(stageTime)} this stage</span><span>{time(total)} total</span>
      {left?<span className="vp-eta">About {time(left)} left · {time(perStep)} per {upscale?'batch':'step'}</span>:null}
    </div>
    {job.stage==='generating'&&!step&&!job.preview_unavailable?<small className="vp-note">The first step takes longest. The preview appears after it finishes. The time estimate starts after two steps.</small>:null}
    {job.stage==='generating'&&offloaded>=OFFLOAD_WARNING_GB?<p className="vp-warning"><AlertTriangle size={14}/><span>{offloaded.toFixed(1)} GB of model weights are in RAM.{onGpu?` ${onGpu.toFixed(1)} GB is on the GPU.`:''} Transfers can increase generation time.</span></p>:null}
    <ol className="vp-steps" aria-label="Stages">{plan.map((item,i)=><li key={item.label} className={i<index?'done':i===index?'current':''} title={`${i+1}. ${item.label}`}><span className="sr-only">{i+1}. {item.label}{i<index?' (done)':i===index?' (current)':''}</span></li>)}</ol>
    {next?<small className="vp-next">Next: {next.label}</small>:null}
  </div>;
}

export type GpuMemory={available:boolean;free_gb?:number;total_gb?:number;resident_image?:{loaded:boolean};job_active?:boolean};

/** Dedicated GPU memory, refreshed while the tab is visible on a local GPU. */
export function useGpuMemory(api:Api,enabled:boolean){
  const [memory,setMemory]=useState<GpuMemory|null>(null);
  useEffect(()=>{if(!enabled)return;let alive=true;const refresh=()=>api('/api/gpu-memory').then((snapshot:GpuMemory)=>{if(alive)setMemory(snapshot);}).catch(()=>{});refresh();const timer=window.setInterval(refresh,15000);return()=>{alive=false;window.clearInterval(timer);};},[api,enabled]);
  return memory;
}

const GIB=1024**3;
/** Model weights plus about 4 GB of working memory for activations. */
export const neededGb=(model:ModelEntry)=>model.bytes/GIB+4;

/** The largest installed, smaller version of the same model family that fits in free GPU memory. */
export function smallerFit(models:ModelEntry[],selected:ModelEntry,freeGb:number){
  if(!selected.model_group)return null;
  return models.filter(m=>m.id!==selected.id&&m.kind===selected.kind&&m.family===selected.family&&m.model_group===selected.model_group&&m.installed&&m.bytes<selected.bytes&&neededGb(m)<=freeGb).sort((a,b)=>b.bytes-a.bytes)[0]||null;
}

/** Warns before submitting when the chosen model will not fit in free GPU memory, and offers a smaller installed version. */
export function VramFitHint({selected,gpu,resident}:{models:ModelEntry[];selected:ModelEntry|undefined;gpu:GpuMemory|null;onUse:(id:string)=>void;resident?:{loaded:boolean}|null}){
  if(!selected?.bytes||!gpu?.available||typeof gpu.free_gb!=='number')return null;
  if(gpu.job_active)return null;
  const need=neededGb(selected);
  if(gpu.free_gb>=need)return null;
  if((gpu.resident_image||resident)?.loaded)return <div className="vs-memory-note" role="status" title="The current free memory does not show how much will be available for this video.">An image model is in GPU memory; it is released before the video starts.</div>;
  return <div className="vs-vram" role="status"><AlertTriangle size={15}/><div>
    <p>{gpu.free_gb.toFixed(1)} of {gpu.total_gb?.toFixed(1)} GB is free now. The model weights use about {(selected.bytes/GIB).toFixed(1)} GB before working memory. Output size, duration and reference inputs also affect memory use.</p>
    <small>Qwen can move part of the model to RAM when needed. This can make generation slower. Your selected model stays unchanged.</small>
  </div></div>;
}
