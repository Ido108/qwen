import {useEffect,useRef,useState} from 'react';
import {Check,Clock,Loader2,ListOrdered,Pause,X,AlertCircle} from 'lucide-react';
import {jobKindLabel,type QueueState,type SavedJob} from './JobQueue';
import {formatDuration} from './duration';
import './queue-activity.css';

export function stageLabel(job:SavedJob){
  if(job.status==='queued')return 'Waiting';
  // Upscale stages (encode, upscale, decode frames) have their own progress text from the worker.
  if(job.kind==='upscale'&&job.message&&job.status==='running')return job.message;
  if(job.stage==='encoding')return job.conditioning_reused?'Reusing prompt and inputs':job.request.images?.length?'Processing prompt and images':'Processing prompt';
  const labels:Record<string,string>={downloading_models:'Downloading model',waiting_for_download:'Waiting for download',loading:'Loading model',loading_model:'Loading model',rewriting:'Preparing prompt',encoding:'Preparing images',generating:'Generating',upscaling:'Upscaling',training:'Training',captioning:'Writing captions',loading_caption_model:'Loading caption model',downloading_caption_model:'Downloading caption model',saving:'Saving result',muxing_audio:'Adding audio',downloading:'Saving result',uploading:'Sending files',waiting:'Waiting for processing',processing:'Processing',complete:'Finishing',decoding:'Decoding'};
  return labels[job.stage]||job.message||'Preparing';
}
export function stepLabel(job:SavedJob){
  return ['generating','upscaling','training','captioning'].includes(job.stage)&&job.total_steps>0
    ?`${job.kind==='upscale'?'Batch':job.stage==='captioning'?'Image':'Step'} ${job.step||0} of ${job.total_steps}`:'';
}

/** One persistent status for the whole queue, independent of the last submitted job. */
export function QueueActivity({state,error,onOpen,onView}:{state:QueueState;error:string;onOpen:()=>void;onView:(job:SavedJob)=>void}){
  const tracked=useRef(new Set<string>());
  const [finished,setFinished]=useState<SavedJob|null>(null);
  const [now,setNow]=useState(Date.now());
  useEffect(()=>{
    const ended=state.recent.filter(job=>tracked.current.has(job.id)&&['succeeded','failed','cancelled'].includes(job.status));
    for(const job of ended)tracked.current.delete(job.id);
    if(ended.length)setFinished(ended[0]);
    if(state.active)tracked.current.add(state.active.id);
    for(const job of state.pending)tracked.current.add(job.id);
  },[state]);
  useEffect(()=>{if(!state.active)return;const timer=window.setInterval(()=>setNow(Date.now()),1000);return()=>clearInterval(timer);},[state.active?.id]);
  const active=state.active,waiting=state.pending.length;
  if(!active&&!waiting&&!state.paused&&!error&&!finished)return null;
  const stage=active?stageLabel(active):state.paused?'Queue paused':waiting?'Starting next job':error?'Connection lost':'';
  const steps=active?stepLabel(active):'';
  const download=active?.stage==='downloading_models';
  const maximum=active?(download?active.download_total_bytes:steps?active.total_steps:0):0;
  const value=active?(download?active.download_completed_bytes:active.step):0;
  const elapsed=active?.started_at?Math.max(0,Math.floor(now/1000-active.started_at)):0;
  return <section className="queue-activity" aria-label="Generation status">
    {active||waiting||state.paused||error?<div className="queue-activity-main">
      <span className="queue-activity-icon" aria-hidden="true">{active?<Loader2 size={20} className="spin"/>:state.paused?<Pause size={20}/>:<Clock size={20}/>}</span>
      <div className="queue-activity-copy">
        <div className="queue-activity-title"><strong role="status">{active?`${jobKindLabel(active)} · ${stage}`:stage}</strong><span>{steps}{steps&&elapsed?' · ':''}{elapsed?formatDuration(elapsed):''}</span></div>
        <div className="queue-activity-detail"><span>{waiting?`${waiting} waiting${state.paused?' · Queue paused':''}`:state.paused?'Queue paused · Current job continues':active?'You can keep editing':''}</span>{error?<span className="queue-reconnecting">Reconnecting · Last known progress</span>:null}</div>
        {active?<progress aria-label={download?'Model download progress':'Current stage progress'} value={maximum?value||0:undefined} max={maximum||1}/>:null}
      </div>
      <button className="button secondary queue-activity-open" onClick={onOpen}><ListOrdered size={16}/><span>Queue</span><b>{waiting+(active?1:0)}</b></button>
    </div>:null}
    {finished?<div className={`queue-finished ${finished.status==='failed'?'failed':''}`}>
      <span role="status">{finished.status==='succeeded'?<Check size={16}/>:finished.status==='failed'?<AlertCircle size={16}/>:<X size={16}/>} {jobKindLabel(finished)} {finished.status==='succeeded'?'ready':finished.status==='failed'?'failed':'cancelled'}</span>
      <button className="text-button" onClick={()=>{if(finished.status==='succeeded')onView(finished);else onOpen();setFinished(null);}}>{finished.status==='succeeded'?'View result':'View details'}</button>
      <button className="icon-button" aria-label="Dismiss job update" onClick={()=>setFinished(null)}><X size={15}/></button>
    </div>:null}
  </section>;
}
