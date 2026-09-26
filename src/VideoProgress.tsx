import {Check,Loader2} from 'lucide-react';
import type {SavedJob} from './JobQueue';
import {ModelDownloadProgress} from './ModelDownloadProgress';

const time=(seconds:number)=>seconds<60?`${seconds}s`:`${Math.floor(seconds/60)}m ${seconds%60}s`;
export function VideoProgress({job,clock}:{job:SavedJob;clock:number}){
  const end=job.finished_at?job.finished_at*1000:clock;
  const total=job.started_at?Math.max(0,Math.round((end-job.started_at*1000)/1000)):0;
  const stageTime=job.stage_started_at?Math.max(0,Math.round((end-job.stage_started_at*1000)/1000)):0;
  const generating=job.stage==='generating';
  const phase=job.stage==='saving'?3:job.stage==='decoding'?2:generating?1:0;
  const step=job.step||0;
  const measured=generating?step:job.stage==='decoding_source'?job.input_frames_done:undefined;
  const maximum=generating?job.total_steps:job.input_frames_total;
  if(job.stage==='downloading_models')return <div className="reference-job-progress" role="status"><div className="video-progress-heading"><Loader2 className="spin" size={18}/><strong>{job.message}</strong></div><ModelDownloadProgress job={job}/><div className="video-progress-times"><span>{time(total)} elapsed</span><span>Generation starts after the download finishes.</span></div></div>;
  return <div className="reference-job-progress" role="status" aria-live="polite">
    <ol className="video-stage-list" aria-label="Video creation stages">{['Prepare inputs','Generate','Decode','Save'].map((name,index)=><li key={name} className={index===phase?'current':index<phase?'done':''}>{index<phase?<Check size={12}/>:<span>{index+1}</span>}{name}</li>)}</ol>
    <div className="video-progress-heading"><Loader2 className="spin" size={18}/><strong>{job.message||'Waiting in queue'}</strong>{generating?<span>{step} / {job.total_steps} steps</span>:null}</div>
    <progress aria-label="Current stage progress" max={maximum||1} value={measured}/>
    <div className="video-progress-times"><span>{time(stageTime)} this stage</span><span>{time(total)} total</span></div>
    {generating&&!step?<small>{job.preview_unavailable?'A live preview is unavailable for this run.':'The first preview appears after the first generation step finishes.'}</small>:null}
  </div>;
}
