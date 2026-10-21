/** Live progress of an image upscale from the fields its worker writes (backend/image_upscale.py Progress). */
export type UpscaleJob={status:string;stage?:string;message?:string;step?:number;total_steps?:number;started_at?:number;
  progress?:number;progress_stage_end?:number;progress_estimate_seconds?:number;progress_at?:number};

/** 0 to 1. Inside a stage the bar moves by elapsed time over the stage estimate and stops at 95% of the stage,
 *  so it never claims a stage is done before the worker says so. */
export function upscaleFraction(job:UpscaleJob,nowMs:number):number{
  if(job.status==='succeeded')return 1;
  if(job.status!=='running')return 0;
  const start=job.progress??0,end=Math.max(start,job.progress_stage_end??start);
  const seconds=job.progress_estimate_seconds,at=job.progress_at;
  const inside=seconds&&at?Math.min(.95,Math.max(0,(nowMs/1000-at)/seconds)):0;
  return Math.min(1,Math.max(0,start+(end-start)*inside));
}

/** 'Stage 3 of 5' while it runs, or '' before the worker reports stages. */
export function upscaleStageLabel(job:UpscaleJob):string{
  return job.status==='running'&&job.total_steps&&job.step&&job.progress!==undefined?`Stage ${job.step} of ${job.total_steps}`:'';
}

export function elapsedLabel(job:UpscaleJob,nowMs:number):string{
  if(!job.started_at)return '';
  const seconds=Math.max(0,Math.floor(nowMs/1000-job.started_at));
  return seconds<60?`${seconds}s`:`${Math.floor(seconds/60)}m ${seconds%60}s`;
}
