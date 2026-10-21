type SamplingRun={sampler?:unknown;scheduler?:unknown;sampling?:unknown};
const text=(value:unknown)=>typeof value==='string'&&value.trim()?value.trim():null;

/** Read recorded execution values. Never borrow current model or form defaults. */
export function resultSampling(run:SamplingRun){
  let sampler=text(run.sampler),scheduler=text(run.scheduler);
  const saved=text(run.sampling);
  if(saved==='turbo-sigmas'){
    scheduler??='turbo-sigmas';
  }else if(saved){
    const parts=/^([a-z0-9_]+)[/-]([a-z0-9_]+)$/i.exec(saved);
    if(parts){sampler??=parts[1];scheduler??=parts[2];}
  }
  return {sampler,scheduler};
}

export function scheduleLabel(value:string|null){
  if(value==='turbo-sigmas')return 'Turbo (fixed)';
  if(value==='ltx-distilled')return 'LTX distilled (fixed)';
  return value??'Not saved';
}
