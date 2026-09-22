export type Timing = {stage_seconds?:Record<string,number>;encoder_load_seconds?:number;diffusion_load_seconds?:number;encoding_seconds?:number;model_reused?:boolean;conditioning_reused?:boolean};

export function GenerationTiming({job}:{job:Timing}){
  if(!job.stage_seconds)return null;
  const times=[['Model setup',(job.stage_seconds.loading||0)+(job.stage_seconds.loading_model||0)],['Prompt & input processing',job.encoding_seconds],['Image generation',job.stage_seconds.generating],['Image decoding',job.stage_seconds.decoding]] as const;
  return <details className="disclosure generation-timing"><summary>Generation time{job.model_reused?<span className="summary-hint">Model reused</span>:null}</summary><dl className="gallery-parameters">{times.filter(([,value])=>value!==undefined).map(([label,value])=><div key={label}><dt>{label}</dt><dd>{value!<0.1?'< 0.1':value!.toFixed(1)}s</dd></div>)}</dl>{job.conditioning_reused?<p>Reused the same prompt and input images.</p>:null}</details>;
}
