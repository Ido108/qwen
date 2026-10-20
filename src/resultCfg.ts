/** Read only the sampler's saved output. Editable request defaults are not evidence. */
export function resultCfg(run:{true_cfg_scale?:number;cfg?:number;guidance?:number}):number|null{
  return [run.true_cfg_scale,run.cfg,run.guidance].find(value=>typeof value==='number'&&Number.isFinite(value)&&value>=0)??null;
}
