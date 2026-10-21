import {useEffect} from 'react';

type SeededJob={id:string;status:string;seed?:number|null;seed_used?:boolean};

/** The newest finished result whose seed was used (FLUX.3 Klein and upscalers report seed_used false). */
export function lastSeed(history:SeededJob[]){
  const job=history.find(item=>item.status==='succeeded'&&item.seed_used!==false&&Number.isInteger(item.seed));
  return job?{id:job.id,seed:job.seed as number}:null;
}

/** While Random seed is on, the seed field shows the newest result's seed, so turning Random off repeats it. */
export function useLastSeed(history:SeededJob[],apply:(seed:number)=>void){
  const newest=lastSeed(history);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(()=>{if(newest)apply(newest.seed);},[newest?.id]);
}
