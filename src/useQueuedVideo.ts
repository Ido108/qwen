import {useEffect,useRef} from 'react';
import type {Api} from './ModelSettings';
import type {SavedJob} from './JobQueue';

/** Reattach after reload and follow the next job without replacing the active preview. */
export function useQueuedVideo(api:Api,kind:string,currentId:string|null,onFound:(job:SavedJob)=>void){
  const callback=useRef(onFound);callback.current=onFound;
  useEffect(()=>{
    if(currentId)return;
    let alive=true,timer:number;
    async function check(){
      try{const queue=await api('/api/queue');if(!alive)return;const job=queue.active?.kind===kind?queue.active:queue.pending?.find((item:SavedJob)=>item.kind===kind);if(job){callback.current(job);return;}}catch{}
      if(alive)timer=window.setTimeout(check,2000);
    }
    check();return()=>{alive=false;clearTimeout(timer);};
  },[api,kind,currentId]);
}
