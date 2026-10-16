import {useEffect,useRef} from 'react';
import {Download,Loader2,X} from 'lucide-react';
import type {Api,ModelState} from './ModelSettings';

const gb=(bytes:number)=>`${(bytes/1e9).toFixed(bytes>=1e10?0:1)} GB`;

/** A small live indicator for the model download that runs in the background, on every page. */
export function DownloadPill({api,state}:{api:Api;state:ModelState|null}){
  const transfer=state?.transfer;const running=transfer?.status==='running';
  const done=transfer?.completed_bytes||0;const total=transfer?.total_bytes||0;
  const samples=useRef<{t:number;b:number}[]>([]);
  useEffect(()=>{if(!running){samples.current=[];return;}samples.current=[...samples.current.filter(x=>Date.now()-x.t<10000),{t:Date.now(),b:done}];},[done,running]);
  if(!running)return null;
  const first=samples.current[0],last=samples.current[samples.current.length-1];
  const speed=first&&last&&last.t>first.t?(last.b-first.b)/((last.t-first.t)/1000):0;
  const left=speed>0&&total?Math.round((total-done)/speed):0;
  const name=transfer?.title||state?.models.find(m=>m.id===transfer?.model_id)?.title||'Model';
  const percent=total?Math.round(100*done/total):0;
  return <div className="download-pill" role="status" aria-live="polite">
    {transfer?.stage==='downloading'?<Download size={14}/>:<Loader2 className="spin" size={14}/>}
    <div className="download-pill-text"><b title={name}>{name}</b>
      <span>{transfer?.stage==='verifying'?'Checking file':transfer?.stage==='preparing'||!total?'Starting':`${percent}% · ${gb(done)} of ${gb(total)}${speed>0?` · ${(speed/1e6).toFixed(0)} MB/s · ${left>=60?`${Math.floor(left/60)}m ${String(left%60).padStart(2,'0')}s`:`${left}s`} left`:''}`}</span>
      <progress value={done} max={total||1}/></div>
    <button type="button" className="icon-button" aria-label="Cancel download" title="Cancel download" onClick={()=>api('/api/models/download/cancel',{method:'POST'}).catch(()=>{})}><X size={14}/></button>
  </div>;
}
