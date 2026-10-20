import {useEffect,useRef} from 'react';
import {Download,Loader2,X} from 'lucide-react';
import type {Api,ModelState} from './ModelSettings';

const gb=(bytes:number)=>`${(bytes/1e9).toFixed(bytes>=1e10?0:1)} GB`;
type Running={key:string;name:string;stage?:string;completed_bytes?:number;total_bytes?:number;cancel:string};

function PillRow({api,item}:{api:Api;item:Running}){
  const done=item.completed_bytes||0;const total=item.total_bytes||0;
  const samples=useRef<{t:number;b:number}[]>([]);
  useEffect(()=>{samples.current=[...samples.current.filter(x=>Date.now()-x.t<10000),{t:Date.now(),b:done}];},[done]);
  const first=samples.current[0],last=samples.current[samples.current.length-1];
  const speed=first&&last&&last.t>first.t?(last.b-first.b)/((last.t-first.t)/1000):0;
  const left=speed>0&&total?Math.round((total-done)/speed):0;
  const percent=total?Math.round(100*done/total):0;
  return <div className="download-pill" role="status" aria-live="polite">
    {item.stage==='downloading'?<Download size={14}/>:<Loader2 className="spin" size={14}/>}
    <div className="download-pill-text"><b title={item.name}>{item.name}</b>
      <span>{item.stage==='verifying'?'Checking file':item.stage==='preparing'||!total?'Starting':`${percent}% · ${gb(done)} of ${gb(total)}${speed>0?` · ${(speed/1e6).toFixed(0)} MB/s · ${left>=60?`${Math.floor(left/60)}m ${String(left%60).padStart(2,'0')}s`:`${left}s`} left`:''}`}</span>
      <progress value={done} max={total||1}/></div>
    <button type="button" className="icon-button" aria-label={`Cancel download of ${item.name}`} title="Cancel download" onClick={()=>api(item.cancel,{method:'POST'}).catch(()=>{})}><X size={14}/></button>
  </div>;
}

/** Live indicators for the model downloads that run in the background, on every page: one per download. */
export function DownloadPill({api,state}:{api:Api;state:ModelState|null}){
  const transfer=state?.transfer;
  const running:Running[]=[
    ...(transfer?.status==='running'?[{key:'models',name:transfer.title||state?.models.find(m=>m.id===transfer.model_id)?.title||'Model',stage:transfer.stage,completed_bytes:transfer.completed_bytes,total_bytes:transfer.total_bytes,cancel:'/api/models/download/cancel'}]:[]),
    ...(state?.downloads||[]).filter(d=>d.status==='running').map(d=>({key:d.requested_id,name:d.title||'Civitai model',stage:d.stage,completed_bytes:d.completed_bytes,total_bytes:d.total_bytes,cancel:`/api/civitai/install/${d.requested_id.split(':')[1]}/cancel`}))];
  if(!running.length)return null;
  return <details className="download-pills"><summary><Download size={14}/>{running.length} {running.length===1?'download':'downloads'} in progress</summary>{running.map(item=><PillRow key={item.key} api={api} item={item}/>)}</details>;
}
