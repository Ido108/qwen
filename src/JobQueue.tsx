import {useCallback,useEffect,useRef,useState} from 'react';
import {ArrowUp,Check,Clock,Film,Image,Loader2,Pause,Play,RotateCcw,X,ListOrdered} from 'lucide-react';
import type {Api} from './ModelSettings';

export type SavedJob={id:string;kind?:string;status:string;stage:string;message:string;request:Record<string,any>;seed:number;created_at:number;started_at?:number;finished_at?:number;elapsed_seconds?:number;step:number;total_steps:number;position?:number;width?:number;height?:number;frames?:number;duration_seconds?:number;image_url?:string;video_url?:string;poster_url?:string;rewritten_prompt?:string;actual_prompt?:string;[key:string]:any};
export type QueueState={paused:boolean;active:SavedJob|null;pending:SavedJob[];recent:SavedJob[]};
export function useQueue(api:Api,enabled:boolean){
  const [state,setState]=useState<QueueState>({paused:false,active:null,pending:[],recent:[]});
  const [error,setError]=useState('');
  const refresh=useCallback(async()=>{try{setState(await api('/api/queue'));setError('');}catch(e){setError((e as Error).message);}},[api]);
  useEffect(()=>{if(!enabled)return;let alive=true;let timer:number;async function poll(){try{const next=await api('/api/queue');if(alive){setState(next);setError('');}}catch(e){if(alive)setError((e as Error).message);}if(alive)timer=window.setTimeout(poll,1500);}poll();return()=>{alive=false;clearTimeout(timer);};},[api,enabled]);
  return {state,error,refresh,count:state.pending.length+(state.active?1:0),revision:state.recent.map(j=>j.id+':'+j.status).join(',')};
}

export function JobQueue({api,state,error,refresh,open,onOpen,onRestore,onView}:{api:Api;state:QueueState;error:string;refresh:()=>Promise<void>;open:boolean;onOpen:(value:boolean)=>void;onRestore:(job:SavedJob)=>void;onView:(job:SavedJob)=>void}){
  const dialog=useRef<HTMLDialogElement>(null);const [acting,setActing]=useState('');const [actionError,setActionError]=useState('');
  const total=state.pending.length+(state.active?1:0);
  useEffect(()=>{if(open&&!dialog.current?.open)dialog.current?.showModal();else if(!open)dialog.current?.close();},[open]);
  async function action(path:string,body?:unknown){setActing(path);setActionError('');try{await api(path,{method:'POST',headers:{'Content-Type':'application/json'},...(body?{body:JSON.stringify(body)}:{})});await refresh();}catch(e){setActionError((e as Error).message);}finally{setActing('');}}
  function row(job:SavedJob,waiting=false){const active=job.id===state.active?.id;const done=job.status==='succeeded';return <article className={`queue-job ${active?'current':''}`} key={job.id} data-job-id={job.id}>
    <div className="queue-job-icon">{active?<Loader2 className="spin" size={20}/>:waiting?<span>{job.position}</span>:done?<Check size={20}/>:job.kind==='video'?<Film size={20}/>:<Image size={20}/>}</div>
    <div className="queue-job-copy"><strong>{job.kind==='video'?'Video':'Image'}<span>{waiting?'Waiting':active?(job.status==='succeeded'?'Finishing':job.message):done?'Completed':job.status==='cancelled'?'Cancelled':'Failed'}</span></strong><p dir="auto">{job.request.prompt}</p><small>{job.request.model_id} · {job.total_steps} steps{job.width?` · ${job.width} × ${job.height}`:''}</small>{active?<progress value={job.stage==='generating'?job.step:undefined} max={job.total_steps}/>:null}{job.status==='failed'?<p className="queue-failure">{job.message}</p>:null}
      <div className="queue-job-actions"><button className="text-button" onClick={()=>{onRestore(job);onOpen(false);}}>Use settings</button>{done?<button className="text-button" onClick={()=>{onView(job);onOpen(false);}}>View result</button>:null}{!active&&!waiting?<button className="text-button" disabled={!!acting} onClick={()=>action(`/api/jobs/${job.id}/retry`)}><RotateCcw size={13}/>{done?'Run again':'Retry'}</button>:null}</div>
    </div><div className="queue-row-tools">{waiting&&job.position!>1?<button className="icon-button" disabled={!!acting} aria-label={`Move job ${job.position} up`} onClick={()=>action(`/api/jobs/${job.id}/move`,{position:job.position!-1})}><ArrowUp size={17}/></button>:null}{active||waiting?<button className="icon-button" disabled={!!acting} aria-label={waiting?`Cancel queued ${job.kind||'image'} ${job.position}`:'Cancel running job'} onClick={()=>action(`/api/jobs/${job.id}/cancel`)}><X size={18}/></button>:null}</div>
  </article>;}
  return <>{total||state.paused||error?<div className="queue-strip"><button onClick={()=>onOpen(true)}><ListOrdered size={18}/><strong>{state.paused?'Queue paused':state.active?'Generating':'Queue'}</strong><span>{state.active?.request.prompt||`${state.pending.length} waiting`}</span><span className="queue-count">{total}</span><span className="queue-strip-link">View queue</span></button>{error?<small>Reconnecting…</small>:null}</div>:null}
    <dialog className="queue-dialog" aria-label="Queue" ref={dialog} onCancel={()=>onOpen(false)} onClick={e=>{if(e.target===dialog.current)onOpen(false);}}><div className="queue-dialog-heading"><div><h2>Queue <span>{total}</span></h2><p>Keep creating. Jobs run one at a time on your GPU.</p></div><button className="icon-button" aria-label="Close queue" onClick={()=>onOpen(false)}><X size={21}/></button></div>
    <div className="queue-controls"><span><Clock size={15}/>{state.pending.length} waiting{state.active?' · 1 running':''}</span><button className="button secondary" disabled={!!acting} onClick={()=>action('/api/queue/control',{paused:!state.paused})}>{state.paused?<Play size={15}/>:<Pause size={15}/>} {state.paused?'Resume queue':'Pause queue'}</button></div>{state.paused?<p className="queue-note">Waiting jobs stay saved. The current job continues.</p>:null}{actionError?<div className="message error" role="alert">{actionError}</div>:null}
    {state.active?row(state.active):null}{state.pending.map(j=>row(j,true))}{!total?<div className="queue-empty"><ListOrdered size={32}/><strong>No waiting jobs</strong><p>Generate an image or video to add it here.</p></div>:null}
    {state.recent.length?<><h3 className="queue-recent-heading">Recent jobs</h3>{state.recent.map(j=>row(j))}</>:null}
    </dialog></>;
}
