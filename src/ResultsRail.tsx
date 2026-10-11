import {ArrowUpRight,ChevronsRight,FolderOpen,ImagePlus,Loader2,PanelRightOpen,RotateCcw} from 'lucide-react';
import type {SavedJob} from './JobQueue';

type Result=SavedJob&{image_url?:string;width?:number;height?:number};

/** Recent image results beside the workspace, with running and waiting jobs on top. */
export function ResultsRail({open,onToggle,running,results,selected,onSelect,onUseAsInput,onUseSettings,onOpenGallery,onOpenFolder,onOpenQueue}:{open:boolean;onToggle:()=>void;running:SavedJob[];results:Result[];selected?:string;
  onSelect:(job:Result)=>void;onUseAsInput:(job:Result)=>void;onUseSettings:(job:Result)=>void;onOpenGallery:()=>void;onOpenFolder:()=>void;onOpenQueue:()=>void}){
  if(!open)return <button className="rail-open" onClick={onToggle} aria-label="Show recent results" title="Show recent results"><PanelRightOpen size={18}/>{running.length?<span>{running.length}</span>:null}</button>;
  return <aside className="results-rail" aria-label="Recent results">
    <div className="rail-head"><h2>Recent</h2><span>{results.length}</span><button className="icon-button" onClick={onToggle} aria-label="Hide recent results" title="Hide"><ChevronsRight size={18}/></button></div>
    {running.length?<div className="rail-running">{running.slice(0,4).map(job=><div className="rail-job" key={job.id}>
      <Loader2 className={job.status==='running'?'spin':''} size={16}/><div><strong>{job.status==='running'?(job.stage==='generating'?`Step ${job.step}/${job.total_steps}`:job.message):`Waiting · #${job.position??''}`}</strong><p>{job.request.prompt}</p>
      {job.status==='running'&&job.stage==='generating'?<progress value={job.step} max={job.total_steps||1}/>:null}</div></div>)}{running.length>4?<button className="text-button rail-more" onClick={onOpenQueue}>+{running.length-4} more in queue</button>:null}</div>:null}
    <div className="rail-grid">{results.map(job=><div className={`rail-item ${selected===job.id?'selected':''}`} key={job.id}>
      <button className="rail-thumb checker" onClick={()=>onSelect(job)} title={job.request.prompt}><img src={`/api/jobs/${job.id}/thumbnail`} alt={job.request.prompt} loading="lazy"/></button>
      <div className="rail-actions"><button onClick={()=>onUseAsInput(job)} title="Use as input" aria-label="Use as input"><ImagePlus size={15}/></button><button onClick={()=>onUseSettings(job)} title="Use settings" aria-label="Use settings"><RotateCcw size={15}/></button></div>
    </div>)}</div>
    {!results.length&&!running.length?<p className="rail-empty">Your images appear here as soon as they finish.</p>:null}
    <div className="rail-foot"><button className="text-button" onClick={onOpenFolder}><FolderOpen size={14}/>Folder</button><button className="text-button" onClick={onOpenGallery}>Gallery<ArrowUpRight size={14}/></button></div>
  </aside>;
}
