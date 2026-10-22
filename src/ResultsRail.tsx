import {UseImageAs} from './UseImageAs';
import {ArrowUpRight,ChevronsRight,FolderOpen,ImagePlus,PanelRightOpen,RotateCcw} from 'lucide-react';
import type {SavedJob} from './JobQueue';

type Result=SavedJob&{image_url?:string;width?:number;height?:number};

/** Recent image results beside the workspace, with running and waiting jobs on top. */
export function ResultsRail({open,onToggle,running,results,selected,onSelect,onUseAsInput,onUseSettings,onOpenGallery,onOpenFolder,onOpenQueue,onShowLive}:{open:boolean;onToggle:()=>void;running:SavedJob[];results:Result[];selected?:string;
  onSelect:(job:Result)=>void;onUseAsInput:(job:Result)=>void;onUseSettings:(job:Result)=>void;onOpenGallery:()=>void;onOpenFolder:()=>void;onOpenQueue:()=>void;onShowLive?:(id:string)=>void}){
  // A running or waiting job as a tile with each step's preview as it denoises.
  const tile=(job:SavedJob)=>{const live=job as SavedJob&{preview_url?:string;preview_updated_at?:number};const going=job.status==='running';
      return <div className="rail-item rail-live" key={job.id} title={job.request.prompt}>
        <div className="rail-thumb checker" role={going&&onShowLive?'button':undefined} tabIndex={going&&onShowLive?0:undefined} title={going?'Show on the big view':undefined} onClick={()=>{if(going)onShowLive?.(job.id);}} onKeyDown={e=>{if(going&&(e.key==='Enter'||e.key===' ')){e.preventDefault();onShowLive?.(job.id);}}}>{live.preview_url?<img src={`${live.preview_url}?v=${live.preview_updated_at}`} alt=""/>:null}
          <span className="rail-live-state">{going?(job.stage==='generating'?`Step ${job.step}/${job.total_steps}`:job.stage==='downloading_models'?'Downloading':job.message||'Preparing'):`Waiting · #${job.position??''}`}</span>
          {going&&job.stage==='generating'&&job.total_steps?<i className="rail-live-bar" style={{width:`${Math.round(100*job.step/job.total_steps)}%`}}/>:null}
        </div></div>;};
  // With Recent hidden, running and waiting jobs still show in a narrow column beside the result.
  if(!open&&running.length)return <aside className="results-rail queue-only" aria-label="Queue">
    <div className="rail-head"><h2>Queue</h2><span>{running.length}</span><button className="icon-button" onClick={onToggle} aria-label="Show recent results" title="Show recent results"><PanelRightOpen size={18}/></button></div>
    <div className="rail-grid rail-queue-grid">{running.map(tile)}</div><button className="text-button rail-more" onClick={onOpenQueue}>View queue</button>
  </aside>;
  // Closed: a slim column whose button sits where the open panel's Hide button is, so the panel opens in place.
  if(!open)return <aside className="results-rail rail-collapsed" aria-label="Recent results">
    <button className="icon-button rail-show" onClick={onToggle} aria-label="Show recent results" title="Show recent results">«</button>
  </aside>;
  return <aside className="results-rail" aria-label="Recent results">
    <div className="rail-head"><h2>Recent</h2><span>{results.length}</span><button className="icon-button" onClick={onToggle} aria-label="Hide recent results" title="Hide"><ChevronsRight size={18}/></button></div>
    {/* Running and waiting jobs are tiles in the grid, with each step's preview as it denoises. */}
    <div className="rail-grid">{running.map(tile)}
      {results.map(job=><div className={`rail-item ${selected===job.id?'selected':''}`} key={job.id}>
      <button className="rail-thumb checker" onClick={()=>onSelect(job)} title={job.request.prompt}><img src={`/api/jobs/${job.id}/thumbnail`} alt={job.request.prompt} loading="lazy"/></button>
      <div className="rail-actions"><UseImageAs jobId={job.id} compact/><button onClick={()=>onUseSettings(job)} title="Use settings" aria-label="Use settings"><RotateCcw size={15}/></button></div>
    </div>)}</div>
    {!results.length&&!running.length?<p className="rail-empty">Your images appear here as soon as they finish.</p>:null}
    <div className="rail-foot"><button className="text-button" onClick={onOpenFolder}><FolderOpen size={14}/>Folder</button><button className="text-button" onClick={onOpenGallery}>Gallery<ArrowUpRight size={14}/></button></div>
  </aside>;
}
