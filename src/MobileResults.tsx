import {UseImageAs} from './UseImageAs';
import {Image as ImageIcon,Expand,RotateCcw} from 'lucide-react';

type Result={id:string;image_url?:string;request:{prompt:string}};
/** Nearby results never move focus or navigate away from the current draft. */
export function MobileResults<T extends Result>({results,onOpen,onUseSettings,onGallery,disabled=false}:{results:T[];onOpen:(url:string)=>void;onUseSettings:(job:T)=>void;onGallery:()=>void;disabled?:boolean}){
  return <section className="mobile-results" aria-label="Recent images">
    <div className="mobile-results-heading"><h3>Recent images</h3><button type="button" onClick={onGallery}>View all</button></div>
    {results.length?<div className="mobile-result-strip">{results.slice(0,10).map((item,index)=><div className="mobile-result-card" key={item.id}><button type="button" className="mobile-result-open" onClick={()=>onOpen(item.image_url!)} aria-label={`Open image ${index+1}`}>
      <img loading="lazy" src={`/api/jobs/${item.id}/thumbnail`} alt={item.request.prompt}/><span><Expand size={13}/>{index===0?'Latest':'Open'}</span>
    </button><button type="button" className="mobile-result-restore" disabled={disabled} onClick={()=>onUseSettings(item)} aria-label={`Use settings from image ${index+1}`} title="Use settings"><RotateCcw size={17}/></button><UseImageAs jobId={item.id} compact disabled={disabled}/></div>)}</div>:<div className="mobile-results-empty"><ImageIcon size={20}/><span>Your images will appear here.</span></div>}
  </section>;
}
