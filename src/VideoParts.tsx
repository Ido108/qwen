import {useState} from 'react';
import type {SavedJob} from './JobQueue';
import {formatClipTime} from './referenceTiming';
import './video-parts.css';

export type VideoPart={index:number;start_seconds:number;end_seconds:number};
type Part=VideoPart;

/** The saved parts of a long video: ready parts while it runs, folded under the whole video when it is done. */
export function VideoParts({job,open=false,active,onPlay}:{job:SavedJob;open?:boolean;active?:number|null;onPlay?:(part:VideoPart)=>void}){
  const parts:Part[]=Array.isArray(job.parts)?job.parts:[];
  const [playing,setPlaying]=useState<number|null>(null);
  if(!parts.length)return null;
  const total=Number(job.total_parts)||parts.length;
  const done=job.status==='succeeded';
  const shown=onPlay?undefined:parts.find(part=>part.index===playing);
  const current=onPlay?active:playing;
  return <details className="disclosure video-parts" open={open}>
    <summary>{done?`Parts of this video (${parts.length})`:`Ready parts: ${parts.length} of ${total}`}</summary>
    <div className="details-body">
      {shown?<figure className="video-part-player">
        <video key={shown.index} controls autoPlay playsInline src={`/api/jobs/${job.id}/parts/${shown.index}`} poster={`/api/jobs/${job.id}/parts/${shown.index}/poster`}/>
        <figcaption>Part {shown.index} · {formatClipTime(shown.start_seconds)} to {formatClipTime(shown.end_seconds)}<a className="text-button" href={`/api/jobs/${job.id}/parts/${shown.index}?download=true`}>Download part</a><button type="button" className="text-button" onClick={()=>setPlaying(null)}>Close</button></figcaption>
      </figure>:null}
      <ol className="video-part-grid">{parts.map(part=><li key={part.index}>
        <button type="button" className={current===part.index?'selected':''} aria-pressed={current===part.index} onClick={()=>onPlay?onPlay(part):setPlaying(part.index)} title={`Play part ${part.index}`}>
          <img src={`/api/jobs/${job.id}/parts/${part.index}/poster`} alt="" loading="lazy"/>
          <span>Part {part.index}<small>{formatClipTime(part.start_seconds)} to {formatClipTime(part.end_seconds)}</small></span>
        </button>
      </li>)}</ol>
    </div>
  </details>;
}
