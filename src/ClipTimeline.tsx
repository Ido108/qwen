import {useEffect,useRef,useState,type KeyboardEvent,type PointerEvent} from 'react';
import {Play,RotateCcw} from 'lucide-react';
import type {UploadedVideo} from './mediaUpload';
import {formatClipTime,REFERENCE_FPS,REFERENCE_MIN_FRAMES,nearestReferenceFrames,sourceClipDefault} from './referenceTiming';

type Edge='start'|'end'|'move';
type Selection={start:number;end:number};
type Drag={edge:Edge;start:number;end:number;pointerTime:number};
const clamp=(value:number,min:number,max:number)=>Math.min(max,Math.max(min,value));
const snap=(seconds:number)=>Math.round(seconds*REFERENCE_FPS)/REFERENCE_FPS;

export function ClipTimeline({source,start,end,maxFrames,onChange,onPreview,onPlay}:{
  source:UploadedVideo;start:number;end:number;maxFrames:number;
  onChange:(selection:Selection)=>void;onPreview:(seconds:number)=>void;onPlay:()=>void;
}){
  const duration=source.duration_seconds||0;
  const minSeconds=REFERENCE_MIN_FRAMES/REFERENCE_FPS;
  const maxSeconds=maxFrames/REFERENCE_FPS;
  const track=useRef<HTMLDivElement>(null);
  const drag=useRef<Drag|null>(null);
  const [stripError,setStripError]=useState(false);
  const [startText,setStartText]=useState(start.toFixed(2));
  const [endText,setEndText]=useState(end.toFixed(2));
  useEffect(()=>setStripError(false),[source.id]);
  useEffect(()=>setStartText(start.toFixed(2)),[start]);
  useEffect(()=>setEndText(end.toFixed(2)),[end]);
  if(!duration||duration<minSeconds-1/REFERENCE_FPS){
    return <div className="clip-timeline clip-unavailable" role="status">{duration?'This video is too short for the model. Select at least 0.9 seconds.':'Reading the video duration…'}</div>;
  }
  const first=clamp(start,0,duration);
  const last=clamp(end,first,duration);
  const position=(clientX:number)=>{
    const bounds=track.current!.getBoundingClientRect();
    return clamp(snap((clientX-bounds.left)/bounds.width*duration),0,duration);
  };
  const moved=(active:Drag,pointerTime:number):Selection=>{
    if(active.edge==='start')return {start:clamp(pointerTime,Math.max(0,active.end-maxSeconds),active.end-minSeconds),end:active.end};
    if(active.edge==='end')return {start:active.start,end:clamp(pointerTime,active.start+minSeconds,Math.min(duration,active.start+maxSeconds))};
    const span=active.end-active.start;
    const nextStart=clamp(snap(active.start+pointerTime-active.pointerTime),0,Math.max(0,duration-span));
    return {start:nextStart,end:nextStart+span};
  };
  function pointerDown(event:PointerEvent<HTMLDivElement>){
    const pointerTime=position(event.clientX);
    const handle=(event.target as HTMLElement).closest('[data-edge]')?.getAttribute('data-edge');
    const edge:Edge=handle==='start'||handle==='end'?handle:'move';
    drag.current={edge,start:first,end:last,pointerTime};
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
    if(!handle&&(pointerTime<first||pointerTime>last)){
      const span=last-first;
      const nextStart=clamp(snap(pointerTime-span/2),0,Math.max(0,duration-span));
      drag.current={edge:'move',start:nextStart,end:nextStart+span,pointerTime};
      onChange({start:nextStart,end:nextStart+span});
    }
  }
  function pointerMove(event:PointerEvent<HTMLDivElement>){
    if(!drag.current)return;
    onChange(moved(drag.current,position(event.clientX)));
  }
  function pointerUp(event:PointerEvent<HTMLDivElement>){
    if(!drag.current)return;
    const selection=moved(drag.current,position(event.clientX));
    onChange(selection);
    onPreview(selection.start);
    drag.current=null;
  }
  function keyboard(event:KeyboardEvent<HTMLButtonElement>,edge:'start'|'end'){
    const direction=event.key==='ArrowLeft'?-1:event.key==='ArrowRight'?1:0;
    if(!direction)return;
    event.preventDefault();
    const delta=direction*(event.shiftKey?1:1/REFERENCE_FPS);
    const selection=edge==='start'
      ?{start:clamp(snap(first+delta),Math.max(0,last-maxSeconds),last-minSeconds),end:last}
      :{start:first,end:clamp(snap(last+delta),first+minSeconds,Math.min(duration,first+maxSeconds))};
    onChange(selection);
    onPreview(selection.start);
  }
  function commitText(edge:'start'|'end'){
    const value=Number(edge==='start'?startText:endText);
    if(Number.isFinite(value)){
      const selection=edge==='start'
        ?{start:clamp(snap(value),Math.max(0,last-maxSeconds),last-minSeconds),end:last}
        :{start:first,end:clamp(snap(value),first+minSeconds,Math.min(duration,first+maxSeconds))};
      onChange(selection);
      onPreview(selection.start);
      setStartText(selection.start.toFixed(2));
      setEndText(selection.end.toFixed(2));
      return;
    }
    setStartText(first.toFixed(2));setEndText(last.toFixed(2));
  }
  const reset=()=>{const selection=sourceClipDefault(duration,maxFrames);onChange(selection);onPreview(0);};
  const outputFrames=nearestReferenceFrames(last-first,maxFrames);
  return <div className="clip-timeline">
    <div className="clip-heading"><div><strong>Source segment</strong><span>Drag the edges to trim. Drag the selection to move it.</span></div><button className="text-button" onClick={reset}><RotateCcw size={14}/>Use from start</button></div>
    <div className="clip-time-ends"><span>00:00.0</span><span>{formatClipTime(duration)}</span></div>
    <div className="clip-track" ref={track} style={{backgroundImage:`url(${source.poster_url})`}} onPointerDown={pointerDown} onPointerMove={pointerMove} onPointerUp={pointerUp} onPointerCancel={()=>{drag.current=null;}}>
      <img className="clip-strip" src={stripError?source.poster_url:`${source.url}/timeline`} onError={()=>setStripError(true)} alt="Video frames across the source timeline" draggable={false}/>
      <div className="clip-dim clip-dim-before" style={{width:`${first/duration*100}%`}}/>
      <div className="clip-dim clip-dim-after" style={{left:`${last/duration*100}%`}}/>
      <div className="clip-selected" style={{left:`${first/duration*100}%`,width:`${(last-first)/duration*100}%`}}/>
      <button type="button" className="clip-handle start" data-edge="start" role="slider" aria-label="Trim start" aria-valuemin={Math.max(0,last-maxSeconds)} aria-valuemax={last-minSeconds} aria-valuenow={first} aria-valuetext={formatClipTime(first)} style={{left:`${first/duration*100}%`}} onKeyDown={event=>keyboard(event,'start')}><span/></button>
      <button type="button" className="clip-handle end" data-edge="end" role="slider" aria-label="Trim end" aria-valuemin={first+minSeconds} aria-valuemax={Math.min(duration,first+maxSeconds)} aria-valuenow={last} aria-valuetext={formatClipTime(last)} style={{left:`${last/duration*100}%`}} onKeyDown={event=>keyboard(event,'end')}><span/></button>
    </div>
    <div className="clip-controls"><label>Start <input type="number" min="0" max={duration-minSeconds} step="0.01" aria-label="Clip start in seconds" value={startText} onChange={event=>setStartText(event.target.value)} onBlur={()=>commitText('start')} onKeyDown={event=>{if(event.key==='Enter')event.currentTarget.blur();}}/><small>s</small></label><label>End <input type="number" min={minSeconds} max={duration} step="0.01" aria-label="Clip end in seconds" value={endText} onChange={event=>setEndText(event.target.value)} onBlur={()=>commitText('end')} onKeyDown={event=>{if(event.key==='Enter')event.currentTarget.blur();}}/><small>s</small></label><button className="button secondary clip-play" onClick={onPlay}><Play size={14}/>Play selection</button></div>
    <p className="clip-summary">Selected {formatClipTime(last-first)} of {formatClipTime(duration)} <span>·</span> Output {formatClipTime(outputFrames/REFERENCE_FPS)} ({outputFrames} frames) <span>·</span> Model limit {formatClipTime(maxSeconds)}</p>
  </div>;
}
