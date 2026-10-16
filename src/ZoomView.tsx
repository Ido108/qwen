import {useEffect,useRef,useState} from 'react';

/** Image viewer: mouse wheel zooms at the pointer, drag moves, double-click resets. */
export function ZoomView({src,alt,className}:{src:string;alt:string;className?:string}){
  const box=useRef<HTMLDivElement>(null);
  const [view,setView]=useState({scale:1,x:0,y:0});
  const drag=useRef<{x:number;y:number;startX:number;startY:number}|null>(null);
  useEffect(()=>setView({scale:1,x:0,y:0}),[src]);
  useEffect(()=>{
    const element=box.current;if(!element)return;
    function wheel(e:WheelEvent){
      e.preventDefault();
      const rect=element!.getBoundingClientRect();
      const px=e.clientX-rect.left-rect.width/2,py=e.clientY-rect.top-rect.height/2;
      setView(old=>{
        const scale=Math.min(16,Math.max(1,old.scale*Math.exp(-e.deltaY*0.0015)));
        if(scale===1)return {scale:1,x:0,y:0};
        const ratio=scale/old.scale;
        // Keep the point under the pointer in place while zooming.
        return {scale,x:px-(px-old.x)*ratio,y:py-(py-old.y)*ratio};
      });
    }
    element.addEventListener('wheel',wheel,{passive:false});
    return()=>element.removeEventListener('wheel',wheel);
  },[]);
  return <div ref={box} className={`zoom-view ${view.scale>1?'zoomed':''} ${className||''}`}
    onPointerDown={e=>{if(view.scale<=1)return;(e.target as HTMLElement).setPointerCapture(e.pointerId);drag.current={x:view.x,y:view.y,startX:e.clientX,startY:e.clientY};}}
    onPointerMove={e=>{const d=drag.current;if(!d)return;setView(v=>({...v,x:d.x+e.clientX-d.startX,y:d.y+e.clientY-d.startY}));}}
    onPointerUp={()=>{drag.current=null;}} onDoubleClick={()=>setView({scale:1,x:0,y:0})}>
    <img src={src} alt={alt} draggable={false} style={{transform:`translate(${view.x}px,${view.y}px) scale(${view.scale})`}}/>
    {view.scale>1?<span className="zoom-badge">{Math.round(view.scale*100)}% · double-click to reset</span>:null}
  </div>;
}
