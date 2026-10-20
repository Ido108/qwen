import {useEffect,useRef,useState,type PointerEvent as ReactPointerEvent,type MouseEvent as ReactMouseEvent} from 'react';
import {Minus,Plus,Maximize} from 'lucide-react';
import {ImageLoadState,useResultImage} from './ProgressiveImage';

/** Image viewer: pinch, wheel, double-tap, pan and explicit zoom controls.
 * At 100%, a one-finger horizontal swipe calls onSwipe; zoomed in, one finger pans. A pinch never swipes. */
export function ZoomView({src,alt,className,onSwipe}:{src:string;alt:string;className?:string;onSwipe?:(step:1|-1)=>void}){
  const box=useRef<HTMLDivElement>(null);
  const img=useRef<HTMLImageElement>(null);
  const [view,setView]=useState({scale:1,x:0,y:0});
  const image=useResultImage(src,view.scale>1);
  const current=useRef(view);current.current=view;
  const pointers=useRef(new Map<number,{x:number;y:number}>());
  const gesture=useRef<{x:number;y:number;distance:number;view:typeof view}|null>(null);
  const lastTap=useRef({at:0,x:0,y:0});
  const lastTouch=useRef(0);
  const down=useRef({x:0,y:0,at:0,moved:false});
  const [failed,setFailed]=useState(false);
  const touchedControl=useRef(0);
  const [swipe,setSwipe]=useState(0);
  const pinched=useRef(false);
  // A browser can suppress the compatibility click immediately after a touch pan.
  // Handle a completed touch directly; keep click for mouse and keyboard activation.
  function control(action:()=>void){return {
    onPointerUp:(e:ReactPointerEvent<HTMLButtonElement>)=>{if(e.pointerType==='mouse')return;e.preventDefault();e.stopPropagation();touchedControl.current=performance.now();action();},
    onClick:(e:ReactMouseEvent<HTMLButtonElement>)=>{if(e.detail===0||performance.now()-touchedControl.current>600)action();},
  };}
  useEffect(()=>{setView({scale:1,x:0,y:0});setSwipe(0);pointers.current.clear();gesture.current=null;setFailed(false);},[src]);
  function bound(next:typeof view){
    const el=box.current,image=img.current;
    if(next.scale<=1)return {scale:1,x:0,y:0};
    if(!el||!image||!image.naturalWidth||!image.naturalHeight)return next;
    const fit=Math.min(image.clientWidth/image.naturalWidth,image.clientHeight/image.naturalHeight);
    const maxX=Math.max(0,(image.naturalWidth*fit*next.scale-el.clientWidth)/2);
    const maxY=Math.max(0,(image.naturalHeight*fit*next.scale-el.clientHeight)/2);
    return {...next,x:Math.max(-maxX,Math.min(maxX,next.x)),y:Math.max(-maxY,Math.min(maxY,next.y))};
  }
  function zoom(scale:number,clientX?:number,clientY?:number){
    const r=box.current?.getBoundingClientRect();if(!r)return;
    const old=current.current,px=(clientX??r.left+r.width/2)-r.left-r.width/2,py=(clientY??r.top+r.height/2)-r.top-r.height/2;
    scale=Math.min(8,Math.max(1,scale));const ratio=scale/old.scale;
    const next=bound({scale,x:px-(px-old.x)*ratio,y:py-(py-old.y)*ratio});current.current=next;setView(next);
  }
  function startGesture(){
    const points=[...pointers.current.values()];if(!points.length){gesture.current=null;return;}
    const a=points[0],b=points[1]??a;
    gesture.current={x:(a.x+b.x)/2,y:(a.y+b.y)/2,distance:Math.hypot(a.x-b.x,a.y-b.y),view:current.current};
  }
  useEffect(()=>{
    const element=box.current;if(!element)return;
    function wheel(e:WheelEvent){
      if((e.target as Element).closest('button'))return;
      e.preventDefault();
      zoom(current.current.scale*Math.exp(-e.deltaY*0.0015),e.clientX,e.clientY);
    }
    element.addEventListener('wheel',wheel,{passive:false});
    return()=>element.removeEventListener('wheel',wheel);
  },[]);
  return <div ref={box} className={`zoom-view ${view.scale>1?'zoomed':''} ${className||''}`} tabIndex={0} aria-label="Image viewer"
    onKeyDown={e=>{if(e.target!==e.currentTarget)return;if(['+','=','-','0'].includes(e.key)){e.preventDefault();zoom(e.key==='0'?1:current.current.scale*(e.key==='-'?1/1.5:1.5));}}}
    onPointerDown={e=>{
      if((e.target as Element).closest('button'))return;
      e.currentTarget.setPointerCapture(e.pointerId);pointers.current.set(e.pointerId,{x:e.clientX,y:e.clientY});
      if(pointers.current.size===1)pinched.current=false;else pinched.current=true;
      down.current={x:e.clientX,y:e.clientY,at:Date.now(),moved:pointers.current.size>1};startGesture();
    }}
    onPointerMove={e=>{
      if(!pointers.current.has(e.pointerId)||!gesture.current)return;
      pointers.current.set(e.pointerId,{x:e.clientX,y:e.clientY});
      if(Math.hypot(e.clientX-down.current.x,e.clientY-down.current.y)>6)down.current.moved=true;
      if(onSwipe&&e.pointerType!=='mouse'&&!pinched.current&&current.current.scale<=1){
        const dx=e.clientX-down.current.x,dy=e.clientY-down.current.y;
        setSwipe(Math.abs(dx)>Math.abs(dy)?dx:0);return;
      }
      const points=[...pointers.current.values()],a=points[0],b=points[1]??a,g=gesture.current;
      const x=(a.x+b.x)/2,y=(a.y+b.y)/2;
      const scale=g.distance?Math.min(8,Math.max(1,g.view.scale*Math.hypot(a.x-b.x,a.y-b.y)/g.distance)):g.view.scale;
      const r=e.currentTarget.getBoundingClientRect(),px=g.x-r.left-r.width/2,py=g.y-r.top-r.height/2,ratio=scale/g.view.scale;
      const next=bound({scale,x:px-(px-g.view.x)*ratio+x-g.x,y:py-(py-g.view.y)*ratio+y-g.y});current.current=next;setView(next);
    }}
    onPointerUp={e=>{
      if(!pointers.current.has(e.pointerId))return;
      if(e.pointerType!=='mouse')lastTouch.current=Date.now();
      if(e.pointerType!=='mouse'&&!down.current.moved&&Date.now()-down.current.at<300){
        const last=lastTap.current;
        if(Date.now()-last.at<350&&Math.hypot(e.clientX-last.x,e.clientY-last.y)<30){zoom(current.current.scale>1?1:2.5,e.clientX,e.clientY);lastTap.current.at=0;}
        else lastTap.current={at:Date.now(),x:e.clientX,y:e.clientY};
      }
      if(onSwipe&&e.pointerType!=='mouse'&&!pinched.current&&current.current.scale<=1){
        const dx=e.clientX-down.current.x,dy=e.clientY-down.current.y;
        if(Math.abs(dx)>60&&Math.abs(dx)>Math.abs(dy)*1.5)onSwipe(dx<0?1:-1);
      }
      setSwipe(0);pointers.current.delete(e.pointerId);startGesture();
    }}
    onPointerCancel={e=>{setSwipe(0);pointers.current.delete(e.pointerId);startGesture();}}
    onDoubleClick={e=>{if(Date.now()-lastTouch.current>500&&!(e.target as Element).closest('button'))zoom(current.current.scale>1?1:2.5,e.clientX,e.clientY);}}>
    <img ref={img} key={image.url} src={image.url} alt={alt} draggable={false} decoding="async" data-quality={image.quality} onLoad={()=>setFailed(false)} onError={()=>setFailed(true)} style={{transform:`translate(${view.x+swipe}px,${view.y}px) scale(${view.scale})`}}/>
    <ImageLoadState {...image} fullSize={view.scale>1}/>
    {failed?<p className="viewer-error" role="alert">The image could not load. Close and open it again.</p>:null}
    <div className="zoom-controls" aria-label="Image zoom">
      <button type="button" aria-label="Zoom out" disabled={view.scale<=1} {...control(()=>zoom(current.current.scale/1.5))}><Minus size={18}/></button>
      <output aria-live="polite">{Math.round(view.scale*100)}%</output>
      <button type="button" aria-label="Zoom in" disabled={view.scale>=8} {...control(()=>zoom(current.current.scale*1.5))}><Plus size={18}/></button>
      <button type="button" aria-label="Fit image" {...control(()=>zoom(1))}><Maximize size={17}/><span>Fit</span></button>
    </div>
  </div>;
}
