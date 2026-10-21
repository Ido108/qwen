import {useEffect,useRef,useState} from 'react';
import {Loader2,Undo2,X} from 'lucide-react';

type Box={x:number;y:number;w:number;h:number};
const RED='rgb(239,68,68)';

/** Draw red boxes on an image (Remove object tool) and return the marked image as a PNG blob. */
export function BoxEditor({src,open,onClose,onApply,title='Mark the object to remove',hint='Drag a box around the object. One or two boxes work best.'}:{src:string;open:boolean;onClose:()=>void;onApply:(png:Blob)=>Promise<void>;title?:string;hint?:string}){
  const dialog=useRef<HTMLDialogElement>(null);const image=useRef<HTMLImageElement>(null);
  const [boxes,setBoxes]=useState<Box[]>([]);const [draft,setDraft]=useState<Box|null>(null);const [saving,setSaving]=useState(false);const [error,setError]=useState('');
  const start=useRef<{x:number;y:number}|null>(null);
  useEffect(()=>{if(open&&!dialog.current?.open){setBoxes([]);setDraft(null);setError('');dialog.current?.showModal();}else if(!open)dialog.current?.close();},[open]);
  function point(e:React.PointerEvent){const r=image.current!.getBoundingClientRect();return {x:Math.min(1,Math.max(0,(e.clientX-r.left)/r.width)),y:Math.min(1,Math.max(0,(e.clientY-r.top)/r.height))};}
  function down(e:React.PointerEvent){(e.target as HTMLElement).setPointerCapture(e.pointerId);start.current=point(e);setDraft({...start.current,w:0,h:0});}
  function move(e:React.PointerEvent){if(!start.current)return;const p=point(e),s=start.current;setDraft({x:Math.min(s.x,p.x),y:Math.min(s.y,p.y),w:Math.abs(p.x-s.x),h:Math.abs(p.y-s.y)});}
  function up(){if(draft&&draft.w>0.01&&draft.h>0.01)setBoxes(b=>[...b,draft]);start.current=null;setDraft(null);}
  async function apply(){
    setSaving(true);setError('');
    try{
      const picture=image.current!;const canvas=document.createElement('canvas');canvas.width=picture.naturalWidth;canvas.height=picture.naturalHeight;
      const context=canvas.getContext('2d')!;context.drawImage(picture,0,0);
      context.strokeStyle=RED;context.lineWidth=Math.max(4,Math.round(Math.max(canvas.width,canvas.height)*0.006));
      for(const b of boxes)context.strokeRect(b.x*canvas.width,b.y*canvas.height,b.w*canvas.width,b.h*canvas.height);
      const blob=await new Promise<Blob|null>(resolve=>canvas.toBlob(resolve,'image/png'));
      if(!blob)throw new Error('Could not create the marked image.');
      await onApply(blob);onClose();
    }catch(e){setError((e as Error).message);}finally{setSaving(false);}
  }
  const shown=draft?[...boxes,draft]:boxes;
  return <dialog ref={dialog} className="box-editor" aria-label={title} onCancel={onClose}>
    <div className="box-editor-head"><div><h2>{title}</h2><p>{hint}</p></div><button className="icon-button" onClick={onClose} aria-label="Close"><X size={20}/></button></div>
    <div className="box-editor-stage"><div className="box-editor-frame" onPointerDown={down} onPointerMove={move} onPointerUp={up}>
      <img ref={image} src={src} alt="Image to mark" draggable={false} crossOrigin="anonymous"/>
      {shown.map((b,i)=><span key={i} className="box-editor-box" style={{left:`${b.x*100}%`,top:`${b.y*100}%`,width:`${b.w*100}%`,height:`${b.h*100}%`}}/>)}
    </div></div>
    {error?<div className="message error"><span>{error}</span></div>:null}
    <div className="recent-inputs-actions"><span>{boxes.length?`${boxes.length} ${boxes.length===1?'box':'boxes'}`:'No box yet'}</span><button className="button secondary" disabled={!boxes.length} onClick={()=>setBoxes(b=>b.slice(0,-1))}><Undo2 size={16}/>Undo</button><button className="button primary" disabled={!boxes.length||saving} onClick={apply}>{saving?<Loader2 className="spin" size={16}/>:null}Use marked image</button></div>
  </dialog>;
}
