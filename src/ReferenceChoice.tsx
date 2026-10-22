import {useLayoutEffect,useId,useRef,useState} from 'react';
import {createPortal} from 'react-dom';
import {Check,ChevronDown,UserRound} from 'lucide-react';

export function ReferenceChoice({personNumber,references,value,onChange,disabled=false,sourceLabel='Source',optionLabel='Person'}:{personNumber:number;references:{id:string;url:string;count?:number}[];value:string|null;onChange:(value:string|null)=>void;disabled?:boolean;sourceLabel?:string;optionLabel?:string}){
  const [open,setOpen]=useState(false);
  const [position,setPosition]=useState({left:0,top:0,width:230});
  const trigger=useRef<HTMLButtonElement>(null);
  const menu=useRef<HTMLDivElement>(null);
  const id=useId();
  const selected=references.findIndex(reference=>reference.id===value);
  const options=[{id:null,url:'',label:'Keep person',count:0},...references.map((reference,index)=>({...reference,label:`${optionLabel} ${index+1}`}))];
  function choose(next:string|null){onChange(next);setOpen(false);trigger.current?.focus();}
  useLayoutEffect(()=>{
    if(!open)return;
    const reposition=()=>{
      const bounds=trigger.current!.getBoundingClientRect();
      const width=Math.min(260,window.innerWidth-24);
      const height=Math.min(options.length*47+10,280);
      setPosition({left:Math.max(12,Math.min(bounds.left,window.innerWidth-width-12)),top:bounds.bottom+height<window.innerHeight-12?bounds.bottom+6:Math.max(12,bounds.top-height-6),width});
    };
    reposition();
    const frame=requestAnimationFrame(()=>menu.current?.querySelector<HTMLButtonElement>('[aria-selected="true"]')?.focus({preventScroll:true}));
    const outside=(event:PointerEvent)=>{if(!menu.current?.contains(event.target as Node)&&!trigger.current?.contains(event.target as Node))setOpen(false);};
    const close=()=>setOpen(false);
    const scroll=(event:Event)=>{if(!menu.current?.contains(event.target as Node))reposition();};
    document.addEventListener('pointerdown',outside);window.addEventListener('resize',close);document.addEventListener('scroll',scroll,true);
    return()=>{cancelAnimationFrame(frame);document.removeEventListener('pointerdown',outside);window.removeEventListener('resize',close);document.removeEventListener('scroll',scroll,true);};
  },[open,options.length]);
  return <>
    <button ref={trigger} type="button" className="person-choice" disabled={disabled} aria-label={`Replacement for ${sourceLabel} ${personNumber}`} aria-haspopup="listbox" aria-expanded={open} aria-controls={open?id:undefined} onClick={()=>setOpen(!open)} onKeyDown={event=>{if(event.key==='ArrowDown'||event.key==='ArrowUp'){event.preventDefault();setOpen(true);}}}>
      {selected>=0?<img src={references[selected].url} alt={`Selected ${optionLabel} ${selected+1}`}/>:<UserRound size={17}/>}
      <span>{selected>=0?`${optionLabel} ${selected+1}`:'Keep person'}</span><ChevronDown size={13}/>
    </button>
    {open&&!disabled?createPortal(<div ref={menu} id={id} role="listbox" aria-label={`Choose replacement for ${sourceLabel} ${personNumber}`} className="person-choice-menu" style={position} onKeyDown={event=>{
      const buttons=Array.from(menu.current!.querySelectorAll<HTMLButtonElement>('[role="option"]'));
      const current=buttons.indexOf(document.activeElement as HTMLButtonElement);
      if(event.key==='Escape'){event.preventDefault();setOpen(false);trigger.current?.focus();}
      if(event.key==='ArrowDown'||event.key==='ArrowUp'){event.preventDefault();buttons[(current+(event.key==='ArrowDown'?1:-1)+buttons.length)%buttons.length]?.focus();}
      if(event.key==='Home'||event.key==='End'){event.preventDefault();buttons[event.key==='Home'?0:buttons.length-1]?.focus();}
      if(event.key==='Tab')setOpen(false);
    }}>{options.map(option=><button type="button" role="option" key={option.id||'keep'} aria-selected={option.id===(selected>=0?value:null)} onClick={()=>choose(option.id)}>{option.url?<img src={option.url} alt=""/>:<span className="person-keep-thumb"><UserRound size={20}/></span>}<span>{option.label}</span>{option.count?<small className="person-choice-count">{option.count} {option.count===1?'image':'images'}</small>:null}{option.id===(selected>=0?value:null)?<Check size={15}/>:null}</button>)}</div>,document.body):null}
  </>;
}
