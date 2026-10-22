import {forwardRef,useId,useImperativeHandle,useRef,useState,type ReactNode,type TextareaHTMLAttributes} from 'react';
import {AtSign,Image as ImageIcon} from 'lucide-react';
import {mentionAt,type MentionOption} from './promptMentions';
import './prompt-input.css';

type Props=Omit<TextareaHTMLAttributes<HTMLTextAreaElement>,'value'|'onChange'> & {
  value:string;onValueChange:(value:string)=>void;mentions:MentionOption[];
  /** More controls beside Mention (for example the people a model knows). */
  tools?:ReactNode;
};

/** Native text editing keeps mobile selection, dictation, IME and undo intact. */
export const PromptInput=forwardRef<HTMLTextAreaElement,Props>(function PromptInput({value,onValueChange,mentions,tools,onKeyDown,onClick,onSelect,onBlur,...props},ref){
  const field=useRef<HTMLTextAreaElement>(null),menu=useRef<HTMLDivElement>(null),selecting=useRef(false);const id=useId();
  useImperativeHandle(ref,()=>field.current!);
  const [match,setMatch]=useState<ReturnType<typeof mentionAt>>(null),[active,setActive]=useState(0);
  const choices=match?mentions.filter(m=>`${m.tag} ${m.label} ${m.detail||''}`.toLocaleLowerCase().includes(match.query)):[];
  const used=mentions.filter(m=>!m.choose&&new RegExp(m.tag+'(?![\\p{L}\\p{N}_])','iu').test(value));
  function track(text:string,caret:number){setMatch(mentionAt(text,caret));setActive(0);}
  function choose(option:MentionOption){
    selecting.current=false;if(!match||props.disabled)return;option.choose?.();
    const tail=value.slice(match.end);const suffix=tail.startsWith(' ')?'':' ';
    const next=value.slice(0,match.start)+option.tag+suffix+tail;
    const caret=match.start+option.tag.length+suffix.length;
    onValueChange(next);setMatch(null);
    requestAnimationFrame(()=>{field.current?.focus({preventScroll:true});field.current?.setSelectionRange(caret,caret);setMatch(null);});
  }
  return <div className="prompt-editor">
    <textarea {...props} ref={field} value={value} aria-autocomplete="list" aria-controls={match?id:undefined} aria-expanded={!!match} aria-activedescendant={match&&choices.length?`${id}-${Math.min(active,choices.length-1)}`:undefined}
      onChange={e=>{onValueChange(e.target.value);track(e.target.value,e.target.selectionStart);}}
      onClick={e=>{track(e.currentTarget.value,e.currentTarget.selectionStart);onClick?.(e);}}
      onSelect={e=>{onSelect?.(e);}}
      onBlur={e=>{if(!selecting.current&&!menu.current?.contains(e.relatedTarget))setMatch(null);onBlur?.(e);}}
      onKeyDown={e=>{
        if(match&&!e.nativeEvent.isComposing){
          if(e.key==='Escape'){e.preventDefault();e.stopPropagation();setMatch(null);return;}
          if(choices.length&&(e.key==='ArrowDown'||e.key==='ArrowUp')){e.preventDefault();const next=(active+(e.key==='ArrowDown'?1:-1)+choices.length)%choices.length;setActive(next);document.getElementById(`${id}-${next}`)?.scrollIntoView({block:'nearest'});return;}
          if(choices.length&&(e.key==='Enter'||e.key==='Tab')&&!e.shiftKey&&!e.ctrlKey&&!e.metaKey){e.preventDefault();e.stopPropagation();choose(choices[Math.min(active,choices.length-1)]);return;}
        }
        onKeyDown?.(e);
      }}/>
    {match?<div ref={menu} id={id} className="prompt-mentions" role="listbox" aria-label="Images and people">
      <div className="prompt-mentions-heading"><AtSign size={14}/>Images and people</div>
      {choices.length?choices.map((option,index)=><button type="button" id={`${id}-${index}`} key={option.key||option.tag} role="option" aria-selected={index===active} disabled={props.disabled} onPointerDown={e=>{selecting.current=true;e.preventDefault();}} onPointerCancel={()=>{selecting.current=false;}} onMouseDown={e=>e.preventDefault()} onClick={()=>choose(option)}>
        {option.image?<img src={option.image} alt="" loading="lazy" decoding="async"/>:<span className="mention-placeholder"><ImageIcon size={18}/></span>}
        <span><strong>{option.label}</strong><small>{option.detail||option.tag}</small></span><b>{option.badge??option.tag}</b>
      </button>):<p>{mentions.length?'No matching image or person.':'Add an image or a person to mention it here.'}</p>}
    </div>:null}
    <div className="prompt-mention-tools"><button type="button" disabled={props.disabled} aria-label="Mention an image or person" title="Or type @ to choose an image or person" onClick={()=>{
      const el=field.current;if(!el)return;const start=el.selectionStart??value.length,end=el.selectionEnd??start;
      const prefix=start>0&&!/\s/.test(value[start-1])?' @':'@';const next=value.slice(0,start)+prefix+value.slice(end);
      onValueChange(next);requestAnimationFrame(()=>{el.focus({preventScroll:true});el.setSelectionRange(start+prefix.length,start+prefix.length);track(next,start+prefix.length);});
    }}><AtSign size={14}/>Mention</button>{tools}{used.length?<div className="prompt-used-mentions">{used.map(m=><button type="button" key={m.tag} title={`${m.tag}: ${m.label}`} disabled={props.disabled} onClick={()=>{const at=value.toLowerCase().indexOf(m.tag.toLowerCase());field.current?.focus({preventScroll:true});field.current?.setSelectionRange(at,at+m.tag.length);}}>{m.image?<img src={m.image} alt=""/>:null}<span>{m.label}</span><small>{m.tag}</small></button>)}</div>:null}</div>
  </div>;
});
