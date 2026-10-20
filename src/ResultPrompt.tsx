import {useEffect,useId,useRef,useState} from 'react';
import {Copy} from 'lucide-react';

/** Read only saved result text. Editing the next request cannot change this. */
export function ResultPrompt({text}:{text?:string}){
  const field=useRef<HTMLParagraphElement>(null),id=useId();
  const [expanded,setExpanded]=useState(false),[overflow,setOverflow]=useState(false),[copied,setCopied]=useState('');
  useEffect(()=>{setExpanded(false);setCopied('');},[text]);
  useEffect(()=>{
    const node=field.current;if(!node||expanded)return;
    const check=()=>setOverflow(node.scrollHeight>node.clientHeight+1);
    check();const observer=new ResizeObserver(check);observer.observe(node);return()=>observer.disconnect();
  },[text,expanded]);
  if(!text?.trim())return null;
  async function copy(){try{await navigator.clipboard.writeText(text!);setCopied('Copied');}catch{setCopied('Could not copy');}}
  return <section className="result-prompt" aria-label="Saved prompt" onKeyDown={e=>e.stopPropagation()}>
    <div className="result-prompt-heading"><strong>Prompt</strong><button type="button" onClick={copy} aria-label="Copy prompt"><Copy size={13}/>{copied||'Copy'}</button></div>
    <p id={id} ref={field} dir="auto" className={expanded?'expanded':''}>{text}</p>
    {overflow||expanded?<button type="button" className="result-prompt-expand" aria-expanded={expanded} aria-controls={id} onClick={()=>setExpanded(value=>!value)}>{expanded?'Show less':'Show more'}</button>:null}
  </section>;
}
