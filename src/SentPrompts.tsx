import {useState} from 'react';
import './prompt-center.css';

export type SentPrompt={label:string;text:string};

/** Everything the app actually sent to a model for this result: rewriter messages, templates and encoder input, in order. */
export function SentPrompts({items}:{items?:SentPrompt[]|null}){
  const [copied,setCopied]=useState<number|null>(null);
  if(!items?.length)return null;
  async function copy(index:number){try{await navigator.clipboard.writeText(items![index].text);setCopied(index);window.setTimeout(()=>setCopied(null),1500);}catch{/* the text stays selectable */}}
  return <section className="sent-prompts"><h3>Sent to the model</h3>
    <p>The full texts, in the order they were sent, including templates and system prompts added under your prompt.</p>
    {items.map((item,index)=><div className="sent-prompt" key={index}>
      <div className="sent-prompt-head"><strong>{item.label}</strong><button type="button" className="text-button" onClick={()=>copy(index)}>{copied===index?'Copied':'Copy'}</button></div>
      <pre dir="auto">{item.text}</pre>
    </div>)}
  </section>;
}
