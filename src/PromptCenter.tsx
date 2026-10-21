import {useEffect,useState} from 'react';
import type {Api} from './ModelSettings';
import './prompt-center.css';

type Option={id:string;label:string;official:boolean};
type Architecture={arch:string;title:string;official_note:string;choice:string;options:Option[];fixed:boolean};
type Prompt={key:string;title:string;group:string;source:string;default:string;text:string;edited:boolean;note:string;read_only?:boolean};

/** Settings > Prompts: the enhancer of each image architecture, and every internal prompt the app sends.
    only: show just these prompt groups (the Training captions tab), without the enhancer choice. */
export function PromptCenter({api,only}:{api:Api;only?:string[]}){
  const [archs,setArchs]=useState<Architecture[]>([]);const [prompts,setPrompts]=useState<Prompt[]>([]);
  const [drafts,setDrafts]=useState<Record<string,string>>({});const [open,setOpen]=useState<string|null>(null);
  const [error,setError]=useState('');const [notice,setNotice]=useState('');const [busy,setBusy]=useState(false);
  useEffect(()=>{api('/api/enhancer-settings').then(r=>setArchs(r.architectures)).catch(e=>setError(e.message));
    api('/api/prompts').then(r=>setPrompts(r.prompts)).catch(e=>setError(e.message));},[api]);
  async function choose(arch:string,choice:string){
    setError('');setNotice('');
    try{setArchs((await api(`/api/enhancer-settings/${arch}`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({choice})})).architectures);setNotice('Saved.');}
    catch(e){setError((e as Error).message);}
  }
  async function save(prompt:Prompt,restore=false){
    setBusy(true);setError('');setNotice('');
    try{
      const next=restore?await api(`/api/prompts/${encodeURIComponent(prompt.key)}`,{method:'DELETE'})
        :await api(`/api/prompts/${encodeURIComponent(prompt.key)}`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:drafts[prompt.key]??prompt.text})});
      setPrompts(next.prompts);setDrafts(d=>{const copy={...d};delete copy[prompt.key];return copy;});
      setNotice(restore?`${prompt.title}: official text restored.`:`${prompt.title}: saved. The next job uses it.`);
    }catch(e){setError((e as Error).message);}finally{setBusy(false);}
  }
  const groups=[...new Set(prompts.map(p=>p.group))].filter(group=>only?only.includes(group):!group.toLowerCase().includes('caption'));
  return <section className="prompt-center">
    {only?null:<><h3>Prompt enhancement</h3>
    <p>Runs only when you check Enhance prompt. Each model type uses its own official enhancer on this computer. Change it here only if you want another one.</p>
    <div className="enhancer-rows">{archs.map(a=><div className="enhancer-row" key={a.arch}>
      <div><strong>{a.title}</strong><small>Official: {a.official_note}</small></div>
      {a.fixed?<span className="enhancer-fixed">Always on</span>:<select aria-label={`Enhancer for ${a.title}`} value={a.choice} onChange={e=>choose(a.arch,e.target.value)}>
        {a.options.map(o=><option key={o.id} value={o.id}>{o.label}</option>)}</select>}
    </div>)}</div>
    <h3>Internal prompts</h3>
    <p>Every text the app adds to your prompt or sends to a model. Official texts come from their sources unchanged. Your edits are kept separately, so updates never erase them. "Restore official" brings the source text back.</p></>}
    {error?<div className="message error" role="alert">{error}</div>:null}{notice?<div className="notice" role="status">{notice}</div>:null}
    {groups.map(group=><div className="prompt-group" key={group}>{only?.length===1?null:<h4>{group}</h4>}
      {prompts.filter(p=>p.group===group).map(p=>{const draft=drafts[p.key]??p.text;const changed=draft!==p.text;return <article className="prompt-item" key={p.key}>
        <button type="button" className="prompt-item-head" aria-expanded={open===p.key} onClick={()=>setOpen(open===p.key?null:p.key)}>
          <span>{p.title}</span>{p.edited?<em>Edited</em>:null}{p.read_only?<em className="muted">Read only</em>:null}<span className="prompt-toggle">{open===p.key?'Hide':'Show'}</span>
        </button>
        {open===p.key?<div className="prompt-item-body">
          {p.source?(p.source.startsWith('http')?<a href={p.source} target="_blank" rel="noreferrer">Source</a>:<small>Source: {p.source}</small>):null}
          {p.note?<p className="prompt-note">{p.note}</p>:null}
          {p.read_only?(p.default?<pre className="prompt-text">{p.default}</pre>:null):<>
            <textarea dir="auto" value={draft} rows={Math.min(24,Math.max(6,draft.split('\n').length+1))} onChange={e=>setDrafts(d=>({...d,[p.key]:e.target.value}))}/>
            <div className="prompt-actions"><button type="button" className="button secondary" disabled={busy||!changed||!draft.trim()} onClick={()=>save(p)}>Save</button>
              <button type="button" className="text-button" disabled={busy||!p.edited} onClick={()=>save(p,true)}>Restore official</button>
              {changed?<button type="button" className="text-button" onClick={()=>setDrafts(d=>{const copy={...d};delete copy[p.key];return copy;})}>Cancel</button>:null}</div>
          </>}
        </div>:null}
      </article>;})}
    </div>)}
  </section>;
}
