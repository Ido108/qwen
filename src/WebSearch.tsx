import {useEffect,useState} from 'react';
import type {Api} from './ModelSettings';
import type {InputImage} from './InputLibrary';

type Hit={url:string;thumb:string;title:string;source:string;width?:number;height?:number};

/** Pictures from a web image search (Google Images through SerpAPI, as in Image Studio). A click imports the
 *  picture as an input; "+ Person" also makes it a new person in People. */
export function WebSearch({api,disabled,onUse,onPeople}:{api:Api;disabled:boolean;onUse:(image:InputImage)=>void;onPeople:()=>void}){
  const [ready,setReady]=useState<boolean|null>(null);const [key,setKey]=useState('');
  const [query,setQuery]=useState('');const [hits,setHits]=useState<Hit[]>([]);
  const [busy,setBusy]=useState('');const [message,setMessage]=useState('');
  useEffect(()=>{api('/api/web-search/status').then(r=>setReady(r.ready)).catch(()=>setReady(false));},[api]);
  async function saveKey(){setMessage('');try{const r=await api('/api/web-search/key',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({key})});setReady(r.ready);setKey('');}catch(e){setMessage((e as Error).message);}}
  async function search(){if(!query.trim())return;setBusy('search');setMessage('');
    try{const r=await api(`/api/web-search/images?q=${encodeURIComponent(query.trim())}`);setHits(r.results);if(!r.results.length)setMessage('No pictures found.');}
    catch(e){setMessage((e as Error).message);}finally{setBusy('');}}
  async function importHit(hit:Hit,person?:string){setBusy(hit.url);setMessage('');
    try{const image=await api('/api/web-search/import',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url:hit.url,person_name:person??null})});
      if(person)onPeople();
      onUse({id:image.id,name:person||image.name,url:image.url,personId:image.person?.id});}
    catch(e){setMessage((e as Error).message);}finally{setBusy('');}}
  function asPerson(hit:Hit){const name=window.prompt('Name for the new person',query.trim());if(name?.trim())void importHit(hit,name.trim());}
  if(ready===false)return <div className="web-search">
    <p className="library-empty">Web search needs a SerpAPI key (the one Image Studio uses).</p>
    <div className="web-search-bar"><input type="password" value={key} onChange={e=>setKey(e.target.value)} placeholder="SerpAPI key" aria-label="SerpAPI key"/><button type="button" className="button secondary" disabled={!key.trim()} onClick={()=>void saveKey()}>Save</button></div>
    {message?<p className="people-message">{message}</p>:null}</div>;
  return <div className="web-search">
    <form className="web-search-bar" onSubmit={e=>{e.preventDefault();void search();}}>
      <input type="search" value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search the web for pictures" aria-label="Search the web for pictures" dir="auto"/>
      <button type="submit" className="button secondary" disabled={disabled||busy==='search'||!query.trim()}>{busy==='search'?'Searching…':'Search'}</button>
    </form>
    {message?<p className="people-message">{message}</p>:null}
    <div className="library-grid web-grid">{hits.map(hit=><div key={hit.url} className="web-hit">
      <button type="button" className="library-tile checker" disabled={disabled||!!busy} onClick={()=>void importHit(hit)} title={`${hit.title}${hit.source?` · ${hit.source}`:''}${hit.width?` · ${hit.width} × ${hit.height}`:''}`} aria-label={`Use ${hit.title||'this picture'}`}>
        <img src={hit.thumb} alt="" loading="lazy" referrerPolicy="no-referrer"/>{busy===hit.url?<span className="web-hit-busy">Importing…</span>:null}</button>
      <button type="button" className="text-button web-hit-person" disabled={disabled||!!busy} onClick={()=>asPerson(hit)} title="Import and make a new person in People">+ Person</button>
    </div>)}</div>
  </div>;
}
