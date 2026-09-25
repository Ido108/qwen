import {Loader2,RefreshCw} from 'lucide-react';

export type SourcePerson={id:string;time:number;preview:string;box:number[];hits:number};

export function PersonMatcher({people,references,assignments,status,error,stale,active,onActive,onAssign,onSeek,onRescan}:{
  people:SourcePerson[];references:{id:string;url:string}[];assignments:Record<string,string|null>;
  status:'idle'|'loading'|'ready'|'error';error:string;stale:boolean;active:string|null;
  onActive:(id:string)=>void;onAssign:(personId:string,referenceId:string|null)=>void;
  onSeek:(seconds:number)=>void;onRescan:()=>void;
}){
  const selected=active&&people.some(person=>person.id===active)?active:people[0]?.id;
  return <section className="person-matcher" aria-label="People in source video">
    <div className="section-heading"><h2>People in video</h2><button className="text-button person-rescan" disabled={status==='loading'} onClick={onRescan}><RefreshCw size={14}/>Scan selected clip</button></div>
    {status==='loading'?<div className="person-scanning" role="status"><Loader2 size={16} className="spin"/>Finding people across the selected clip…</div>:null}
    {status==='error'?<p className="person-analysis-error" role="alert">Could not detect people: {error} You can still use the reference images without person assignments.</p>:null}
    {stale&&status!=='loading'?<p className="person-analysis-error">The selected clip changed. Scan it again before generating with person assignments.</p>:null}
    {status==='ready'&&!people.length?<p className="reference-help">No faces were found in this segment. You can still use the reference images directly.</p>:null}
    {people.length?<><p className="person-instruction">Select a person, then choose a reference image. Keep asks the model to preserve that person; check the result.</p><div className="person-cards">{people.map((person,index)=><div key={person.id} className={`person-card ${selected===person.id?'active':''}`}><button className="person-photo" onClick={()=>{onActive(person.id);onSeek(person.time);}} aria-label={`Show Person ${index+1} in source video`}><img src={person.preview} alt={`Person ${index+1}`}/></button><div className="person-card-body"><button className="person-tag" onClick={()=>{onActive(person.id);onSeek(person.time);}}>@Person{index+1}</button><span>{person.box[0]+person.box[2]/2<0.38?'Left':person.box[0]+person.box[2]/2>0.62?'Right':'Center'} · {person.time.toFixed(1)}s</span><select aria-label={`Replacement for Person ${index+1}`} value={assignments[person.id]||''} onChange={event=>{onActive(person.id);onAssign(person.id,event.target.value||null);}}><option value="">Keep person</option>{references.map((reference,referenceIndex)=><option key={reference.id} value={reference.id}>Use @Ref{referenceIndex+1}</option>)}</select></div></div>)}</div>
      {references.length?<div className="person-reference-picker"><strong>Reference for {selected?.replace('person-','Person ')}</strong><div>{references.map((reference,index)=><button key={reference.id} className={selected&&assignments[selected]===reference.id?'active':''} onClick={()=>selected&&onAssign(selected,reference.id)} title={`Assign @Ref${index+1} to ${selected}`}><img src={reference.url} alt={`Reference ${index+1}`}/><span>@Ref{index+1}</span></button>)}<button className={selected&&!assignments[selected]?'active':''} onClick={()=>selected&&onAssign(selected,null)}><span className="person-keep-icon">—</span><span>Keep</span></button></div></div>:null}
    </>:null}
  </section>;
}
