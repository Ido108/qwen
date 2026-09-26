import {Loader2,RefreshCw} from 'lucide-react';
import {ReferenceChoice} from './ReferenceChoice';

export type SourcePerson={id:string;time:number;preview:string;box:number[];hits:number};

export function PersonMatcher({people,references,assignments,status,error,stale,active,onActive,onAssign,onSeek,onRescan}:{
  people:SourcePerson[];references:{id:string;url:string}[];assignments:Record<string,string|null>;
  status:'idle'|'loading'|'ready'|'error';error:string;stale:boolean;active:string|null;
  onActive:(id:string)=>void;onAssign:(personId:string,referenceId:string|null)=>void;
  onSeek:(seconds:number)=>void;onRescan:()=>void;
}){
  return <section className="person-matcher" aria-label="People in source video">
    <div className="section-heading"><h2>People in video</h2><button className="text-button person-rescan" disabled={status==='loading'} onClick={onRescan}><RefreshCw size={13}/>Scan selected clip</button></div>
    {status==='loading'?<div className="person-scanning" role="status"><Loader2 size={15} className="spin"/>Finding people…</div>:null}
    {status==='error'?<p className="person-analysis-error" role="alert">Could not detect people: {error} You can still use the first reference for the main person.</p>:null}
    {stale&&status!=='loading'?<p className="person-analysis-error">The clip changed. Scan it again to update the people.</p>:null}
    {status==='ready'&&!people.length?<p className="reference-help">No faces found. The first reference will guide the main person.</p>:null}
    {people.length?<div className="person-cards">{people.map((person,index)=><div key={person.id} className={`person-card ${active===person.id?'active':''}`}>
      <button className="person-photo" onClick={()=>{onActive(person.id);onSeek(person.time);}} aria-label={`Show Person ${index+1} in source video`}><img src={person.preview} alt={`Person ${index+1}`}/></button>
      <div className="person-card-body"><button className="person-tag" onClick={()=>{onActive(person.id);onSeek(person.time);}}>@Person{index+1}</button>
        <ReferenceChoice personNumber={index+1} references={references} value={assignments[person.id]||null} onChange={value=>{onActive(person.id);onAssign(person.id,value);}}/>
      </div>
    </div>)}</div>:null}
  </section>;
}
