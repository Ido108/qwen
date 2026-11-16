import {Loader2,RefreshCw} from 'lucide-react';
import {ReferenceChoice} from './ReferenceChoice';

export type SourcePerson={id:string;time:number;preview:string;box:number[];hits:number;appearances?:[number,number[]][]};

export function PersonMatcher({people,references,assignments,status,error,stale,active,onActive,onAssign,onSame,onSeek,onRescan}:{
  people:SourcePerson[];references:{id:string;url:string;count?:number}[];assignments:Record<string,string|null>;
  status:'idle'|'loading'|'ready'|'error';error:string;stale:boolean;active:string|null;
  onActive:(id:string)=>void;onAssign:(personId:string,referenceId:string|null)=>void;
  onSame:(personId:string,sameAsId:string)=>void;
  onSeek:(seconds:number)=>void;onRescan:()=>void;
}){
  return <section className="person-matcher" aria-label="People in source video">
    <div className="section-heading"><h2>People in video</h2><button className="text-button person-rescan" disabled={status==='loading'} onClick={onRescan}><RefreshCw size={13}/>Scan selected clip</button></div>
    {status==='loading'||(stale&&status!=='error')?<div className="person-scanning" role="status"><Loader2 size={15} className="spin"/>Finding people…</div>:null}
    {status==='error'?<p className="person-analysis-error" role="alert">Could not detect people: {error} You can still use the first uploaded person for the main person.</p>:null}
    {status==='ready'&&!people.length?<p className="reference-help">No faces found. The first uploaded person will guide the main person.</p>:null}
    {people.length?<div className="person-cards">{people.map((person,index)=><div key={person.id} className={`person-card ${active===person.id?'active':''}`}>
      <button className="person-photo" onClick={()=>{onActive(person.id);onSeek(person.time);}} aria-label={`Show Source ${index+1} in source video`}><img src={person.preview} alt={`Source person ${index+1}`}/></button>
      <div className="person-card-body"><button className="person-tag" onClick={()=>{onActive(person.id);onSeek(person.time);}}>@Source{index+1}</button>
        <ReferenceChoice personNumber={index+1} references={references} value={assignments[person.id]||null} onChange={value=>{onActive(person.id);onAssign(person.id,value);}}/>
        {people.length>1?<select aria-label={`Source ${index+1} is the same person as`} value="" onChange={event=>{if(event.target.value)onSame(person.id,event.target.value);}}>
          <option value="">Same person as…</option>
          {people.map((other,otherIndex)=>other.id===person.id?null:<option key={other.id} value={other.id}>@Source{otherIndex+1}</option>)}
        </select>:null}
      </div>
    </div>)}</div>:null}
  </section>;
}
