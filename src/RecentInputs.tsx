import {useEffect,useRef,useState} from 'react';
import {Check,History,Loader2,X} from 'lucide-react';
import type {Api} from './ModelSettings';

export type InputImage={id:string;name:string;url:string};
type Upload={id:string;name:string;url:string;thumb_url:string};
type Result={id:string;request:{prompt:string};image_url?:string};

/** Pick previously used input images or earlier results as new inputs. */
export function RecentInputs({api,open,onClose,onAdd,limit,results}:{api:Api;open:boolean;onClose:()=>void;onAdd:(images:InputImage[])=>void;limit:number;results:Result[]}){
  const dialog=useRef<HTMLDialogElement>(null);
  const [source,setSource]=useState<'uploads'|'results'>('uploads');
  const [uploads,setUploads]=useState<Upload[]>([]);const [total,setTotal]=useState(0);
  const [selected,setSelected]=useState<string[]>([]);const [loading,setLoading]=useState(false);const [error,setError]=useState('');
  useEffect(()=>{if(open&&!dialog.current?.open)dialog.current?.showModal();else if(!open)dialog.current?.close();if(open){setSelected([]);setError('');load(0);}},[open]);
  async function load(offset:number){setLoading(true);try{const page=await api(`/api/uploads?offset=${offset}&limit=60`);setUploads(old=>offset?[...old,...page.items]:page.items);setTotal(page.total);}catch(e){setError((e as Error).message);}finally{setLoading(false);}}
  function toggle(id:string){setSelected(old=>old.includes(id)?old.filter(x=>x!==id):old.length>=limit?old:[...old,id]);}
  async function add(){
    setLoading(true);setError('');
    try{
      const images:InputImage[]=[];
      for(const id of selected){
        if(source==='uploads'){const item=uploads.find(u=>u.id===id)!;images.push({id:item.id,name:item.name,url:item.url});}
        else images.push(await api(`/api/jobs/${id}/reuse`,{method:'POST'}));
      }
      onAdd(images);onClose();
    }catch(e){setError((e as Error).message);}finally{setLoading(false);}
  }
  const items=source==='uploads'?uploads.map(u=>({id:u.id,thumb:u.thumb_url,label:u.name})):results.map(r=>({id:r.id,thumb:`/api/jobs/${r.id}/thumbnail`,label:r.request.prompt}));
  return <dialog ref={dialog} className="recent-inputs" aria-label="Recent images" onCancel={onClose} onClick={e=>{if(e.target===dialog.current)onClose();}}>
    <div className="recent-inputs-head"><div><h2><History size={19}/>Recent images</h2><p>Select up to {limit} {limit===1?'image':'images'} to add as input.</p></div><button className="icon-button" onClick={onClose} aria-label="Close"><X size={20}/></button></div>
    <div className="segmented" role="tablist"><button role="tab" aria-selected={source==='uploads'} onClick={()=>{setSource('uploads');setSelected([]);}}>Uploaded inputs <span>{total}</span></button><button role="tab" aria-selected={source==='results'} onClick={()=>{setSource('results');setSelected([]);}}>My results <span>{results.length}</span></button></div>
    {error?<div className="message error"><span>{error}</span></div>:null}
    <div className="recent-inputs-grid">{items.map(item=>{const index=selected.indexOf(item.id);return <button key={item.id} className={`recent-input checker ${index>=0?'selected':''}`} onClick={()=>toggle(item.id)} title={item.label} aria-pressed={index>=0}>
      <img src={item.thumb} alt="" loading="lazy"/>{index>=0?<span className="recent-input-order">{index+1}</span>:null}</button>;})}
      {!items.length&&!loading?<p className="recent-empty">{source==='uploads'?'Images you add as input appear here.':'Your finished images appear here.'}</p>:null}
    </div>
    {source==='uploads'&&uploads.length<total?<button className="text-button recent-more" disabled={loading} onClick={()=>load(uploads.length)}>Show more</button>:null}
    <div className="recent-inputs-actions"><span>{selected.length?`${selected.length} selected`:'Click images to select them'}</span><button className="button secondary" onClick={onClose}>Cancel</button><button className="button primary" disabled={!selected.length||loading} onClick={add}>{loading?<Loader2 className="spin" size={17}/>:<Check size={17}/>}Add {selected.length||''} as input</button></div>
  </dialog>;
}
