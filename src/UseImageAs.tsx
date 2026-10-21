import {createContext,useContext,useEffect,useRef,useState,type ReactNode} from 'react';
import {ArrowUpRight} from 'lucide-react';
import type {InputImage} from './InputLibrary';
import {apiRequest} from './session';
import './use-image-as.css';

const destinations={
  edit:['Image editing','#images'],
  'swap-scene':['Swap · source picture','#swap'],
  'swap-person':['Swap · replacement person','#swap'],
  references:['Video · reference image','#multi-reference'],
  background:['Video · background image','#multi-reference'],
  first:['Video · first frame','#keyframes'],
  last:['Video · last frame','#keyframes'],
  orbit:['360 Orbit','#orbit'],
  replacement:['Video to video · replacement person','#video-to-video'],
  training:['Training image','#train'],
} as const;
type Destination=keyof typeof destinations;
type Handler=(image:InputImage)=>void;
type Router={register:(id:Destination,handler:Handler)=>()=>void;send:(jobId:string,target:Destination)=>Promise<void>};
const Context=createContext<Router|null>(null);
export function ImageDestinations({children}:{children:ReactNode}){
  const handlers=useRef(new Map<Destination,Handler>());
  const [router]=useState<Router>(()=>({
    register(id,handler){handlers.current.set(id,handler);return()=>{if(handlers.current.get(id)===handler)handlers.current.delete(id);};},
    async send(jobId,target){
      const image:InputImage=await apiRequest(`/api/jobs/${encodeURIComponent(jobId)}/reuse`,{method:'POST'});
      const handle=handlers.current.get(target);
      if(!handle)throw new Error('This destination is not ready. Try again.');
      handle(image);
      window.dispatchEvent(new Event('qwen-image-used'));
      // Dispatch even when the hash is unchanged: the target can be hidden by a preview.
      window.history.replaceState(null,'',destinations[target][1]);
      window.dispatchEvent(new HashChangeEvent('hashchange'));
      requestAnimationFrame(()=>window.scrollTo({top:0,behavior:'instant'}));
    },
  }));
  return <Context.Provider value={router}>{children}</Context.Provider>;
}
export function useImageDestination(id:Destination,handler:Handler){
  const router=useContext(Context);const current=useRef(handler);current.current=handler;
  useEffect(()=>router?.register(id,image=>current.current(image)),[router,id]);
}
export function UseImageAs({jobId,disabled=false,compact=false}:{jobId:string;disabled?:boolean;compact?:boolean}){
  const router=useContext(Context);const [busy,setBusy]=useState(false);const [error,setError]=useState('');
  return <span className={`use-image-as ${compact?'compact':''}`}>
    <label><ArrowUpRight size={15}/><select aria-label="Use image as" title="Use as" value="" disabled={disabled||busy||!router} onChange={async event=>{
      const target=event.target.value as Destination;if(!target||!router)return;setBusy(true);setError('');
      try{await router.send(jobId,target);}catch(e){setError((e as Error).message);}finally{setBusy(false);}
    }}><option value="" disabled>{busy?'Opening…':'Use as'}</option>{Object.entries(destinations).map(([id,[label]])=><option key={id} value={id}>{label}</option>)}</select></label>
    {error?<small role="alert">{error}</small>:null}
  </span>;
}
