import {createContext,useContext,useEffect,useRef,useState,type ReactNode} from 'react';
import {Check,Save,ChevronDown} from 'lucide-react';
import {apiRequest} from './session';

type Patch=Record<string,unknown>;
type Values=Record<string,Patch>;
type Store=Record<string,Record<string,Values>>;
export type DefaultBindings=Record<string,{value:Patch;apply:(patch:any)=>void}>;
type State={data:Store;ready:boolean;selections:Map<string,string>;save:(model:string,scope:string,field:string,value:Patch|null)=>Promise<void>};
const StoreContext=createContext<State|null>(null);
type Scope={model:string;scope:string;bindings:DefaultBindings;parent:Scope|null};
const FieldContext=createContext<Scope|null>(null);
export function bindSettings(values:Record<string,any>,apply:(patch:any)=>void,fields:Record<string,string|string[]>):DefaultBindings{
  return Object.fromEntries(Object.entries(fields).map(([label,keys])=>[label,{value:Object.fromEntries((Array.isArray(keys)?keys:[keys]).map(key=>[key,values[key]??null])),apply}]));
}
export function ModelDefaultsRoot({children}:{children:ReactNode}){
  const [data,setData]=useState<Store>({});const [ready,setReady]=useState(false);
  const selections=useRef(new Map<string,string>());
  useEffect(()=>{let alive=true;let timer:number;const load=async()=>{try{const saved=await apiRequest('/api/model-defaults');if(alive){setData(saved);setReady(true);}}catch{if(alive)timer=window.setTimeout(load,4000);}};load();return()=>{alive=false;clearTimeout(timer);};},[]);
  async function save(model:string,scope:string,field:string,value:Patch|null){
    const saved=await apiRequest(`/api/model-defaults/${encodeURIComponent(model)}`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({scope,field,value})});
    setData(old=>({...old,[model]:{...old[model],[scope]:saved}}));
  }
  return <StoreContext.Provider value={{data,ready,selections:selections.current,save}}>{children}</StoreContext.Provider>;
}
/** Apply saved settings only on an explicit model/scope change, or to a new draft.
 * A restored job keeps its snapshot, including when it changes the selected model. */
export function ModelDefaultsScope({model,scope,bindings,children,draftKey,restoreKey,restoreModel}:{model:string;scope:string;bindings:DefaultBindings;children:ReactNode;draftKey?:string;restoreKey?:unknown;restoreModel?:string}){
  const store=useContext(StoreContext);const last=useRef<string|null>(null);const restored=useRef(restoreKey);
  const parent=useContext(FieldContext);
  const hasDraft=useRef(draftKey?!!localStorage.getItem(draftKey):true);
  const skipNext=useRef<string|null>(null);if(restored.current!==restoreKey){restored.current=restoreKey;skipNext.current=restoreModel??model;}
  useEffect(()=>{
    if(!store?.ready||!model)return;
    const key=model+'/'+scope;
    if(skipNext.current){if(skipNext.current===model){last.current=key;skipNext.current=null;store.selections.set(scope,model);}return;}
    if(last.current===key)return;
    const first=last.current===null;last.current=key;
    const previous=store.selections.get(scope);store.selections.set(scope,model);
    if(first&&hasDraft.current&&(!previous||previous===model))return;
    const patches=new Map<(patch:any)=>void,Patch>();
    for(const [field,value] of Object.entries(store.data[model]?.[scope]||{})){
      const binding=bindings[field];if(binding)patches.set(binding.apply,{...patches.get(binding.apply),...value});
    }
    // Run after the parent validates the new model and applies its factory settings.
    queueMicrotask(()=>{if(last.current===key&&!skipNext.current)for(const [apply,patch] of patches)apply(patch);});
  },[store?.ready,model,scope,restoreKey]);
  return <FieldContext.Provider value={{model,scope,bindings,parent}}>{children}</FieldContext.Provider>;
}
export function useSettingDefault(label:string){
  let field=useContext(FieldContext);const store=useContext(StoreContext);
  while(field&&!field.bindings[label])field=field.parent;
  const binding=field?.bindings[label];const saved=field&&store?.data[field.model]?.[field.scope]?.[label];
  const target=field;
  return {binding,saved,ready:!!store?.ready,save:binding&&target&&store?((value:Patch|null)=>store.save(target.model,target.scope,label,value)):undefined};
}
export function SaveDefaultButton({label,disabled=false,onRecommended}:{label:string;disabled?:boolean;onRecommended:()=>void}){
  const {binding,saved,ready,save}=useSettingDefault(label);const [busy,setBusy]=useState(false);const [error,setError]=useState('');
  if(!binding||!save)return null;
  const same=!!saved&&JSON.stringify(saved)===JSON.stringify(binding.value);
  return <span className="setting-save-wrap"><span className="setting-save-buttons"><button type="button" className={`setting-save ${same?'saved':''}`} disabled={disabled||busy||!ready}
    aria-label={`Set ${label} as model default`} title="Set as model default" onClick={async event=>{event.preventDefault();event.stopPropagation();setBusy(true);setError('');try{await save(binding.value);}catch(e){setError((e as Error).message);}finally{setBusy(false);}}}>
    {same?<Check size={13}/>:<Save size={13}/>}<span>{busy?'Saving…':same?'Saved default':'Set as default'}</span>
  </button>{saved?<details className="setting-default-menu"><summary aria-label={`More defaults for ${label}`}><ChevronDown size={13}/></summary><button type="button" disabled={disabled||busy} onClick={async event=>{const menu=event.currentTarget.closest('details');setBusy(true);setError('');try{await save(null);onRecommended();menu?.removeAttribute('open');}catch(e){setError((e as Error).message);}finally{setBusy(false);}}}>Use recommended default</button></details>:null}</span>{error?<small role="alert">{error}</small>:null}</span>;
}
