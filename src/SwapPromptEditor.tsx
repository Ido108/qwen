import {subjectLabel} from './ImageLabel';
import {useEffect,useRef,useState} from 'react';
import {Pencil,History,RotateCcw,X,Eye,Save} from 'lucide-react';
import type {Api} from './ModelSettings';
import type {ImageTool} from './ImageSettings';
import type {SwapState} from './SwapPanel';
import {PromptInput} from './PromptInput';
import {displayImageMentions,imageMentionOptions} from './promptMentions';
import './swap-prompt-editor.css';

type Version={id:string;text:string;created_at:number|null;label:string};
type Saved={current:string;text:string;versions:Version[]};
export function SwapPromptEditor({api,tool,value,onChange,disabled}:{api:Api;tool:ImageTool;value:SwapState;onChange:(next:SwapState)=>void;disabled:boolean}){
  const [open,setOpen]=useState(false),[saved,setSaved]=useState<Saved|null>(null),[draft,setDraft]=useState(''),[error,setError]=useState(''),[busy,setBusy]=useState(false),[preview,setPreview]=useState(''),[chosen,setChosen]=useState('');
  const dialog=useRef<HTMLDialogElement>(null),latest=useRef(value);latest.current=value;
  useEffect(()=>{if(open)dialog.current?.showModal();else dialog.current?.close();},[open]);
  async function load(useSaved=false){setBusy(true);setError('');try{const next=await api(`/api/swap-prompts/${tool.id}`);setSaved(next);if(!useSaved)setDraft(displayImageMentions(value.promptTemplate??next.text));setChosen('');}catch(e){setError((e as Error).message);}finally{setBusy(false);}}
  async function start(){setOpen(true);setSaved(null);setDraft('');setPreview('');await load();}
  async function save(restoreId?:string){if(!saved)return;setBusy(true);setError('');try{
    const next=await api(`/api/swap-prompts/${tool.id}`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:draft,expected:saved.current,restore_id:restoreId})});
    setSaved(next);setDraft(displayImageMentions(next.text));setChosen('');setPreview('');onChange({...latest.current,promptTemplate:undefined,promptRevision:undefined});
    if(!restoreId)setOpen(false);
  }catch(e){setError((e as Error).message);}finally{setBusy(false);}}
  async function showPreview(){setBusy(true);setError('');try{const next=await api(`/api/swap-prompts/${tool.id}/preview`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt:'Preview',images:value.slots.map(i=>i?.id),image_labels:value.slots.map(i=>subjectLabel(i?.subjectLabel)),swap_selection:value.selection,swap_extra:value.extra,swap_pose_source:value.poseSource??1,swap_style_source:value.styleSource??1,swap_style_mode:value.styleMode??'realistic',swap_prompt_template:draft})});setPreview(next.prompt);}catch(e){setError((e as Error).message);}finally{setBusy(false);}}
  const references=[0,1].map(i=>value.slots[i]??{id:`slot-${i}`,name:i?'Replacement person':'Scene',url:''});
  const mentions=[...imageMentionOptions(references),{tag:'@PoseSource',label:'Head pose and expression',detail:`Picture ${value.poseSource??1}`},{tag:'@StyleSource',label:'Color and style',detail:`Picture ${value.styleSource??1}`}];
  const version=saved?.versions.find(v=>v.id===chosen);
  return <><button type="button" className="text-button swap-prompt-open" disabled={disabled} onClick={start}><Pencil size={14}/>Edit swap prompt</button>
    <dialog ref={dialog} className="swap-prompt-dialog" onCancel={e=>{e.preventDefault();if(!busy)setOpen(false);}} aria-labelledby="swap-prompt-title">
      <header><div><h2 id="swap-prompt-title">Swap prompt</h2><small>{tool.title}</small></div><button type="button" className="icon-button" disabled={busy} aria-label="Close prompt editor" onClick={()=>setOpen(false)}><X size={19}/></button></header>
      <div className="swap-prompt-body">
        {!saved&&!error?<p>Loading saved prompt…</p>:null}
        {saved?<><label htmlFor="swap-template">Instructions</label><PromptInput id="swap-template" value={draft} onValueChange={text=>{setDraft(text);setPreview('');}} mentions={mentions} rows={9} maxLength={6000} disabled={busy} dir="auto"/>
          <p className="swap-editor-hint">PoseSource and StyleSource follow your picture choices. Face selections, image style and extra instructions are added to this prompt.</p>
          <button type="button" className="text-button" disabled={busy||!value.selection} onClick={showPreview}><Eye size={15}/>View full prompt</button>
          {!value.selection?<small>Select the faces to view the full prompt.</small>:null}
          {preview?<pre className="swap-prompt-preview" aria-label="Full prompt sent to the model">{preview}</pre>:null}
          <details className="swap-prompt-history"><summary><History size={15}/>Version history <span>{saved.versions.length}</span></summary>
            <div className="swap-version-list">{saved.versions.map(v=><button type="button" key={v.id} aria-pressed={chosen===v.id} onClick={()=>setChosen(v.id)}><span>{v.label}</span><small>{v.created_at?new Date(v.created_at*1000).toLocaleString():''}{v.id===saved.current?' · Current':''}</small></button>)}</div>
            {version?<div className="swap-version-preview"><p>{displayImageMentions(version.text)}</p><button type="button" className="button secondary" disabled={busy||version.id===saved.current} onClick={()=>save(version.id)}><RotateCcw size={14}/>Restore this version</button></div>:null}
          </details></>:null}
        {error?<div className="message error" role="alert">{error}<button type="button" className="text-button" disabled={busy} onClick={()=>load(true)}>Reload saved versions</button></div>:null}
      </div>
      <footer><span>Saved for this swap type on all devices.</span><button type="button" className="button secondary" disabled={busy} onClick={()=>setOpen(false)}>Cancel</button><button type="button" className="button primary" disabled={busy||!saved||!draft.trim()} onClick={()=>save()}><Save size={15}/>{busy?'Saving…':'Save prompt'}</button></footer>
    </dialog></>;
}
