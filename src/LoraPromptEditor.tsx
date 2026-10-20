import {useEffect,useId,useRef,useState} from 'react';
import {Pencil,X} from 'lucide-react';
import {apiRequest} from './session';
import type {ModelEntry} from './ModelSettings';

export function LoraPromptEditor({model}:{model:ModelEntry}){
  const [open,setOpen]=useState(false),[text,setText]=useState(''),[error,setError]=useState(''),[busy,setBusy]=useState(false),[loaded,setLoaded]=useState(false);
  const dialog=useRef<HTMLDialogElement>(null);const field=useId();
  useEffect(()=>{if(!open){dialog.current?.close();return;}dialog.current?.showModal();let alive=true;
    setBusy(true);setLoaded(false);setText('');setError('');apiRequest(`/api/loras/${encodeURIComponent(model.id)}/settings`).then(value=>{if(alive){setText(value.prompt_addition||'');setLoaded(true);}}).catch(e=>{if(alive)setError(e.message);}).finally(()=>{if(alive)setBusy(false);});
    return()=>{alive=false;};
  },[open,model.id]);
  async function save(){setBusy(true);setError('');try{
    const value=await apiRequest(`/api/loras/${encodeURIComponent(model.id)}/settings`,{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt_addition:text})});
    window.dispatchEvent(new CustomEvent('qwen-lora-settings',{detail:{id:model.id,...value}}));setOpen(false);
  }catch(e){setError((e as Error).message);}finally{setBusy(false);}}
  return <><button type="button" className="text-button lora-edit-prompt" onClick={()=>setOpen(true)} aria-label={`Edit prompt addition for ${model.title}`}><Pencil size={14}/>Prompt addition</button>
    <dialog ref={dialog} className="lora-prompt-dialog" aria-label="LoRA prompt addition" onCancel={()=>setOpen(false)} onClick={e=>{if(e.target===dialog.current)setOpen(false);}}>
      <div className="lora-prompt-heading"><div><h2>Prompt addition</h2><p>{model.title}</p></div><button type="button" className="icon-button" aria-label="Close prompt addition" onClick={()=>setOpen(false)}><X size={20}/></button></div>
      <label htmlFor={field}>Text to add when this LoRA is on</label>
      <textarea id={field} rows={7} maxLength={4000} dir="auto" disabled={busy||!loaded} value={text} onChange={e=>setText(e.target.value)} placeholder="Describe the style, details, or words you want this LoRA to use."/>
      <p>Your prompt stays unchanged. Each enabled LoRA adds its saved text to the prompt sent for generation. Strength 0 skips this text.</p>
      {model.trained_words?.length?<p>Required words are also added: {model.trained_words.join(', ')}</p>:null}
      {error?<p role="alert" className="message error">{error}</p>:null}
      <div className="lora-prompt-actions"><button type="button" className="text-button" disabled={busy||!loaded} onClick={()=>setText('')}>Clear text</button><button type="button" className="button secondary" onClick={()=>setOpen(false)}>Cancel</button><button type="button" className="button primary" disabled={busy||!loaded} onClick={save}>{busy?(loaded?'Saving…':'Loading…'):'Save'}</button></div>
    </dialog></>;
}
