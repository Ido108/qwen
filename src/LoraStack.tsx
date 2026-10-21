import {ResetButton} from './SettingLabel';
import {useEffect,useLayoutEffect,useRef,useState} from 'react';
import type {ModelEntry} from './ModelSettings';
import type {AuthorSettings,LoraChoice,LoraMeta} from './loraChoices';
import {loraFit,replaceLora} from './loraChoices';
import {LoraPromptEditor} from './LoraPromptEditor';
export type {LoraChoice} from './loraChoices';
export {enabledLoras} from './loraChoices';

/** The user's LoRAs (for example from Civitai) on top of the selected model, with a strength each. */
export function LoraStack({value,onChange,loras,family,onBrowse,disabled,compact=false,model,onAuthorSettings,hasModel,pixels}:{value:LoraChoice[];onChange:(next:LoraChoice[])=>void;loras:ModelEntry[];family:string;pixels?:number;
  prompt:string;onPrompt:(next:string)=>void;onBrowse:()=>void;disabled:boolean;compact?:boolean;
  model?:{quantization?:string;fp8_plain?:boolean}|null;onAuthorSettings?:(settings:AuthorSettings&{family:string})=>void;hasModel?:(family:string)=>boolean}){
  const [picking,setPicking]=useState<{replaceId?:string}|null>(null);
  const [search,setSearch]=useState('');
  const picker=useRef<HTMLDivElement>(null);
  const anchor=useRef<HTMLButtonElement|null>(null);
  const stack=useRef<HTMLDivElement>(null);
  const focusId=useRef<string|null>(null);
  useLayoutEffect(()=>{
    const panel=picker.current;
    if(!panel)return;
    if(!picking){panel.hidePopover();restoreFocus();return;}
    function place(){
      if(!panel||!anchor.current?.isConnected)return;
      const box=anchor.current.getBoundingClientRect(),viewport=window.visualViewport;
      const left=viewport?.offsetLeft||0,top=viewport?.offsetTop||0;
      const width=viewport?.width||window.innerWidth,height=viewport?.height||window.innerHeight;
      const below=top+height-box.bottom-12,above=box.top-top-12;
      const up=below<230&&above>below;
      const panelWidth=Math.min(380,Math.max(270,box.width),width-16);
      panel.style.width=`${panelWidth}px`;
      panel.style.maxHeight=`${Math.max(120,Math.min(360,up?above:below,height-16))}px`;
      panel.style.left=`${Math.max(left+8,Math.min(box.left,left+width-panelWidth-8))}px`;
      // Measure after the height cap applies so short lists stay next to the row.
      panel.style.top=`${Math.max(top+8,up?box.top-panel.getBoundingClientRect().height-6:box.bottom+6)}px`;
    }
    panel.showPopover();place();
    const followScroll=(event:Event)=>{if(!panel.contains(event.target as Node))place();};
    window.addEventListener('resize',place);
    window.addEventListener('scroll',followScroll,true);
    window.visualViewport?.addEventListener('resize',place);
    window.visualViewport?.addEventListener('scroll',place);
    return()=>{
      window.removeEventListener('resize',place);
      window.removeEventListener('scroll',followScroll,true);
      window.visualViewport?.removeEventListener('resize',place);
      window.visualViewport?.removeEventListener('scroll',place);
    };
  },[picking,search]);
  useEffect(()=>{setPicking(null);},[family]);
  const usable=loras.filter(l=>l.installed&&(l.families||[]).includes(family));
  useEffect(()=>{const saved=(event:Event)=>{const change=(event as CustomEvent).detail;if(value.some(v=>v.id===change.id))onChange(value.map(v=>{if(v.id!==change.id)return v;const {prompt_addition,...choice}=v;return choice;}));};window.addEventListener('qwen-lora-settings',saved);return()=>window.removeEventListener('qwen-lora-settings',saved);},[value,onChange]);
  const entries=new Map(loras.map(lora=>[lora.id,lora]));
  const selectedIds=new Set(value.map(choice=>choice.id));
  const active=value.map(choice=>({choice,lora:entries.get(choice.id),fit:loraFit(entries.get(choice.id) as LoraMeta||{id:choice.id},family,pixels)})).filter(x=>x.lora);
  // Runtime LoRAs on 4-bit or unscaled 8-bit weights: the small LoRA change is partly lost to rounding.
  const weakBase=!!active.some(({choice,fit})=>choice.enabled!==false&&fit.usable)&&(model?.quantization?.toLowerCase()==='nvfp4'||!!model?.fp8_plain);
  function choose(lora:ModelEntry){
    if(disabled||!usable.some(item=>item.id===lora.id))return;
    if(picking?.replaceId){
      if(selectedIds.has(lora.id)&&lora.id!==picking.replaceId)return;
      onChange(replaceLora(value,picking.replaceId,lora.id));
    }else{
      if(selectedIds.has(lora.id))return;
      onChange([...value,{id:lora.id,strength:lora.default_strength??1}]);
    }
    focusId.current=lora.id;
    setPicking(null);
  }
  function openPicker(button:HTMLButtonElement,replaceId?:string){anchor.current=button;setSearch('');setPicking({replaceId});}
  function restoreFocus(){
    const id=focusId.current;focusId.current=null;
    if(id)Array.from(stack.current?.querySelectorAll<HTMLButtonElement>('.lora-select')||[]).find(button=>button.dataset.loraId===id)?.focus({preventScroll:true});
  }
  const filtered=usable.filter(lora=>`${lora.title} ${lora.civitai?.creator||''} ${(lora.trained_words||[]).join(' ')}`.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase()));
  const setStrength=(id:string,strength:number)=>onChange(value.map(v=>v.id===id?{...v,strength}:v));
  return <div ref={stack} className={`lora-stack lora-controls ${compact?'compact':''}`}>
    <div className="lora-head"><b>LoRAs</b><span>{active.length?`${active.filter(({choice,fit})=>choice.enabled!==false&&fit.usable).length} on · ${active.length} saved`:compact?'':'Optional'}</span>
      <button type="button" className="lora-add" disabled={disabled} onClick={e=>openPicker(e.currentTarget)} title="Add a LoRA" aria-haspopup="dialog" aria-expanded={!!picking&&!picking.replaceId}>+ Add LoRA</button></div>
    {weakBase?<p className="lora-notice">LoRA effect is weaker on NVFP4/FP8 files; an INT8 or BF16 model keeps it.</p>:null}
    {active.map(({choice,lora,fit})=>{const enabled=choice.enabled!==false;const fits=fit.usable;const lora_=lora as ModelEntry&LoraMeta;return <div className={`lora-control ${enabled?'':'off'} ${fits?'':'unfit'}`} key={choice.id}>
      <div className="lora-control-heading">
        <button type="button" className="lora-select" data-lora-id={choice.id} disabled={disabled} onClick={e=>openPicker(e.currentTarget,choice.id)}
          aria-label={`Replace ${lora!.title}`} aria-haspopup="dialog" aria-expanded={picking?.replaceId===choice.id} title={`Replace ${lora!.title}`}>
          {lora!.preview?<img src={lora!.preview} alt="" loading="lazy"/>:<span className="lora-noimg"/>}
          <strong>{lora!.title}</strong><span className="lora-caret" aria-hidden="true">▾</span>
        </button>
        <button type="button" className="lora-switch" role="switch" aria-checked={enabled} aria-label={`Use ${lora!.title}`} disabled={disabled}
          onClick={()=>onChange(value.map(v=>v.id===choice.id?{...v,enabled:!enabled}:v))}>
          <span className="lora-switch-track" aria-hidden="true"/><span>{enabled?'On':'Off'}</span>
        </button>
        <button type="button" className="lora-remove" disabled={disabled} aria-label={`Remove ${lora!.title}`} onClick={()=>onChange(value.filter(v=>v.id!==choice.id))}><span aria-hidden="true">×</span></button>
      </div>
      <div className="lora-strength-row"><span>Strength</span>
        <input type="range" min="0" max="2" step="0.05" value={choice.strength} disabled={disabled||!enabled} aria-label={`${lora!.title} strength`}
          onChange={e=>setStrength(choice.id,Number(e.target.value))}/>
        <input type="number" min="0" max="2" step="0.05" value={choice.strength} disabled={disabled||!enabled} aria-label={`${lora!.title} strength value`} title="Strength, 0 to 2"
          onChange={e=>setStrength(choice.id,Math.max(0,Math.min(2,Number(e.target.value)||0)))}/>
        <ResetButton label={`${lora!.title} strength`} value={lora!.default_strength??1} disabled={disabled} onReset={()=>setStrength(choice.id,lora!.default_strength??1)}/>
        <LoraPromptEditor model={lora!} compact disabled={disabled}/>
      </div>
      {fit.blocked?<p className="lora-compatibility">{fit.blocked}</p>:fit.notice?<p className="lora-notice">{fit.notice}{fit.settings&&onAuthorSettings&&(hasModel?.(fit.settings.family)??true)?<> <button type="button" className="text-button lora-author-settings" disabled={disabled} onClick={()=>onAuthorSettings(fit.settings!)}>Use author settings</button></>:null}</p>:null}
      {lora_.test_file?<p className="lora-trigger">{lora_.test_file}</p>:null}
    </div>;})}
    <div ref={picker} popover="auto" role="dialog" className="lora-picker" aria-label={picking?.replaceId?'Replace LoRA':'Add a LoRA'}
      onToggle={e=>{if((e.nativeEvent as ToggleEvent).newState==='closed')setPicking(null);}}
      onKeyDown={e=>{
        if(e.key==='Escape'){setPicking(null);anchor.current?.focus();return;}
        if(e.key!=='ArrowDown'&&e.key!=='ArrowUp')return;
        const buttons=Array.from(picker.current?.querySelectorAll<HTMLButtonElement>('.lora-option:not(:disabled)')||[]);
        if(!buttons.length)return;
        e.preventDefault();
        const index=buttons.indexOf(document.activeElement as HTMLButtonElement);
        buttons[index<0?(e.key==='ArrowDown'?0:buttons.length-1):(index+(e.key==='ArrowDown'?1:-1)+buttons.length)%buttons.length].focus();
      }}>
      <div className="lora-picker-heading"><strong>{picking?.replaceId?'Replace LoRA':'Add a LoRA'}</strong><button type="button" className="icon-button" onClick={()=>{setPicking(null);anchor.current?.focus();}} aria-label="Close"><span aria-hidden="true">×</span></button></div>
      <div className="lora-search"><input type="search" aria-label="Search LoRAs" placeholder="Search LoRAs" value={search} onChange={e=>setSearch(e.target.value)}/></div>
      <div className="lora-picker-grid">{(picking?filtered:[]).map(lora=><button type="button" key={lora.id} className="lora-option" disabled={disabled||(selectedIds.has(lora.id)&&lora.id!==picking?.replaceId)||!loraFit(lora as LoraMeta,family).usable} aria-label={lora.title} onClick={()=>choose(lora)}>
        {lora.preview?<img src={lora.preview} alt="" loading="lazy"/>:<span className="lora-noimg"/>}<strong>{lora.title}</strong>
        <small>{lora.id===picking?.replaceId?'Current LoRA':selectedIds.has(lora.id)?'Already added':(lora as LoraMeta).test_file?'Test file':(lora as LoraMeta).accelerator&&family.endsWith('turbo')?'Speed LoRA, not for Turbo':lora.civitai?.creator||'Installed'}</small></button>)}
        {!filtered.length?<p className="recent-empty">{usable.length?'No matching LoRAs.':'You have no LoRAs for this model yet.'}</p>:null}</div>
      <div className="lora-picker-footer"><button type="button" className="text-button" onClick={()=>{setPicking(null);onBrowse();}}>Browse Civitai</button></div>
    </div>
  </div>;
}
