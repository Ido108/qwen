import type {ReactNode} from 'react';
import {RotateCcw} from 'lucide-react';
import './setting-reset.css';
import {SaveDefaultButton,useSettingDefault} from './ModelDefaults';

export function ResetButton({label,value,onReset,disabled=false,withSave=true}:{label:string;value?:string|number;onReset:()=>void;disabled?:boolean;withSave?:boolean}){
  const {saved,binding}=useSettingDefault(label);
  const button=<button type="button" className="setting-reset" disabled={disabled}
    aria-label={`Reset ${label}`} title={saved?`Reset ${label} to your saved default`:`Reset ${label}${value===undefined?'':` to ${value}`}`}
    onClick={event=>{event.preventDefault();event.stopPropagation();if(saved&&binding)binding.apply(saved);else onReset();}}>
    <RotateCcw size={13}/><span>Reset</span>
  </button>;
  return withSave&&binding?<span className="setting-actions"><SaveDefaultButton label={label} disabled={disabled} onRecommended={onReset}/>{button}</span>:button;
}
export function SettingLabel({htmlFor,children,label,value,onReset,disabled=false}:{htmlFor?:string;children:ReactNode;label:string;value?:string|number;onReset:()=>void;disabled?:boolean}){
  return <div className="setting-heading"><label htmlFor={htmlFor}>{children}</label><span className="setting-actions"><SaveDefaultButton label={label} disabled={disabled} onRecommended={onReset}/><ResetButton withSave={false} label={label} value={value} onReset={onReset} disabled={disabled}/></span></div>;
}
/** Small ▼ ▲ buttons beside a slider's value for one-step changes. */
export function Nudge({value,step,min,max,onChange,label,disabled=false}:{value:number;step:number;min:number;max:number;onChange:(next:number)=>void;label:string;disabled?:boolean}){
  const set=(next:number)=>onChange(Number(Math.min(max,Math.max(min,next)).toFixed(2)));
  return <span className="nudge">
    <button type="button" aria-label={`Lower ${label}`} title={`Lower ${label}`} disabled={disabled||value<=min} onClick={e=>{e.preventDefault();set(value-step);}}>▼</button>
    <button type="button" aria-label={`Raise ${label}`} title={`Raise ${label}`} disabled={disabled||value>=max} onClick={e=>{e.preventDefault();set(value+step);}}>▲</button>
  </span>;
}
