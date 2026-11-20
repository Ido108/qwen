import type {PortraitField,PortraitGroup,PortraitSettings,PortraitValue} from './portraitControls';

function Field({field,value,onChange,prefix}:{field:PortraitField;value:PortraitValue;onChange:(value:PortraitValue)=>void;prefix:string}){
  const id=`${prefix}-${field.key}`;
  const hint=field.hint?<small>{field.hint}</small>:null;
  if(field.type==='checkbox')return <div className="field portrait-check"><label className="checkbox-label"><input id={id} type="checkbox" checked={value===true} onChange={e=>onChange(e.target.checked)}/>{field.label}</label>{hint}</div>;
  if(field.type==='select'){
    const options=field.options||[];
    const index=Math.max(0,options.findIndex(([option])=>option===value));
    return <div className="field"><label htmlFor={id}>{field.label}</label><select id={id} value={index} onChange={e=>onChange(options[Number(e.target.value)][0])}>{options.map(([option,label],i)=><option key={String(option)} value={i}>{label}</option>)}</select>{hint}</div>;
  }
  return <div className="field"><label htmlFor={id}>{field.label}</label><input id={id} type="number" min={field.min} max={field.max} step={field.step} value={value===null||value===undefined?'':Number(value)} onChange={e=>{const next=e.target.valueAsNumber;if(Number.isFinite(next))onChange(Math.min(field.max??next,Math.max(field.min??next,next)));}}/>{hint}</div>;
}

export function PortraitGroupFields({group,settings,onChange,prefix}:{group:PortraitGroup;settings:PortraitSettings;onChange:(key:string,value:PortraitValue)=>void;prefix:string}){
  const fields=group.fields.filter(field=>!field.hidden);
  if(group.hidden||!fields.length)return null;
  return <>{group.description?<p className="portrait-group-note">{group.description}</p>:null}<div className="portrait-fields">{fields.map(field=><Field key={field.key} field={field} prefix={prefix} value={settings[field.key]??field.value} onChange={value=>onChange(field.key,value)}/>)}</div></>;
}

/** Main groups as panels, the other groups in one Advanced settings disclosure. */
export function PortraitFields({groups,settings,onChange,prefix,skip=[]}:{groups:PortraitGroup[];settings:PortraitSettings;onChange:(key:string,value:PortraitValue)=>void;prefix:string;skip?:string[]}){
  const shown=groups.filter(group=>!group.hidden&&!skip.includes(group.title)&&group.fields.some(field=>!field.hidden));
  const main=shown.filter(group=>group.primary),advanced=shown.filter(group=>!group.primary);
  return <>
    {main.map(group=><section key={group.title} className="panel vs-panel portrait-panel"><div className="section-heading"><h2>{group.title}</h2></div><PortraitGroupFields group={group} settings={settings} onChange={onChange} prefix={prefix}/></section>)}
    {advanced.length?<details className="disclosure portrait-advanced"><summary>Advanced settings <span className="summary-hint">{advanced.map(group=>group.title).join(' · ')}</span></summary><div className="details-body">
      {advanced.map(group=><div key={group.title} className="portrait-advanced-group"><h3>{group.title}</h3><PortraitGroupFields group={group} settings={settings} onChange={onChange} prefix={prefix}/></div>)}
    </div></details>:null}
  </>;
}
