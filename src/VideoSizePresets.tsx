import {SettingLabel} from './SettingLabel';
import {fitSize} from './videoModels';
type Rule={step:number;maxSide:number;maxPixels:number};
const ratios=[['16:9',16/9],['9:16',9/16],['1:1',1],['4:3',4/3],['3:4',3/4]] as const;
export function VideoSizePresets({id,width,height,rule,onChange,defaultSize,disabled=false}:{id:string;width:number;height:number;rule:Rule;defaultSize?:{width:number;height:number};onChange:(size:{width:number;height:number})=>void;disabled?:boolean}){
  const ratio=width/height,area=width*height;
  const resetSize=defaultSize||fitSize(1344,768,rule);
  const aspect=ratios.find(([,r])=>Math.abs(Math.log(ratio/r))<.045)?.[0]||'current';
  const tiers=[{name:'Small',pixels:Math.min(400000,rule.maxPixels)},{name:'Medium',pixels:Math.min(750000,rule.maxPixels)},{name:'Maximum',pixels:rule.maxPixels}];
  const size=(r:number,pixels:number)=>fitSize(Math.sqrt(pixels*r),Math.sqrt(pixels/r),rule);
  const selectedTier=tiers.find(t=>{const s=size(ratio,t.pixels);return s.width===width&&s.height===height;});
  return <div className="video-size-presets"><div className="field"><SettingLabel htmlFor={id+'-aspect'} label="Aspect ratio" disabled={disabled} onReset={()=>onChange(size(resetSize.width/resetSize.height,Math.min(area,rule.maxPixels)))}>Aspect ratio</SettingLabel><select id={id+'-aspect'} disabled={disabled} value={aspect} onChange={e=>{const r=ratios.find(([name])=>name===e.target.value)?.[1];if(r)onChange(size(r,selectedTier?.pixels||Math.min(area,rule.maxPixels)));}}>{aspect==='current'?<option value="current">Current · {width}:{height}</option>:null}{ratios.map(([name])=><option key={name} value={name}>{name}{name==='16:9'?' · Landscape':name==='9:16'?' · Portrait':name==='1:1'?' · Square':''}</option>)}</select></div><div className="field"><SettingLabel htmlFor={id+'-resolution'} label="Resolution" disabled={disabled} onReset={()=>onChange(size(ratio,resetSize.width*resetSize.height))}>Resolution</SettingLabel><select id={id+'-resolution'} disabled={disabled} value={selectedTier?.name||'current'} onChange={e=>{const tier=tiers.find(t=>t.name===e.target.value);if(tier)onChange(size(ratio,tier.pixels));}}>{!selectedTier?<option value="current">{width} × {height} · {(area/1e6).toFixed(2)} MP</option>:null}{tiers.map(t=>{const s=size(ratio,t.pixels);return <option key={t.name} value={t.name}>{t.name} · {s.width} × {s.height} · {(s.width*s.height/1e6).toFixed(2)} MP</option>;})}</select></div></div>;
}
