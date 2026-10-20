import {useState,type CSSProperties} from 'react';
import {ProgressiveImage} from './ProgressiveImage';

/** Before/after slider: the input image on the left of the line, the result on the right. */
export function CompareView({before,after,alt}:{before:string;after:string;alt:string}){
  const [position,setPosition]=useState(50);
  return <div className="compare-view" style={{'--compare':`${position}%`} as CSSProperties}>
    <ProgressiveImage className="compare-after" src={after} alt={alt} fullSize/>
    <img className="compare-before" src={before} alt="Input image" aria-hidden="true"/>
    <div className="compare-line" aria-hidden="true"><span/></div>
    <span className="compare-tag compare-tag-before">Input</span><span className="compare-tag compare-tag-after">Result</span>
    <input className="compare-range" type="range" min="0" max="100" step="0.5" value={position} aria-label="Compare input and result"
      onChange={e=>setPosition(Number(e.target.value))} onKeyDown={e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight')e.stopPropagation();}}/>
  </div>;
}
