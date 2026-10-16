import type {ModelEntry} from './ModelSettings';

/** The three video pages that create video with a diffusion model (Upscale is separate). */
export type VideoTask='create'|'references'|'video-to-video';

export const isLtx=(m?:ModelEntry)=>m?.family==='ltx25'||m?.family==='ltx25-msr';
export const isRef2va=(m?:ModelEntry)=>m?.kind==='reference-video'&&m?.family==='minimax-h3-ref2va';

/** Which pages can run a model. LTX 2.5 transformers run Create video without the MSR LoRA and
 * Reference images with it; MiniMax H3 Ref2VA runs Video to video and reference images only. */
export function videoTasks(m:ModelEntry):VideoTask[]{
  if(m.kind==='multi-reference'&&m.family==='ltx25-msr')return ['create','references'];
  if(m.kind==='video')return ['create'];
  if(isRef2va(m))return ['video-to-video','references'];
  return [];
}

export const familyLabel=(m:ModelEntry)=>isLtx(m)?'LTX 2.5':m.family==='fasth3'?'FastH3':m.family==='minimax-h3'?'MiniMax H3':isRef2va(m)?'MiniMax H3 Ref2VA':m.family||'Other';

/** Short capability list for the "Supports:" hint. */
export function supports(m:ModelEntry):string[]{
  if(isLtx(m))return ['text','first frame','last frame','reference images','voices'];
  if(isRef2va(m))return ['source video','reference images','voices'];
  if(m.family==='fasth3')return ['text'];
  if(m.family==='minimax-h3')return ['text','first frame','last frame'];
  return [];
}

/** Input modes for Create video. */
export const createModes=(m?:ModelEntry)=>!m?[]:isLtx(m)?(m.modes||['text','image']):(m.modes||[]);

export function videoModelsFor(models:ModelEntry[]|undefined,task:VideoTask){
  return (models||[]).filter(m=>videoTasks(m).includes(task));
}

/** Model title for a page: the LTX files are shared, so "MSR V2" only names them on Reference images. */
export const pageTitle=(m:ModelEntry,task:VideoTask)=>task==='create'&&m.family==='ltx25-msr'?m.title.replace(/\s*MSR V2/,''):m.title;

/** Models grouped by family, in catalog order, for <optgroup>. `first` puts the page's main family on top. */
export function groupByFamily(models:ModelEntry[],first?:string){
  const groups=new Map<string,ModelEntry[]>();
  for(const m of models){const key=familyLabel(m);groups.set(key,[...(groups.get(key)||[]),m]);}
  return Array.from(groups.entries()).sort(([a],[b])=>Number(b===first)-Number(a===first));
}

// Frame counts at 24 fps. MiniMax H3 uses 17n+5 frames, LTX 2.5 uses 8n+1 frames.
export const H3_FRAMES=[56,73,124,175,243,362];
export const REF2VA_FRAMES=[73,124,175,243,362];
export const LTX_FRAMES=[49,73,97,121,193,241,361];
export const nearestFrames=(value:number,options:number[])=>options.reduce((best,next)=>Math.abs(next-value)<Math.abs(best-value)?next:best,options[0]);

/** Create video size rule for one model and input mode. LTX text and first-frame video runs a half-size
 * pass and a 2x refinement pass (multiples of 64); LTX first and last frame is one full-size pass. */
export function createSizeRule(m:ModelEntry|undefined,lastFrame:boolean){
  if(isLtx(m))return lastFrame?{step:32,maxSide:1536,maxPixels:1024*1024}:{step:64,maxSide:1536,maxPixels:1536*1024};
  return {step:32,maxSide:1344,maxPixels:1344*768};
}
export const sizeFits=(width:number,height:number,rule:{step:number;maxSide:number;maxPixels:number})=>width>=256&&height>=256&&width<=rule.maxSide&&height<=rule.maxSide&&width%rule.step===0&&height%rule.step===0&&width*height<=rule.maxPixels&&width/height>=0.25&&width/height<=4;

/** The closest valid size with the same aspect ratio. */
export function fitSize(width:number,height:number,rule:{step:number;maxSide:number;maxPixels:number}){
  const ratio=width/height;let w=width,h=height;
  const scale=Math.min(1,rule.maxSide/w,rule.maxSide/h,Math.sqrt(rule.maxPixels/(w*h)));
  w=Math.max(256,Math.round(w*scale/rule.step)*rule.step);h=Math.max(256,Math.round(h*scale/rule.step)*rule.step);
  while(w*h>rule.maxPixels||w>rule.maxSide||h>rule.maxSide){if(w/h>=ratio&&w>256)w-=rule.step;else h-=rule.step;}
  return {width:w,height:h};
}
