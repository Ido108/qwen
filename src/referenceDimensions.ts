// The MiniMax H3 reference models share a 32-pixel grid. This app uses a
// 1344 x 768 pixel budget; match the source shape independently of size.
export type AspectMode='auto'|'16:9'|'9:16'|'1:1'|'4:3'|'3:4'|'21:9'|'9:21'|'custom';
export type SizeTier='draft'|'standard'|'high'|'custom';

export const ASPECT_OPTIONS:{value:AspectMode;label:string;ratio?:number}[]=[
  {value:'auto',label:'Match source'},
  {value:'16:9',label:'16:9 · Landscape',ratio:16/9},
  {value:'9:16',label:'9:16 · Portrait',ratio:9/16},
  {value:'1:1',label:'1:1 · Square',ratio:1},
  {value:'4:3',label:'4:3 · Landscape',ratio:4/3},
  {value:'3:4',label:'3:4 · Portrait',ratio:3/4},
  {value:'21:9',label:'21:9 · Wide',ratio:21/9},
  {value:'9:21',label:'9:21 · Tall',ratio:9/21},
  {value:'custom',label:'Custom ratio'},
];

export const SIZE_OPTIONS:{value:SizeTier;label:string;pixels:number}[]=[
  {value:'draft',label:'Draft · faster',pixels:350_000},
  {value:'standard',label:'Standard',pixels:590_000},
  {value:'high',label:'High · more detail',pixels:980_000},
  {value:'custom',label:'Custom width and height',pixels:0},
];

export const REFERENCE_MAX_PIXELS=1344*768;
export function referenceSizeValid(width:number,height:number){
  const aspect=width/height;
  return Number.isInteger(width)&&Number.isInteger(height)&&width>=256&&height>=256&&
    width<=1344&&height<=1344&&width%32===0&&height%32===0&&
    width*height<=REFERENCE_MAX_PIXELS&&aspect>=0.25&&aspect<=4;
}

export function aspectFor(mode:AspectMode,sourceWidth:number,sourceHeight:number,customWidth:number,customHeight:number){
  if(mode==='auto')return sourceWidth>0&&sourceHeight>0?sourceWidth/sourceHeight:16/9;
  if(mode==='custom')return customWidth>0&&customHeight>0?customWidth/customHeight:16/9;
  return ASPECT_OPTIONS.find(option=>option.value===mode)?.ratio||16/9;
}

export function nearestReferenceSize(aspect:number,tier:Exclude<SizeTier,'custom'>){
  const ratio=Math.max(0.25,Math.min(4,aspect));
  const target=SIZE_OPTIONS.find(option=>option.value===tier)!.pixels;
  let best={width:1024,height:576,score:Infinity};
  for(let width=256;width<=1344;width+=32){
    for(let height=256;height<=1344;height+=32){
      const area=width*height;
      if(area>REFERENCE_MAX_PIXELS)continue;
      const shapeError=Math.abs(Math.log(width/height/ratio));
      const sizeError=Math.abs(Math.log(area/target));
      const score=shapeError*5+sizeError*.7;
      if(score<best.score)best={width,height,score};
    }
  }
  return {width:best.width,height:best.height};
}
