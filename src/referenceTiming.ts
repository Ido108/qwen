export const REFERENCE_FPS=24;
export const REFERENCE_MIN_FRAMES=22;
export const REFERENCE_MAX_FRAMES=362;

export function nearestReferenceFrames(seconds:number,maxFrames=REFERENCE_MAX_FRAMES){
  const maximum=Math.min(REFERENCE_MAX_FRAMES,maxFrames);
  const count=Math.round((seconds*REFERENCE_FPS-5)/17);
  return Math.min(maximum,Math.max(REFERENCE_MIN_FRAMES,5+17*count));
}

export function sourceClipDefault(duration:number,maxFrames=REFERENCE_MAX_FRAMES){
  const end=Math.min(duration,maxFrames/REFERENCE_FPS);
  return {start:0,end,frames:nearestReferenceFrames(end,maxFrames)};
}

export function formatClipTime(seconds:number){
  const minutes=Math.floor(seconds/60);
  const remainder=(seconds-minutes*60).toFixed(1).padStart(4,'0');
  return `${String(minutes).padStart(2,'0')}:${remainder}`;
}
