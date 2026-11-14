export const REFERENCE_FPS=24;
export const REFERENCE_MIN_FRAMES=22;
export const REFERENCE_MAX_FRAMES=362;

export function nearestReferenceFrames(seconds:number,maxFrames=REFERENCE_MAX_FRAMES){
  // Infinity: a face swap runs in parts, so the selection has no length limit
  const maximum=Number.isFinite(maxFrames)?Math.min(REFERENCE_MAX_FRAMES,maxFrames):Infinity;
  const count=Math.round((seconds*REFERENCE_FPS-5)/17);
  return Math.min(maximum,Math.max(REFERENCE_MIN_FRAMES,5+17*count));
}

export function sourceClipDefault(duration:number,maxFrames=REFERENCE_MAX_FRAMES){
  // The first selection is at most one pass long; a longer one is chosen on the timeline
  const end=Math.min(duration,Math.min(REFERENCE_MAX_FRAMES,maxFrames)/REFERENCE_FPS);
  return {start:0,end,frames:nearestReferenceFrames(end,maxFrames)};
}

export function formatClipTime(seconds:number){
  const minutes=Math.floor(seconds/60);
  const remainder=(seconds-minutes*60).toFixed(1).padStart(4,'0');
  return `${String(minutes).padStart(2,'0')}:${remainder}`;
}
