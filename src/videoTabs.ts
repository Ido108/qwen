/** The video pages, and the input mode of the page that creates video from text or from a first frame. */
export type VideoSection='orbit'|'create'|'reference'|'multi-reference'|'upscale'|'animate'|'lipsync'|'extract-audio';
export type VideoMode='text'|'image';

export const VIDEO_HASHES=['#video','#keyframes','#text-to-video','#reference','#video-to-video','#multi-reference','#upscale','#orbit','#animate','#lip-sync','#extract-audio'];

/** The page a link opens. Video's own entry (#video) is Image to video: the picture is the first frame.
 * Reference images, where a picture is a subject and not a frame, is its own page. */
export function videoEntry(hash:string):{section:VideoSection;mode?:VideoMode}{
  if(hash==='#orbit')return {section:'orbit'};
  if(hash==='#upscale')return {section:'upscale'};
  if(hash==='#animate')return {section:'animate'};
  if(hash==='#lip-sync')return {section:'lipsync'};
  if(hash==='#extract-audio')return {section:'extract-audio'};
  if(hash==='#reference'||hash==='#video-to-video')return {section:'reference'};
  if(hash==='#multi-reference')return {section:'multi-reference'};
  return {section:'create',mode:hash==='#text-to-video'?'text':'image'};
}

const HASHES:Record<Exclude<VideoSection,'create'>,string>={orbit:'#orbit',reference:'#video-to-video','multi-reference':'#multi-reference',upscale:'#upscale',animate:'#animate',lipsync:'#lip-sync','extract-audio':'#extract-audio'};
export const videoHash=(section:VideoSection,mode:VideoMode)=>section==='create'?(mode==='text'?'#text-to-video':'#keyframes'):HASHES[section];

/** The page a "Send to" target opens. */
export function sendSection(target:string):VideoSection{
  if(target==='upscale')return 'upscale';
  if(target==='lipsync'||target==='lipsync-audio')return 'lipsync';
  if(target==='animate'||target==='animate-audio')return 'animate';
  if(target==='extract-audio')return 'extract-audio';
  return 'reference';
}
