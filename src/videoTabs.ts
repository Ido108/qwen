/** The video pages, and the input mode of the page that creates video from text or from a first frame. */
export type VideoSection='orbit'|'create'|'reference'|'multi-reference'|'upscale';
export type VideoMode='text'|'image';

export const VIDEO_HASHES=['#video','#keyframes','#text-to-video','#reference','#video-to-video','#multi-reference','#upscale','#orbit'];

/** The page a link opens. Video's own entry (#video) is Image to video: the picture is the first frame.
 * Reference images, where a picture is a subject and not a frame, is its own page. */
export function videoEntry(hash:string):{section:VideoSection;mode?:VideoMode}{
  if(hash==='#orbit')return {section:'orbit'};
  if(hash==='#upscale')return {section:'upscale'};
  if(hash==='#reference'||hash==='#video-to-video')return {section:'reference'};
  if(hash==='#multi-reference')return {section:'multi-reference'};
  return {section:'create',mode:hash==='#text-to-video'?'text':'image'};
}

export const videoHash=(section:VideoSection,mode:VideoMode)=>section==='orbit'?'#orbit':section==='create'?(mode==='text'?'#text-to-video':'#keyframes'):section==='reference'?'#video-to-video':section==='multi-reference'?'#multi-reference':'#upscale';
