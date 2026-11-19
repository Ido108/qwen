export type OutputAudio='generated'|'source'|'mute';
// The output sound when a source video is selected. A source with no sound gives a video with no sound:
// with "Generate audio" the model made up speech for a silent source (2026-11-19).
// A source with sound, selected after a silent one, gets the usual setting again.
export function soundForSource(hasAudio:boolean,previousWasSilent:boolean,current:OutputAudio,usual:OutputAudio):OutputAudio{
  if(!hasAudio)return 'mute';
  return previousWasSilent&&current==='mute'?usual:current;
}
