import {useState} from 'react';
import {ArrowUpRight,Loader2} from 'lucide-react';
import type {UploadedVideo} from './mediaUpload';
import './send-video.css';

export type VideoDestination='faceswap'|'crossview'|'reference'|'upscale';
export type VideoInput={video:UploadedVideo;target:VideoDestination;key:number};
export type SendVideo=(jobId:string,target:VideoDestination,part?:'final'|'subject')=>Promise<void>;

export function SendVideoTo({jobId,onSend,part='final'}:{jobId:string;onSend:SendVideo;part?:'final'|'subject'}){
  const [busy,setBusy]=useState(false),[error,setError]=useState('');
  async function send(target:VideoDestination){setBusy(true);setError('');try{await onSend(jobId,target,part);}catch(e){setError((e as Error).message);}finally{setBusy(false);}}
  return <div className="send-video-control">
    <label className="send-video-select">{busy?<Loader2 size={14} className="spin"/>:<ArrowUpRight size={14}/>}
      <select aria-label={part==='subject'?'Send first-stage video to':'Send video to'} value="" disabled={busy} onChange={e=>{if(e.target.value)void send(e.target.value as VideoDestination);}}>
        <option value="" disabled>{busy?'Preparing video…':'Send to'}</option>
        <option value="faceswap">Replace subjects</option><option value="crossview">Camera view</option><option value="reference">Reference-guided</option><option value="upscale">Upscale</option>
      </select>
    </label>
    {error?<small role="alert" className="send-video-error">{error}</small>:null}
  </div>;
}
