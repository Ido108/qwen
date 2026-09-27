import {useEffect,useState} from 'react';
import {Download,Film} from 'lucide-react';
import type {SavedJob} from './JobQueue';
import type {ModelState} from './ModelSettings';
import {VideoProgress} from './VideoProgress';
import {GenerationTiming} from './GenerationTiming';
import {SendVideoTo,type SendVideo} from './SendVideoTo';
import './video-output.css';

export function VideoOutput({job,result,busy,clock,models,onSendVideo,upscale=false}:{job:SavedJob|null;result:SavedJob|null;busy:boolean;clock:number;models:ModelState|null;onSendVideo:SendVideo;upscale?:boolean}){
  const [view,setView]=useState<'live'|'result'>(busy?'live':'result');
  const [previewError,setPreviewError]=useState(false),[playError,setPlayError]=useState(false);
  useEffect(()=>{setView(busy?'live':'result');},[busy,job?.id]);
  useEffect(()=>{setPreviewError(false);},[job?.preview_url,job?.preview_updated_at]);
  useEffect(()=>{setPlayError(false);},[result?.id,result?.video_url]);
  const live=busy&&view==='live';
  const preview=live&&job?.preview_url&&!previewError;
  return <>
    <div className="section-heading"><h2>{live?'Live preview':upscale?'Upscaled video':'Result'}</h2><small>{live?(job?.preview_chunk?`Section ${job.preview_chunk}`:job?.preview_step?`Step ${job.preview_step}`:'In progress'):result?.width?`${result.width} × ${result.height} · ${result.duration_seconds?.toFixed(1)}s`:'MP4'}</small></div>
    {busy&&result?<div className="reference-preview-tabs"><button className={view==='live'?'active':''} onClick={()=>setView('live')}>Live preview</button><button className={view==='result'?'active':''} onClick={()=>setView('result')}>Previous result</button></div>:null}
    <div className="shared-video-output">
      {preview?<img src={`${job!.preview_url}?v=${job!.preview_updated_at}`} alt={upscale?'Upscaled frames from the latest completed section':'Approximate video from the latest generation step'} onError={()=>setPreviewError(true)}/>:live?<div className="video-empty"><Film size={40}/><strong>{previewError?'Preview could not be loaded':job?.preview_unavailable?'Live preview is unavailable':'Preparing your preview'}</strong><span>{job?.preview_unavailable||previewError?'Processing continues. The finished video will appear here.':upscale?'The preview appears when the first section has finished.':'The first preview appears after the first generation step.'}</span></div>:result?.video_url?<video key={result.id+result.video_url} controls playsInline preload="metadata" poster={result.poster_url} src={result.video_url} onError={()=>setPlayError(true)}/>:<div className="video-empty"><Film size={40}/><strong>Your video will appear here</strong></div>}
    </div>
    {preview?<p className="reference-preview-caption">{upscale?'Completed section · updates as more frames finish.':'Intermediate result · updates after each generation step.'}</p>:null}
    {busy&&job?<VideoProgress job={job} clock={clock}/>:null}
    {!live&&result?<><GenerationTiming job={result} models={models}/><div className="result-actions"><SendVideoTo jobId={result.id} onSend={onSendVideo}/><a className="button secondary" href={`${result.video_url}?download=true`}><Download size={15}/>Download MP4</a><a className="text-button" href={`/api/jobs/${result.id}/metadata`}>Download settings</a></div>{playError?<small role="alert" className="send-video-error">This video could not be played. Try downloading it to view the saved file.</small>:null}</>:null}
  </>;
}
