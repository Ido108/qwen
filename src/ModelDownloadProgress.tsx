export type DownloadStatus={stage:string;download_completed_bytes?:number;download_total_bytes?:number};
export function ModelDownloadProgress({job}:{job:DownloadStatus}){
  if(job.stage!=='downloading_models')return null;
  const done=job.download_completed_bytes||0,total=job.download_total_bytes||0;
  return <div className="model-download-progress"><progress aria-label="Model download progress" max={total||1} value={total?done:undefined}/>{total?<small>{(done/1e9).toFixed(2)} / {(total/1e9).toFixed(2)} GB · {Math.floor(100*done/total)}%</small>:null}</div>;
}
