import {useState} from 'react';
import {apiRequest,throughMoise} from './session';

/** Opens the folder of a finished job's file on this computer, with the file selected. */
export function ShowInFolder({jobId,className='text-button'}:{jobId:string;className?:string}){
  const [error,setError]=useState('');
  // Through Moise the app runs on another device; the folder would open on the computer, not here.
  if(throughMoise)return null;
  async function open(){setError('');try{await apiRequest(`/api/jobs/${jobId}/show-in-folder`,{method:'POST'});}catch(e){setError((e as Error).message);}}
  return <><button type="button" className={className} onClick={()=>void open()}>Show in folder</button>{error?<small role="alert" className="send-video-error">{error}</small>:null}</>;
}
