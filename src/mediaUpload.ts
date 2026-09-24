export type UploadedVideo={id:string;name:string;url:string;poster_url:string;bytes:number;width:number;height:number;fps:number|null;duration_seconds:number|null;has_audio:boolean};

export function uploadVideo(file:File,token:string,onProgress:(fraction:number)=>void):Promise<UploadedVideo>{
  return new Promise((resolve,reject)=>{
    const xhr=new XMLHttpRequest();
    xhr.open('PUT','/api/video-uploads');
    xhr.setRequestHeader('X-Studio-Token',token);
    xhr.setRequestHeader('X-File-Name',encodeURIComponent(file.name));
    xhr.upload.onprogress=event=>{if(event.lengthComputable)onProgress(event.loaded/event.total);};
    xhr.onerror=()=>reject(new Error('Upload failed. Check the connection and try again.'));
    xhr.onabort=()=>reject(new Error('Upload cancelled.'));
    xhr.onload=()=>{
      let body:any={};try{body=JSON.parse(xhr.responseText);}catch{}
      if(xhr.status>=200&&xhr.status<300)resolve(body as UploadedVideo);
      else reject(new Error(typeof body.detail==='string'?body.detail:`Upload failed (${xhr.status}).`));
    };
    xhr.send(file);
  });
}
