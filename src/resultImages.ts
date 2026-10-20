/** Browser-only, bounded decoded-image cache. Never store private images on a shared service. */
const ready=new Map<string,HTMLImageElement>();
type Pending={image:HTMLImageElement;promise:Promise<void>;users:number;cancel:()=>void};
const pending=new Map<string,Pending>();
const MAX_READY=12;

export function resultSources(src:string){
  if(!/\/api\/jobs\/[a-f0-9]{32}\/image$/.test(src))return null;
  return {thumbnail:src.replace(/\/image$/,'/thumbnail'),display:src.replace(/\/image$/,'/display'),original:src};
}
export function imageReady(src:string){
  const image=ready.get(src);
  if(image){ready.delete(src);ready.set(src,image);}
  return !!image;
}

export function acquireImage(src:string,priority:'high'|'low'='high'){
  if(imageReady(src))return {promise:Promise.resolve(),release:()=>{}};
  let entry=pending.get(src);
  if(!entry){
    const image=new Image();image.decoding='async';image.fetchPriority=priority;
    let cancel=()=>{};
    const promise=new Promise<void>((resolve,reject)=>{
      let ended=false;
      const finish=(error?:Error)=>{
        if(ended)return;ended=true;image.onload=null;image.onerror=null;pending.delete(src);
        if(error){reject(error);return;}
        ready.set(src,image);
        while(ready.size>MAX_READY)ready.delete(ready.keys().next().value!);
        resolve();
      };
      image.onload=()=>{image.decode().then(()=>finish(),()=>finish());};
      image.onerror=()=>finish(new Error('The image could not load.'));
      cancel=()=>{finish(new Error('Image selection changed.'));image.removeAttribute('src');};
    });
    entry={image,promise,users:0,cancel};pending.set(src,entry);image.src=src;
  }else if(priority==='high')entry.image.fetchPriority='high';
  entry.users++;
  const current=entry;let released=false;
  return {promise:current.promise,release:()=>{
    if(released)return;released=true;current.users--;
    if(!current.users&&pending.get(src)===current)current.cancel();
  }};
}

export function canPrefetch(){
  const network=(navigator as Navigator&{connection?:{saveData?:boolean;effectiveType?:string}}).connection;
  return !network?.saveData&&!['slow-2g','2g'].includes(network?.effectiveType||'');
}
