import {useEffect,useState,type ImgHTMLAttributes} from 'react';
import {acquireImage,canPrefetch,imageReady,resultSources} from './resultImages';
import './result-images.css';

export function useResultImage(src:string,fullSize=false){
  const [state,setState]=useState({source:'',url:'',quality:'thumbnail',failed:false});
  const [attempt,setAttempt]=useState(0);
  const sources=resultSources(src);
  const cached=sources&&(imageReady(src)?{url:src,quality:'original'}:imageReady(sources.display)?{url:sources.display,quality:'display'}:null);
  const current=state.source===src?state:null;
  const selected=cached||current?.url&&current||{url:sources?.thumbnail||src,quality:sources?'thumbnail':'original'};
  useEffect(()=>{
    const images=resultSources(src);if(!images)return;
    let alive=true;const leases:ReturnType<typeof acquireImage>[]=[];
    setState(old=>old.source===src?{...old,failed:false}:{source:src,url:'',quality:'thumbnail',failed:false});
    async function show(url:string,quality:string){
      const lease=acquireImage(url);leases.push(lease);await lease.promise;
      if(alive)setState({source:src,url,quality,failed:false});
    }
    (async()=>{
      try{
        if(imageReady(src)){await show(src,'original');return;}
        try{await show(images!.display,'display');}
        catch(error){if(!alive)return;await show(src,'original');return;}
        if(alive&&fullSize)await show(src,'original');
      }catch{if(alive)setState(old=>({...old,source:src,failed:true}));}
    })();
    return()=>{alive=false;leases.forEach(lease=>lease.release());};
  },[src,fullSize,attempt]);
  const failed=!!current?.failed;
  return {...selected,failed,busy:!!sources&&!failed&&(selected.quality==='thumbnail'||fullSize&&selected.quality!=='original'),retry:()=>setAttempt(n=>n+1)};
}

/** At most the previous and next viewing copies. Cancel obsolete work on fast navigation. */
export function useResultNeighbors(urls:(string|undefined)[],selected?:string|null,enabled=true){
  const index=urls.indexOf(selected||'');
  const next=index>=0?urls[index+1]:undefined,previous=index>0?urls[index-1]:undefined;
  useEffect(()=>{
    if(!enabled||!selected||!canPrefetch())return;
    let alive=true;const leases:ReturnType<typeof acquireImage>[]=[];
    const timer=setTimeout(async()=>{
      // Give the selected image priority over nearby results on a slow connection.
      const current=resultSources(selected);
      if(current){const lease=acquireImage(current.display,'low');leases.push(lease);try{await lease.promise;}catch{return;}}
      for(const url of [next,previous]){
        if(!alive)return;const sources=url&&resultSources(url);if(!sources)continue;
        const lease=acquireImage(sources.display,'low');leases.push(lease);try{await lease.promise;}catch{/* Selection or connection changed. */}
      }
    },250);
    return()=>{alive=false;clearTimeout(timer);leases.forEach(lease=>lease.release());};
  },[selected,next,previous,enabled]);
}

export function ImageLoadState({busy,failed,retry,fullSize=false}:{busy:boolean;failed:boolean;retry:()=>void;fullSize?:boolean}){
  if(failed)return <button type="button" className="image-load-state image-load-retry" onClick={e=>{e.stopPropagation();retry();}}>Could not load image · Retry</button>;
  return busy?<span className="image-load-state" role="status">{fullSize?'Loading full size…':'Loading image…'}</span>:null;
}

export function ProgressiveImage({src,alt,onOpen,fullSize=false,...props}:ImgHTMLAttributes<HTMLImageElement>&{src:string;onOpen?:()=>void;fullSize?:boolean}){
  const image=useResultImage(src,fullSize);
  const picture=<img {...props} key={image.url} src={image.url} alt={alt} decoding="async" data-quality={image.quality}/>;
  return <>{onOpen?<button className="result-image-button" type="button" onClick={onOpen} aria-label="Preview image">{picture}</button>:picture}<ImageLoadState {...image} fullSize={fullSize}/></>;
}
