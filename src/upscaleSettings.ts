import type {ModelEntry} from './ModelSettings';
export type UpscaleParams={model_id:string;source_video:string;resolution:number;size_mode:'short_side'|'scale';scale:2|4;batch_size:number;chunk_size:number;temporal_overlap:number;seed:number;preserve_audio:boolean;output_dir:string;memory_mode:'auto'|'low_memory'|'gpu';color_correction:string;clip_len:number;flash_chunk_size:number;flash_sampling:string;flash_sparse_ratio:number;flash_local_range:number;flash_conditioning:number;flash_cache_format:string;flash_decoder:'tiny'|'full';flash_color:string};
export const upscaleInitial:UpscaleParams={model_id:'seedvr2-3b-fp16',source_video:'',resolution:1080,size_mode:'short_side',scale:2,batch_size:21,chunk_size:100,temporal_overlap:1,seed:100,preserve_audio:true,output_dir:'',memory_mode:'auto',color_correction:'lab',clip_len:8,flash_chunk_size:33,flash_sampling:'streaming_faithful_lowvram',flash_sparse_ratio:2,flash_local_range:11,flash_conditioning:1,flash_cache_format:'int8',flash_decoder:'tiny',flash_color:'off'};
export function modelDefaults(model:ModelEntry):Partial<UpscaleParams>{
  const d=model.upscale_defaults;
  return {model_id:model.id,batch_size:d?.batch_size||5,chunk_size:d?.chunk_size||100,temporal_overlap:d?.temporal_overlap??1,color_correction:d?.color_correction||'lab',clip_len:model.family==='swiftvr'?d?.clip_len||8:8,flash_chunk_size:model.family==='flashvsr'?d?.clip_len||33:33,flash_sampling:'streaming_faithful_lowvram',flash_sparse_ratio:2,flash_local_range:11,flash_conditioning:1,flash_cache_format:'int8',flash_decoder:'tiny',flash_color:'off'};
}
export function upscaleDimensions(width:number,height:number,p:UpscaleParams){
  const factor=p.size_mode==='scale'?p.scale:p.resolution/Math.min(width,height);
  return [width,height].map(side=>Math.max(2,Math.floor(side*factor/2+.5)*2));
}
