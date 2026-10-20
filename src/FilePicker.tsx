import {useRef,type ReactNode} from 'react';
import './file-picker.css';

export const IMAGE_ACCEPT='image/*,.heic,.heif';

/** The native input covers the control, so a touch opens the device picker directly. */
export function FilePicker({children,className='',disabled=false,multiple=false,label,title,onFiles,accept=IMAGE_ACCEPT,directory=false}:{
  children:ReactNode;className?:string;disabled?:boolean;multiple?:boolean;label?:string;title?:string;onFiles:(files:File[])=>void;accept?:string;directory?:boolean;
}){
  const input=useRef<HTMLInputElement>(null);
  return <label className={`file-picker ${className}`} role="button" tabIndex={disabled?-1:0} aria-disabled={disabled} aria-label={label} title={title}
    onKeyDown={event=>{if(!disabled&&(event.key==='Enter'||event.key===' ')){event.preventDefault();input.current?.click();}}}>
    {children}
    <input ref={input} type="file" accept={accept} multiple={multiple} disabled={disabled} tabIndex={-1} aria-hidden="true" {...(directory?{webkitdirectory:'',directory:''}:{})}
      onChange={event=>{const files=Array.from(event.currentTarget.files||[]);event.currentTarget.value='';if(files.length)onFiles(files);}}/>
  </label>;
}
