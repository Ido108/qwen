import {useEffect,useLayoutEffect,useRef,useState,type CSSProperties,type ReactNode,type RefObject} from 'react';
import {createPortal} from 'react-dom';

/** A photo picker beside its anchor on desktop and in a bottom sheet on mobile.
 * Native modality keeps touches and keyboard focus inside until selection or close. */
export function LibraryPopover({open,onClose,anchor,children}:{open:boolean;onClose:()=>void;anchor:RefObject<HTMLElement|null>;children:ReactNode}){
  const box=useRef<HTMLDialogElement>(null);const [style,setStyle]=useState<CSSProperties>({});
  useLayoutEffect(()=>{if(!open){box.current?.close();return;}
    const previous=document.activeElement as HTMLElement|null;
    function place(){
      const r=anchor.current?.getBoundingClientRect();if(!r)return;
      const vw=window.innerWidth,vh=window.visualViewport?.height||window.innerHeight,gap=10,edge=12,maxHeight=Math.round(vh*0.7);
      if(vw<760||(vw<=1000&&matchMedia('(pointer:coarse)').matches)){setStyle({});return;}
      // Wide windows: open to the right of the controls, over the result. Narrow windows: open below the input panel.
      if(vw-r.right>=420+gap+edge)setStyle({left:r.right+gap,top:Math.max(edge,Math.min(r.top,vh-maxHeight-edge)),width:Math.min(620,vw-r.right-gap-edge),maxHeight});
      else{const top=Math.max(edge,Math.min(r.bottom+6,vh-maxHeight-edge));setStyle({left:Math.max(edge,r.left),top,width:Math.min(r.width,vw-2*edge),maxHeight});}
    }
    place();box.current?.showModal();window.addEventListener('resize',place);window.visualViewport?.addEventListener('resize',place);
    return()=>{window.removeEventListener('resize',place);window.visualViewport?.removeEventListener('resize',place);box.current?.close();previous?.focus({preventScroll:true});};
  },[open,anchor]);
  useEffect(()=>{if(!open)return;
    const original=document.body.style.overflow;document.body.style.overflow='hidden';
    return()=>{document.body.style.overflow=original;};
  },[open]);
  return createPortal(<dialog ref={box} className="library-popover" aria-label="Image library" style={style}
    onCancel={e=>{e.preventDefault();onClose();}}
    onClick={e=>{if(e.target!==box.current)return;const r=box.current.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)onClose();}}>{children}</dialog>,document.body);
}
