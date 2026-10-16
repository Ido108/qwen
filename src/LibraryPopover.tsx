import {useEffect,useLayoutEffect,useRef,useState,type CSSProperties,type ReactNode,type RefObject} from 'react';

/** A floating panel next to (or below) its anchor. It closes on Escape and on a click outside the panel and the anchor.
 * It stays mounted while closed, so its content keeps its state. */
export function LibraryPopover({open,onClose,anchor,children}:{open:boolean;onClose:()=>void;anchor:RefObject<HTMLElement|null>;children:ReactNode}){
  const box=useRef<HTMLDivElement>(null);const [style,setStyle]=useState<CSSProperties>({});
  useLayoutEffect(()=>{if(!open)return;
    function place(){
      const r=anchor.current?.getBoundingClientRect();if(!r)return;
      const vw=window.innerWidth,vh=window.innerHeight,gap=10,edge=12,maxHeight=Math.round(vh*0.6);
      // Wide windows: open to the right of the controls, over the result. Narrow windows: open below the input panel.
      if(vw-r.right>=420+gap+edge)setStyle({left:r.right+gap,top:Math.max(edge,Math.min(r.top,vh-maxHeight-edge)),width:Math.min(620,vw-r.right-gap-edge),maxHeight});
      else{const top=Math.max(edge,Math.min(r.bottom+6,vh-maxHeight-edge));setStyle({left:Math.max(edge,r.left),top,width:Math.min(r.width,vw-2*edge),maxHeight});}
    }
    place();window.addEventListener('resize',place);window.addEventListener('scroll',place,true);
    return()=>{window.removeEventListener('resize',place);window.removeEventListener('scroll',place,true);};
  },[open,anchor]);
  useEffect(()=>{if(!open)return;
    const down=(e:MouseEvent)=>{const target=e.target as Element;if(box.current?.contains(target)||anchor.current?.contains(target)||target.closest?.('dialog'))return;onClose();};
    const key=(e:KeyboardEvent)=>{if(e.key==='Escape'&&!document.querySelector('dialog[open]'))onClose();};
    document.addEventListener('mousedown',down);document.addEventListener('keydown',key);
    return()=>{document.removeEventListener('mousedown',down);document.removeEventListener('keydown',key);};
  },[open,onClose,anchor]);
  return <div ref={box} className="library-popover" role="dialog" aria-label="Image library" hidden={!open} style={style}>{children}</div>;
}
