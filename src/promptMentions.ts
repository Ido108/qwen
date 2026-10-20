import type {InputImage,Person} from './InputLibrary';

export type MentionOption={tag:string;label:string;image?:string;detail?:string;choose?:()=>void;key?:string};
export function displayImageMentions(text:string){return text.replace(/<\s*(?:image|picture)\s*(\d+)\s*>/gi,(_,n)=>`@Image${Number(n)}`).replaceAll('<pose_source>','@PoseSource').replaceAll('<style_source>','@StyleSource');}
export function imageMentionOptions(images:(InputImage|null)[],people:Person[]=[],add?:(image:InputImage)=>void,limit=images.length):MentionOption[]{
  const current=images.flatMap((image,i)=>{
    if(!image)return [];
    const owners=people.filter(p=>p.image_ids.includes(image.id));
    const person=image.personId?owners.find(p=>p.id===image.personId):owners.length===1?owners[0]:undefined;
    const subject=image.subjectLabel?.trim()||'person';
    return [{tag:`@Image${i+1}`,label:person?.name||image.name||`Image ${i+1}`,image:image.url,detail:`${subject[0].toUpperCase()+subject.slice(1)} · Image ${i+1}`}];
  });
  if(!add||images.filter(Boolean).length>=limit)return current;
  const next=images.indexOf(null)>=0?images.indexOf(null):images.length;
  return [...current,...people.filter(p=>!images.some(i=>i&&p.image_ids.includes(i.id))).map(p=>({
    key:p.id,tag:`@Image${next+1}`,label:p.name,image:p.thumb_url,detail:'Add from People',
    choose:()=>add({id:p.image_id,name:p.name,url:p.url,personId:p.id})
  }))];
}

export function mentionAt(value:string,caret:number){
  const match=value.slice(0,caret).match(/(?:^|[^\p{L}\p{N}_@])(@[^\s@<>]*)$/u);
  return match?{start:caret-match[1].length,end:caret,query:match[1].slice(1).toLocaleLowerCase()}:null;
}
