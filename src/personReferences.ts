export type ReferencePerson={id:string;images:string[]};
export const personImages=(people:ReferencePerson[])=>people.flatMap(person=>person.images);
export function remapPersonTags(text:string,before:ReferencePerson[],after:ReferencePerson[]){
  let removed=false;
  const value=text.replace(/@Person(\d+)\b/gi,(_,number:string)=>{const old=before[Number(number)-1];const index=after.findIndex(person=>person.id===old?.id);if(index<0){removed=true;return '';}return `@Person${index+1}`;});
  return removed?'':value;
}
export function migratePeople<T extends {reference_images:string[];reference_people?:ReferencePerson[];person_assignments:Record<string,string|null>;prompt:string;replacement_prompt:string;workflow_prompts?:Partial<Record<string,string>>}>(saved:T){
  if(saved.reference_people)return {...saved,reference_people:saved.reference_people};
  const people=saved.reference_images.map((image,index)=>({id:`ref-person-legacy-${index+1}`,images:[image]}));
  const names=new Map(people.map(person=>[person.images[0],person.id]));
  const prompt=(value:string)=>value.replace(/@Person(\d+)/gi,'@Source$1').replace(/@Ref(\d+)/gi,'@Person$1');
  return {...saved,reference_people:people,person_assignments:Object.fromEntries(Object.entries(saved.person_assignments).map(([key,value])=>[key,value?names.get(value)||value:null])),prompt:prompt(saved.prompt),replacement_prompt:prompt(saved.replacement_prompt),workflow_prompts:Object.fromEntries(Object.entries(saved.workflow_prompts||{}).map(([key,value])=>[key,prompt(value||'')]))};
}
