export type LoraChoice={id:string;strength:number;enabled?:boolean;prompt_addition?:string|null};
/** Steps, CFG, sampler, scheduler and VAE from the author's own example graphs. */
export type AuthorSettings={steps?:number;cfg?:number;sampler?:string;scheduler?:string;vae_id?:string};
export type LoraMeta={id:string;families?:string[];accelerator?:{steps:number|null;steps_by_size?:[number,number][]}|null;
  trained_on?:{family:string|null;label:string;settings?:AuthorSettings|null}|null;test_file?:string};
export type LoraFit={usable:boolean;blocked?:string;notice?:string;settings?:AuthorSettings&{family:string}};

/** Replace one slot without moving it or copying another adapter's prompt text. */
export function replaceLora(value:LoraChoice[],currentId:string,nextId:string){
  if(currentId===nextId||value.some(choice=>choice.id===nextId))return value;
  return value.map(choice=>{
    if(choice.id!==currentId)return choice;
    const {prompt_addition,...settings}=choice;
    return {...settings,id:nextId};
  });
}

/** A model change keeps every saved choice (a LoRA for another model waits in the list); only removed LoRAs go. */
export function compatibleLoras(value:LoraChoice[],models:{id:string;families?:string[]}[],_family:string){
  return value.filter(choice=>models.some(model=>model.id===choice.id));
}

/** Whether a LoRA applies to this model, and what to tell the user when it was made for another one. */
/** Speed LoRA steps for this output size: some authors publish more steps for larger images. */
export function speedSteps(fast:{steps:number|null;steps_by_size?:[number,number][]},pixels?:number){
  const table=fast.steps_by_size;
  if(!table?.length||!pixels)return fast.steps;
  return (table.find(([limit])=>pixels<=limit)??table[table.length-1])[1];
}

export function loraFit(lora:LoraMeta,family:string,pixels?:number):LoraFit{
  if(!(lora.families||[]).includes(family))return {usable:false,blocked:'Not for this model.'};
  const fast=lora.accelerator;
  if(fast){
    if(family.endsWith('turbo'))return {usable:false,blocked:`Speed LoRA: this Turbo model is already distilled, so it is not used. Use it with a Standard model${fast.steps?` at ${fast.steps} steps`:''}.`};
    const steps=speedSteps(fast,pixels);
    return steps?{usable:true,notice:`Speed LoRA: use ${steps} steps and CFG 1${fast.steps_by_size&&pixels?' at this size':''}.`,settings:{steps,cfg:1,family}}:{usable:true};
  }
  const trained=lora.trained_on;
  if(trained?.family&&trained.family!==family){
    const settings=trained.settings&&Object.keys(trained.settings).length?{...trained.settings,family:trained.family}:undefined;
    return {usable:true,notice:`Trained on ${trained.label}. With this model the effect can be weaker.`,settings};
  }
  return {usable:true};
}

/** Sampling the active LoRAs ask for on this model and size: a speed LoRA's steps and CFG 1 first, then the
 * author's own steps, CFG, sampler and schedule when the LoRA was made for this same model family. */
export function loraSampling(active:{id:string}[],metas:(LoraMeta&{title?:string})[],family:string,pixels?:number):{values:AuthorSettings;note:string}|null{
  const used=active.map(choice=>metas.find(meta=>meta.id===choice.id)).filter((meta):meta is LoraMeta&{title?:string}=>!!meta);
  for(const meta of used){
    const fit=loraFit(meta,family,pixels);
    if(meta.accelerator&&fit.usable&&fit.settings?.steps)
      return {values:{steps:fit.settings.steps,cfg:1},note:`Steps ${fit.settings.steps} and CFG 1 from the speed LoRA${meta.accelerator.steps_by_size&&pixels?' for this size':''}.`};
  }
  for(const meta of used){
    const s=meta.trained_on?.family===family?meta.trained_on.settings:null;
    if(!s)continue;
    const values:AuthorSettings={...(s.steps?{steps:s.steps}:{}),...(s.cfg?{cfg:s.cfg}:{}),...(s.sampler?{sampler:s.sampler}:{}),...(s.scheduler?{scheduler:s.scheduler}:{})};
    const parts=[values.steps?`steps ${values.steps}`:'',values.cfg?`CFG ${values.cfg}`:'',values.sampler?`sampler ${values.sampler}`:'',values.scheduler?`schedule ${values.scheduler}`:''].filter(Boolean);
    if(parts.length)return {values,note:`${parts.join(', ').replace(/^./,c=>c.toUpperCase())} from the author of ${meta.title||'this LoRA'}.`};
  }
  return null;
}

/** Keep bypass choices in the draft, but send only applicable weights to a job. */
export function enabledLoras(value:LoraChoice[],models:LoraMeta[],family:string){
  return value.filter(choice=>{
    const lora=models.find(model=>model.id===choice.id);
    return choice.enabled!==false&&!!lora&&loraFit(lora,family).usable;
  }).map(({id,strength,prompt_addition})=>({id,strength,...(prompt_addition!=null?{prompt_addition}:{})}));
}
