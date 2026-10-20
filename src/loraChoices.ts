export type LoraChoice={id:string;strength:number;enabled?:boolean;prompt_addition?:string|null};

export function compatibleLoras(value:LoraChoice[],models:{id:string;families?:string[]}[],family:string){
  return value.filter(choice=>models.some(model=>model.id===choice.id&&model.families?.includes(family)));
}

/** Keep bypass choices in the draft, but send only applicable weights to a job. */
export function enabledLoras(value:LoraChoice[],models:{id:string;families?:string[]}[],family:string){
  return value.filter(choice=>choice.enabled!==false&&models.some(model=>model.id===choice.id&&model.families?.includes(family)))
    .map(({id,strength,prompt_addition})=>({id,strength,...(prompt_addition!=null?{prompt_addition}:{})}));
}
