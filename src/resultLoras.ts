/** The engine saves names and strengths after it applies the complete LoRA stack.
 * Keep this text intact: names can contain commas, parentheses and Unicode.
 * Missing metadata is different from a recorded run without LoRAs.
 */
export function resultLoras(run:{applied_lora?:string|null}):string|null{
  if(run.applied_lora===null)return 'None';
  if(typeof run.applied_lora==='string')return run.applied_lora.trim()||'None';
  return null;
}
