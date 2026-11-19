import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';
const source=readFileSync(new URL('../src/formFields.ts',import.meta.url),'utf8');
const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {formFields}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
const tab=readFileSync(new URL('../src/ReferenceTab.tsx',import.meta.url),'utf8');
// The form of Video to video as the screen defines it
const initial=new Function(`return ${tab.match(/const initial:Params=(\{.*?\});/s)[1]};`)();

test('a form restored from a repaired video keeps its settings and not the repair',()=>{
  // The request of the job that failed on 2026-11-19: a new video of 362 frames with the repair of another video
  const saved={workflow:'faceswap',model_id:'h3-ref-full-q8_0',source_video:'516515b392aa4114a78e61d2e9cf4fb0',frames:362,steps:25,width:576,height:1024,
    prompt:'the person',reference_images:['a'],repair_of:'d34026bc'.padEnd(32,'0'),repair_from_part:17,kind:'reference-video',execution:'local',
    sampling_profile:{steps:25},steps_mode:'manual',long_parts:'continue',continue_run:0};
  const restored={...initial,...formFields(initial,saved)};
  assert.deepEqual(Object.keys(restored).sort(),Object.keys(initial).sort(),'only the fields of the form');
  for(const key of ['repair_of','repair_from_part','kind','execution','sampling_profile','steps_mode','long_parts','continue_run'])assert.equal(key in restored,false,key);
  for(const key of ['model_id','source_video','frames','steps','width','height','prompt'])assert.equal(restored[key],saved[key],key);
  assert.deepEqual(restored.reference_images,['a']);
});

test('a missing or broken saved draft leaves the form as it is',()=>{
  for(const saved of [null,undefined,'text',5,[]])assert.deepEqual({...initial,...formFields(initial,saved)},initial);
  assert.deepEqual(formFields(initial,{steps:null,seed:0}),{steps:null,seed:0});
});

test('the saved draft and a restored job both pass through the form fields',()=>{
  assert.match(tab,/function loadDraft\(\):Params\{[^\n]*\{\.\.\.initial,\.\.\.formFields\(initial,saved\),/);
  assert.match(tab,/setParams\(migratePeople\(\{\.\.\.initial,\.\.\.formFields\(initial,saved\.request\),/);
  assert.equal(/\{\.\.\.initial,\.\.\.saved[,.]/.test(tab),false,'no place spreads a whole saved request into the form');
});
