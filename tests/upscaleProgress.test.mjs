import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';
const source=readFileSync(new URL('../src/upscaleProgress.ts',import.meta.url),'utf8');
const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {upscaleFraction,upscaleStageLabel,elapsedLabel}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
test('the bar moves inside a stage by time and never passes 95% of the stage',()=>{
  const job={status:'running',progress:.4,progress_stage_end:.8,progress_estimate_seconds:10,progress_at:100,step:3,total_steps:5,started_at:90};
  assert.equal(upscaleFraction(job,100_000),.4);
  assert.ok(Math.abs(upscaleFraction(job,105_000)-.6)<1e-9);
  assert.ok(Math.abs(upscaleFraction(job,200_000)-(.4+.4*.95))<1e-9);
  assert.equal(upscaleFraction({status:'queued'},0),0);
  assert.equal(upscaleFraction({status:'succeeded'},0),1);
  assert.equal(upscaleStageLabel(job),'Stage 3 of 5');
  assert.equal(upscaleStageLabel({status:'running',step:1,total_steps:1}),'');
  assert.equal(elapsedLabel(job,165_000),'1m 15s');
});
