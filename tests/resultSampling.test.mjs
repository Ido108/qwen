import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';
const source=readFileSync(new URL('../src/resultSampling.ts',import.meta.url),'utf8');
const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {resultSampling,scheduleLabel}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));

test('actual saved sampler and schedule win over legacy labels and request values',()=>{
  assert.deepEqual(resultSampling({sampler:'res_2m',scheduler:'beta',sampling:'euler-simple',request:{sampler:'res_2s',scheduler:'normal'}}),
    {sampler:'res_2m',scheduler:'beta'});
});
test('older image records retain exact sampler names with either saved separator',()=>{
  assert.deepEqual(resultSampling({sampling:'res_2s_ode-beta'}),{sampler:'res_2s_ode',scheduler:'beta'});
  assert.deepEqual(resultSampling({sampling:'euler/simple'}),{sampler:'euler',scheduler:'simple'});
});
test('fixed schedules stay distinct; missing fields are not invented from the request',()=>{
  assert.deepEqual(resultSampling({sampling:'turbo-sigmas',request:{sampler:'res_2m',scheduler:'beta'}}),
    {sampler:null,scheduler:'turbo-sigmas'});
  assert.equal(scheduleLabel('turbo-sigmas'),'Turbo (fixed)');
  assert.equal(scheduleLabel('ltx-distilled'),'LTX distilled (fixed)');
  assert.deepEqual(resultSampling({request:{sampler:'euler'}}),{sampler:null,scheduler:null});
  assert.deepEqual(resultSampling({sampling:{sampler:'euler'},sampler:NaN}),{sampler:null,scheduler:null});
});
