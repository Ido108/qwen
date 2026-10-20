import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';
const source=readFileSync(new URL('../src/resultCfg.ts',import.meta.url),'utf8');
const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {resultCfg}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
test('completed result uses actual sampler CFG, not mutable request defaults',()=>{
  assert.equal(resultCfg({true_cfg_scale:1,cfg:2,request:{cfg:7}}),1);
  assert.equal(resultCfg({cfg:4.5}),4.5);
  assert.equal(resultCfg({guidance:1}),1);
  assert.equal(resultCfg({request:{cfg:7}}),null);
  assert.equal(resultCfg({cfg:NaN}),null);
  assert.equal(resultCfg({cfg:0}),0);
});
