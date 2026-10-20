import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';
const source=readFileSync(new URL('../src/videoModels.ts',import.meta.url),'utf8');
const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {videoTasks}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
test('new LTX and H3 reference imports reach the reference page by architecture',()=>{
  assert.deepEqual(videoTasks({id:'future-distilled',kind:'video',family:'ltx25'}),['create','references']);
  assert.deepEqual(videoTasks({id:'future-dev',kind:'video',family:'ltx25'}),['create','references']);
  assert.ok(videoTasks({id:'future-ref-turbo',kind:'reference-video',family:'minimax-h3-ref2va'}).includes('references'));
  assert.deepEqual(videoTasks({id:'future-fl2va',kind:'video',family:'minimax-h3'}),['create']);
  assert.deepEqual(videoTasks({kind:'video-lora',family:'ltx25'}),[]);
});
