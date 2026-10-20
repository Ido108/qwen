import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';
const source=readFileSync(new URL('../src/resultLoras.ts',import.meta.url),'utf8');
const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {resultLoras}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));

test('result keeps the applied tool and user LoRAs, including exact names and strengths',()=>{
  const saved='Head swap (1.45), Ink, paper (v2) | דיו (0.6), Light (0)';
  assert.equal(resultLoras({applied_lora:saved,request:{loras:[{id:'new-selection',strength:2}]}}),saved);
  // A renamed, deleted or bypassed current selection cannot change this saved fact.
  assert.equal(resultLoras({applied_lora:saved,request:{loras:[]}}),saved);
});

test('known no-LoRA run differs from missing metadata; never infer use from a request',()=>{
  assert.equal(resultLoras({applied_lora:null,request:{loras:[{id:'disabled',enabled:false}]}}),'None');
  assert.equal(resultLoras({applied_lora:''}),'None');
  assert.equal(resultLoras({request:{loras:[{id:'selected-only',strength:1}]}}),null);
  assert.equal(resultLoras({lora:'Loaded before a failed run (1)'}),null);
  assert.equal(resultLoras({applied_lora:42}),null);
});
