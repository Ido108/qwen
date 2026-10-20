import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';

const source=readFileSync(new URL('../src/loraChoices.ts',import.meta.url),'utf8');
const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {enabledLoras,compatibleLoras}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
const models=[{id:'portrait',families:['image']},{id:'style',families:['image']},{id:'motion',families:['video']}];

test('bypass leaves the saved strength intact and excludes weights from a job',()=>{
  const draft=[{id:'portrait',strength:.6,enabled:false},{id:'style',strength:.85}];
  const saved=structuredClone(draft);
  assert.deepEqual(enabledLoras(draft,models,'image'),[{id:'style',strength:.85}]);
  assert.deepEqual(draft,saved);
  draft[0].enabled=true;
  assert.deepEqual(enabledLoras(draft,models,'image'),[{id:'portrait',strength:.6},{id:'style',strength:.85}]);
});

test('old drafts stay enabled; zero strength and disabled state remain distinct',()=>{
  assert.deepEqual(enabledLoras([{id:'portrait',strength:0},{id:'style',strength:1.5,enabled:true}],models,'image'),
    [{id:'portrait',strength:0},{id:'style',strength:1.5}]);
});

test('a model change preserves choices but sends only compatible LoRAs',()=>{
  const draft=[{id:'portrait',strength:.6},{id:'motion',strength:1},{id:'missing',strength:1}];
  assert.deepEqual(enabledLoras(draft,models,'video'),[{id:'motion',strength:1}]);
  assert.deepEqual(enabledLoras(draft,models,'image'),[{id:'portrait',strength:.6}]);
  assert.equal(draft.length,3);
});

test('architecture changes remove incompatible selections and retain saved additions',()=>{
  const selected=[{id:'portrait',strength:1.5,prompt_addition:'Paper.'},{id:'motion',strength:1}];
  assert.deepEqual(compatibleLoras(selected,models,'image'),[selected[0]]);
  assert.deepEqual(enabledLoras(selected,models,'image'),[selected[0]]);
  assert.equal(selected.length,2);
});
