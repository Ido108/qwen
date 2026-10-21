import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';

const source=readFileSync(new URL('../src/loraChoices.ts',import.meta.url),'utf8');
const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {enabledLoras,compatibleLoras,replaceLora,loraFit,speedSteps}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
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

test('architecture changes keep every saved selection and send only compatible ones',()=>{
  const selected=[{id:'portrait',strength:1.5,prompt_addition:'Paper.'},{id:'motion',strength:1},{id:'deleted',strength:1}];
  assert.deepEqual(compatibleLoras(selected,models,'image'),[selected[0],selected[1]]);
  assert.deepEqual(enabledLoras(selected,models,'image'),[selected[0]]);
  assert.equal(selected.length,3);
});

test('replacement keeps the slot, strength and bypass but drops the previous adapter prompt',()=>{
  const draft=[{id:'portrait',strength:1.45,enabled:false,prompt_addition:'Old portrait words.'},{id:'style',strength:.7}];
  const next=replaceLora(draft,'portrait','new-portrait');
  assert.deepEqual(next,[{id:'new-portrait',strength:1.45,enabled:false},draft[1]]);
  assert.equal(draft[0].prompt_addition,'Old portrait words.');
  assert.equal(replaceLora(draft,'portrait','style'),draft);
  assert.equal(replaceLora(draft,'portrait','portrait'),draft);
  assert.deepEqual(replaceLora(draft,'missing','new'),draft);
});

test('large stacks retain every compatible enabled selection in order',()=>{
  const catalog=Array.from({length:100},(_,i)=>({id:`style-${i}`,families:['image']}));
  const draft=catalog.map(({id})=>({id,strength:.7}));
  assert.deepEqual(enabledLoras(compatibleLoras(draft,catalog,'image'),catalog,'image'),draft);
});

const qwen=['qwen-image-2.1','qwen-image-2.1-turbo'];
test('a speed LoRA is blocked on Turbo models and suggests its steps on Standard',()=>{
  const speed={id:'viggle',families:qwen,accelerator:{steps:4}};
  assert.equal(loraFit(speed,'qwen-image-2.1-turbo').usable,false);
  assert.match(loraFit(speed,'qwen-image-2.1-turbo').blocked,/already distilled/);
  assert.deepEqual(loraFit(speed,'qwen-image-2.1').settings,{steps:4,cfg:1,family:'qwen-image-2.1'});
  assert.deepEqual(enabledLoras([{id:'viggle',strength:1}],[speed],'qwen-image-2.1-turbo'),[]);
  assert.deepEqual(enabledLoras([{id:'viggle',strength:1}],[speed],'qwen-image-2.1'),[{id:'viggle',strength:1}]);
});
test('a LoRA trained on another variant is sent with a notice and the author settings',()=>{
  const style={id:'grain',families:qwen,trained_on:{family:'qwen-image-2.1',label:'Qwen-Image-2.1 Standard',settings:{steps:50,cfg:3,sampler:'res_2m',scheduler:'beta',vae_id:'vae'}}};
  const fit=loraFit(style,'qwen-image-2.1-turbo');
  assert.equal(fit.usable,true);
  assert.match(fit.notice,/Trained on Qwen-Image-2.1 Standard/);
  assert.deepEqual(fit.settings,{steps:50,cfg:3,sampler:'res_2m',scheduler:'beta',vae_id:'vae',family:'qwen-image-2.1'});
  assert.deepEqual(loraFit(style,'qwen-image-2.1'),{usable:true});
  assert.deepEqual(loraFit({...style,trained_on:{family:null,label:'Qwen-Image-2.1'}},'qwen-image-2.1-turbo'),{usable:true});
  assert.equal(loraFit(style,'krea-2').blocked,'Not for this model.');
});

test('speed LoRA steps follow the author size table', () => {
  const fast = {steps: 4, steps_by_size: [[1048576, 4], [2359296, 6], [4194304, 8]]};
  assert.equal(speedSteps(fast, 1024 * 1024), 4);
  assert.equal(speedSteps(fast, 1536 * 1536), 6);
  assert.equal(speedSteps(fast, 2048 * 2048), 8);
  assert.equal(speedSteps(fast, 2688 * 1536), 8);
  assert.equal(speedSteps({steps: 4}, 2048 * 2048), 4);
});
