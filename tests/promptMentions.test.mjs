import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';
const source=readFileSync(new URL('../src/promptMentions.ts',import.meta.url),'utf8');
const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {mentionAt,imageMentionOptions,displayImageMentions}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
test('mention search accepts Hebrew names and ignores email addresses',()=>{
  assert.equal(mentionAt('name@example.com',16),null);
  assert.deepEqual(mentionAt('Use @משה',8),{start:4,end:8,query:'משה'});
  assert.equal(mentionAt('@Image1 ',8),null);
});
test('a library person uses the next free slot and preserves selected identities',()=>{
  const people=[{id:'p',name:'Person name',image_id:'p1',image_ids:['p1','p2'],url:'/p1',thumb_url:'/thumb'}];
  let added;
  const options=imageMentionOptions([null,{id:'scene',name:'Scene',url:'/scene'}],people,image=>added=image,2);
  assert.equal(options[0].tag,'@Image2');assert.equal(options[1].tag,'@Image1');
  options[1].choose();assert.equal(added.personId,'p');
  const selected=imageMentionOptions([{id:'p2',name:'file',url:'/p2'}],people,()=>{},1);
  assert.equal(selected.length,1);assert.equal(selected[0].label,'Person name');
});
test('canonical mentions become readable without changing other text',()=>{
  assert.equal(displayImageMentions('Use <image 1>, <Picture 2> and <style_source>.'),'Use @Image1, @Image2 and @StyleSource.');
});
test('labels describe the subject without changing its tag or guessing shared-photo ownership',()=>{
  const image={id:'shared',name:'Photo',url:'/photo',subjectLabel:'dog'};
  const people=['Alice','Bob'].map(name=>({id:name,name,image_ids:['shared']}));
  const option=imageMentionOptions([image],people)[0];
  assert.equal(option.tag,'@Image1');assert.equal(option.label,'Photo');assert.equal(option.detail,'Dog · Image 1');
  assert.equal(imageMentionOptions([{...image,personId:'Bob'}],people)[0].label,'Bob');
});
