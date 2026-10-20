import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';
class FakeImage{
  static instances=[];
  constructor(){FakeImage.instances.push(this);}
  decode(){return Promise.resolve();}
  removeAttribute(){this.cancelled=true;}
}
globalThis.Image=FakeImage;
const code=ts.transpileModule(readFileSync(new URL('../src/resultImages.ts',import.meta.url),'utf8'),{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {acquireImage,imageReady,resultSources}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));

test('a shared image survives one viewer closing and cancels when nobody needs it',async()=>{
  const first=acquireImage('/shared'),second=acquireImage('/shared');
  const image=FakeImage.instances.at(-1);
  const rejected=assert.rejects(first.promise,/selection changed/);
  first.release();assert.equal(image.cancelled,undefined);
  second.release();await rejected;assert.equal(image.cancelled,true);assert.equal(imageReady('/shared'),false);
  const retry=acquireImage('/shared');const newImage=FakeImage.instances.at(-1);
  assert.notEqual(newImage,image);newImage.onload();await retry.promise;
  assert.equal(imageReady('/shared'),true);retry.release();
});

test('decoded image reuse is bounded and keeps recently viewed images',async()=>{
  for(let i=0;i<13;i++){
    const lease=acquireImage('/cached-'+i);FakeImage.instances.at(-1).onload();await lease.promise;lease.release();
  }
  assert.equal(imageReady('/cached-0'),false);assert.equal(imageReady('/cached-1'),true);
  const count=FakeImage.instances.length;await acquireImage('/cached-1').promise;
  assert.equal(FakeImage.instances.length,count);
  const next=acquireImage('/extra');FakeImage.instances.at(-1).onload();await next.promise;
  assert.equal(imageReady('/cached-1'),true);assert.equal(imageReady('/cached-2'),false);
});

test('viewing URLs preserve the remote prefix and never rewrite downloads or inputs',()=>{
  const root='/qwen-studio/api/jobs/'+'a'.repeat(32);
  assert.deepEqual(resultSources(root+'/image'),{thumbnail:root+'/thumbnail',display:root+'/display',original:root+'/image'});
  assert.equal(resultSources(root+'/image?download=true'),null);
  assert.equal(resultSources('/api/uploads/example.png'),null);
});
