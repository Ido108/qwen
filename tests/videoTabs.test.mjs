import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';
const source=readFileSync(new URL('../src/videoTabs.ts',import.meta.url),'utf8');
const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {videoEntry,videoHash,VIDEO_HASHES}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
const main=readFileSync(new URL('../src/main.tsx',import.meta.url),'utf8');

test('Image to video is the first frame page, and reference images is its own page',()=>{
  // 2026-11-19: the tab named "Image to video" opened the reference images page, where a picture is not a frame
  assert.deepEqual(videoEntry('#video'),{section:'create',mode:'image'});
  assert.deepEqual(videoEntry(''),{section:'create',mode:'image'});
  assert.deepEqual(videoEntry('#keyframes'),{section:'create',mode:'image'});
  assert.deepEqual(videoEntry('#text-to-video'),{section:'create',mode:'text'});
  assert.deepEqual(videoEntry('#multi-reference'),{section:'multi-reference'});
  assert.deepEqual(videoEntry('#video-to-video'),{section:'reference'});
  assert.deepEqual(videoEntry('#reference'),{section:'reference'});
  assert.deepEqual(videoEntry('#upscale'),{section:'upscale'});
  assert.deepEqual(videoEntry('#orbit'),{section:'orbit'});
});

test('every page writes a link that opens the same page again',()=>{
  for(const [section,mode] of [['create','image'],['create','text'],['multi-reference','image'],['reference','image'],['upscale','text'],['orbit','image']]){
    const hash=videoHash(section,mode);
    assert.ok(VIDEO_HASHES.includes(hash),hash);
    assert.equal(videoEntry(hash).section,section);
    if(section==='create')assert.equal(videoEntry(hash).mode,mode);
  }
});

test('the tabs of the screen open what their names say',()=>{
  const tab=name=>main.match(new RegExp(`<button role="tab" aria-selected=\\{([^}]*)\\} onClick=\\{\\(\\)=>([^}]*)\\}>${name}</button>`));
  assert.equal(tab('Image to video')[2],"showVideo('create','image')");
  assert.equal(tab('Image to video')[1],"videoSection==='create'&&videoMode==='image'");
  assert.equal(tab('Text to video')[2],"showVideo('create','text')");
  assert.equal(tab('Reference images')[2],"showVideo('multi-reference')");
  // The phone list has the same three entries
  for(const option of ['<option value="create:image">Image to video</option>','<option value="create:text">Text to video</option>','<option value="multi-reference">Reference images</option>'])assert.ok(main.includes(option),option);
});
