import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';
const source=readFileSync(new URL('../src/sourceSound.ts',import.meta.url),'utf8');
const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
const {soundForSource}=await import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));
const tab=readFileSync(new URL('../src/ReferenceTab.tsx',import.meta.url),'utf8');

test('a source video with no sound gives a video with no sound',()=>{
  // 2026-11-19: a 15 s source with no sound stream and "Generate audio": the model made up speech
  for(const current of ['generated','source','mute'])assert.equal(soundForSource(false,false,current,'generated'),'mute');
  assert.equal(soundForSource(false,true,'mute','generated'),'mute');
});

test('a source with sound keeps the chosen setting, and gets the usual one again after a silent source',()=>{
  for(const current of ['generated','source','mute'])assert.equal(soundForSource(true,false,current,'generated'),current);
  assert.equal(soundForSource(true,true,'mute','generated'),'generated');
  assert.equal(soundForSource(true,true,'generated','generated'),'generated','chosen by hand for the silent source: kept');
});

test('the form applies it when a source is selected and says that the source has no sound',()=>{
  assert.match(tab,/function acceptSource\(video:UploadedVideo,target\?:Workflow\)\{const previousWasSilent=source\?\.has_audio===false;setSource\(video\);/);
  assert.match(tab,/output_audio:soundForSource\(video\.has_audio,previousWasSilent,p\.output_audio,initial\.output_audio\)/);
  assert.match(tab,/<option value="source" disabled=\{source\?\.has_audio===false\}>Keep source audio<\/option>/);
  assert.match(tab,/\{source\?\.has_audio===false&&<small className="reference-help">The source video has no sound\.<\/small>\}/);
});
