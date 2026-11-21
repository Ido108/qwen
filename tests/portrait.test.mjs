import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';
const load=async name=>{const source=readFileSync(new URL(`../src/${name}`,import.meta.url),'utf8');
  const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
  return import('data:text/javascript;base64,'+Buffer.from(code).toString('base64'));};
const {videoEntry,videoHash,sendSection,VIDEO_HASHES}=await load('videoTabs.ts');
const {portraitGroups,portraitDefaults,portraitOptions,portraitModel,ENGINES,SYNC_MODELS}=await load('portraitControls.ts');
const main=readFileSync(new URL('../src/main.tsx',import.meta.url),'utf8');
const backend=readFileSync(new URL('../backend/portrait_settings.py',import.meta.url),'utf8');
const allowed=name=>new Set([...backend.match(new RegExp(`${name} = \\{([^}]*)\\}`))[1].matchAll(/'([a-z_0-9]+)'/g)].map(m=>m[1]));

test('the new video pages open from their links and their tabs',()=>{
  for(const [hash,section,name] of [['#animate','animate','Animate image'],['#lip-sync','lipsync','Lip sync'],['#extract-audio','extract-audio','Extract audio']]){
    assert.deepEqual(videoEntry(hash),{section});
    assert.equal(videoHash(section,'image'),hash);
    assert.ok(VIDEO_HASHES.includes(hash));
    assert.ok(main.includes(`onClick={()=>showVideo('${section}')}>${name}</button>`),name);
    assert.ok(main.includes(`<option value="${section}">${name}</option>`),`phone list: ${name}`);
  }
  // The pages that existed keep their links
  assert.equal(videoHash('upscale','text'),'#upscale');assert.equal(videoHash('reference','image'),'#video-to-video');
});

test('every Send to target opens the page that takes it',()=>{
  assert.equal(sendSection('lipsync'),'lipsync');assert.equal(sendSection('lipsync-audio'),'lipsync');
  assert.equal(sendSection('animate'),'animate');assert.equal(sendSection('animate-audio'),'animate');
  assert.equal(sendSection('extract-audio'),'extract-audio');assert.equal(sendSection('upscale'),'upscale');
  for(const target of ['faceswap','crossview','reference'])assert.equal(sendSection(target),'reference');
});

test('every setting of every engine and lip sync model is a key the backend passes on',()=>{
  const animate=new Set([...allowed('ANIMATION_OPTIONS'),...allowed('SYNC_OPTIONS')]);
  const sync=allowed('SYNC_OPTIONS');
  for(const engine of ENGINES.map(e=>e.id))for(const model of SYNC_MODELS.map(m=>m[0])){
    for(const variant of [{},{tps_alignment:'face',mrfa_alignment:'face',fsrt_alignment:'face',sync_area:'lips',face_analysis:'mediapipe'}]){
      const settings=portraitDefaults('animate',{animation_engine:engine,sync_model:model,...variant});
      for(const key of Object.keys(portraitOptions('animate',settings)))assert.ok(animate.has(key),`${engine}/${model}: ${key}`);
      const syncSettings=portraitDefaults('sync',{sync_model:model,...variant});
      for(const key of Object.keys(portraitOptions('sync',syncSettings)))assert.ok(sync.has(key),`sync ${model}: ${key}`);
    }
  }
});

test('a new engine starts from its own defaults and keeps the shared ones',()=>{
  const imtalker=portraitDefaults('animate',{animation_engine:'imtalker'});
  assert.equal(imtalker.composition,false);assert.equal(imtalker.imtalker_crop_context,1.6);assert.equal(imtalker.lipsync,false);
  // Lip sync is off until it is checked; then the model settings are sent
  for(const engine of ENGINES.map(e=>e.id)){const d=portraitDefaults('animate',{animation_engine:engine});assert.equal(d.lipsync,false,engine);assert.equal(portraitOptions('animate',d).animation_engine,engine);assert.equal(d.composition,false,engine);
    assert.equal(portraitOptions('animate',d).lipsync,false);assert.ok(!('sync_model' in portraitOptions('animate',d)),engine);
    assert.equal(portraitOptions('animate',{...d,lipsync:true}).sync_model,'musetalk');}
  assert.equal(portraitModel({kind:'animate',request:{options:{animation_engine:'imtalker'}}}),'IMTalker');
  const tps=portraitDefaults('animate',{animation_engine:'tpsmm'});
  assert.equal(tps.tps_checkpoint,'vox');assert.equal(tps.tps_mode,'relative');
  // Turning lip sync off sends no lip sync model settings
  const off=portraitOptions('animate',{...imtalker,lipsync:false});
  assert.equal(off.lipsync,false);assert.ok(!('sync_model' in off));assert.ok(!('sync_batch_size' in off));
  // LatentSync shows its own steps and guidance, MuseTalk its batch size
  const latent=portraitOptions('sync',portraitDefaults('sync',{sync_model:'latentsync'}));
  assert.equal(latent.sync_steps,20);assert.ok(!('sync_batch_size' in latent));
  assert.equal(portraitOptions('sync',portraitDefaults('sync',{sync_model:'musetalk'})).sync_batch_size,0);
  for(const model of ['tbdub','tbdub_teacher']){
    assert.deepEqual(portraitOptions('sync',portraitDefaults('sync',{sync_model:model})),{sync_model:model});
  }
});

test('the result line names the engine and the lip sync model',()=>{
  assert.equal(portraitModel({kind:'animate',request:{options:{animation_engine:'liveportrait',lipsync:true,sync_model:'latentsync'}}}),'LivePortrait + LatentSync 1.6');
  assert.equal(portraitModel({kind:'animate',request:{options:{animation_engine:'tpsmm',lipsync:false}}}),'Thin-Plate Spline Motion Model (TPSMM)');
  assert.equal(portraitModel({kind:'lipsync',request:{options:{sync_model:'wav2lip_gan'}}}),'Wav2Lip GAN');
  assert.equal(portraitModel({kind:'face-restore',request:{}}),'GFPGAN 1.4');
});

test('no label or hint of the portrait tools has an em dash or an en dash',()=>{
  for(const name of ['portraitControls.ts','PortraitInputs.tsx','PortraitFields.tsx','AnimateTab.tsx','LipSyncTab.tsx','ExtractAudioTab.tsx'])
    assert.ok(!/[\u2013\u2014]/.test(readFileSync(new URL(`../src/${name}`,import.meta.url),'utf8')),name);
  for(const engine of ENGINES.map(e=>e.id))for(const group of portraitGroups('animate',portraitDefaults('animate',{animation_engine:engine})))
    for(const field of group.fields)assert.ok(!/[\u2013\u2014]/.test(JSON.stringify(field)),field.key);
});
