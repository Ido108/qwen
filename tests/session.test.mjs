import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import ts from 'typescript';

globalThis.window={location:{pathname:'/'}};

let sequence=0;
async function client(){
  const source=readFileSync(new URL('../src/session.ts',import.meta.url),'utf8');
  const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
  return import('data:text/javascript;base64,'+Buffer.from(code+`\n// instance ${sequence++}`).toString('base64'));
}
const response=(body,status=200)=>new Response(JSON.stringify(body),{status,headers:{'Content-Type':'application/json'}});

test('all screens share outage backoff and one recovery probe',async()=>{
  const api=await client();let calls=0;
  globalThis.fetch=async()=>{calls++;return response({detail:'Qwen is reconnecting'},503);};
  await assert.rejects(api.apiRequest('/api/models'));
  await assert.rejects(api.apiRequest('/api/queue'));
  await assert.rejects(api.apiRequest('/api/gpu-memory'));
  assert.equal(calls,1);
  assert.equal(api.connectionState().status,'offline');
  api.retryConnection();
  globalThis.fetch=async()=>{calls++;await new Promise(r=>setTimeout(r,20));return response({ok:true});};
  const [a,b]=await Promise.all([api.apiRequest('/api/models'),api.apiRequest('/api/models')]);
  assert.deepEqual(a,b);assert.equal(calls,2);assert.equal(api.connectionState().status,'ready');
});

test('uncertain generation failure is never replayed after recovery',async()=>{
  const api=await client();api.rememberSession('ready');let generations=0;
  globalThis.fetch=async()=>{generations++;return response({detail:'Bad gateway'},502);};
  await assert.rejects(api.apiRequest('/api/generate',{method:'POST',body:'{}'}));
  api.retryConnection();globalThis.fetch=async()=>response({active:null});
  await api.apiRequest('/api/queue');
  assert.equal(generations,1);
});

test('first mutation waits for a session, including restored automatic person scans',async()=>{
  const api=await client();const calls=[];
  globalThis.fetch=async(path,init)=>{calls.push(path);if(path==='/api/session')return response({token:'new'});assert.equal(init.headers.get('X-Studio-Token'),'new');return response({people:[]});};
  await api.apiRequest('/api/video-uploads/example/people',{method:'POST',body:'{}'});
  assert.deepEqual(calls,['/api/session','/api/video-uploads/example/people']);
});
test('server restart renews once, preserves request body and does not submit twice',async()=>{
  const api=await client();api.rememberSession('old');let handled=0;const calls=[];const body=new FormData();body.append('prompt','keep my choices');
  globalThis.fetch=async(path,init)=>{calls.push(path);if(path==='/api/session')return response({token:'new'});assert.equal(init.body,body);if(init.headers.get('X-Studio-Token')==='old')return response({code:'session_expired'},403);handled++;return response({id:'one'});};
  assert.deepEqual(await api.apiRequest('/api/generate',{method:'POST',body}),{id:'one'});
  assert.equal(handled,1);assert.equal(calls.length,3);
});
test('parallel rejected actions share a single session refresh',async()=>{
  const api=await client();api.rememberSession('old');let refreshes=0;let handled=0;
  globalThis.fetch=async(path,init)=>{if(path==='/api/session'){refreshes++;await new Promise(r=>setTimeout(r,20));return response({token:'new'});}if(init.headers.get('X-Studio-Token')==='old')return response({code:'session_expired'},403);handled++;return response({ok:true});};
  await Promise.all([1,2,3].map(()=>api.apiRequest('/api/scan',{method:'POST'})));
  assert.equal(refreshes,1);assert.equal(handled,3);
});
test('permission failures and uncertain network errors are never retried',async()=>{
  const api=await client();api.rememberSession('ready');let attempts=0;
  globalThis.fetch=async()=>{attempts++;return response({detail:'Local access only.'},403);};
  await assert.rejects(()=>api.apiRequest('/api/generate',{method:'POST'}),/Local access only/);assert.equal(attempts,1);
  globalThis.fetch=async()=>{attempts++;throw new Error('Network lost');};
  await assert.rejects(()=>api.apiRequest('/api/generate',{method:'POST'}),/Network lost/);assert.equal(attempts,2);
});
test('persistent expiry cannot cause a retry loop',async()=>{
  const api=await client();api.rememberSession('old');let posts=0;
  globalThis.fetch=async(path)=>{if(path==='/api/session')return response({token:'new'});posts++;return response({code:'session_expired',detail:'Refresh the app and try again.'},403);};
  await assert.rejects(()=>api.apiRequest('/api/scan',{method:'POST'}),/Refresh/);assert.equal(posts,2);
});

test('remote sign-in retry retains the selected photos',async()=>{
  globalThis.window.location.pathname='/qwen-studio/';
  globalThis.localStorage={getItem:()=> 'test-dashboard-key'};
  try{
    const api=await client();api.rememberSession('ready');
    const body=new FormData();body.append('files',new File(['image'],'phone.heic',{type:'image/heic'}));
    let posts=0,signIns=0;
    globalThis.fetch=async(path,init)=>{
      if(path==='/api/auth/verify'){signIns++;return response({ok:true});}
      assert.equal(path,'/api/uploads');assert.equal(init.body,body);
      return ++posts===1?response({code:'moise_sign_in'},401):response([{id:'photo.png'}]);
    };
    assert.deepEqual(await api.apiRequest('/api/uploads',{method:'POST',body}),[{id:'photo.png'}]);
    assert.equal(signIns,1);assert.equal(posts,2);
  }finally{globalThis.window.location.pathname='/';delete globalThis.localStorage;}
});

test('XHR video uploads also renew the session and retain the same file',async()=>{
  const source=readFileSync(new URL('../src/session.ts',import.meta.url),'utf8');
  const code=ts.transpileModule(source,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
  const sessionUrl='data:text/javascript;base64,'+Buffer.from(code+`\n// upload ${sequence++}`).toString('base64');
  const session=await import(sessionUrl);session.rememberSession('old');
  const uploadSource=readFileSync(new URL('../src/mediaUpload.ts',import.meta.url),'utf8').replace("'./session'",JSON.stringify(sessionUrl));
  const uploadCode=ts.transpileModule(uploadSource,{compilerOptions:{target:ts.ScriptTarget.ES2022,module:ts.ModuleKind.ESNext}}).outputText;
  const {uploadVideo}=await import('data:text/javascript;base64,'+Buffer.from(uploadCode).toString('base64'));
  const file=new File(['video bytes'],'clip.mp4',{type:'video/mp4'});let attempts=0;let accepted=0;
  globalThis.fetch=async()=>response({token:'new'});
  globalThis.XMLHttpRequest=class {
    constructor(){this.upload={};this.headers={};}
    open(method,path){assert.equal(method,'PUT');assert.equal(path,'/api/video-uploads');}
    setRequestHeader(key,value){this.headers[key]=value;}
    send(sent){assert.equal(sent,file);attempts++;this.status=this.headers['X-Studio-Token']==='old'?403:200;if(this.status===200)accepted++;this.responseText=JSON.stringify(this.status===403?{code:'session_expired'}:{id:'uploaded-once'});queueMicrotask(()=>this.onload());}
  };
  assert.equal((await uploadVideo(file,'stale-component-prop',()=>{})).id,'uploaded-once');
  assert.equal(attempts,2);assert.equal(accepted,1);
});
