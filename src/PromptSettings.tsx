import {useEffect,useRef,useState} from 'react';
import {Check,ExternalLink,Loader2,LogOut,RefreshCw} from 'lucide-react';
import type {Api} from './ModelSettings';

export type EnhancementState={connected:boolean;expired:boolean;email:string;model:string;models:{id:string;name:string}[];error?:string};
const copy={
  en:{title:'Prompt enhancement',description:'Improve your prompt with Gemini through Antigravity.',connect:'Connect Google Account',connected:'Connected',disconnected:'Not connected',disconnect:'Disconnect',model:'AI model',privacy:'When enabled, your prompt and input images are sent to Google for enhancement. Image and video generation use the processing location selected above.',waiting:'Waiting for Google sign-in…',open:'Open sign-in window',manual:'Paste callback URL',manualHelp:'If the connection does not finish automatically, copy the full localhost address after signing in and paste it here.',submit:'Complete connection',cancel:'Cancel',expired:'This sign-in request expired. Connect again.',failed:'Could not connect. Try again.',test:'Test connection',tested:'Connection and selected model are working.',reconnect:'Reconnect',legacy:'Unused local prompt models',legacyHelp:'Prompt enhancement now uses your Google account. These old files are not used.'},
  he:{title:'שיפור פרומפט',description:'שיפור הפרומפט עם Gemini דרך Antigravity.',connect:'התחברות עם Google',connected:'מחובר',disconnected:'לא מחובר',disconnect:'ניתוק',model:'מודל AI',privacy:'כשהאפשרות פעילה, הפרומפט ותמונות הקלט נשלחים ל־Google לשיפור. יצירת התמונה מתבצעת במחשב הזה.',waiting:'ממתין להשלמת ההתחברות ב־Google…',open:'פתיחת חלון ההתחברות',manual:'הדבקת כתובת החזרה',manualHelp:'אם החיבור לא הושלם אוטומטית, העתיקו את כתובת ה־localhost המלאה לאחר ההתחברות והדביקו כאן.',submit:'השלמת החיבור',cancel:'ביטול',expired:'תוקף בקשת ההתחברות פג. התחברו שוב.',failed:'החיבור נכשל. נסו שוב.',test:'בדיקת חיבור',tested:'החיבור והמודל שנבחר עובדים.',reconnect:'חיבור מחדש',legacy:'מודלי פרומפט מקומיים שאינם בשימוש',legacyHelp:'שיפור הפרומפט משתמש כעת בחשבון Google. הקבצים הישנים האלה אינם בשימוש.'},
  zh:{title:'提示词改写',description:'通过 Antigravity 使用 Gemini 改写提示词。',connect:'连接 Google 帐号',connected:'已连接',disconnected:'未连接',disconnect:'断开连接',model:'AI 模型',privacy:'启用后，提示词和输入图片会发送给 Google 进行改写。图片生成在本机运行。',waiting:'等待 Google 登录…',open:'打开登录窗口',manual:'粘贴回调网址',manualHelp:'如果连接未自动完成，请在登录后复制完整的 localhost 网址并粘贴到此处。',submit:'完成连接',cancel:'取消',expired:'登录请求已过期，请重新连接。',failed:'连接失败，请重试。',test:'测试连接',tested:'连接和所选模型工作正常。',reconnect:'重新连接',legacy:'未使用的本地提示词模型',legacyHelp:'提示词改写现在使用 Google 帐号，这些旧文件不再使用。'}
};
export const promptSettingsCopy=copy;

export function PromptSettings({lang,api,state,onState,busy}:{lang:'en'|'he'|'zh';api:Api;state:EnhancementState|null;onState:(next:EnhancementState)=>void;busy:boolean}){
  const s=copy[lang];
  const [flow,setFlow]=useState<{state:string;authUrl:string;automaticCallback:boolean}|null>(null);
  const [input,setInput]=useState('');const [working,setWorking]=useState(false);
  const [error,setError]=useState('');const [notice,setNotice]=useState('');
  const popup=useRef<Window|null>(null);const mounted=useRef(true);
  useEffect(()=>{mounted.current=true;api('/api/oauth/antigravity/status').then(next=>{if(mounted.current)onState(next);}).catch(e=>setError(e.message));return()=>{mounted.current=false;};},[api,onState]);
  useEffect(()=>{
    if(!flow)return;let alive=true;let timer:number;
    const poll=async()=>{
      try{
        const result=await api(`/api/oauth/antigravity/poll/${encodeURIComponent(flow.state)}`);
        if(!alive)return;
        if(result.status==='completed'){
          setFlow(null);setWorking(false);
          if(result.success){onState(await api('/api/oauth/antigravity/status'));setInput('');setError('');popup.current?.close();}
          else setError(result.error||s.failed);
          return;
        }
        if(result.status==='expired'){setFlow(null);setError(s.expired);return;}
      }catch{/* Poll again after a temporary local connection interruption. */}
      if(alive)timer=window.setTimeout(poll,1500);
    };
    poll();return()=>{alive=false;clearTimeout(timer);};
  },[flow,api,onState,s]);
  async function connect(){
    setError('');setNotice('');setInput('');setWorking(true);
    // Open synchronously from the click, so popup blockers do not hide sign-in.
    popup.current=window.open('about:blank','image_studio_antigravity','width=640,height=760,scrollbars=yes');
    try{const next=await api('/api/oauth/antigravity/start',{method:'POST'});setFlow(next);if(popup.current)popup.current.location.href=next.authUrl;}
    catch(e){popup.current?.close();setError((e as Error).message);}
    finally{setWorking(false);}
  }
  async function finish(){
    setWorking(true);setError('');
    try{await api('/api/oauth/antigravity/exchange',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({callbackInput:input.trim()})});onState(await api('/api/oauth/antigravity/status'));setFlow(null);setInput('');popup.current?.close();}
    catch(e){setError((e as Error).message);}finally{setWorking(false);}
  }
  async function act(action:'disconnect'|'test'){
    setWorking(true);setError('');setNotice('');
    try{const next=await api(`/api/oauth/antigravity/${action}`,{method:'POST'});if(action==='test'){onState(next.connection);setNotice(s.tested);}else{onState(next);setFlow(null);setInput('');}}
    catch(e){setError((e as Error).message);}finally{setWorking(false);}
  }
  return <section className="prompt-settings">
    <div className="connection-heading"><div><h3>{s.title}</h3><p>{s.description}</p></div><span className={`connection-status ${state?.connected?'connected':''}`}>{state?.connected?<Check size={13}/>:<span className="status-dot"/>}{state?.connected?s.connected:s.disconnected}</span></div>
    <div className="google-connection"><span className="google-mark" aria-hidden="true">G</span><div className="connection-account"><strong>Google · Antigravity</strong>{state?.email?<span dir="ltr">{state.email}</span>:null}</div>
      {state?.connected?<button className="text-button" disabled={busy||working} onClick={()=>act('disconnect')}><LogOut size={15}/>{s.disconnect}</button>:<button className="button secondary" disabled={working} onClick={connect}>{working?<Loader2 size={15} className="spin"/>:<ExternalLink size={15}/>} {s.connect}</button>}
    </div>
    {flow?<div className="oauth-flow" aria-live="polite"><div className="oauth-wait"><Loader2 size={16} className="spin"/><strong>{s.waiting}</strong></div><a className="text-button" href={flow.authUrl} target="image_studio_antigravity" rel="noreferrer"><ExternalLink size={14}/>{s.open}</a>
      <details open={!flow.automaticCallback}><summary>{s.manual}</summary><p>{s.manualHelp}</p><label className="sr-only" htmlFor="oauth-callback">{s.manual}</label><textarea id="oauth-callback" dir="ltr" autoComplete="off" spellCheck={false} rows={2} placeholder="http://localhost:1455/auth/callback?code=…&state=…" value={input} onChange={e=>setInput(e.target.value)}/><div className="oauth-actions"><button className="button secondary" disabled={working||!input.trim()} onClick={finish}>{s.submit}</button><button className="text-button" onClick={()=>{setFlow(null);setInput('');popup.current?.close();}}>{s.cancel}</button></div></details>
    </div>:null}
    <div className="connection-model"><label htmlFor="enhancement-model">{s.model}</label><select id="enhancement-model" value={state?.model||'gemini-3.7-flash'} disabled={working} onChange={async e=>{setNotice('');setError('');try{onState(await api('/api/enhancement/model',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:e.target.value})}));}catch(e){setError((e as Error).message);}}}>{(state?.models||[]).map(model=><option key={model.id} value={model.id}>{model.name}</option>)}</select>{state?.connected?<button className="text-button" disabled={working||busy} onClick={()=>act('test')}>{working?<Loader2 className="spin" size={14}/>:<RefreshCw size={14}/>} {s.test}</button>:null}</div>
    <p className="connection-privacy">{s.privacy}</p>
    {error||state?.error?<div className="message error" role="alert">{error||state?.error}</div>:null}{notice?<div className="connection-success" role="status"><Check size={15}/>{notice}</div>:null}
  </section>;
}
