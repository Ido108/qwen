// Retry only a request rejected by the session guard before its handler ran.
let token = '';
let refreshing: Promise<string> | null = null;
export type ConnectionState={status:'ready'|'offline'|'sign-in';message:string;retryAt:number};
let connection:ConnectionState={status:'ready',message:'',retryAt:0};
let failures=0;
const listeners=new Set<()=>void>();
const reads=new Map<string,Promise<any>>();
let recovery:Promise<any>|null=null;
export const connectionState=()=>connection;
export function watchConnection(listener:()=>void){listeners.add(listener);return()=>{listeners.delete(listener);};}
function publish(next:ConnectionState){connection=next;listeners.forEach(fn=>fn());}
export function retryConnection(){failures=0;publish({...connection,retryAt:0});}
function unavailable(status:number){
  failures++;
  const signIn=status===401;
  publish({status:signIn?'sign-in':'offline',message:signIn?'Sign in to Moise to reconnect.':'Connection lost. Reconnecting. Your saved settings stay here.',retryAt:Date.now()+(signIn?30000:Math.min(15000,1000*2**Math.min(failures,4)))});
}

export function rememberSession(value: unknown) {
  if (typeof value === 'string' && value) token = value;
}

export async function sessionToken(refresh = false): Promise<string> {
  if (!refresh && token) return token;
  if (!refreshing) refreshing = (async () => {
    let response = await fetch('/api/session', {cache: 'no-store'});
    if (response.status === 401 && throughMoise && await signInAgain()) response = await fetch('/api/session', {cache: 'no-store'});
    // An old detached server may still be serving the newly built frontend
    // until the user next restarts. It already exposes the token in config.
    if (response.status === 404) response = await fetch('/api/config', {cache: 'no-store'});
    if (!response.ok) throw new Error('Could not connect to Qwen. Try again.');
    const body = await response.json();
    if (!body.token) throw new Error('Could not connect to Qwen. Try again.');
    rememberSession(body.token);
    return token;
  })().finally(() => { refreshing = null; });
  return refreshing;
}

export function expiredSession(status: number, body: {code?: string; detail?: string}) {
  return status === 403 && (body.code === 'session_expired' || body.detail === 'Refresh the app and try again.');
}

export async function renewSession(rejected: string) {
  return rejected !== token && token ? token : sessionToken(true);
}

// Through Moise (/qwen-studio/), an expired session can be renewed. The dashboard keeps its key in
// this browser (same origin), so sign in again once and retry. The address is built from parts so the remote
// proxy, which adds /qwen-studio to the app's /api/ paths, never rewrites it.
export const throughMoise = window.location.pathname.startsWith('/qwen-studio/');
let signingIn: Promise<boolean> | null = null;

export function signInAgain(): Promise<boolean> {
  if (!signingIn) signingIn = (async () => {
    try {
      const key = localStorage.getItem('dashboardSecretKey');
      if (!key) return false;
      const response = await fetch(['', 'api', 'auth', 'verify'].join('/'), {method: 'POST', headers: {Authorization: `Bearer ${key}`}, credentials: 'same-origin'});
      return response.ok;
    } catch { return false; }
  })().finally(() => { setTimeout(() => { signingIn = null; }, 5000); });
  return signingIn;
}

async function request(path: string, init: RequestInit = {}) {
  const mutation = !['GET', 'HEAD', 'OPTIONS'].includes((init.method || 'GET').toUpperCase());
  let sent = mutation ? await sessionToken() : token;
  for (let attempt = 0; ; attempt++) {
    const headers = new Headers(init.headers);
    if (sent) headers.set('X-Studio-Token', sent);
    let response:Response;
    try{response=await fetch(path, {...init, headers});}
    catch(error){if(!(error instanceof DOMException&&error.name==='AbortError'))unavailable(0);throw error;}
    const body = await response.json().catch(() => ({detail: response.statusText}));
    if (response.status === 401 && throughMoise && attempt === 0 && await signInAgain()) continue;
    if (!response.ok) {
      if (mutation && attempt === 0 && expiredSession(response.status, body)) {
        sent = await renewSession(sent);
        continue;
      }
      if([401,502,503,504].includes(response.status))unavailable(response.status);
      throw new Error(typeof body.detail === 'string' ? body.detail : (body.detail || []).map((e: {msg: string}) => e.msg).join(' ')||'Could not connect. Try again.');
    }
    failures=0;if(connection.status!=='ready')publish({status:'ready',message:'',retryAt:0});
    if (path === '/api/config') rememberSession(body.token);
    return body;
  }
}

export function apiRequest(path:string,init:RequestInit={}):Promise<any>{
  const read=['GET','HEAD'].includes((init.method||'GET').toUpperCase());
  if(read&&reads.has(path))return reads.get(path)!;
  if(Date.now()<connection.retryAt)return Promise.reject(new Error(connection.message));
  // One request probes a failed connection. Other screens wait for its result.
  if(recovery)return recovery.then(()=>apiRequest(path,init));
  const pending=request(path,init);
  if(connection.status!=='ready'){
    recovery=pending;
    pending.then(()=>{recovery=null;},()=>{recovery=null;});
  }
  if(read){reads.set(path,pending);pending.then(()=>reads.delete(path),()=>reads.delete(path));}
  return pending;
}
