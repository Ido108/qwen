// Retry only a request rejected by the session guard before its handler ran.
let token = '';
let refreshing: Promise<string> | null = null;

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

// Through Moise (/qwen-studio/), Moise's own session ends with every Moise restart. The dashboard keeps its key in
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

export async function apiRequest(path: string, init: RequestInit = {}) {
  const mutation = !['GET', 'HEAD', 'OPTIONS'].includes((init.method || 'GET').toUpperCase());
  let sent = mutation ? await sessionToken() : token;
  for (let attempt = 0; ; attempt++) {
    const headers = new Headers(init.headers);
    if (sent) headers.set('X-Studio-Token', sent);
    const response = await fetch(path, {...init, headers});
    const body = await response.json().catch(() => ({detail: response.statusText}));
    if (response.status === 401 && throughMoise && attempt === 0 && await signInAgain()) continue;
    if (!response.ok) {
      if (mutation && attempt === 0 && expiredSession(response.status, body)) {
        sent = await renewSession(sent);
        continue;
      }
      throw new Error(typeof body.detail === 'string' ? body.detail : (body.detail || []).map((e: {msg: string}) => e.msg).join(' '));
    }
    if (path === '/api/config') rememberSession(body.token);
    return body;
  }
}
