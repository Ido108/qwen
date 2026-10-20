import {useSyncExternalStore} from 'react';
import {connectionState,retryConnection,watchConnection} from './session';
export function ConnectionNotice(){
  const state=useSyncExternalStore(watchConnection,connectionState);
  if(state.status==='ready')return null;
  return <aside className="connection-notice" role="status"><span>{state.message}</span>{state.status==='sign-in'?<a href={['','dashboard.html'].join('/')} target="_blank" rel="noreferrer">Sign in</a>:null}<button onClick={retryConnection}>Try now</button></aside>;
}
