/** 42s, 1m 05s, 1h 02m 05s. */
export function formatDuration(seconds:number){
  const whole=Math.max(0,Math.round(seconds)),hours=Math.floor(whole/3600),minutes=Math.floor(whole%3600/60),rest=whole%60;
  const pad=(n:number)=>String(n).padStart(2,'0');
  return hours?`${hours}h ${pad(minutes)}m ${pad(rest)}s`:minutes?`${minutes}m ${pad(rest)}s`:`${rest}s`;
}
