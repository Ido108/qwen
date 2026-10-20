'use strict';
const {app, BrowserWindow, shell} = require('electron');
const target = new URL(process.argv.find(value => value.startsWith('--url='))?.slice(6) || 'http://127.0.0.1:47821');
if (target.protocol !== 'http:' || target.hostname !== '127.0.0.1') throw new Error('Use the local Qwen server URL.');
const checkClose = process.argv.includes('--check-close');
function external(url) { if (/^https?:\/\//.test(url)) shell.openExternal(url); }
app.whenReady().then(async () => {
  const window = new BrowserWindow({width:1440, height:960, minWidth:360, minHeight:480,
    show:!checkClose, title:'Qwen', autoHideMenuBar:true,
    icon:require('node:path').join(__dirname, '../public/branding/icon-128.png'),
    webPreferences:{nodeIntegration:false, contextIsolation:true, sandbox:true}});
  window.webContents.setWindowOpenHandler(({url}) => {external(url); return {action:'deny'};});
  window.webContents.on('will-navigate', (event, url) => {
    if (new URL(url).origin !== target.origin) {event.preventDefault(); external(url);}
  });
  try {
    await window.loadURL(target.href);
    if (checkClose) {console.log('Qwen desktop page loaded. Closing only this window.'); window.close();}
  } catch (error) {console.error(error.message); app.exit(1);}
});
app.on('window-all-closed', () => app.quit());
