/** Settings of the portrait tools (Animate image, Lip sync, Face restore, Extract audio). Each key is a field of the
 *  portrait engine's own request; backend/portrait_settings.py passes only these keys on, and the engine checks them. */
export type PortraitTool='animate'|'sync'|'enhance'|'extract';
export type PortraitValue=string|number|boolean|null;
export type PortraitSettings=Record<string,PortraitValue>;
type Choice=[PortraitValue,string];
export type PortraitField={key:string;label:string;value:PortraitValue;type:'select'|'number'|'checkbox';options?:Choice[];min?:number;max?:number;step?:number;hint:string;hidden?:boolean};
export type PortraitGroup={title:string;description?:string;fields:PortraitField[];primary?:boolean;hidden?:boolean};

const select=(key:string,label:string,value:PortraitValue,options:Choice[],hint=''):PortraitField=>({key,label,value,options,hint,type:'select'});
const number=(key:string,label:string,value:PortraitValue,min:number,max:number,step=1,hint=''):PortraitField=>({key,label,value,min,max,step,hint,type:'number'});
const check=(key:string,label:string,value:boolean,hint=''):PortraitField=>({key,label,value,hint,type:'checkbox'});

export const SYNC_MODELS:Choice[]=[
  ['musetalk','MuseTalk 1.5: fast, 256 px'],
  ['tbdub','TBDub Student: 2 steps'],
  ['tbdub_teacher','TBDub Teacher: 30 steps'],
  ['latentsync','LatentSync 1.6: detailed, 512 px'],
  ['wav2lip_gan','Wav2Lip GAN: original, 96 px'],
  ['wav2lip','Wav2Lip: original, 96 px'],
];
export const SYNC_INFO:Record<string,string>={
  musetalk:'Recommended for quick results and everyday use. One model pass per frame. Included in the first install.',
  tbdub:'512 px dubbing for one face, at 25 fps.',
  tbdub_teacher:'512 px dubbing for one face, at 25 fps. Uses the 30-step teacher.',
  latentsync:'Temporal diffusion for more detailed lips. Slower; needs an NVIDIA GPU with at least 18 GB. Downloads 4.72 GB on first use.',
  wav2lip_gan:'The original generator with visual-quality training. Its 96 pixel face output can look soft. Downloads 416 MB on first use.',
  wav2lip:'The original lip sync generator. Prioritizes speech alignment; its 96 pixel face output can look soft. Downloads 416 MB on first use.',
};
export const ENGINES:{id:string;title:string;detail:string;url:string}[]=[
  {id:'imtalker',title:'IMTalker: video-driven renderer',detail:'512 × 512 generation with identity-adapted motion. Transfers head pose, gaze and expression directly from the video. Downloads 1.98 GB on first use.',url:'https://github.com/bigai-nlco/IMTalker'},
  {id:'liveportrait',title:'LivePortrait',detail:'21 learned 3D keypoints with appearance, warping, stitching and retargeting networks. 512 × 512 animated portrait. Included in the first install.',url:'https://github.com/KlingAIResearch/LivePortrait'},
  {id:'performrecast',title:'PerformRecast: expression transfer',detail:'Transfers facial expressions and keeps the head pose of the picture. 512 px renderer. Downloads 854 MB on first use. Choose another engine when head motion must also transfer.',url:'https://github.com/youku-aigc/PerformRecast'},
  {id:'personalive',title:'PersonaLive: video-driven diffusion',detail:'A streaming diffusion model: the portrait follows the head pose and expression of each video frame. 512 × 512 at 25 fps, 4 steps. First use downloads 15.5 GB and installs a separate 6 GB environment. NVIDIA GPU with 12 GB.',url:'https://github.com/GVCLab/PersonaLive'},
  {id:'tpsmm',title:'Thin-Plate Spline Motion Model (TPSMM)',detail:'50 learned keypoints, 10 TPS transforms, dense motion and inpainting. 256 × 256 (VoxCeleb) or 384 px (TED, upper body). Included in the first install.',url:'https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model'},
  {id:'mrfa',title:'MRFA: dense motion refinement',detail:'Ten keypoints and local shape transforms, refined in six learned stages. 256 × 256. Downloads 2.46 GB on first use (874 MB stays). Slower than TPSMM.',url:'https://github.com/JialeTao/MRFA'},
  {id:'fsrt',title:'FSRT: Vox256',detail:'Ten pose keypoints plus a separate expression encoder. 256 × 256. Downloads 236 MB on first use. Can soften detail.',url:'https://github.com/andrerochow/fsrt'},
];

const BLEND:Choice[]=[['smooth','Match edges and blend (recommended)'],['original','Original edge blend']];
const ALIGN=(face='Separate face crops'):Choice[]=>[['source','Match video to source image'],['face',face],['resize','Original full-frame resize']];

/** The groups of fields for a tool, with the same conditions as the engine's own app. */
export function portraitGroups(tool:PortraitTool,s:PortraitSettings):PortraitGroup[]{
  const result:PortraitGroup[]=[];
  const engine=String(s.animation_engine??'imtalker');
  const alignKey=engine==='fsrt'?'fsrt_alignment':engine==='mrfa'?'mrfa_alignment':'tps_alignment';
  const hideMatch=!['tpsmm','fsrt','mrfa'].includes(engine)||(s[alignKey]??'source')!=='source';
  if(tool==='animate'){
    if(engine==='tpsmm')result.push({title:'TPSMM settings',description:'Vox is trained on faces; TED is trained on upper-body videos.',fields:[
      select('tps_checkpoint','Trained checkpoint','vox',[['vox','VoxCeleb: faces, 256 px (original)'],['ted','TED-talks: upper body, 384 px']],'TED downloads 335 MB on first use. Use a similarly framed upper-body picture and video.'),
      select('tps_mode','Motion mode','relative',[['relative','Relative'],['standard','Standard (absolute keypoints)'],['avd','AVD (trained disentanglement)']],'Relative keeps the identity of the picture as the starting pose.'),
      {...check('tps_adapt_scale','Adapt movement scale',true,'Scale relative motion by the source and reference keypoint areas.'),hidden:(s.tps_mode??'relative')!=='relative'},
      check('tps_preserve_detail','Reproject source detail (experimental)',false,'Off keeps the trained Vox output. Reprojection can sharpen texture but weaken mouth shapes.'),
      number('motion','Motion multiplier',1,0,1.8,.05,'1 = original motion amplitude.'),
      select('tps_alignment','Prepare the driving video','source',[['source','Match video to source image'],['face','Separate face crops'],['resize','Original notebook: resize full frames']],'Match crops a square around the face of the picture and fits the video into it, fixed for each scene.'),
      select('tps_sampling','Model input resizing','cached',[['cached','Anti-aliased resize, cached filters (recommended)'],['original','Original notebook resize'],['fast','OpenCV area resize']],'OpenCV uses a different filter and can change motion.'),
      {...number('source_crop_scale','Source crop scale',2.3,1.5,4,.05,'Higher values include more area around the face of the picture.'),hidden:(s.tps_alignment??'source')==='resize'},
      {...number('driving_crop_scale','Driving crop scale',2.3,1.5,4,.05,'Use matching crop proportions for picture and video.'),hidden:(s.tps_alignment??'source')!=='face'},
      {...number('crop_vy','Vertical crop offset',-.125,-.5,.5,.025,'Negative values move the crop upward.'),hidden:(s.tps_alignment??'source')==='resize'},
      {...check('composition','Keep the original background',false,'Places the animated crop on the still picture.'),hidden:(s.tps_alignment??'source')!=='face'},
    ]});
    else if(engine==='performrecast')result.push({title:'PerformRecast settings',description:'Expression transfer only. The head pose and position of the picture stay.',fields:[
      select('perform_mode','Expression transfer','relative',[['relative','Relative: changes from the reference'],['replacement','Replacement: expression of the video']],'Relative keeps the resting expression of the picture.'),
      select('perform_expression','Expression strength',1,[[.5,'Reduced: 0.5'],[.75,'Gentle: 0.75'],[1,'Original amplitude: 1.0'],[1.25,'Stronger: 1.25']]),
      check('composition','Keep the original background',false,'Places the animated crop on the still picture.'),
      {...select('composition_blend','Background blending','smooth',BLEND),hidden:s.composition!==true},
      number('source_crop_scale','Source crop scale',2.3,1.5,4,.05,'Higher values include more area around the face of the picture.'),
      number('driving_crop_scale','Driving crop scale',2.3,1.5,4,.05,'Use matching crop proportions for picture and video.'),
      number('crop_vy','Vertical crop offset',-.125,-.5,.5,.025,'Negative values move the crop upward.'),
    ]});
    else if(engine==='mrfa')result.push({title:'MRFA settings',fields:[
      select('mrfa_checkpoint','Trained checkpoint','vox',[['vox','Vox: portrait videos'],['celebvhq','CelebV-HQ: alternative portrait training']],'Both checkpoints are installed together.'),
      check('mrfa_relative','Relative motion',true,'Apply changes from the reference to the picture. Off transfers the absolute keypoints of the video.'),
      {...check('mrfa_relative_jacobian','Relative local shape',true,'Transfer changes in the scale and rotation of each learned region.'),hidden:s.mrfa_relative===false},
      {...check('mrfa_adapt_scale','Adapt movement scale',true,'Scale relative motion by the source and reference keypoint areas.'),hidden:s.mrfa_relative===false},
      select('mrfa_alignment','Prepare the driving video','source',ALIGN(),'The crop stays fixed for each scene, so head movement and expressions stay.'),
      {...number('source_crop_scale','Source crop scale',2.3,1.5,4,.05),hidden:(s.mrfa_alignment??'source')!=='face'},
      {...number('driving_crop_scale','Driving crop scale',2.3,1.5,4,.05),hidden:s.mrfa_alignment!=='face'},
      {...number('crop_vy','Vertical crop offset',-.125,-.5,.5,.025),hidden:(s.mrfa_alignment??'source')!=='face'},
      {...check('composition','Keep the original background',false,'Places the animated crop on the still picture.'),hidden:(s.mrfa_alignment??'source')!=='face'},
      {...select('composition_blend','Background blending','smooth',BLEND),hidden:(s.mrfa_alignment??'source')!=='face'},
    ]});
    else if(engine==='fsrt')result.push({title:'FSRT settings',description:'Relative pose with the expression transferred directly.',fields:[
      check('fsrt_relative','Relative keypoint motion',true,'Keep the head proportions of the picture and apply changes from the reference.'),
      {...check('fsrt_adapt_scale','Adapt movement scale',true),hidden:s.fsrt_relative===false},
      select('fsrt_alignment','Prepare the driving video','source',ALIGN()),
      select('fsrt_chunk_size','Pixels processed per batch',4096,[[1024,'1,024: least memory'],[2048,'2,048'],[4096,'4,096 (recommended)'],[8192,'8,192'],[16384,'16,384: more memory']],'Controls memory use, not output resolution.'),
      {...number('source_crop_scale','Source crop scale',2.3,1.5,4,.05),hidden:(s.fsrt_alignment??'source')!=='face'},
      {...number('driving_crop_scale','Driving crop scale',2.3,1.5,4,.05),hidden:s.fsrt_alignment!=='face'},
      {...number('crop_vy','Vertical crop offset',-.125,-.5,.5,.025),hidden:(s.fsrt_alignment??'source')!=='face'},
      {...check('composition','Keep the original background',false,'Places the animated crop on the still picture.'),hidden:(s.fsrt_alignment??'source')!=='face'},
      {...select('composition_blend','Background blending','smooth',BLEND),hidden:(s.fsrt_alignment??'source')!=='face'},
    ]});
    else if(engine==='personalive')result.push({title:'PersonaLive settings',description:'Fixed values of the released pipeline: 512 px, 25 fps, 4 steps, seed 42.',fields:[
      select('personalive_crop_context','Face size in the frame',2,[[2,'Standard: head and shoulders (recommended)'],[1.6,'Closer: face fills about 60%'],[2.4,'Wider: more shoulders and background']],'Picture and video are cropped to a square with the face at this size, fixed for each scene.'),
      select('scene_handling','Scene changes','auto',[['auto','Automatic: frame each scene separately'],['single','One continuous scene']]),
      check('composition','Keep the original background',false,'Places the animated crop on the still picture.'),
      select('composition_blend','Blend with body and background','smooth',BLEND),
    ]});
    else if(engine==='imtalker')result.push({title:'IMTalker settings',description:'512 px renderer. Identity detail and head proportions can change; compare a preview.',fields:[
      select('imtalker_crop_context','Face size in the frame',1.6,[[1.6,'Standard: face fills about 60% (recommended)'],[1.4,'Closer: face fills about 70%'],[1.8,'Wider: more hair and shoulders']],'The crop of the video stays fixed for each scene, so head movement stays.'),
      select('imtalker_face_fit','Video face fit','original',[['original','Standard: same face box size in picture and video'],['box','Face box of the IMTalker release'],['place','Exact position and size: eyes and mouth in the same place, no stretch'],['stretch','Stretch: eyes and mouth in the same place'],['shape','Stretch and shape: eyes, mouth, nose and face outline in the same place']],'The video is never rotated.'),
      select('scene_handling','Scene changes','auto',[['auto','Automatic: frame each scene separately'],['single','One continuous scene']]),
      check('composition','Keep the original background',false,'Places the animated crop on the still picture.'),
      select('composition_blend','Blend with body and background','smooth',BLEND),
    ]});
    else result.push({title:'LivePortrait settings',fields:[
      select('mode','Animation region','all',[['all','All'],['expression','Expression'],['pose','Head pose'],['mouth','Lips'],['eyes','Eyes']]),
      check('relative_motion','Relative motion',true,'Apply changes relative to the reference frame. Off uses the absolute motion of the video.'),
      number('motion','Driving multiplier',1,0,1.8,.05,'Scales the keypoint motion after stitching.'),
      check('stitching','Stitching',true),
      check('normalize_lip','Normalize source lips',true,'Close an open mouth of the picture before relative animation.'),
      check('composition','Keep the original background',false,'Places the animated crop on the still picture.'),
      number('source_crop_scale','Source crop scale',2.3,1.5,4,.05),
      number('driving_crop_scale','Driving crop scale',2.3,1.5,4,.05),
      number('crop_vy','Vertical crop offset',-.125,-.5,.5,.025),
    ]});
    const quick:Record<string,string>={tpsmm:'tps_checkpoint',mrfa:'mrfa_checkpoint',fsrt:'fsrt_relative',imtalker:'imtalker_crop_context',personalive:'personalive_crop_context',performrecast:'perform_expression',liveportrait:'mode'};
    const model=result[0];
    const advanced=model.fields.filter(field=>field.key!==quick[engine]);
    model.fields=model.fields.filter(field=>field.key===quick[engine]);
    model.primary=true;
    model.description=undefined;
    result.push({title:'Model and framing options',fields:advanced});
    if(!['imtalker','personalive'].includes(engine))result.push({title:'Face matching',description:'How the reference face is found and aligned.',fields:[
      select('face_analysis','Face analysis','original',[['original','Existing analysis: 203 landmarks'],['mediapipe','Google Face Mesh: 478 landmarks and expressions']],s.face_analysis==='mediapipe'?'First use installs about 100 MB.':'No extra download.'),
      {...check('face_stretch','Stretch the video face to the proportions of the picture',false,'On: eyes and mouth of the video land on those of the picture. Off: one scale for width and height.'),hidden:hideMatch},
      {...select('mesh_shape','Bend the video face to the shape of the picture',0,[[0,'Off (recommended)'],[.35,'Conservative: also match nose and outline'],[.7,'Stronger: compare a preview'],[1,'Full: nose and outline matched']],'The bend stays fixed on the screen and also reaches the neck and shoulders.'),hidden:hideMatch},
    ]});
    if(!['imtalker','personalive'].includes(engine))result.push({title:'Reference frame and tracking',fields:[
      select('scene_handling','Scene changes','auto',[['auto','Automatic: reference for each scene'],['single','One reference for the whole video']],'Recommended for edited videos: a cut is not treated as a facial expression.'),
      {...select('driving_crop_mode','Driving crop tracking','fixed',[['fixed','Fixed to the reference frame'],['tracked','Follow the face in each frame']],'Fixed keeps head rotation and movement.'),hidden:['fsrt','mrfa','performrecast'].includes(engine)||engine==='tpsmm'&&(s.tps_alignment??'source')!=='face'},
      select('reference_samples','Reference search detail',48,[[24,'Quick: 24 samples'],[48,'Balanced: 48 samples (recommended)'],[96,'Detailed: 96 samples'],[240,'Extensive: 240 samples']]),
      {...select('redetect_interval','Face detection frequency',30,[[30,'Every 30 frames (recommended)'],[10,'Every 10 frames: fast head movement'],[1,'Every frame: slower']]),hidden:['fsrt','mrfa','performrecast'].includes(engine)||engine==='tpsmm'&&(s.tps_alignment??'source')!=='face'},
      {...select('smoothing','Motion smoothing',0,[[0,'Off: keep fast expressions'],[.25,'Light'],[.6,'Stronger']]),hidden:['fsrt','mrfa','performrecast'].includes(engine)},
    ]});
    if(engine==='liveportrait')result.push({title:'Optional motion calibration',fields:[
      select('lip','Additional lip correction',1,[[0,'Off: use model motion'],[.5,'Moderate correction'],[1,'Full correction (recommended)']],'This does not turn off mouth animation.'),
      select('eye','Additional eye correction',0,[[0,'Off: use model motion (recommended)'],[.5,'Moderate correction'],[1,'Full correction']],'This does not turn off blinking.'),
      check('adapt','Adapt face geometry',false,'A constant vertical alignment correction, limited to 0.88 to 1.12.'),
    ]});
  }
  if(tool==='animate'||tool==='sync'){
    const enabled=tool==='sync'||s.lipsync===true,model=String(s.sync_model??'musetalk');
    result.push({title:'Lip sync',primary:true,description:tool==='animate'?'Optional second pass: match the animated mouth to the audio.':undefined,fields:[
      ...(tool==='animate'?[check('lipsync','Lip sync after animation',false,'Head movement and expressions still come from the driving video.')]:[]),
      {...select('sync_model','Lip sync model','musetalk',SYNC_MODELS,SYNC_INFO[model]),hidden:!enabled},
      ...(model==='latentsync'?[number('sync_steps','Denoising steps',20,10,60,1,'20 is the default of the model. More steps take longer.'),number('sync_guidance','Audio guidance',1.5,1,3,.1,'Higher values follow the audio more strongly.')]
        :[select('sync_batch_size','Processing batch size',0,[[0,'Automatic: use available memory'],[1,'1 frame: least memory'],[2,'2 frames'],[4,'4 frames'],[8,'8 frames: more memory'],[16,'16 frames: most memory']])]).map(field=>({...field,hidden:!enabled})),
    ]});
    result.push({title:'Lip sync advanced settings',hidden:!enabled,fields:[
      ...(model==='latentsync'?[number('sync_seed','Seed',1247,0,2147483647,1,'Keep this fixed when you compare settings.')]:[]),
      ...(model==='musetalk'?[
        select('sync_crop_mode','Face crop','anatomical',[['anatomical','Nose and chin landmarks (recommended)'],['legacy','Previous eye-and-mouth estimate']]),
        number('sync_crop_shift','Crop shift (256 pixel face)',0,-32,32,1,'Moves the model crop up or down.'),
        select('sync_area','Replaced area','face',[['face','Lower face: lips, jaw and chin'],['lips','Lips only']],'Lips only keeps the jaw of the video.'),
        {...number('sync_margin','Mouth mask margin',.24,0,.5,.01,'Space around the lips, relative to mouth width.'),hidden:(s.sync_area??'face')!=='lips'},
      ]:model.startsWith('wav2lip')?[
        select('wav2lip_face_box','Face box','detector',[['detector','Face detector of the Wav2Lip release (recommended)'],['landmarks','Box around the face landmarks']]),
        number('wav2lip_pad_bottom','Chin padding (pixels)',10,0,80,1,'Includes the lower jaw in the model crop.'),
      ]:[]),
      number('sync_blend','Mouth blend',1,0,1,.05,'1 = generated mouth; 0 = unchanged pixels of the video.'),
      number('sync_feather','Mask feather',.04,.01,.15,.01,'Blur radius relative to mouth width.'),
    ]});
  }
  if(tool==='enhance')result.push({title:'Face restore',primary:true,fields:[
    number('enhance_blend','Restore strength',1,0,1,.05,'0 keeps the face of the video; 1 applies the full GFPGAN result.'),
    select('enhance_scale','Output size',1,[[1,'Same size as the video'],[2,'2× (background enlarged)']],'GFPGAN restores faces at 512 × 512.'),
  ]});
  if(tool==='extract')result.push({title:'Export',primary:true,fields:[select('audio_format','Audio format','wav',[['wav','WAV'],['mp3','MP3 (320 kbps)']])]});
  if(tool==='animate'){
    for(const group of result)for(const field of group.fields){
      if(field.key==='composition_blend')field.hidden=field.hidden||s.composition!==true;
    }
  }
  if(tool==='animate'||tool==='sync'){
    const sync=result.find(group=>group.title==='Lip sync')!;
    const extra=result.find(group=>group.title==='Lip sync advanced settings')!;
    extra.fields.unshift(...sync.fields.filter(field=>!['lipsync','sync_model'].includes(field.key)));
    sync.fields=sync.fields.filter(field=>['lipsync','sync_model'].includes(field.key));
  }
  if((tool==='sync'||tool==='animate')&&String(s.sync_model??'').startsWith('tbdub')){
    for(const group of result)if(group.title.toLowerCase().startsWith('lip')){
      group.fields=group.fields.filter(field=>field.key==='lipsync'||field.key==='sync_model');
    }
  }
  return result;
}

/** All defaults of a tool for the given choices, so a new choice (for example another engine) starts from its defaults. */
export function portraitDefaults(tool:PortraitTool,current:PortraitSettings):PortraitSettings{
  const out:PortraitSettings=tool==='animate'?{animation_engine:'imtalker',...current}:{...current};
  for(let pass=0;pass<3;pass++)for(const group of portraitGroups(tool,out))for(const field of group.fields)if(!(field.key in out))out[field.key]=field.value;
  return out;
}

export function changeLipModel(tool:PortraitTool,settings:PortraitSettings,profiles:Record<string,PortraitSettings>,model:string){
  const next={...settings},previous:PortraitSettings={};
  for(const key of Object.keys(next)){
    if(key!=='sync_model'&&(key.startsWith('sync_')||key.startsWith('wav2lip_'))){
      previous[key]=next[key];
      delete next[key];
    }
  }
  return {lipProfiles:{...profiles,[String(settings.sync_model||'musetalk')]:previous},
    settings:portraitDefaults(tool,{...next,...profiles[model],sync_model:model})};
}

/** The values a job sends: every visible field of every visible group. Hidden fields keep the engine's own default. */
export function portraitOptions(tool:PortraitTool,settings:PortraitSettings):PortraitSettings{
  const out:PortraitSettings=tool==='animate'?{animation_engine:settings.animation_engine??'imtalker'}:{};
  for(const group of portraitGroups(tool,settings))if(!group.hidden)for(const field of group.fields)if(!field.hidden&&settings[field.key]!==undefined)out[field.key]=settings[field.key];
  return out;
}

export const engineTitle=(id:string)=>ENGINES.find(e=>e.id===id)?.title.split(':')[0]||id;
export const syncTitle=(id:string)=>String(SYNC_MODELS.find(([value])=>value===id)?.[1]??id).split(':')[0];

/** The model line of a portrait result (run details, queue). */
export function portraitModel(job:{kind?:string;request:Record<string,any>}){
  const o=job.request?.options||{};
  if(job.kind==='animate')return engineTitle(o.animation_engine||'imtalker')+(o.lipsync===true?` + ${syncTitle(o.sync_model||'musetalk')}`:'');
  if(job.kind==='lipsync')return syncTitle(o.sync_model||'musetalk');
  if(job.kind==='face-restore')return 'GFPGAN 1.4';
  if(job.kind==='extract-audio')return String(job.request?.audio_format||'wav').toUpperCase();
  return '';
}
export const PORTRAIT_KINDS=['animate','lipsync','face-restore','extract-audio'];
