"""Video job output: real H.264 MP4, optional AAC audio, thumbnail and provenance."""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import time
from .common import resolve_image, resolve_video, write_json


def save_mp4(path, frames, audio, fps=24):
    import av
    import numpy as np
    frame_count, height, width, _ = frames.shape
    if frame_count < 1:
        raise RuntimeError('The model returned no video frames.')
    with av.open(str(path), 'w', options={'movflags': '+faststart'}) as output:
        stream = output.add_stream('libx264', rate=fps)
        stream.width, stream.height = width, height
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'medium'}
        audio_stream = None
        if audio:
            waveform = audio['waveform'][0].numpy()
            if not np.isfinite(waveform).all():
                raise RuntimeError('The model returned invalid audio samples. Retry the video.')
            rate = int(audio['sample_rate'])
            target = round(frame_count / fps * rate)
            waveform = np.pad(waveform[:, :target], ((0, 0), (0, max(0, target - waveform.shape[1]))))
            layout = 'stereo' if waveform.shape[0] == 2 else 'mono'
            audio_stream = output.add_stream('aac', rate=rate)
            audio_stream.layout = layout
            audio_stream.bit_rate = 192000
        for index, pixels in enumerate(frames):
            array = pixels.numpy()
            if not np.isfinite(array).all():
                raise RuntimeError('The model returned invalid video pixels. Retry the video.')
            frame = av.VideoFrame.from_ndarray((array.clip(0, 1) * 255).round().astype(np.uint8), format='rgb24')
            frame.pts, frame.time_base = index, Fraction(1, fps)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
        if audio_stream:
            for offset in range(0, waveform.shape[1], 1024):
                frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(waveform[:, offset:offset + 1024]), format='fltp', layout=layout)
                frame.sample_rate = rate
                frame.pts, frame.time_base = offset, Fraction(1, rate)
                for packet in audio_stream.encode(frame):
                    output.mux(packet)
            for packet in audio_stream.encode():
                output.mux(packet)


from .prompt_registry import LTX_VIDEO as LTX_ENHANCE_SYSTEM  # editable in Settings > Internal prompts


def enhance_ltx_video(prompt, images, request, update):
    """Antigravity rewrite in the prose style of the official LTX-2.5 prompts."""
    import base64
    import io
    from . import antigravity
    model_id = request.get('enhancement_model') or antigravity.selected_model()
    parts = []
    for name, image in images.items():
        copy = image.copy()
        copy.thumbnail((1024, 1024))
        stream = io.BytesIO()
        copy.save(stream, 'PNG')
        parts.extend([{'text': name.replace('_', ' ')},
                      {'inlineData': {'mimeType': 'image/png', 'data': base64.b64encode(stream.getvalue()).decode()}}])
    parts.append({'text': f'Duration: {request["frames"]/24:.2f} seconds. Audio: {request["audio"]}. Brief: {prompt}'})
    update(stage='rewriting', message='Enhancing video prompt with Antigravity...', rewrite_provider='Antigravity', rewrite_model=model_id)
    from .prompt_registry import get
    system = get('video:ltx-enhance', LTX_ENHANCE_SYSTEM)
    update(sent_prompt={'label': f'Enhance with Antigravity {model_id}: system message', 'text': system})
    update(sent_prompt={'label': f'Enhance with Antigravity {model_id}: user message', 'text': '\n'.join('[image]' if 'inlineData' in part else part['text'] for part in parts)})
    text = antigravity.generate_text(model_id, system, parts).strip()
    if not text or len(text) > 16000:
        raise RuntimeError('Prompt enhancement returned an invalid response. Retry or turn off Enhance prompt.')
    update(rewritten_prompt=text, sent_prompt={'label': 'Enhance: rewritten prompt', 'text': text})
    return text


def run(request, job, directory, update):
    import torch
    from PIL import Image
    from .models import entry, is_ltx_video
    from .video_engine import generate
    if not torch.cuda.is_available():
        raise RuntimeError('NVIDIA CUDA is unavailable. Check the graphics driver and restart Qwen.')
    torch.set_float32_matmul_precision('high')
    torch.set_num_threads(8)
    reference=request.get('kind')=='reference-video'
    if reference:
        from .reference_sampling import reference_sampling
        sampling=reference_sampling(entry(request['model_id']),request.get('quality'),request.get('steps'))
        request.update(quality=sampling['quality'],steps=sampling['steps'])
    images = {}
    source_audio=None
    if reference:
        from .reference_media import read_frames,read_audio
        from .reference_timing import probe_video_duration
        update(stage='decoding_source',message='Decoding selected source video…')
        source=resolve_video(request['source_video'])
        clip_start=request.get('source_start_seconds',0)
        clip_end=request.get('source_end_seconds')
        images['source_video']=read_frames(source,request['frames'],start_seconds=clip_start,end_seconds=clip_end,
            progress=lambda done,total:update(input_frames_done=done,input_frames_total=total))
        update(stage='reading_audio',message='Preparing source audio…')
        source_audio=read_audio(source,request['frames']/24,start_seconds=clip_start,end_seconds=clip_end) if request['source_audio'] or request['output_audio']=='source' else None
        if request['source_audio']:
            images['source_audio']=source_audio
        if request['workflow']=='crossview':
            if request.get('warp_video'):
                warp_path=resolve_video(request['warp_video'])
                selected_duration=(clip_end-clip_start) if clip_end is not None else request['frames']/24
                warp_duration=probe_video_duration(warp_path)
                if warp_duration+1/24<selected_duration:
                    raise ValueError('The camera guide video must cover the selected source segment.')
                warp_start=clip_start if clip_end is not None and warp_duration+1/24>=clip_end else 0
                images['warp_video']=read_frames(warp_path,request['frames'],start_seconds=warp_start,
                                                 end_seconds=warp_start+selected_duration)
        update(stage='reading_references',message='Preparing reference images…')
        for index,image_id in enumerate(request['reference_images'],1):
            with Image.open(resolve_image(image_id)) as source_image:
                images[f'ref_image_{index}']=source_image.convert('RGB')
                images[f'ref_image_{index}'].save(directory/f'reference-{index}.png')
        from .reference_prompt import build_reference_prompt, reference_instructions
        instructions=reference_instructions(request)
        if request['enhance'] and instructions:
            from .prompt_enhancer import enhance_reference
            camera_request={**request,'reference_images':[],'reference_people':[]} if request['workflow']=='crossview' else request
            instructions=enhance_reference(instructions,images,camera_request,update)
        prompt=build_reference_prompt(request,instructions)
    else:
        for name in ('first_frame', 'last_frame'):
            if request.get(name):
                with Image.open(resolve_image(request[name])) as source:
                    images[name] = source.convert('RGB')
                    if request.get('workflow') == 'orbit':
                        from .orbit import prepare_image
                        images[name] = prepare_image(images[name])
                    images[name].save(directory / f'{name}.png')
        prompt = request['prompt']
        if request['enhance'] and is_ltx_video(entry(request['model_id'])):
            prompt = enhance_ltx_video(prompt, images, request, update)
        elif request['enhance']:
            from .prompt_enhancer import enhance_video
            prompt = enhance_video(prompt, images, request, update)
    from .lora_prompts import internal_prompt
    prompt, lora_words = internal_prompt(prompt, request)
    update(lora_prompt_words=lora_words)
    from . import sent_prompts
    sent_prompts.clear()
    update(sent_prompt={'label': 'Prompt sent to the video model (after enhancement and LoRA words)', 'text': prompt})
    if reference and request['workflow']=='crossview' and request.get('replace_subjects_first'):
        steps=request['steps']
        def stage_progress(stage_number):
            def relay(**values):
                values['phase']='Replacing subjects' if stage_number==0 else 'Changing camera view'
                if values.get('message'):
                    values['message']=values['phase']+' · '+values['message']
                if 'total_steps' in values:
                    values['total_steps']=steps*2
                if values.get('stage')=='generating':
                    values['step']=values.get('step',0)+stage_number*steps
                update(**values)
            return relay
        replacement_instructions=reference_instructions(request,replacement=True)
        replacement_request={**request,'workflow':'faceswap','warp_video':None,
                             'output_audio':'mute'}
        if request['enhance'] and replacement_instructions:
            from .prompt_enhancer import enhance_reference
            replacement_instructions=enhance_reference(replacement_instructions,images,
                replacement_request,stage_progress(0))
        replacement_prompt=build_reference_prompt(request,replacement_instructions,replacement=True)
        replacement_prompt, replacement_words = internal_prompt(replacement_prompt, replacement_request)
        update(replacement_lora_prompt_words=replacement_words)
        update(subject_replacement_prompt=replacement_prompt)
        update(sent_prompt={'label': 'Subject replacement pass prompt', 'text': replacement_prompt})
        first_frames,_=generate(replacement_request,replacement_prompt,images,stage_progress(0),preview_directory=directory)
        intermediate=directory/'subject-replacement.mp4'
        save_mp4(intermediate,first_frames,None)
        update(subject_replacement_ready=True)
        second_images={'source_video':first_frames}
        if 'warp_video' in images:
            second_images['warp_video']=images['warp_video']
        if request['source_audio'] and source_audio is not None:
            second_images['source_audio']=source_audio
        camera_request={**request,'reference_images':[],'reference_people':[]}
        del first_frames,images
        frames,audio=generate(camera_request,prompt,second_images,stage_progress(1),preview_directory=directory)
    elif not reference and is_ltx_video(entry(request['model_id'])):
        # LTX 2.5 runs the official ComfyUI LTX workflows; MiniMax H3 keeps its own engine.
        from .ltx_engine import generate_video
        frames, audio = generate_video(request, prompt, images, update, preview_directory=directory)
    else:
        frames, audio = generate(request, prompt, images, update,preview_directory=directory)
    if reference:
        if request['output_audio']=='source':
            if source_audio is None:
                raise ValueError('The source video has no audio track. Choose generated audio or mute.')
            audio=source_audio
        elif request['output_audio']=='mute':
            audio=None
    if len(frames) != request['frames']:
        raise RuntimeError(f'The decoder returned {len(frames)} frames; expected {request["frames"]}.')
    update(stage='saving', message='Saving MP4…')
    destination = Path(request['output_dir']) / (time.strftime('%Y-%m-%d_%H-%M-%S') + '_' + job['id'][:8] + '.mp4')
    save_mp4(destination, frames, audio)
    poster = directory / 'poster.png'
    Image.fromarray((frames[0].numpy().clip(0, 1) * 255).round().astype('uint8')).save(poster)
    sent_prompts.flush(update)
    update(status='succeeded', stage='complete', message='Video saved.', output_path=str(destination),
        video_url=f'/api/jobs/{job["id"]}/video', poster_url=f'/api/jobs/{job["id"]}/poster',
        finished_at=time.time(), elapsed_seconds=round(time.time() - job['started_at'], 1),
        duration_seconds=round(len(frames) / 24, 3), fps=24, frames=len(frames),
        width=int(frames.shape[2]),height=int(frames.shape[1]),
        has_audio=audio is not None, audio_sample_rate=audio['sample_rate'] if audio else None,
        actual_prompt=prompt, model_revision=entry(request['model_id'])['revision'])
    write_json(destination.with_suffix('.json'), job)
