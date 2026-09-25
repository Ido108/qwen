"""Verify two visible identities are separable across a short source video."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import av
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from backend.reference_people import analyze_people
    from backend.video_worker import save_mp4

    with av.open(str(args.source)) as media:
        source=next(media.decode(video=0)).to_image().convert('RGB')
    with Image.open(args.reference) as opened:
        reference=opened.convert('RGB')
    canvas=Image.new('RGB',(1024,512),'#24212c')
    for index,photo in enumerate((source,reference)):
        width,height=photo.size
        cropped=photo.crop((0,0,width,int(height*0.66)))
        canvas.paste(ImageOps.fit(cropped,(512,512)),(index*512,0))
    args.output.mkdir(parents=True,exist_ok=True)
    canvas.save(args.output/'two-people-source.jpg',quality=90)
    video=args.output/'two-people-source.mp4'
    frame=torch.from_numpy(np.array(canvas).astype(np.float32)/255).unsqueeze(0).repeat(48,1,1,1)
    save_mp4(video,frame,None)
    people=analyze_people(video,0,2)
    print([(item['id'],item['hits'],item['box'][0]) for item in people])
    assert len(people)>=2, 'Two visible identities were merged or one was missed.'
    assert any(item['box'][0]<0.5 for item in people) and any(item['box'][0]>0.5 for item in people), 'Left/right mapping is incorrect.'


if __name__=='__main__':
    main()
