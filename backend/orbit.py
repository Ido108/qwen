"""Author's FL2VA 360 orbit contract. Keep it separate from generic video drafts."""
from pydantic import BaseModel, Field
from .common import VideoRequest, VIDEOS_DIR, LoraChoice

SOURCE = 'https://huggingface.co/pablodawson/MiniMax-H3-360-Orbit-LoRA'
LORA_ID = 'h3-360-orbit'
MODEL_ID = 'h3-int8'
PROMPT = ('One frozen instant. Only the camera moves. In a continuous 360 orbit. '
          'Preserve every person and object in exactly the same world position, orientation, shape and pose throughout the shot. '
          'Airborne objects remain suspended at the captured height and angle: no wobbling, shaking, spinning, drifting, falling or continued action. '
          'Keep faces, hands, clothing, liquids and the background motionless while retaining their natural appearance. '
          'Camera parallax is the only source of apparent movement. No cuts, zoom, morphing or added objects.')


class OrbitRequest(BaseModel):
    image: str = Field(min_length=1)
    model_id: str = MODEL_ID
    strength: float = Field(default=1.0, ge=0.5, le=1.5)
    seed: int = Field(default=0, ge=0, le=2147483647)
    randomize_seed: bool = True
    output_dir: str = str(VIDEOS_DIR)


def video_request(body: OrbitRequest) -> VideoRequest:
    from .models import entry
    item=entry(body.model_id)
    sampling=profile(item,entry(LORA_ID))
    if sampling is None:
        raise ValueError('Choose an Orbit model from the list. This model cannot use the required start and end frames.')
    return VideoRequest(model_id=body.model_id, workflow='orbit', prompt=PROMPT, mode='image',
                        first_frame=body.image, last_frame=body.image, width=768, height=768,
                        frames=73, steps=sampling['steps'], quality='standard', audio=False, enhance=False,
                        seed=body.seed, randomize_seed=body.randomize_seed, output_dir=body.output_dir,
                        loras=[LoraChoice(id=LORA_ID, strength=body.strength)])


def profile(item, lora):
    """Select by architecture and keyframe support, never by checkpoint ID.

    A checkpoint author's step setting wins. Otherwise use the Orbit author's
    recommendation rather than the general H3 workflow's default step count.
    """
    if (item.get('kind')!='video' or item.get('family') not in lora.get('families',[])
            or item.get('family')!='minimax-h3' or 'image' not in item.get('modes',[])):
        return None
    from .video_sampling import video_sampling
    defaults=item.get('generation_defaults',{})
    specific=(defaults.get('embedded_turbo') or (defaults.get('author_settings') or {}).get('steps')
              or (defaults.get('field_sources') or {}).get('steps') in ('model','version','file'))
    steps=defaults.get('steps') if specific else lora['generation_defaults']['steps']
    result=video_sampling(item,'standard',int(steps))
    result['steps_source']='model' if specific else 'orbit'
    return result


def validate(body: VideoRequest):
    expected = video_request(OrbitRequest(image=body.first_frame or '', model_id=body.model_id, seed=body.seed,
                             randomize_seed=body.randomize_seed, output_dir=body.output_dir,
                             strength=body.loras[0].strength if len(body.loras)==1 else 1))
    for key in ('model_id','prompt','mode','last_frame','width','height','frames','steps','quality','audio','enhance','loras'):
        if getattr(body,key) != getattr(expected,key):
            raise ValueError('Use the 360 Orbit page to create an orbit with its required model and settings.')


def prepare_image(image):
    from PIL import Image, ImageOps
    # Crop once before both encodes. H3 otherwise stretches the first keyframe
    # but crops the last one, so a non-square upload would not close the loop.
    return ImageOps.fit(image.convert('RGB'), (768,768), method=Image.Resampling.LANCZOS)
