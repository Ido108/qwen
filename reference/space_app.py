import gradio as gr
import numpy as np
import random
import requests
import io
import time
# import spaces
import uuid
from datetime import datetime

from PIL import Image

import os
import base64
import json

# ============== 配置参数 ==============
# 日志目录，可通过环境变量 LOG_DIR 自定义
LOG_DIR = os.environ.get("LOG_DIR", "./generation_logs_paper_case")

# ============== API 配置 ==============
API_ENDPOINT = "https://poc-dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
EDIT_MODEL = "pre-qwen-image-2.1-pro-yunqi"
T2I_MODEL = "pre-qwen-image-2.1-pro-yunqi"

MAX_INPUT_IMAGES = 10
DEFAULT_LANGUAGE = "en"

# ============== 预设分辨率 ==============
SIZE_PRESETS_2K = {
    "2688x1536 (16:9)": (2688, 1536),
    "1536x2688 (9:16)": (1536, 2688),
    "2048x2048 (1:1)": (2048, 2048),
    "2368x1728 (4:3)": (2368, 1728),
    "1728x2368 (3:4)": (1728, 2368),
}
SIZE_PRESETS_1K = {
    "1344x768 (16:9)": (1344, 768),
    "768x1344 (9:16)": (768, 1344),
    "1184x864 (4:3)": (1184, 864),
    "864x1184 (3:4)": (864, 1184),
    "1024x1024 (1:1)": (1024, 1024),
}
# 合并所有预设，添加自定义选项
ALL_SIZE_PRESETS = {"自定义 (Custom)": None}
ALL_SIZE_PRESETS.update({f"2K - {k}": v for k, v in SIZE_PRESETS_2K.items()})
ALL_SIZE_PRESETS.update({f"1K - {k}": v for k, v in SIZE_PRESETS_1K.items()})


# ============== 日志记录类 ==============
class GenerationLogger:
    """记录生成日志：JSON(prompt信息) + PNG(图片)"""

    def __init__(self, log_dir="./logs"):
        """
        初始化日志记录器

        Args:
            log_dir: 日志存储目录
        """
        self.log_dir = log_dir
        self._ensure_dir()


    def _ensure_dir(self):
        """确保日志目录存在"""
        os.makedirs(self.log_dir, exist_ok=True)

    def set_log_dir(self, log_dir):
        """
        动态设置日志目录

        Args:
            log_dir: 新的日志目录路径
        """
        self.log_dir = log_dir
        self._ensure_dir()

    def log_generation(self, original_prompt, enhanced_prompt, image, seed, params, gpu_id=None, input_images=None, username=None, api_response=None):
        """
        保存一次生成的完整记录

        Args:
            original_prompt: 用户输入的原始prompt
            enhanced_prompt: LLM改写后的prompt
            image: 生成的PIL图片对象
            seed: 使用的随机种子
            params: 生成参数字典
            gpu_id: 使用的GPU ID
            input_images: 输入图片列表（PIL Image对象）
            username: 触发生成的用户名

        Returns:
            str: 日志文件的基础名称（不含扩展名）
        """
        self._ensure_dir()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        base_name = f"{timestamp}_{unique_id}"

        # 保存输入图片
        input_image_paths = []
        if input_images and len(input_images) > 0:
            for i, img in enumerate(input_images):
                input_path = os.path.join(self.log_dir, f"{base_name}.{i}.png")
                img.save(input_path)
                input_image_paths.append(os.path.abspath(input_path))
            print(f"[Logger] 已保存 {len(input_images)} 张输入图片")

        # 保存输出图片
        image_path = os.path.join(self.log_dir, f"{base_name}.png")
        image.save(image_path)

        # 保存JSON日志
        json_path = os.path.join(self.log_dir, f"{base_name}.json")
        log_data = {
            "timestamp": timestamp,
            "username": username,
            "original_prompt": original_prompt,
            "enhanced_prompt": enhanced_prompt,
            "seed": seed,
            "gpu_id": gpu_id,
            "parameters": params,
            "input_images": input_image_paths,
            "image_path": os.path.abspath(image_path),
            "api_response": api_response
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        print(f"[Logger] 日志已保存: {base_name}.json / {base_name}.png")
        return base_name



def validate_image_count(images):
    if images is not None and len(images) > MAX_INPUT_IMAGES:
        raise gr.Error("最多支持 10 张输入图片。 / Up to 10 input images are supported.")


def encode_image(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def call_image_api(prompt, mode="t2i", images=None, size="1024*1024", negative_prompt=" ", seed=None, prompt_extend=True):
    """
    调用 POC API 生成图片

    Args:
        prompt: 文本提示词
        mode: "t2i" 或 "edit"
        images: PIL Image 列表 (edit 模式需要)
        size: 尺寸字符串如 "1024*1024"
        negative_prompt: 负向提示词
        seed: 随机种子
        prompt_extend: 是否开启API侧提示词智能改写

    Returns:
        tuple: (PIL.Image, dict) 生成的图片和完整API响应
    """
    validate_image_count(images)

    # 使用独立的 IMAGE_API_KEY 环境变量
    api_key = os.environ.get('IMAGE_API_KEY')
    if not api_key:
        raise EnvironmentError("IMAGE_API_KEY environment variable is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 构造 messages
    content = [{"text": prompt}]

    if mode == "edit":
        model = EDIT_MODEL
        # 添加图片到 content
        if images and len(images) > 0:
            # 图片放在文本之前
            content = []
            content.append({"text": prompt})
            for img in images:
                img_base64 = encode_image(img)
                content.append({"image": f"data:image/png;base64,{img_base64}"})
    else:
        model = T2I_MODEL

    # 构造请求体
    payload = {
        "model": model,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        },
        "parameters": {
            "watermark": False,
            "negative_prompt": negative_prompt,
            "prompt_extend": prompt_extend,
            "debug": True
        }
    }

    # 传入 seed 参数
    if seed is not None:
        payload["parameters"]["seed"] = int(seed)

    # 传入 size 参数（T2I 始终传，Edit 仅在指定时传）
    if size:
        payload["parameters"]["size"] = size

    print(f"[API] Calling {model} API...")
    print(f"[API] Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"[API] Prompt: {prompt}")
    if mode == "edit" and images:
        print(f"[API] Input images count: {len(images)}")

    # 发送请求
    response = requests.post(API_ENDPOINT, headers=headers, json=payload)
    print(response)

    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")

    result = response.json()
    print(f"[API] Response: {json.dumps(result, ensure_ascii=False)[:500]}...")

    # 解析响应 - 处理同步和异步两种情况
    if "output" in result:
        output = result["output"]

        # 检查是否是异步任务
        if "task_id" in output and "task_status" in output:
            task_id = output["task_id"]
            task_status = output["task_status"]
            print(f"[API] Async task created: {task_id}, status: {task_status}")

            # 轮询等待任务完成
            image, async_result = poll_task_result(task_id, api_key)
            return image, async_result

        # 同步响应 - 直接返回结果
        if "choices" in output and len(output["choices"]) > 0:
            choice = output["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                for item in choice["message"]["content"]:
                    if "image" in item:
                        image_url = item["image"]
                        return download_image(image_url), result

        # 另一种响应格式
        if "results" in output and len(output["results"]) > 0:
            image_url = output["results"][0].get("url")
            if image_url:
                return download_image(image_url), result

    raise Exception(f"Failed to parse API response: {result}")


def poll_task_result(task_id, api_key, max_retries=120, interval=2):
    """
    轮询异步任务结果

    Args:
        task_id: 任务ID
        api_key: API密钥
        max_retries: 最大重试次数
        interval: 轮询间隔(秒)

    Returns:
        tuple: (PIL.Image, dict) 生成的图片和完整API响应
    """
    task_url = f"https://poc-dashscope.aliyuncs.com/api/v1/tasks/{task_id}"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }

    for i in range(max_retries):
        response = requests.get(task_url, headers=headers)
        if response.status_code != 200:
            print(f"[API] Task polling failed: {response.status_code}")
            time.sleep(interval)
            continue

        result = response.json()
        output = result.get("output", {})
        task_status = output.get("task_status", "UNKNOWN")

        print(f"[API] Task {task_id} status: {task_status} (attempt {i+1}/{max_retries})")

        if task_status == "SUCCEEDED":
            # 获取生成的图片
            if "results" in output and len(output["results"]) > 0:
                image_url = output["results"][0].get("url")
                if image_url:
                    return download_image(image_url), result
            raise Exception(f"Task succeeded but no image found in response: {result}")

        elif task_status == "FAILED":
            error_msg = output.get("message", "Unknown error")
            raise Exception(f"Task failed: {error_msg}")

        elif task_status in ["PENDING", "RUNNING"]:
            time.sleep(interval)
            continue

        else:
            print(f"[API] Unknown task status: {task_status}")
            time.sleep(interval)

    raise Exception(f"Task {task_id} timed out after {max_retries * interval} seconds")


def download_image(url):
    """
    从URL下载图片

    Args:
        url: 图片URL

    Returns:
        PIL.Image: 下载的图片
    """
    print(f"[API] Downloading image from: {url[:100]}...")
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download image: {response.status_code}")

    with Image.open(io.BytesIO(response.content)) as image:
        return image.convert("RGBA")




# 初始化日志记录器
print(f"Initializing logger with directory: {LOG_DIR}")
logger = GenerationLogger(log_dir=LOG_DIR)


# --- UI Constants and Helpers ---
MAX_SEED = np.iinfo(np.int32).max

# --- Stage 2: Image Generation Function ---
def generate_image_stage(
    input_images,
    original_prompt,
    custom_size,
    log_dir,
    seed=42,
    randomize_seed=False,
    height=1024,
    width=1024,
    negative_prompt=" ",
    prompt_extend=True,
    username=None,
    progress=gr.Progress(track_tqdm=True),
):
    """
    使用 POC API 生成图片。
    根据是否有输入图片自动判断模式：有图片 -> Edit，无图片 -> T2I。
    改写由 API 侧 prompt_extend 参数控制。
    """
    validate_image_count(input_images)

    # 更新日志目录（如果用户修改了）
    if log_dir and log_dir.strip():
        logger.set_log_dir(log_dir.strip())

    # 使用传入的 negative_prompt（默认为空格）
    if not negative_prompt or not negative_prompt.strip():
        negative_prompt = " "

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # Load input images into PIL Images from gallery
    pil_images = []
    if input_images is not None and len(input_images) > 0:
        for item in input_images:
            try:
                if isinstance(item, Image.Image):
                    pil_images.append(item.convert("RGB"))
                elif isinstance(item, tuple) and len(item) > 0 and isinstance(item[0], Image.Image):
                    pil_images.append(item[0].convert("RGB"))
                elif isinstance(item, str):
                    pil_images.append(Image.open(item).convert("RGB"))
                elif hasattr(item, "name"):
                    pil_images.append(Image.open(item.name).convert("RGB"))
            except Exception as e:
                print(f"[Warning] Failed to load input image: {e}")
                continue
        print(f"[API] Loaded {len(pil_images)} input images for editing")

    # 根据是否有输入图片自动判断模式
    is_edit_mode = len(pil_images) > 0
    mode = "edit" if is_edit_mode else "t2i"

    # 构造 size：仅在用户开启自定义尺寸时传，否则由 API 自动决定
    if custom_size:
        size = f"{width}*{height}"
    else:
        size = None

    print(f"[API] Mode: {mode} (auto-detected: {'has input images' if is_edit_mode else 'no input images'})")
    print(f"[API] Prompt: '{original_prompt[:100]}...'" if len(original_prompt) > 100 else f"[API] Prompt: '{original_prompt}'")
    print(f"[API] Negative Prompt: '{negative_prompt}'")
    print(f"[API] Prompt Extend: {prompt_extend}")
    print(f"[API] Input images count: {len(pil_images)}")
    print(f"[API] Seed: {seed}, Size: {size if size else 'auto (not specified)'}")

    # 调用 API 生成图片
    image, api_response = call_image_api(
        prompt=original_prompt,
        mode=mode,
        images=pil_images if len(pil_images) > 0 else None,
        size=size,
        negative_prompt=negative_prompt,
        seed=seed,
        prompt_extend=prompt_extend
    )

    # Only show actual API rewrite text; never substitute the original prompt.
    rewritten_prompt = extract_rewritten_prompt(api_response) if prompt_extend else ""
    enhanced_prompt = rewritten_prompt or original_prompt

    # 记录生成日志
    params = {
        "mode": mode,
        "height": height,
        "width": width,
        "negative_prompt": negative_prompt,
        "prompt_extend": prompt_extend,
        "input_images_count": len(pil_images),
        "api_model": EDIT_MODEL if is_edit_mode else T2I_MODEL
    }

    log_name = logger.log_generation(
        original_prompt=original_prompt,
        enhanced_prompt=enhanced_prompt,
        image=image,
        seed=seed,
        params=params,
        gpu_id=None,
        input_images=pil_images if len(pil_images) > 0 else None,
        username=username,
        api_response=api_response
    )

    print(f"[API] Generation complete, logged as: {log_name}")

    return image, seed, rewritten_prompt


def extract_rewritten_prompt(api_response):
    """Read the text returned alongside the image when parameters.debug is true."""
    if not isinstance(api_response, dict):
        return ""
    output = api_response.get("output") or {}
    debug_info = output.get("debug_info") or {}
    rewrite_info = debug_info.get("rewrite_debug_info") or {}
    rewritten = rewrite_info.get("rewritten_prompt")
    if isinstance(rewritten, str) and rewritten.strip():
        return rewritten
    # Some responses expose only the prompt actually passed to generation.
    actual_prompt = debug_info.get("actual_prompt")
    if output.get("rewrite_status") == "success" and isinstance(actual_prompt, str) and actual_prompt.strip():
        return actual_prompt
    # Compatibility with responses that include rewrite text beside the image.
    choices = output.get("choices") or []
    if not choices:
        return ""
    content = (choices[0].get("message") or {}).get("content") or []
    return "\n\n".join(
        item["text"] for item in content
        if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip()
    )


def make_placeholder_image(text, width=512, height=320, bg_color=(30, 30, 30), text_color=(200, 200, 200)):
    """生成一张带有居中文字的占位图片"""
    from PIL import ImageDraw, ImageFont
    img = Image.new("RGBA", (width, height), (*bg_color, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - tw) // 2
    y = (height - th) // 2
    draw.text((x, y), text, fill=text_color, font=font)
    return img


# --- Unified generate_with_enhance generator ---
def generate_with_enhance(
    input_images,
    original_prompt,
    enable_extend,
    custom_size,
    log_dir,
    seed,
    randomize_seed,
    height,
    width,
    negative_prompt,
    request: gr.Request,
):
    """
    生成图片，prompt_extend 由 API 侧处理。
    Yields intermediate results so the UI updates progressively.
    """
    username = request.username if request else None
    print(f"[API] Request from user: {username}")

    yield make_placeholder_image("Generating image..."), seed, ""

    image, seed, rewritten_prompt = generate_image_stage(
        input_images, original_prompt,
        custom_size, log_dir, seed, randomize_seed, height, width,
        negative_prompt=negative_prompt,
        prompt_extend=enable_extend,
        username=username,
    )
    yield image, seed, rewritten_prompt


def make_example_loader(images, text, extend):
    """Bind each example to a zero-argument callback without late binding."""
    def load_example():
        return list(images), text, extend
    return load_example


# UI translations are registered once; callbacks update only the current session.
localized_components = []
localized_properties = []


def localize(component, **properties):
    localized_components.append(component)
    localized_properties.append(properties)
    for name, values in properties.items():
        setattr(component, name, values[1 if DEFAULT_LANGUAGE == "en" else 0])
    return component


def switch_language(language):
    index = 1 if language == "en" else 0
    return [gr.update(**{name: values[index] for name, values in props.items()})
            for props in localized_properties]


EXAMPLE_TITLES_EN = {
    "电影角色三视图": "Film character turnaround",
    "小羊肖恩生日故事": "Shaun the Sheep birthday story",
    "民国漫剧分镜": "Period drama storyboard",
    "祈年殿拆解科普卡": "Temple architecture infographic",
    "AI 进化时间轴": "AI evolution timeline",
    "唇膏广告分镜": "Lipstick commercial storyboard",
    "巨兽对决故事板": "Giant creature battle storyboard",
}

ENGLISH_EXAMPLES = [
    ("Editorial portrait", "Create an editorial portrait of a botanist in a sunlit greenhouse, surrounded by ferns and delicate orchids. Natural skin texture, linen clothing, soft morning backlight, subtle film grain, medium-format photography, calm expression, no text or watermark."),
    ("Typography poster", 'Design a refined travel poster for a fictional night train. Render the headline exactly as "THE MIDNIGHT EXPRESS" and the subtitle "A journey under the stars". A silver train curves through dark blue mountains beneath a crescent moon. Art Deco geometry, ivory and gold lettering, clear typographic hierarchy, generous margins, print-ready composition.'),
    ("Six-panel storyboard", "Create a six-panel cinematic storyboard about a small robot restoring an abandoned rooftop garden. Show: arrival at dawn, discovery of a dried seedling, repairing an irrigation pipe, planting new seeds, the first rain, and a lush garden at sunset. Keep the robot's round yellow body and blue eyes consistent in every panel. Clear panel borders, expressive visual storytelling, detailed environments, no captions."),
    ("Product photography", "Photograph a translucent emerald perfume bottle on pale limestone beside a shallow pool. Rippling sunlight reflects through the glass onto the stone. A single olive branch frames the upper left corner. Luxury product photography, realistic refraction, crisp bottle edges, soft shadows, uncluttered composition, no logo or text."),
]


# --- Examples and UI Layout ---
EXAMPLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
with open(os.path.join(EXAMPLE_DIR, "cases.json"), encoding="utf-8") as case_file:
    DEMO_CASES = json.load(case_file)
with open(os.path.join(EXAMPLE_DIR, "generated_references.json"), encoding="utf-8") as reference_file:
    GENERATED_REFERENCES = {case["prompt"]: case for case in json.load(reference_file)}


def add_generated_reference(prompt):
    reference = GENERATED_REFERENCES[prompt]
    localize(gr.Gallery(
        value=[os.path.join(EXAMPLE_DIR, name) for name in reference["outputs"]],
        columns=1, height=280, interactive=False,
    ), label=("参考结果", "Reference output"))



def add_case_examples(cases, input_images, prompt, enable_extend):
    """Show bundled reference images and load full prompts in their original order."""
    for case in cases:
        paths = [os.path.join(EXAMPLE_DIR, name) for name in case["inputs"]]
        references = [os.path.join(EXAMPLE_DIR, name) for name in case["outputs"]]
        with gr.Accordion(case["title"], open=False) as panel:
            localize(panel, label=(case["title"], case["title_en"]))
            with gr.Row():
                if paths:
                    localize(gr.Gallery(
                        value=[(path, str(index)) for index, path in enumerate(paths, 1)],
                        columns=min(len(paths), 5), height=280, interactive=False,
                    ), label=("输入参考图（按编号顺序）", "Input references (in numbered order)"))
                if references:
                    localize(gr.Gallery(value=references, columns=1, height=280, interactive=False),
                             label=("参考结果", "Reference output"))
            localize(gr.Textbox(value=case["prompt"], lines=4, max_lines=20, interactive=False),
                     label=("完整提示词", "Full prompt"))
            button = localize(gr.Button(), value=("使用此示例", "Use this example"))
            button.click(
                fn=make_example_loader(paths, case["prompt"], case["prompt_extend"]),
                inputs=[], outputs=[input_images, prompt, enable_extend], queue=False,
            )


css = """
#col-container {
    margin: 0 auto;
    max-width: 1024px;
}
#edit_text{margin-top: -62px !important}
"""

with gr.Blocks(title="Qwen Image 2.1 Demo", css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.HTML('<a href="https://huggingface.co/Qwen/Qwen-Image-2.1" target="_blank"><img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Image/image2.1/logo.png" alt="Qwen-Image Logo" width="400" style="display: block; margin: 0 auto;"></a>')
        language = gr.Radio(choices=[("中文", "zh"), ("English", "en")], value=DEFAULT_LANGUAGE, label="语言 / Language")
        instructions = gr.Markdown("""
## Qwen Image 2.1 Demo 使用说明
1. 如果不输入图片，默认进入文生图模式；如果输入图片，则进入图像编辑模式（支持1-10张图片）。
2. 默认"Enable Prompt Extend"为开启状态，API会自动进行提示词智能改写。如果希望直接使用原始提示词，可以关闭该选项。
3. 下方提供了包括文生图，图生图的测试样例，可以作为模型基础能力的参考。
4. 生成透明图时，建议提示词遵循以下格式，将中间的省略号替换为具体画面描述：

   `这是一张带有透明度的RGBA图像。.... 该图像具有alpha通道，背景是透明的。`
        """)
        with gr.Row():
            with gr.Column():
                input_images = gr.Gallery(
                    label="Input Images (for editing)",
                    show_label=True,
                    type="pil",
                    interactive=True,
                    columns=5,
                    height="auto"
                )
            with gr.Column():
                result = gr.Image(label="Result", show_label=False, type="pil", image_mode="RGBA", format='png')

        # Prompt Input
        with gr.Row():
            prompt = gr.Text(
                    label="Prompt",
                    lines=6,
                    max_lines=24,
                    show_label=True,
                    placeholder="Describe what you want to generate or edit...",
                    container=True,
            )
        with gr.Row():
            enable_extend = gr.Checkbox(
                label="Enable Prompt Extend (API侧提示词智能改写)",
                value=True,
            )
            generate_button = gr.Button("Generate Image", variant="primary")

        rewritten_prompt_output = localize(
            gr.Textbox(value="", lines=4, max_lines=20, interactive=False),
            label=("改写结果", "Rewritten prompt"),
            placeholder=("开启智能改写并生成图片后，API 返回的改写提示词会显示在这里；未返回时留空。",
                         "After generation with prompt enhancement enabled, the API rewrite appears here. Blank if none is returned."),
        )
        enable_extend.change(fn=lambda: "", inputs=[], outputs=[rewritten_prompt_output], queue=False)



        with gr.Accordion("Advanced Settings", open=False) as advanced:
            # 日志目录配置
            log_dir_input = gr.Textbox(
                label="Log Directory (日志保存目录)",
                show_label=True,
                value=LOG_DIR,
                placeholder="输入日志保存目录路径，例如: ./generation_logs",
                interactive=True,
                visible=False,
            )

            # gr.Markdown(f"**当前使用 API 模式** (Edit: {EDIT_MODEL}, T2I: {T2I_MODEL})")

            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            )

            randomize_seed = gr.Checkbox(label="Randomize seed", value=True)

            negative_prompt_input = gr.Textbox(
                label="Negative Prompt (负向提示词)",
                value=" ",
                placeholder="输入不希望出现的内容",
                lines=2,
                interactive=True,
            )

            # 自定义输出尺寸
            custom_size = gr.Checkbox(
                label="自定义输出尺寸 (Customize output size)",
                value=False,
                info="关闭时由 API/模型自动决定输出尺寸；开启后使用下方分辨率设置",
            )

            # 分辨率预设选择
            resolution_help = gr.Markdown("**分辨率设置 (Resolution Settings) — 需开启上方自定义尺寸选项**")
            with gr.Row():
                resolution_preset = gr.Dropdown(
                    choices=list(ALL_SIZE_PRESETS.keys()),
                    value="2K - 2688x1536 (16:9)",
                    label="预设分辨率 (Resolution Preset)",
                    info="选择预设分辨率或自定义",
                )

            with gr.Row():
                width = gr.Slider(
                    label="Width",
                    minimum=256,
                    maximum=2688,
                    step=8,
                    value=2688,
                )

                height = gr.Slider(
                    label="Height",
                    minimum=256,
                    maximum=2688,
                    step=8,
                    value=1536,
                )

            # 分辨率预设回调函数
            def update_resolution(preset_name):
                if preset_name == "自定义 (Custom)" or ALL_SIZE_PRESETS.get(preset_name) is None:
                    # 自定义模式，不改变当前值
                    return gr.update(), gr.update()
                w, h = ALL_SIZE_PRESETS[preset_name]
                return w, h

            resolution_preset.change(
                fn=update_resolution,
                inputs=[resolution_preset],
                outputs=[width, height],
            )

        # --- Examples ---
        examples_heading = gr.Markdown("### 示例")

        localize(gr.Markdown(), value=("**精选文生图示例**", "**Featured text-to-image examples**"))
        add_case_examples([case for case in DEMO_CASES if not case["inputs"]],
                          input_images, prompt, enable_extend)
        edit_heading = gr.Markdown("**图像编辑示例**")
        add_case_examples([case for case in DEMO_CASES if case["inputs"]],
                          input_images, prompt, enable_extend)

        # T2I examples (no input images)
        t2i_heading = gr.Markdown("**中文文生图示例**")
        t2i_examples = [
                ["""生成一张真人电影质感的单角色三视图设定参考图。干净灰色背景，左侧为一张较大的完整头肩半身细节图，必须是同一角色从头部到肩部/胸口的连续完整特写，清楚展示面部、发型、上身服装、配饰和材质细节；左侧不要拆成多个局部小图、不要拼贴多个细节框、不要只给眼睛/衣料/配饰等碎片特写。右侧展示同一角色正面、侧面、背面全身三视图，全身可见。角色设定如下：丹尼尔·斯通，纯白背景，无其他人物和场景；丹尼尔·斯通的单人全身立绘，正对镜头站立，表情自然；30岁青年男性，2020年代现代美国，欧洲裔白人外貌，浅肤色，身高约185cm，9头身，魁梧健壮体型，肌肉发达，2020年代棕色短寸头，发质粗硬，眼窝深陷，瞳孔呈灰蓝色，鼻梁高挺且鼻头宽大，唇形厚实且唇色苍白，方下颌轮廓分明，面部皮肤粗糙并带有污垢痕迹，穿着2020年代脏污灰色连帽卫衣配同色系工装裤，脚穿2020年代黑色防滑工装靴，双手佩戴破旧皮革手套，双手自然下垂；光照均匀，高画质，手部完美，无文字水印。 三视图中面部特征、身形比例、发型、服装、配饰、鞋履、姿态和关键视觉元素保持一致；表情自然中性，站姿稳定，比例统一，皮肤和布料金属等材质细节清楚，电影级写实光照，高画质，无文字、无logo、无水印。""", True],
                ["""3D粘土漫画，生成小羊肖恩\n镜头一：清晨的青苔底农场暖意融融，小羊肖恩发现农夫正在精心装饰巧克力生日蛋糕。镜头二：随即心生主意，独自带着小羊提米靠近厨房。镜头三：它悄悄避开熟睡的牧羊犬，带着提米溜进屋内。镜头四：不料提米意外打翻面粉、蹭损蛋糕，场面十分狼狈。镜头五：关键时刻肖恩灵机一动，伸出手指，在厚厚的白色粉末上流畅地画了起来。\n首先是一个巨大的、歪歪扭拙却充满爱意的爱心；接着是"HAPPY BIRTHDAY"的字样。随后，他指挥提米将散落的鲜红草莓摆放在爱心周围，又将从窗外采摘的野花（雏菊、矢车菊）插在蛋糕受损的边缘，巧妙地遮盖了瑕疵。原本狼藉的桌面，瞬间变成了一幅质朴而温馨的田园画作。肖恩退后一步，满意地拍了拍手上的面粉，脸上露出自豪的微笑。镜头六：农夫归来后，看见这份质朴又温馨的布置开怀大笑，这场小小的意外，最终变成了农场治愈又暖心的生日惊喜。""", True],
                ["""一组连贯短视频故事板分镜图，竖版漫剧短剧叙事节奏，2.5D 拟真人细腻质感，民国新旧交替时代背景，主角是清冷民国女学生，配儒雅留洋青年配角，镜头遵循全景 - 中景 - 近景 - 特写顺序；镜头内容： 老街全景雨雾、 二人撑伞对视中景、女生垂眸特写、青年递信手部特写、 巷口转身远景、灯下拆信近景、含泪侧脸特写、留白空镜；每格底部标注简短剧情字幕，画面低饱和复古色调，柔和分层柔光，人物五官写实精致，布料纹理清晰，无杂乱水印，分镜边框规整，镜头衔接流畅，叙事紧凑适配短视频，整体光影层次丰富，全程统一 2.5D 拟真人画风，人物形象全程不崩，场景道具贴合民国时代设定""", True],
                ["""竖版祈年殿拆解科普卡，纯白色宣纸底色，左竖排书法"祈年殿"、小字天坛与红印章，附线稿和简介；主体从上到下分层拆解全建筑构件，右侧虚线序号标注名称，中式手绘，配色棕、深蓝、朱红、米白。""", True],
                ["""生成一张计算机/技术主题的"AI进化历程"时间演化卡，采用线性顺序的时间轴/编年结构，风格为手绘/涂鸦。背景用大地棕褐色系单色渐变，搭配手绘涂鸦元素与统一科普插图。版式采用卡片网格、上图下文，图文留白清晰。标题用粗黑标题体，正文用中文手写体，重点文字采用渐变色填充字与双色调文字，文字信息整体规整易读，完整呈现AI发展节点。""", True],
                ["""生成一张原创高端美妆广告分镜展示板，主题为"LUNE 丝绒唇膏 15 秒广告脚本"。整体像真实品牌团队制作的商业广告脚本视觉板，包含 5 个连续分镜画面 + 时间轴 + 字幕 / 台词 + 镜头说明 + 转场提示。排版清晰、统一、精致、高级，具有女性向、轻奢、电影感、美妆广告质感。主色调为 奶油白、黑色、酒红、玫瑰豆沙、暖金色，灯光柔和，肤质细腻，材质高级。""", True],
                ["""核心大纲 1. 场景：3DCG质感，废弃都市遗迹（黄昏），尘土漫天，绯红色夕阳，地面有巨大脚印。 2. 人物：炎狱巨蜥（熔岩鳞片、喷烈焰）、霜牙巨象（雪白、喷寒气）、幸存人类（远景蜷缩）。 3. 剧情脉络： （1）对峙（0-20秒）：两只巨兽在废墟对峙，人类蜷缩避险，氛围压迫。 （2）开战（21-50秒）：炎狱巨蜥喷烈焰，霜牙巨象喷寒气对抗，引发冲击，高楼坍塌。 （3）激战（51-90秒）：双方互相攻击，均受重伤，嘶吼震彻废墟。 （4）落幕（91-110秒）：两只巨兽重伤对峙，镜头拉远，定格巨兽与废墟，留下悬念""", True],
            ]
        for example_title, example in zip(['电影角色三视图', '小羊肖恩生日故事', '民国漫剧分镜', '祈年殿拆解科普卡', 'AI 进化时间轴', '唇膏广告分镜', '巨兽对决故事板'], t2i_examples):
            with gr.Accordion(example_title, open=False) as example_panel:
                localize(example_panel, label=(example_title, EXAMPLE_TITLES_EN[example_title]))
                add_generated_reference(example[0])
                gr.Markdown(example[0])
                example_button = localize(gr.Button(), value=("使用此提示词", "Use this prompt"))
                example_button.click(
                    fn=make_example_loader([], example[0], example[1]),
                    inputs=[],
                    outputs=[input_images, prompt, enable_extend],
                    queue=False,
                )

        english_heading = localize(gr.Markdown(), value=("**英文文生图示例**", "**English text-to-image examples**"))
        for title, text in ENGLISH_EXAMPLES:
            with gr.Accordion(title, open=False):
                add_generated_reference(text)
                gr.Markdown(text)
                button = localize(gr.Button(), value=("使用此提示词", "Use this prompt"))
                button.click(fn=make_example_loader([], text, True), inputs=[],
                             outputs=[input_images, prompt, enable_extend], queue=False)

    localize(instructions, value=(instructions.value, """## Qwen Image 2.1 Demo
1. Generate from text without an input image, or upload 1–10 images to edit or combine them. Refer to images by their upload order in your prompt.
2. Prompt enhancement is enabled by default. Turn it off to use your original prompt directly.
3. Expand an example to read its full prompt, then click its button to fill the input. Chinese and English prompts are both supported.
4. For transparent image generation, use the following prompt format and replace `xxxxx` with your image description:

   `This is an RGBA image with transparency. xxxxx The image has alpha channel and the background is transparent.`
"""))
    localize(input_images, label=("输入图片（编辑用，最多 10 张）", "Input images (editing, up to 10)"))
    localize(result, label=("生成结果", "Result"))
    localize(prompt, label=("提示词", "Prompt"), placeholder=("描述想生成或编辑的内容，可按上传顺序引用第 1–10 张图…", "Describe what to generate or edit; refer to images 1–10 in upload order…"))
    localize(enable_extend, label=("智能改写提示词", "Enhance prompt"))
    localize(generate_button, value=("生成图片", "Generate image"))
    localize(advanced, label=("高级设置", "Advanced settings"))
    localize(log_dir_input, label=("日志保存目录", "Log directory"), placeholder=("输入日志保存目录", "Enter a log directory"))
    localize(seed, label=("随机种子", "Seed"))
    localize(randomize_seed, label=("随机种子自动变化", "Randomize seed"))
    localize(negative_prompt_input, label=("负向提示词", "Negative prompt"), placeholder=("输入不希望出现的内容", "Describe what to exclude"))
    localize(custom_size, label=("自定义输出尺寸", "Customize output size"), info=("关闭时自动选择尺寸；开启后使用下方设置", "When off, size is chosen automatically. Enable to use the settings below."))
    localize(resolution_help, value=("**分辨率设置 — 需开启自定义尺寸**", "**Resolution settings — enable custom output size first**"))
    preset_values = list(ALL_SIZE_PRESETS)
    localize(resolution_preset, label=("预设分辨率", "Resolution preset"), info=("选择预设分辨率或自定义", "Choose a preset or custom dimensions"), choices=(
        [("自定义" if ALL_SIZE_PRESETS[v] is None else v, v) for v in preset_values],
        [("Custom" if ALL_SIZE_PRESETS[v] is None else v, v) for v in preset_values]))
    localize(width, label=("宽度", "Width"))
    localize(height, label=("高度", "Height"))
    localize(examples_heading, value=("### 示例", "### Examples"))
    localize(t2i_heading, value=("**中文文生图示例**", "**Chinese text-to-image examples**"))
    localize(edit_heading, value=("**图像编辑示例**", "**Image-editing examples**"))
    language.change(fn=switch_language, inputs=[language], outputs=localized_components, queue=False)
    input_images.upload(fn=validate_image_count, inputs=[input_images], outputs=[], queue=False)

    # Generate Image button event (with optional prompt enhancement)
    generate_button.click(
        fn=generate_with_enhance,
        inputs=[
            input_images,     # input_images (有图片->Edit模式, 无图片->T2I模式)
            prompt,           # original_prompt
            enable_extend,    # enable_extend (是否开启API侧改写)
            custom_size,      # custom_size (是否自定义输出尺寸)
            log_dir_input,    # log_dir
            seed,
            randomize_seed,
            height,
            width,
            negative_prompt_input,  # negative_prompt (负向提示词)
        ],
        outputs=[result, seed, rewritten_prompt_output],
        concurrency_limit=2
    )

demo.queue(default_concurrency_limit=10, max_size=20)

if __name__ == "__main__":
    # 使用 os.path.realpath 解析真实路径，避免 NAS 软链接导致 Gradio 文件权限检查失败（404）
    _script_dir = os.path.realpath(os.path.dirname(os.path.abspath(__file__)))
    _allowed = [
        os.path.join(_script_dir, "examples"),
        os.path.realpath(EXAMPLE_DIR),
    ]
    demo.launch(
        server_name="0.0.0.0",
        allowed_paths=_allowed,
    )
