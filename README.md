# Qwen - Images & Video

אפליקציה מקומית ליצירה ועריכת תמונות, עם Qwen-Image-2.1 ב־BF16 וב־GGUF, ובחירת מודלים דרך חלון ההגדרות.

## הפעלה

להתקנה מלאה ולהפעלה: לחיצה כפולה על **Setup.bat**. בהפעלות הבאות: **Start Qwen.bat**, או **Qwen** בתיקיית REPOS.
החלון נפתח כמו אפליקציה. הפעלה נוספת משתמשת בשרת שכבר פועל.

המתקין מכין סביבת Python נפרדת, מתקין Node.js מקומי אם חסר, בונה את הממשק, מאמת את תמונות הדוגמה ובודק CUDA. אין צורך להתקין Python, Node.js או uv ידנית ואין צורך בהרשאות מנהל. נדרש כרטיס NVIDIA ודרייבר תואם. ההתקנה שומרת את מיקומי הקבצים ב־`local-settings.json`; במחשב חדש נבחר כונן עם מקום פנוי. הורדת משקלי המודלים מתבצעת **מתוך Settings בממשק**, עם התקדמות, ביטול, ניסיון נוסף ואימות SHA-256. כשהמודלים חסרים, חלון ההגדרות נפתח אוטומטית. המתקין אינו מוריד מודלי תמונה או LLM מקומי ברקע.

1. כתבו פרומפט. אפשר לגרור, לבחור או להדביק עד 10 תמונות כדי לערוך או לשלב אותן.
2. השאירו את שיפור הפרומפט פעיל, או כבו כדי להשתמש ישירות בטקסט שכתבתם.
3. בחרו **Auto 1K** ליצירה מהירה יותר או **Auto 2K** לפירוט גבוה יותר. בשניהם יחס התמונה אוטומטי. **Custom size** מאפשר גודל מדויק.
4. בשדה **Steps** אפשר לשנות את מספר הצעדים (1–80), או לחזור לברירת המחדל של המודל. לחצו על יצירת תמונה. התוצאה נשמרת כ־PNG, כולל שקיפות ופרטי היצירה.

הממשק כולו באנגלית. אפשר לכתוב פרומפטים בשפות שונות ולשמור טקסט מדויק בשפת המקור. הגודל והצעדים גלויים ליד הפרומפט; גודל מותאם כולל את כל 10 גדלי 1K/2K של הספייס. ההגדרות המתקדמות כוללות Seed, Seed אקראי, פרומפט שלילי ותיקיית שמירה. כל 37 הדוגמאות המקוריות זמינות עם תמונות הקלט והתוצאה והפרומפט המלא. טעינת דוגמה שומרת על סדר התמונות ועל הגדרת שיפור הפרומפט המקורית.

ברירת המחדל לתמונות היא **25 צעדים**, לפי קובצי ComfyUI הרשמיים ל[יצירה](https://github.com/Comfy-Org/workflow_templates/blob/371a7b7171bbd11e9cc92ef615ba5ad223d7e5b4/templates/image_qwen_image_2_1_t2i.json) ול[עריכה](https://github.com/Comfy-Org/workflow_templates/blob/371a7b7171bbd11e9cc92ef615ba5ad223d7e5b4/templates/image_qwen_image_2_1_image_edit.json), שאליהם מפנה גם [מאגר GGUF](https://huggingface.co/abenzerps/Qwen-Image-2.1-GGUF). דוגמאות Diffusers משתמשות ב־40, אבל האפליקציה משתמשת ב־ComfyUI עם Euler/simple. לכל מודל רשומה ברירת מחדל ומקור בקטלוג, עם קישור ליד Steps. בחירת מודל אחר מחזירה לברירת המחדל שלו; אפשר לשנות ידנית. בווידאו: MiniMax H3 רגיל 20, Turbo עם LoRA שמונה, ו־FastH3 V2 שמונה. פחות צעדים יכולים לפגוע בפרטים או בנאמנות לפרומפט. Auto 1K מכוון לכמיליון פיקסלים ו־Auto 2K לכארבעה מיליון, עם עיגול לממדים הנתמכים. הגדרות ישנות נשארות ב־Auto 2K.

## קבצים במחשב

- אפליקציה וקוד: `C:\Users\ido\REPOS\qwen-image-studio`
- מודלים: `E:\AI\qwen-image-studio\models`
- סביבת Python על ה־SSD הפנימי: `D:\AI\qwen-image-studio\venv`
- תמונות: `E:\AI\qwen-image-studio\images`
- פרטי משימות ותמונות המקור: `E:\AI\qwen-image-studio\jobs`
- רשימת מודלים מותקנים ואימות הורדות: `E:\AI\qwen-image-studio\installed-models.json`
- תקלות הפעלה: `E:\AI\qwen-image-studio\logs\server.err.log`

כונן E חייב להיות מחובר בזמן השימוש. יצירת התמונות פועלת מקומית. שיפור הפרומפט משתמש ב־Gemini דרך Antigravity לאחר התחברות לחשבון Google, ללא API Key. כשהשיפור פעיל, הפרומפט ותמונות הקלט נשלחים ל־Google; כשהוא כבוי אין קריאה לענן. `Download Models.bat` מוריד Q8 ו־Q4_K_M עם הקבצים הנדרשים להם, משלים הורדות שנקטעו ובודק SHA-256 מול Hugging Face. מודלי תמונה נוספים זמינים בהגדרות. אין הורדה או טעינה של LLM מקומי לשיפור הפרומפט. `Setup.bat` מתקן את סביבת ההרצה המבודדת. אין שינויים בסביבות Python של אפליקציות אחרות. אפשר להגדיר `QWEN_DATA_DIR` ו־`QWEN_RUNTIME_DIR` להתקנה במיקום אחר.

## חיבור שיפור הפרומפט

פתחו הגדרות ולחצו **התחברות עם Google**. ההתחברות נפתחת בחלון Google ומסתיימת אוטומטית באפליקציה. אם אפליקציה אחרת תופסת את פורט החזרה, אפשר להדביק את כתובת ה־localhost המלאה בשדה שמופיע בהגדרות. יש בחירת מודל, בדיקת חיבור וניתוק.

החיבור נשמר מוצפן באמצעות Windows DPAPI, עבור משתמש Windows הנוכחי. אין אסימוני גישה בפרונט, במשימות או בקובצי התמונות. רענון ההתחברות מתבצע לפני כל בקשה כשצריך. מודלי ה־PE המקומיים אינם נטענים. מקודד הטקסט של מודל התמונה עדיין נחוץ ליצירת תמונות מקומית.

## בחירת מודלים

כפתור המודל בכותרת או גלגל השיניים פותחים את ההגדרות. לכל מודל יש הורדה, בחירה והסרה עם אישור. הורדה כוללת את הקבצים הנדרשים; הסרה אינה מוחקת תמונות שנוצרו.

| מודל תמונה | גודל קובץ | מקודד אוטומטי |
| --- | --- | --- |
| Q8_0 | 7.59GB | Qwen3-VL 8B INT8 |
| Q4_K_M | 4.60GB | Qwen3-VL 8B INT8 |
| Q6_K | 5.88GB | Qwen3-VL 8B INT8 |
| Q5_K_M | 5.22GB | Qwen3-VL 8B INT8 |
| Q4_0 | 4.05GB | Qwen3-VL 8B INT8 |
| BF16 רשמי | 14.23GB | Qwen3-VL 8B BF16 |

הורדה חדשה של BF16 כוללת רק את משקלי יצירת התמונה, עם המקודד וה־VAE המשותפים כתלויות. לא מורידים את העותקים הכפולים שהיו בחבילת Diffusers המקורית. בהתקנה ישנה מוצג הנפח שקיים בפועל, והסרה מאושרת מסירה גם את הקבצים הישנים של אותה חבילה.

קובצי ה־GGUF מגיעים מהמאגר `abenzerps/Qwen-Image-2.1-Uncensored-GGUF`. המקודד INT8 שוקל 9.35GB; BF16 שוקל 17.53GB. ה־VAE התואם, `qwen_image_2.1_vae_bf16.safetensors`, שוקל 0.68GB ונבחר אוטומטית. אפשר לשנות את המקודד בהגדרות. Q8 ו־Q4_K_M עם INT8 ו־VAE יחד דורשים כ־22.2GB. שני משפרי הפרומפט המקומיים הישנים, 18.84GB כל אחד, אינם בשימוש. אפשר להסיר אותם מתוך ההגדרות לאחר אישור, בלי לפגוע בשיפור הפרומפט דרך Google.

המאגר מזהיר מתקלה ב־Q8: וקטורי RMSNorm של 128 ערכים נשמרו כ־136 בתים. המתאם המקומי מפענח את 64 הווקטורים האלה לפני הטעינה. המטריצות הגדולות נשארות מקוונטטות; קובצי המודל המקוריים אינם משתנים.

## ההבדל מהספייס

הספייס הרשמי משתמש בשרת Alibaba ובמודל `pre-qwen-image-2.1-pro-yunqi`. כאן רצים המשקלים הפתוחים הרשמיים של Qwen מקומית. לכן אותו Seed אינו הבטחה לאותה תמונה בדיוק. שיפור הפרומפט משתמש ב־Gemini דרך Antigravity, עם הוראות הכתיבה המקוריות של Qwen ליצירה ולעריכה.

- מנוע התמונה הוא ComfyUI בגרסה קבועה, ללא ממשק ComfyUI או גרף שצריך לערוך. GGUF נטען באמצעות ComfyUI-GGUF; BF16 משתמש במשקלי ה־Transformer הרשמיים. CUDA 13 מפעיל את ההאצות המתאימות ל־RTX 4090.
- המודלים משתחררים בסיום משימה ובביטול. אין טעינה כבדה בזמן פתיחת הממשק.
- הגודל האוטומטי נקבע לפי פלט שיפור הפרומפט, או לפי תמונת קלט; ברירת המחדל ללא אלה היא 2048×2048.
- גדלים מותאמים שאינם כפולה של 32 נוצרים בגודל הנתמך הקרוב כלפי מעלה ומותאמים לגודל המדויק. הפעולה מתועדת בפרטי התמונה.
- פרומפט שלילי מפעיל CFG=2; ללא פרומפט שלילי נשמרת ברירת המחדל הרשמית CFG=1. פרומפט שלילי מוסיף חישוב.
- ברירת המחדל לתמונות: 25 צעדים, לפי תהליכי ComfyUI הרשמיים שבהם האפליקציה משתמשת. אפשר לשנות את מספר הצעדים בממשק.
- ליצירת רקע שקוף יש להשתמש בנוסח המופיע בעזרה: `This is an RGBA image with transparency. [Description]. The image has alpha channel and the background is transparent.` עצם שמירת PNG ב־RGBA אינה מבטיחה שהמודל ייצור רקע שקוף.
- אין הבטחה שכל גודל וכל שילוב של 10 תמונות ייכנסו בזיכרון הפנוי בכל רגע. כשל מדווח במסך, עם אפשרות להקטין את הגודל ולנסות שוב. תמונות שכבר הושלמו נשמרות.

## מקור וגרסאות

- [Space](https://huggingface.co/spaces/Qwen/Qwen-Image-2.1), revision `e4eefeaed9b527f9b7bd98c0e3d3d29e1180606f`
- [המודל הראשי](https://huggingface.co/Qwen/Qwen-Image-2.1), revision `790c92633540aa0cb11d9abf19eb46d861714758`
- [שיפור פרומפט ליצירה](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I), revision `f3ed7985c788ad75b3ab7223e0c4c51e2a43545b`
- [שיפור פרומפט לעריכה](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I), revision `72927bc08afc99b7888ceb7d7d51a12db3700bbd`
- [קוד רשמי](https://github.com/QwenLM/Qwen-Image-2.1), קבצי שיפור הפרומפט נשמרו ב־`reference/`.
- [מאגר GGUF](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF), revision `e92386d3b86fd77ae1650534815ab3f6b85d46ba`.
- [ComfyUI](https://github.com/Comfy-Org/ComfyUI), revision `b0f4b7b294ce482a2e071d9d762c133d38c7aa07`; [ComfyUI-GGUF](https://github.com/leejet/ComfyUI-GGUF), revision `f912d5e5c25921e41eae2c0131eeb4d350e7c165`. רישיונות הקוד נכללים בתיקיות `vendor`. קוד Diffusers ב־`reference` נשמר להשוואה בלבד.
- הגרסאות המותקנות מופיעות ב־`requirements-lock.txt` וב־`package-lock.json`. תיעוד התאמת OAuth למואיז ול־Weboss נמצא ב־`reference/oauth-integration.md`.
- המודלים וקוד Qwen כפופים ל־Qwen Research License. נוסח הרישיון נמצא ב־`reference/Qwen-Research-LICENSE.txt` ובתיקיות המודלים.

## עיצוב

הקונספט נמצא ב־`reference/design-concept.png`. האייקון נמצא ב־`public/branding/`. שניהם נוצרו בכלי Image Gen המובנה; התמונות בדוגמאות הן תמונות המקור מהספייס ואינן מוצגות כתוצאות מקומיות.

כיוון העיצוב: אותם אזורי עבודה ואותו סדר פעולות, רקע לבן ולבנדר בהיר, טיפוגרפיית Segoe UI, כפתור יצירה סגול, אזור תוצאה עם תצוגת שקיפות. אין מסכי ניווט מיותרים. האייקון: שלושה סרטים סגולים שלובים כצמצם מצלמה, על רקע שקוף.

## פיתוח ובדיקות

```powershell
npm run build
D:\AI\qwen-image-studio\venv\Scripts\python.exe -m unittest discover -s tests -v
```

השרת מקשיב רק ב־`127.0.0.1`, ונפתח בפורט 47821 או בפורט פנוי עד 47840. פעולות כתיבה דורשות אסימון מקומי. אי אפשר לקבל דרך נתיבי התמונות קובץ שרירותי מהמחשב.


## Video

Open the **Video** tab in the same app. Choose **Text to video** or **Image to video**, write a prompt, choose duration and size, and click **Generate video**. Image mode accepts a first frame and an optional last frame; its size initially follows the first image. The player supports playback and MP4 download. History restores previous settings and keeps video outputs separate from images.

- **FastH3 8-Step V2 Q4_K_M**: fast text-to-video with native audio. Uses the real VSA-H3 CUDA kernels. Eight sampling steps are fixed to the distilled model's schedule.
- **MiniMax H3 Q4_0**: text-to-video and first/last-frame image-to-video. Standard mode uses 20 steps; **Turbo** uses the official 8-step LoRA. Both follow the official CFG 1 workflow.
- Settings also offers FastH3 Q5_K_M / Q6_K / INT8 and MiniMax H3 Q8_0 / INT8. Required H3 text encoder and video/audio VAEs are selected automatically and shared across the models.
- Every MiniMax H3 variant supports first/last-frame images. FastH3 V2 variants support text only, as documented by their authors. The model selector labels these capabilities; choosing image mode keeps the selected model and offers an explicit switch to MiniMax H3 when needed.
- On Windows, video inference checks NVIDIA dedicated VRAM as well as the CUDA budget, accounts for other applications, and streams weights from RAM when necessary. The conditioning encoder is released before keyframe VAE encoding. This preserves the requested frame count and resolution.
- Video models and shared files occupy about 48.45 GB for both installed Q4 models plus Turbo. No duplicate text encoder or VAE is downloaded for each video model. Model weights are downloaded, resumed and verified from **Settings**.
- Output is H.264 MP4 at 24 fps, with optional AAC stereo audio. Durations follow the model's supported frame grid; the exact duration is shown in the interface. Smaller Draft sizes are available. The models jointly generate audio/video even when audio is excluded from the saved MP4.
- Video prompt enhancement uses the same Antigravity connection. The required H3 conditioning encoder is not a prompt-enhancement LLM.

See [video research and exact model contracts](reference/video-research.md) and [validation results](reference/validation.md). FastH3 is limited to the upstream distilled text-to-video task; image inputs use MiniMax H3. The separate H3 reference-to-video checkpoint is not included in the first/last-frame model.

## Video to video and upscaling

The **Video to video** tab has three separate workflows. **Replace subjects** uses the MiniMax H3 Ref2VA FaceSwap LoRA and ordered person reference images. **Camera view** uses the CrossView Warp LoRA, MoGe 2 depth estimation, and the camera controls. It does not require or apply FaceSwap. **Reference-guided** uses the Ref2VA model with the source video and reference images without either specialized LoRA.

In Camera view, open **Replace subjects before changing camera view** only if both changes are wanted. The checkbox creates a complete subject-replacement MP4 first, then uses that generated video as the source for CrossView. The intermediate MP4 can be viewed and downloaded from the result and gallery. Optional controls, including this checkbox, camera options, and the manual camera guide, start collapsed. A generative replacement or viewpoint change may alter details outside the intended subject or reveal areas absent from the source; it cannot guarantee unchanged pixels.

Settings offers the merged eight-step MiniMax H3 Ref2VA Turbo INT8 checkpoint, the standard INT8 model, and the experimental 10Eros-Max Ref2VA Q4_K_M / Q8_0 GGUF models. Each model selects its matching H3 text encoder, video/audio VAEs, FaceSwap or CrossView LoRA, and MoGe 2 model when needed. Turbo uses eight steps by default; 10Eros-Max uses 25. Steps remain editable. Video and image models use the same Settings download, verification, and removal controls.

The **Upscale** tab runs SeedVR2 at the source frame rate and aspect ratio. FP8 and Q4_K_M GGUF are available for the 3B model; the 7B FP8 model is another Settings option. The default frame batch of five with one overlapping frame avoids redundant processing. VAE tiling is enabled automatically at 1080p and above. Source audio is retained by default and can be changed under Advanced settings. Finished videos appear in the shared gallery with a small source thumbnail, saved parameters, and a full-size overlay.

The [LTX 2.5 Multiple Subject Reference](https://huggingface.co/LiconStudio/LTX-2.5-Multiple-Subject-Reference) LoRA is listed in Settings. Its separate LTX 2.5 base model, Gemma text encoder, and VAEs require access to the [gated Lightricks model](https://huggingface.co/Lightricks/LTX-2.5) through the Hugging Face account used on this computer. The LoRA is designed for ordered image references and optional paired audio, not the FaceSwap or CrossView video-to-video workflows. LTX generation is not exposed while its required base files are unavailable.

## Queue and gallery

Image and video requests share a persistent queue. The form stays editable while jobs are running or waiting; each click saves a separate snapshot of the prompt, input images, model, dimensions, steps, output folder and resolved seed. Only one worker uses the GPU at a time. Image jobs with the same model and encoder reuse that worker; video and model changes release it before starting the next worker.

- **Queue** shows the active job, waiting order and recent results/errors. Waiting jobs can move up or be cancelled individually. **Pause queue** lets the active job finish while preserving waiting jobs; **Resume queue** continues processing. A failed or cancelled job does not stall the queue.
- Waiting jobs, their order and the paused state survive server restarts. A job interrupted during inference is shown as failed with a retry action. Queued model files cannot be removed until the jobs finish or are cancelled.
- **Gallery** combines existing and new images and videos, with type filters, prompt/model search, pagination, lightweight previews and full-size playback. **Use settings** loads the original prompt, input images, model, dimensions, steps and actual seed into the relevant form. **Run again** submits those settings directly to the queue with the seed fixed. Prompt enhancement, when enabled in the saved request, runs again and may produce a different rewrite.
- The details view includes source images, parameters, the enhanced prompt, media download and full JSON settings. Image/video drafts, including uploaded input references, survive browser refresh. Original uploads and generated files stay on the local data drive.

## Consecutive images

The image model, conditioning encoder and VAE stay constructed between jobs. GPU weights remain resident when memory permits; ComfyUI can evict a component to its retained CPU weights when more working memory is needed. On Windows, this restores the original read-only CPU views instead of copying GPU data into mapped weights. Changing image model/encoder or starting video closes the image worker first. Cancellation and inference errors discard the worker; the next job starts fresh.

The most recent prompt/input conditioning is also reused when the actual prompt, negative prompt and ordered image pixels are unchanged. Changing the seed, step count or output dimensions does not require repeating that processing. New text or images are processed normally. An empty negative prompt no longer incurs an unused negative encoding pass at CFG 1; nonempty negatives retain the official CFG 2 behavior. Prompt enhancement remains a separate optional cloud operation.

Progress shows the current stage's time and the total job time separately. **Generation time** in the result/gallery details shows model setup, prompt/input processing, generation and decoding. **Settings → Release model memory** frees the idle image worker without deleting any files. It is disabled while jobs are running or waiting.
