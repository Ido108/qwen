# Master task list (deduplicated, newest wording wins)

Status key:
- DONE+VERIFIED: the assistant reports a real test (live job, browser check, measured numbers).
- DONE-UNVERIFIED: built or claimed, no real test reported (or tested only by unit tests or an agent with no reported check).
- PARTIAL: part done, part open.
- NOT DONE: not built, or still open.
- BROKEN: the user said later that it does not work, and no later fix is confirmed.
- Strict rule used: if the user complained again after a claimed fix, the item is at most PARTIAL until a new test after that complaint.
- "Refs" are the msg keys shown in Part 1 as "(msg N)".

Git: origin/main ends at be5901a (pushed 2026-10-22 09:42 UTC). 15 local commits are NOT pushed: 1786778, 688ed6f, 0a6e167, 29fe604, b3c3c4d, bc15d53, 3f8f8dc, 9fae7e8, 9adfeb7, d94b5b6, c68f6b1, 1ff563c, abda7d3, 2a5a967, a80ad52 (a80ad52 is the newest). Uncommitted: docs/progress-2026-10-22.md, scripts/openclaw-manual.ps1 (not ours).

## Swap

| ID | Task (newest wording) | Refs | Status | Evidence |
|---|---|---|---|---|
| SW1 | Krea 2 swap must really swap the face (now only the hair changes, "does not work at all") | 186, 191, 236, N1 | BROKEN | User 13:08 "Krea 2 swap does not work at all"; assistant 12:15 saw the Krea result kept the scene's hair bun; progress page S1 BROKEN; audit agent running |
| SW2 | Swap perfect for every model (Qwen Standard, Qwen Turbo, Krea 2), exact BFS author setup | 185, 196, N3 | PARTIAL | Qwen Standard done (SW3); Krea broken; Qwen Turbo not separately verified; audit agent running |
| SW3 | Qwen swap = BFS author workflow: author prompt, Pruna 8-step LoRA, 8 steps, CFG 1, deis_2m / simple, square box | 179, 184, 185 | DONE+VERIFIED | c68f6b1, 9adfeb7; job 627974f7; after the assistant broke it by saving the Krea prompt into the Qwen editor, the Qwen prompt was restored and the user wrote "now it worked" (13:12) |
| SW4 | Find what broke Swap since it worked | N2, N3 | DONE+VERIFIED | Cause named: the assistant saved the Krea prompt as the Qwen Head swap prompt; restored; user confirmed it worked |
| SW5 | One head picture only; pick ONE person in the scene; no multi-face swap; default pick = person with the head's gender; 1 face + 1 head is automatic | 187, 205, 203 | PARTIAL | a80ad52; browser test only (man head chose Man 1, woman head chose Woman 2); no real job reported |
| SW6 | Scene face scan starts as soon as the scene is added (not after the head) | 227 | PARTIAL | a80ad52; browser: picker 0.4 s after scene; no real job |
| SW7 | Face detection must work and be visible on the scene picture, also with one face | 224, 232, N1, N2 | BROKEN | Hang fixed in a80ad52 (scan 1.55 s, then 0.16 to 0.29 s); but at 13:11 and 13:12 the user said it finds no faces and asked where the scene face detection is; progress page S15 OPEN |
| SW8 | Labels: "Person" first, then Male or Female; scene slot has no label (it is an input image) | 203, 229, N2 | BROKEN | a80ad52 shipped "Woman / Man"; user rejected it at 13:12 |
| SW9 | Person photo arrows (1/2) on the picture; person dropdown does not stretch or push down | 203, 229, 230 | PARTIAL | a80ad52; checked only on a screenshot; user complained at 12:56 and 12:59 before this commit; not confirmed |
| SW10 | Swap result shows (small) which scene picture and which head picture were used: "who swapped what" | 213, 237 | NOT DONE | User repeated it at 13:10; progress page S2 OPEN |
| SW11 | Swap Steps, CFG, sampler, schedule follow the model and swap type; Swap defaults saved apart from the model's general default | 121, 183, 198, 200, 202, 206, 207 | DONE+VERIFIED | d94b5b6, 3f8f8dc; browser: Turbo 8/1, Standard 40/1 (8 with Pruna), Moody 8/1; Swap CFG default saved under Swap only |
| SW12 | Settings on the Swap page must match what the output says | 209 | PARTIAL | 1ff563c (Swap own size), restore brings the size back; no test reported that compares page and output |
| SW13 | Swap speed: the 03:33 PM result was best but very slow; explain why it was good; make it fast | 198, 209, 211 | PARTIAL | 1 MP default and Pruna 8 steps: 0.6 s per step against 9.5 s at 2 MP (measured); the 2 MP cliff is not fixed (progress S8 OPEN); no explanation of the 03:33 PM result given to the user |
| SW14 | "Keep original face" name, 0.1 to 2.0, one-line higher / lower hint that is correct per model | 172, 173, 175, 177 | PARTIAL | 29fe604, b3c3c4d; direction proven only on Qwen by the user's test; Krea direction not tested (progress S12, S16) |
| SW15 | Swap prompt editor saves any text; tags do not touch the internal prompt; base and head sent in the correct order | 181 | DONE+VERIFIED | 3f8f8dc; live job, sent text checked |
| SW16 | Head pose / Color and style only where they change the prompt; face comes exactly from the head picture | 168 | DONE+VERIFIED | 0a6e167, bc15d53; checked on a live job ("expression of person 2 in image1" bug fixed) |
| SW17 | Krea swap must not use Qwen face picks | 186 | DONE+VERIFIED | 9fae7e8; job ran after the fix (the swap result itself is SW1) |
| SW18 | Replace a picture by clicking it; no "Replace" and "Library" links; no Upload button | 188 | DONE+VERIFIED | 9fae7e8; browser |
| SW19 | Swap LoRAs available in Edit / Add LoRA | 135 | DONE+VERIFIED | 4d0b71f, be5901a; browser check |
| SW20 | Restore task on a Swap output opens the correct Swap task | 132 | DONE-UNVERIFIED | Could not reproduce in 4 paths; the question to the user ("which button?") was never answered |
| SW21 | Sent prompts shown in Swap Details too | 72 | DONE-UNVERIFIED | Claimed in the 16:08 report ("Sent to the model in every result"); no Swap-specific check reported |
| SW22 | Swap Qwen Standard slow (2 min per image) | 123, 128, 130 | DONE+VERIFIED | Cause BF16 + CFG 2 + 40 steps; automatic INT8 copies (4d0b71f) measured about 3x faster |
| SW23 | BFS repo: put each other item into its feature | 196, 203 | DONE-UNVERIFIED | Qwen and Krea parts placed; FLUX.2 Klein, Qwen Edit 2511, LTX-2 dropped by the user (item 136) |
| SW24 | Questions answered: face-swap-sx-cu and Gemma 4; Qwen Edit 2511 vs 2.1; "is this the author's prompt?" | 176, 180, 184 | DONE | Answered (the workflow has no Gemma node; 2511 is older; the first line was the Krea prompt) |

## Krea

| ID | Task | Refs | Status | Evidence |
|---|---|---|---|---|
| K1 | Krea 2 identity edit implemented | 2, 48 | DONE+VERIFIED | 5a28ffc; edit-test files 26 and 27 keep identity |
| K2 | Krea identity inputs: Person + optional Scene | 59 | DONE-UNVERIFIED | Built by the UI agent (16:16 report); no test reported |
| K3 | Krea identity output is cropped / a two-panel diptych | 69 | NOT DONE | Investigation started before the first compaction; no later report says it is fixed (only "copies the pose, a LoRA limit") |
| K4 | Krea Auto at 1K = 3:4 or 4:3, never 1:1 | 151, 215 | DONE+VERIFIED | abda7d3; job 80e4752d 880 x 1168; user complained at 12:47 after the first fix (0a6e167); not confirmed by the user since |
| K5 | Krea identity strength slider | 163 | DONE+VERIFIED | 688ed6f; measured 1.0 about 30% faster than 2.0 |
| K6 | Krea photo jobs extremely slow (28.7 s per step, spill) | 147, 149, 153, 155, 157, 161, 163 | DONE+VERIFIED | 688ed6f, 0a6e167; 2K with 4 and 9 GB taken: 3.1 to 3.4 s per step; all 6 families same speed as before |
| K7 | Why only the identity LoRA can do identity; the wrong "cannot use person photos" text | 2, 3 | DONE | Answered and text fixed in the first session |

## Enhance

| ID | Task | Refs | Status | Evidence |
|---|---|---|---|---|
| E1 | Labels / @mentions are lost when Enhance runs | 234 | NOT DONE | Reproduced: rewritten text says "her", "the person", no image reference; progress E1 OPEN |
| E2 | More enhancer text models (not only 4) | 234 | NOT DONE | Progress E2 OPEN |
| E3 | Text model menu beside "Enhance prompt" (Gemma and the others) | 218 | PARTIAL | 2a5a967 built; not checked on screen |
| E4 | Enhance keeps the requested shot ("medium shot", no forced close-up) | 145 | DONE+VERIFIED | 29fe604; 11 of 12 tests |
| E5 | Local official enhancer (Qwen PE), uncensored, Hebrew in, English out, per-model templates | 4, 65, 68, 73 | DONE+VERIFIED | Live jobs (Qwen, edit, Krea, MOSS, Hebrew sign) |
| E6 | Enhance only when checked, off by default; checkbox visible | 67, 70 | DONE-UNVERIFIED | Claimed in the reports; no explicit test reported |
| E7 | Enhancer choice in Settings, default = each model's own enhancer | 81 | DONE+VERIFIED | Screens checked on desktop and phone (20:07 report) |
| E8 | Enhance speed and cache | 98, 102, 111 | DONE+VERIFIED | 23 to 130 s down to 9 to 12 s; repeat 0.01 s; Q6 135 to 158 tok/s |
| E9 | No Google / cloud anywhere, local only | 87, 100, 104 | DONE+VERIFIED | 4e7f320; old route 404; screens checked |
| E10 | Gemma 4 uncensored downloaded; other models can use it | 105, 107 | DONE+VERIFIED | 8a70e4c; Qwen through Gemma job 11.7 s; not offered for Z-Image (it wrote 1 sentence) |
| E11 | MOSS official enhancer (SKILL.md) | 74 | DONE+VERIFIED | MOSS Enhance job 9 s in the app |
| E12 | Settings editor for every internal prompt | 68 | DONE+VERIFIED | 34 prompts in Settings, Prompts tab after restart |

## Layout/UI

| ID | Task | Refs | Status | Evidence |
|---|---|---|---|---|
| L1 | Compact layout on all pages, desktop and mobile: no wasted space, no huge blocks, no extra text, no empty scrolling | 143, 165 | PARTIAL | 1786778, 688ed6f; the user complained again (10:22) after 1786778; Swap panel complaints continued to 12:59; no confirmation |
| L2 | Steps and CFG always visible, sliders, up / down arrows | 121, 165, 171 | DONE+VERIFIED | 4d0b71f, 688ed6f, 0a6e167; browser and phone widths |
| L3 | One-line "Higher / Lower" hints | 172 | DONE-UNVERIFIED | 0a6e167; no screen check reported for the hint text |
| L4 | Stop button in the top banner | 159 | DONE-UNVERIFIED | 688ed6f; not clicked in a test (progress L4) |
| L5 | Recent panel opens at its button; no collision with the queue | 167 | DONE+VERIFIED | Browser at 1440, 1280, 1024 |
| L6 | Settings in 2 columns, no endless scroll | 134 | DONE+VERIFIED | 4d0b71f; 430, 390, 375, 320 px |
| L7 | Internal LoRAs group with checkbox on by default | 141 | DONE+VERIFIED | be5901a; browser 1440 and 390 px |
| L8 | Queue rows compact with thumbnails; "Queue paused" text bug; proactive UX audit | 84, 86 | DONE-UNVERIFIED | Done by agents in the first session; no test listed in the reports |

## Gallery/Viewer

| ID | Task | Refs | Status | Evidence |
|---|---|---|---|---|
| G1 | Live preview develops on the big Result picture, whole picture, not cropped | 192, 203, 217, 220 | DONE+VERIFIED | abda7d3, 2a5a967; browser 4 of 4; after 2 user complaints, not confirmed by the user |
| G2 | Preview only at job start or tile click; stepping shows results | 222 | DONE+VERIFIED | 2a5a967; browser |
| G3 | Result panel ‹ › and arrow keys | 197 | DONE+VERIFIED | 9adfeb7; browser |
| G4 | Run time beside the date and time on gallery cards | 44, 51, 164 | DONE-UNVERIFIED | 688ed6f; not checked on screen (progress L6) |
| G5 | Stage times on click; total and net inference time | 45 | DONE-UNVERIFIED | Built in the first session |
| G6 | Upscale result shows source size | 39 | DONE-UNVERIFIED | Claimed (gallery card 3 lines) |
| G7 | Full prompt as sent in Details | 71 | DONE-UNVERIFIED | Claimed |
| G8 | Full-screen viewer, zoom inside bounds, nothing covering the image | 82 | DONE-UNVERIFIED | Built by an agent, claimed checked |
| G9 | Viewer buttons clean, Details first | 113 | DONE+VERIFIED | 6625973; 320, 390, 1366 px |
| G10 | Image upscale for outputs and inputs, several models, Advanced parameters, progress | 109, 115, 117 | PARTIAL | 6625973; 3B and Nomos2 tested; 7B and 7B Sharp not tested |

## Library/People

| ID | Task | Refs | Status | Evidence |
|---|---|---|---|---|
| P1 | Library always opens on People | 188, 225 | DONE+VERIFIED | a80ad52; browser; user complained at 12:55 after 9fae7e8; not confirmed since |
| P2 | Gender preset on every person in People (scan all People photos once, saved per person), shown as Person + Male / Female | 203, N2 | NOT DONE | Progress L9 OPEN; label format rejected (SW8) |
| P3 | Select and delete uploaded inputs | 126 | DONE+VERIFIED | 4d0b71f; API and UI test on own test picture |
| P4 | Serenity people apart from @mentions, with a filter | 77, 83 | DONE+VERIFIED | GPU test with 2 people |

## Mobile

| ID | Task | Refs | Status | Evidence |
|---|---|---|---|---|
| M1 | Phone download opens the share sheet (Save Image) | 190 | DONE-UNVERIFIED | 9fae7e8; not tested on a real phone |
| M2 | iPhone shows Swap correctly | 120 | DONE+VERIFIED | WebKit at 390 px |

## Speed/VRAM

| ID | Task | Refs | Status | Evidence |
|---|---|---|---|---|
| S1 | 5090 optimization of all models and architectures; LoRA fixes | 1, 7, 11 | DONE+VERIFIED | Measured table (Qwen 0.48 to 0.15 s per step, etc.) |
| S2 | Default steps and CFG per model; Turbo vs Standard classification | 9, 11, 25 | DONE+VERIFIED | Weight measurement; REDQW21 fixed |
| S3 | Steps change automatically with resolution; recommended settings load on model, size, LoRA change; seed follows the last result; "40 steps" group label | 23, 57, 58, 61 | DONE-UNVERIFIED | Built by the UI agent; no test listed |
| S4 | Video upscale stuck, misleading stage labels, fastest upscaler settings, SwiftVR works | 29, 31, 33, 35, 36, 47, 49, 53 | DONE+VERIFIED | SeedVR2 741 to 179 s, FlashVSR 132 s, SwiftVR 31 s, output file checked |
| S5 | Video 3 to 8 minutes | 41, 43 | DONE+VERIFIED | Orbit 117 to 79 s (39 s repeat) |
| S6 | ComfyUI memory research and best use | 159 | PARTIAL | Research done; faster offloading (pinned memory, async transfer, dynamic VRAM) not applied |
| S7 | Job after Stop failed ("Inference tensors") | 163 | DONE+VERIFIED | 2 of 2 |
| S8 | Qwen swap at 2 MP is 9.5 s per step | 211 | NOT DONE | Progress S8 OPEN; 1 MP default is a workaround |
| S9 | low_vram, cable, local prompt model questions | 4, 5, 6 | DONE | Answered; low_vram off in training except Krea |

## Models/Downloads

| ID | Task | Refs | Status | Evidence |
|---|---|---|---|---|
| D1 | All models on D:\models, E: copies deleted, unrelated files deleted | 10, 54, 55, 65 | DONE+VERIFIED | 650 GB freed, test job after deletion. Open note: one locked Stable Yogi E: copy (12.8 GB) was "to delete after restart"; no later report. D: is nearly full, INT8 copies and tool LoRAs went to E: |
| D2 | Hugging Face and ModelScope sources | 2, 3 | DONE+VERIFIED | Pushed c5d5c1c |
| D3 | MOSS Image, Turbo, GGUF, SFT GGUF, Serenity | 13, 75, 77 | DONE+VERIFIED | GPU tests; Serenity v1 INT8 not tested |
| D4 | MOSS image-to-image with identity | 85 | DONE | i2i built and tested; identity not possible with MOSS, user told |
| D5 | Hugging Face alternatives; "not supported" label; 8 new tools | 118 | DONE-UNVERIFIED | 6625973; unit tests; not every new tool tested on the GPU |
| D6 | Q6 enhancer download | 119 | DONE+VERIFIED | sha256, live jobs |
| D7 | Main civitai.com: architectures, LoRAs, face swaps, Turbo models | 170 | PARTIAL | Same catalog as civitai.red; "Qwen 2" tag fixed; new architectures (Anima etc.) not added; the user then said no old models |
| D8 | FLUX 3 output looks bad | 21 | DONE-UNVERIFIED | Greedy decoding, CFG 4; GPU P3 power state needs a user setting |
| D9 | Edit test on all models with the user's photo, results folder | 15, 17, 20 | DONE+VERIFIED | E:\AI\qwen-image-studio\validation\edit-test-2026-10-21 |
| D10 | Lightricks 2.3 question | 63 | DONE-UNVERIFIED | No explicit answer found in the kept reports |

## Settings

| ID | Task | Refs | Status | Evidence |
|---|---|---|---|---|
| ST1 | Settings clearer everywhere; Training captions page rebuilt | 87 | DONE-UNVERIFIED | Agent rebuilt it; no screen test listed |
| ST2 | CFG exact per model; LoRA parameters, multiple LoRAs; 2K steps questions | 25, 27, 28 | DONE | Answered and fixed in the first session |

## Other

| ID | Task | Refs | Status | Evidence |
|---|---|---|---|---|
| O1 | Delete the test images of children; never use children in tests | 137 | DONE+VERIFIED | Jobs 960c4ba5, 7e5cde2a and the log folder removed; memory rule saved |
| O2 | Log viewer red only for real errors | 50 | DONE+VERIFIED | Checked on the user's log lines |
| O3 | MOSS training pipeline; captioning per model; training research | 78, 80, 87 | DONE-UNVERIFIED | Test training ran; full training from the Train page in the UI not tested |
| O4 | Training sample 404 log | 88 | DONE-UNVERIFIED | Deleted trainings hidden; no check reported |
| O5 | Full plan with agents, full task map and progress page; do not stop until all is fixed | 230, 237 | PARTIAL | docs/progress-2026-10-22.md draft exists (uncommitted); this map is the second input |
| O6 | Push the batch | 142 | NOT DONE | 15 local commits after be5901a; the user has not said "push" since 09:42 UTC |
| O7 | Hebrew text inside pictures | 139 | NOT DONE | User set it to the lowest priority |
| O8 | Google API-key call count | 93, 96 | CANCELLED | User: "not for here" |
| O9 | "js" | 97 | N/A | Unclear, the user did not answer |
| O10 | User actions in NVIDIA settings (Sysmem Fallback "Prefer No Sysmem Fallback", Power "Prefer maximum performance" for the venv python) | none | UNKNOWN | Asked of the user; no confirmation in the transcript |
| O11 | lucide icons still on about 30 screens (global CLAUDE.md rule, not a request in this session) | none | NOT DONE | Stated in the 16:08 report |

# Part 1: Chronological list of every user request

Source: transcript ef77dbda (the only project transcript changed today; the 2 other files from 2026-10-21 hold no user text). Timestamps are UTC (local time is UTC+3). Duplicate deliveries of the same mid-turn message are merged. Task notifications, compaction summaries and system text are left out.

1. [2026-10-21 09:25] (msg 1) Area: Speed/VRAM
   - HE: תעשה את כל הבדיקות האפשריות בכל מקום אפשרי בהקשר כל המודלים וכל המקומות הקשורים גיטהאב האגינגפייס לא משנה מה-  ...
   - EN: Run every possible check (GitHub, Hugging Face, anywhere) on all models and architectures: are the settings optimal for the RTX 5090
   - EN: Check each model and each architecture separately
   - EN: Find why LoRAs do not work properly
   - EN: Use 8 research agents
2. [2026-10-21 09:28] (msg 2) Area: Krea, Models/Downloads
   - HE: וגם שים לב כרגע האפליקציה מתארת את קריאה 2 כמודל שלא יודע להשתמש בתמונות דמויות וזה לא נכון אז תוודא שאתה בודק ...
   - EN: Fix the app text that says Krea 2 cannot use person photos (it is wrong)
   - EN: Check conradlocke/krea2-identity-edit
   - EN: Check whether the installed Krea models can do identity too
   - EN: Add model sources other than Civitai
3. [2026-10-21 09:32] (msg 3) Area: Models/Downloads
   - HE: לא צריך שתבחן את זה לגבי האיידנטיטי רק תבדוק למה זה יכול והשאר של קריאה לא יכולים, וגם תביא עוד מקומות להביא מ ...
   - EN: No need to test identity; explain why that LoRA can do it and the other Krea models cannot
   - EN: Add more model sources that match the architectures, Hugging Face too
4. [2026-10-21 09:34] (msg 4) Area: Enhance
   - HE: מה זה לוקאל פרומפט מודל? יש תהליך שזה כותב פ]רומפט עם מודל פנימי לוקאלי?
   - EN: Explain 'local prompt model': is there a process that writes prompts with a local model?
5. [2026-10-21 09:35] (msg 5) Area: Speed/VRAM
   - HE: האם כבל יותר מהיר יכול לשפר את זה או החיבור עצמו לא מאפשר יותר מפאקינג 0.6 גיגה??
   - EN: Can a faster cable make the USB drive faster than 0.6 GB/s?
6. [2026-10-21 09:36] (msg 6) Area: Speed/VRAM (training)
   - HE: low_vram is on למה???? בשביל מה ??
   - EN: Why is low_vram on? What is it for?
7. [2026-10-21 09:37] (msg 7) Area: Speed/VRAM
   - HE: בקיצור תעשה את כל כל האופטימיזציות להכל כל פיפס כל נקודה כדי שכל מודל יעבוד בצורה הכי מושלמת לא משנה באיזה מתא ...
   - EN: Do all optimizations everywhere, so every model works perfectly on any architecture I choose
8. [2026-10-21 09:38] (msg 9) Area: Settings
   - HE: ותבדוק האם הסטפס שמוגרדרים לכל מודל כדיפולט הם באמת נונים ותסדר את זה
   - EN: Check that the default steps of each model are right, and fix them
9. [2026-10-21 09:44] (msg 10) Area: Models/Downloads
   - HE: פיניתי מלא מלא מקום ב D תעביר הכל לשם לתיקיית מודלס בצורה מושלמת לפי התיקיות ותוודא שהאפליקציה יודעת לשלךף מהמ ...
   - EN: Move all models to D:\models in the correct folders
   - EN: Make the app load models from the correct places
   - EN: Make the models already on D: work in the app
10. [2026-10-21 09:56] (msg 11) Area: Models/Downloads
   - HE: והאם כל ההתממשקות של קודה וטרנספורמרס או וואטאבר לכל ארכיטקטורה מתבצעת בצורה מושלמת?? והאם שמת את מה שטורבו בט ...
   - EN: Is the CUDA / transformers integration correct for each architecture?
   - EN: Are Turbo models classified as Turbo and the others as Standard?
   - EN: Why are there no LoRAs for Turbo models?
11. [2026-10-21 11:24] (msg 13) Area: Models/Downloads
   - HE: https://civitai.red/models/3005406/moss-image תוסיף ותחקור לגבי הארכיטקטורה הזאת שיקבל אותה בצורה מושלמת ותוסי ...
   - EN: Add MOSS Image (civitai 3005406) fully: research the architecture, add Turbo, GGUF and all variants
12. [2026-10-21 12:01] (msg 15) Area: Other (testing)
   - HE: תבדוק בכולם אדיט עם בנאדם שמחזיק שלט שכתוב עליו את שם המודל, לבדוק שהמודלים יודעים להכניס בנאדם לתוך התמונה בד ...
   - EN: Edit test on every model: a person holding a sign with the model name; check each model puts the person in like Qwen 2.1 does
13. [2026-10-21 12:02] (msg 17) Area: Other (testing)
   - HE: תנסה עם תמונה שלי ido
   - EN: Use the user's own photo 'ido' for the edit test
14. [2026-10-21 12:02] (msg 19) Area: Other
   - HE: כמובן אבל אל תדלג על המשימות האחרות
   - EN: Do the edit test but do not skip the other tasks
15. [2026-10-21 12:04] (msg 20) Area: Other (testing)
   - HE: כמבחן האדיט נגמר תפתח לי תיקיה עם כל מה שיצא ששם הקובץ הוא המודל המדויק והתנאים
   - EN: When the edit test ends, open a folder with all outputs, each file named by exact model and settings
16. [2026-10-21 12:18] (msg 21) Area: Models/Downloads
   - HE: [Image #1] למה פלוקס 3 יוצא כזה פח?
   - EN: Why does FLUX 3 output look so bad?
17. [2026-10-21 12:26] (msg 23) Area: Settings
   - HE: תוודרזולוציה שאני קובע אם צריך שינוי בסטפס בשביל זה בכל מודל ואם לא לאז לא אבל אם כן אז שישתנה אוטומטית להמלצה
   - EN: For each model: if the resolution I set needs other steps, change the steps automatically to the recommendation
18. [2026-10-21 12:26] (msg 25) Area: Settings
   - HE: וגם ווידאת בכולם שה cfg מדויק למודל? והאם סידרת את הלורות והמודלים ולורה מקבלת פרמטרים נכונים אם מפעילים? כולל ...
   - EN: Check that CFG is exact for each model
   - EN: Check that LoRAs get correct parameters when enabled, including several LoRAs
   - EN: Do not change things only to show work
19. [2026-10-21 12:27] (msg 27) Area: Settings
   - HE: וגם מי ביקש לעשות 2 קיי? והאם אתה בטוח שצריך את כל הסטפס האלו בשביל 2 קיי? אם כן אז תגיד כן אל תמציא לי תשובה
   - EN: Who asked for 2K? Are all those steps needed for 2K? Do not invent an answer
20. [2026-10-21 12:31] (msg 28) Area: Settings
   - HE: [Image #2] 1 תסביר לי את הפער בין רזולוצית האאוט]פוט לבין שכתוב למטה 2048 2- למה עשית 40 דטפס בעצם? זה מה שצרי ...
   - EN: Explain the gap between the output resolution and the 2048 written below
   - EN: Why 40 steps: is that what this model needs?
21. [2026-10-21 12:32] (msg 29) Area: Video (upscale), Speed/VRAM
   - HE: [Image #3] ולמה הגיפיו לא עובד?
   - EN: Why is the GPU not working (upscale screenshot)?
22. [2026-10-21 12:34] (msg 31) Area: Speed/VRAM
   - HE: לא הבנתי מה עוזר שלוקח 38298392 שעות לטעון מודל ? מה הולך פה לא יכול להיות שזה תקין
   - EN: Model load takes forever; that cannot be correct
23. [2026-10-21 12:35] (msg 33) Area: Video (upscale)
   - HE: וגם אם זה לא אפסקיילינג כרגע למה כתוב שהשלב הוא אפסקיילינג?
   - EN: Why does the stage say Upscaling when it is not upscaling?
24. [2026-10-21 12:36] (msg 35) Area: Video (upscale)
   - HE: זה כבר 6 דקות זה לא הולך לזוז
   - EN: 6 minutes and it does not move
25. [2026-10-21 12:39] (msg 36) Area: Video (upscale)
   - HE: [Image #4] שוב לא זז זה מראה שזה משתמש ב100 אחוז גיפיו אבל כלום לא קורה ולא ממשיך מבאצ' 1 למה לעזאזל זה לא עוב ...
   - EN: Stuck at batch 1 with 100% GPU; the logs are misleading; make it clear what happens
26. [2026-10-21 12:45] (msg 39) Area: Gallery/Viewer
   - HE: שבתוצאה יהיה כתוב גם על הסורס איזה גודל הוא יאן לי מושג בכלל בכמה זה גדל
   - EN: Upscale result must show the source size too
27. [2026-10-21 12:45] (msg 41) Area: Video, Speed/VRAM
   - HE: זה מטורף שלוקח 3 דקות לסרטון של 5 שניות על 5090
   - EN: 3 minutes for a 5 s video on a 5090 is too slow
28. [2026-10-21 13:14] (msg 43) Area: Video, Speed/VRAM
   - HE: 8 דק חתיכת חולה נפש
   - EN: 8 minutes (same complaint)
29. [2026-10-21 13:23] (msg 44) Area: Gallery/Viewer
   - HE: [Image #6] שיהיה בקטן לצד שם המודל את זמן האינפרנס שלו בכל אחד בקטן בדיוק בגודל של שם המודל, לצידו
   - EN: Show the inference time beside the model name on every item, at the same small size
30. [2026-10-21 13:25] (msg 45) Area: Gallery/Viewer
   - HE: [Image #7] ושפה יהיה כתוב גם זמן לכל דבר בלחיצה, ובתצוגה הראשית זמן כולל וזמן אינפרנס נטו
   - EN: Time of each stage on click; in the main view total time and net inference time
31. [2026-10-21 13:25] (msg 47) Area: Video (upscale)
   - HE: ותוודא שהאפסקיילרים מקבלים את האופטימיזציה שכל אחד מהם יהיה הכי מהיר מכל הטסטים שעשית
   - EN: Every upscaler must use the fastest settings found in all tests
32. [2026-10-21 13:34] (msg 48) Area: Krea
   - HE: Krea 2 identity editing: possible only with krea2-identity-edit and its own node pack. Not implemented, as you ...
   - EN: Implement Krea 2 identity edit (the user never asked to skip it)
33. [2026-10-21 13:34] (msg 49) Area: Video (upscale)
   - HE: SwiftVR: not tested (20 GB download), so it keeps its original settings. לא הוספת את זה?
   - EN: Add and test SwiftVR
34. [2026-10-21 13:42] (msg 50) Area: Other (logs)
   - HE: [pasted text] מה קורה זה לא זז ולמה [Image #9] דברים שהם לא שגיאה מופיעים באדום בלוגים?
   - EN: Why does the job not move (pasted log)?
   - EN: Why are normal lines red in the logs?
35. [2026-10-21 13:45] (msg 51) Area: Gallery/Viewer
   - HE: [Image #10] ואני רוצה גם תאריך ושעה על כל אחד
   - EN: Date and time on every gallery card
36. [2026-10-21 13:46] (msg 53) Area: Video (upscale)
   - HE: וידאתץ שהסוויפט אכן מייצא וידאו אפסקיילד ?
   - EN: Verify SwiftVR really exports an upscaled video
37. [2026-10-21 13:47] (msg 54) Area: Models/Downloads
   - HE: כן תמחק תשאיר רק בדי ותמחק מדי את כל מה שלא קשור לתוכנה שלנו (שאריות של קומפי וכדומה
   - EN: Delete the E: copies, keep models only on D:
   - EN: Delete from D: everything not related to the app (ComfyUI leftovers etc.)
38. [2026-10-21 13:55] (msg 55) Area: Models/Downloads
   - HE: איםה הליסט? אני לא יודע להגיד בדיוק פשוט תסתכל על קבצים שלא רלוונטים לשום דבר באפליקציה שלנו
   - EN: No list needed: find yourself the files not related to the app
39. [2026-10-21 13:57] (msg 57) Area: Settings
   - HE: וכשאתה מסיים תרסטרט את האפליקציה ושלום על ישראל תוודא שתמיד ההגדרות המומלצות טעונות בכל דבר שאני עושה כולל מוד ...
   - EN: Restart the app when done
   - EN: Always load the recommended settings: on model change, size change and LoRA add
40. [2026-10-21 14:00] (msg 58) Area: Settings
   - HE: [Image #11] ואגב למה כתוב פה 40 סטפס אם יש כאלה שהם 20 וכו?
   - EN: Why does the group label say 40 steps when some models use 20?
41. [2026-10-21 14:01] (msg 59) Area: Krea
   - HE: [Image #12] טזה מה זה אומר לעזאזל חייב 2 תמונות??? ואם לא חייב, למה לעזאזל זה לא עושה פשוט אינפוט של פרסון ואי ...
   - EN: Krea identity: are 2 pictures required? Use a Person input and an optional Scene input
42. [2026-10-21 14:04] (msg 61) Area: Settings
   - HE: וגם תוודא שהסיד כל פעם מתעדכן לסיד של התמונה האחרונה שנוצרה (כך שאם מסמן החוצה את ראנדום זה פועל על הסיד האחרו ...
   - EN: The seed always updates to the seed of the last generated image
43. [2026-10-21 14:06] (msg 63) Area: Models/Downloads
   - HE: רגע אבל יש גם לייטריקס 2 3 שכן האפליקציה משתמשת נכון? מה שראית הם לא שלנו כן?
   - EN: The app uses Lightricks (LTX) 2.3 files, right? The files you saw are not ours?
44. [2026-10-21 14:14] (msg 65) Area: Enhance, Models/Downloads
   - HE: קיצר בלי לשאול תמחק את כל אבל עכל המודלים שלא קשורים לאפליקציה ותבדוק מה המודל המקומי הכי טוב שירוץ הכי קל לעב ...
   - EN: Without asking, delete all models not related to the app
   - EN: Find the best local enhancer model (research, what Qwen uses) and make it work perfectly, uncensored too
45. [2026-10-21 14:14] (msg 67) Area: Enhance
   - HE: וקדימה תעשה הכל ותדחף וכמובן לאנהאנס פרומפט רק אם לוחץ צ'ק על זזה דיפולט שלא
   - EN: Do everything and push
   - EN: Enhance prompt runs only when its checkbox is on; default off
46. [2026-10-21 14:45] (msg 68) Area: Enhance, Settings
   - HE: תוודא שזה משתמש באנהאנס וההגדרות שלו מהקוון בצורה נכונה ולא שאתה ממציא חדש- וגם שיהיה לי אופציה בהגדרות לתקן ו ...
   - EN: Use Qwen's own enhancer and its settings correctly, do not invent new ones
   - EN: Settings option to view and edit every internal prompt
47. [2026-10-21 15:06] (msg 69) Area: Krea
   - HE: [krea2edit] reference placement: last ref='beside (right of target)', first ref='aligned (trained)' האדיט של ק ...
   - EN: Krea edit keeps giving cropped images of the person; check the internal prompt
48. [2026-10-21 15:08] (msg 70) Area: Enhance
   - HE: ואיפה לעזאזל הצ'קמ[Image #13] ארק של אנהאנס פרומפט?!
   - EN: Where is the Enhance prompt checkbox?
49. [2026-10-21 15:08] (msg 71) Area: Gallery/Viewer
   - HE: ואני רוצה שבתצוגות של תמנונות שיש בהם פרומפט פנימי שנשלח יש בדיטיילס את הפרומפט המלא שנשלח כמו שהוא כדי שאוכל  ...
   - EN: Details of every image that used an internal prompt shows the full prompt as sent
50. [2026-10-21 15:09] (msg 72) Area: Swap
   - HE: וגם בסוואפ ובכל מקום אחר אפשרי שיש מתחת עוד גאידנס פרומפט שעובר!!!
   - EN: The same in Swap and everywhere a guidance prompt is added
51. [2026-10-21 15:10] (msg 73) Area: Enhance
   - HE: אגב האנהאנס פרומפט גם מקבל בעברית ומוצא פרומפט באנגלית נכון? וזה עושה אופטימיזציה לכל מודל ואך פרומפטם אצלו עו ...
   - EN: Does Enhance take Hebrew and give English? Does it optimize the prompt for each model?
52. [2026-10-21 15:12] (msg 74) Area: Enhance
   - HE: https://github.com/OpenMOSS/MOSS-Image/blob/main/skills/moss-image-prompt-enhancer/SKILL.md תבדוק גם לגבי זה
   - EN: Check the MOSS official prompt-enhancer SKILL.md
53. [2026-10-21 15:14] (msg 75) Area: Models/Downloads
   - HE: https://huggingface.co/axiomofmind/Moss-Image-SFT-GGUF
   - EN: Check axiomofmind/Moss-Image-SFT-GGUF
54. [2026-10-21 15:18] (msg 77) Area: Models/Downloads, Library/People
   - HE: https://huggingface.co/malcolmrey/moss-serenity וגם זה תבדוק ותכניס את כל הרלוונטים בעיקר טורבו פנימה ואני רוצ ...
   - EN: Add malcolmrey/moss-serenity: everything relevant, mainly Turbo
   - EN: A small helper in tagging to insert the collection's people when on that model
55. [2026-10-21 15:22] (msg 78) Area: Other (training)
   - HE: תעשה פייפליין מלא של אימון לורה דמות לmoss בדיוק כמו האחרים . אני מוודא שוב האם עשית מערך שעושה קאפשנינג מושלם ...
   - EN: Full MOSS character LoRA training pipeline like the others
   - EN: Is captioning correct for each model's dataset preparation?
56. [2026-10-21 15:22] (msg 80) Area: Other (training)
   - HE: תסתכל בדברים כמו ai-toolkit וכד ודברים שקשורים באימון לורות לקוון, לקריאה 2 ולmoss ותוודא שיש הבנה מלאה ואדפטצ ...
   - EN: Research ai-toolkit and similar for Qwen, Krea 2 and MOSS LoRA training; best exact parameters per architecture
57. [2026-10-21 15:25] (msg 81) Area: Settings, Enhance
   - HE: ולמה אין לי בהגדרות שום בחירה של מודלים לאנהאנס פרומפט? וכמוןבן דיפולט לכל אחד מהם שיש לו אנהאנס מובנה להשתמש  ...
   - EN: Settings: a choice of enhancer model
   - EN: Default for each model with a built-in enhancer = its own enhancer
58. [2026-10-21 15:27] (msg 82) Area: Gallery/Viewer
   - HE: [Image #14] [Image #15] ולמה לעזאזל הפרוויו בכל אחד מהם שונה לחלוטין ונראה כזה חרא בגלירה, בשניהם אני מבקש שזה ...
   - EN: Gallery and result previews: one design, full screen, not a small modal
   - EN: Zoom stays inside the bounds, no cropping
   - EN: Nothing covers the image (fit / zoom bar)
59. [2026-10-21 15:31] (msg 83) Area: Library/People, Models/Downloads
   - HE: [Image #16] חתיכת אידיוט מפגר צתפריד בין מנשן של התמונות לבין כשזה על מוס אז מוס סריניטי פיפל1!!!!!!!! ככה אני ...
   - EN: Separate @image mentions from the MOSS Serenity people, with a filter
   - EN: Serenity Turbo must not be listed under Base
60. [2026-10-21 15:34] (msg 84) Area: Layout/UI
   - HE: [Image #17] באיזה פאקינג אולם לא רואים את התמונה לצד כל אחד כזה ולמה כל אחד מקבל לבוק כזה עצום??? שהתמונה שם ת ...
   - EN: Queue: a thumbnail beside each item, compact rows, open the image from there
   - EN: Find other obvious UX problems yourself
61. [2026-10-21 15:35] (msg 85) Area: Models/Downloads
   - HE: [pasted text] תבדוק מה אפשר לעשות עם המידע הזה כולל מחקר כדי שיעבוד אימאג' טו אימאג' עם איידנטיטי באפליקציה של ...
   - EN: Research how to make MOSS image-to-image with identity work in the app
62. [2026-10-21 15:36] (msg 86) Area: Layout/UI
   - HE: [Image #19] ומה זה הבאג הזה למה הטקסט קבוע שם
   - EN: Bug: the 'Queue paused' text stays fixed there
63. [2026-10-21 15:37] (msg 87) Area: Settings, Other (training)
   - HE: [Image #20] תוודא שהסטינגס בכל פאקינג מקום בסטינגס מעודכנים ופי אלף יותר ברורים למה זה עוד עם גוגל!?!!?!? ולמה ...
   - EN: Settings everywhere up to date and much clearer
   - EN: Why is Google still there?
   - EN: The Training captions settings page is not understandable
   - EN: Full research on captioning and LoRA training from image datasets; fix everything
64. [2026-10-21 15:55] (msg 88) Area: Other (training)
   - HE: [pasted text]
   - EN: (Pasted log) training sample images return 404
65. [2026-10-21 15:55] (msg 90) Area: Other
   - HE: אל תעצור את המשימות רק תבין מהזה אולי האייג'נט כבר מודע לזה
   - EN: Do not stop the tasks; the agent may already know about this log
66. [2026-10-21 16:07] (msg 92) Area: Speed/VRAM
   - HE: שים לבס כרגע זה משתמש ב80 אחוז גיפיו אם זה בסדר תמשיך אבל שיהיו עירניים לניצול גיפיו
   - EN: GPU at 80%: continue if OK, but watch GPU use
67. [2026-10-21 16:30] (msg 93) Area: Other
   - HE: תבדוק מה לעזאזל היו כמות השימושים בקריאות לגוגל עם איפיאיי ולא דרך האואוות או אנטיגרווויטי בחודש האחרון
   - EN: How many Google calls with an API key (not OAuth / Antigravity) in the last month?
68. [2026-10-21 16:30] (msg 96) Area: Other
   - HE: סליחה זה לא לפה
   - EN: Cancel: the previous message was for another place
69. [2026-10-21 16:51] (msg 97) Area: Other
   - HE: js
   - EN: 'js' (unclear, no follow-up)
70. [2026-10-21 16:51] (msg 98) Area: Enhance
   - HE: מה זה החרא הזה למה לאנהאנס פרומפט לוקח ככ הרבה זמן האם זה משתמש נכון במודל עם הכרטיס מסך??? ולמה לעזאזל שאני כ ...
   - EN: Why does Enhance take so long? Does it use the GPU correctly?
   - EN: Cache the enhanced prompt when the same prompt repeats
71. [2026-10-21 19:00] (msg 99) Area: Enhance, Video
   - HE: תעשה פוש וטפל גם בפאת׳ של הוידאו
   - EN: Push
   - EN: Handle the video path too (cache)
72. [2026-10-21 19:34] (msg 100) Area: Enhance
   - HE: לא!!!!!! תעיף את הטסטים עם גוגל אני רוצה רק עם מודל מקומי ושיהיה מהיר וטוב
   - EN: Remove the tests with Google; only a local model, fast and good
73. [2026-10-21 19:35] (msg 102) Area: Enhance
   - HE: והמודלים של שאר המודלים אין כאלה יותר מהירים מהסיוט הזה שקורה עכשיו זה לא מסתדר לי כמו. הזמן שצריך לחכות לפאקי ...
   - EN: Are there faster enhancer models for the other model types? The wait is too long on a 5090
74. [2026-10-21 19:59] (msg 104) Area: Enhance, Settings
   - HE: אין קלאוד תעיף תקלאוד מהאפליקציה זהו שחרר מזה אמרתי לך אלף פעם רק מקומי
   - EN: Remove the cloud from the app completely; local only
75. [2026-10-21 20:24] (msg 105) Area: Models/Downloads
   - HE: תעשה דאונלואד לגמה ופוש נו
   - EN: Download Gemma and push
76. [2026-10-21 20:26] (msg 107) Area: Enhance
   - HE: תוודא שאם אני אוצה גם מודלים אחרים יוכלו לעבוד עם ג׳מה ותמצא גם ג׳מה אנסנסורד
   - EN: Other models must be able to use Gemma if I choose it
   - EN: Find an uncensored Gemma
77. [2026-10-21 20:30] (msg 109) Area: Gallery/Viewer (upscale)
   - HE: למה לעזאזל אין שום מקום של אפסקייל לתמונה? תחפש את ההכי טוב שעובד לתמונות ושיהיה פינת אפסקייל גם לאאוטפוטים ול ...
   - EN: An image upscale place for outputs and inputs, with the best image upscaler
78. [2026-10-21 20:33] (msg 111) Area: Enhance
   - HE: שים לב שעדיין לוקח 20 שניות לפאקינג שיט הזה לכתוב פרומפט בפרומפט תמונה מסכן למה זה מה זה פאקינג כותב זה לא הגי ...
   - EN: Enhance still takes 20 s for a simple image prompt: too slow
79. [2026-10-21 21:13] (msg 113) Area: Gallery/Viewer
   - HE: לא לא זה לא נראה טובה יא מזדיין תוודא שהכפתורים והכל מסודרים פי אלף טוב כמו כן הדיטליס הכפתור דיטיילס מוחבא תצ ...
   - EN: Viewer buttons look bad: arrange them much better
   - EN: The Details button is hidden
80. [2026-10-21 21:16] (msg 115) Area: Gallery/Viewer (upscale)
   - HE: לא הבנתי יש רק מודל אחד של אפסקייל באפסקיילים? ואין שום פרמטרים לאפסקייל כדי לעשות פיינטיון אם ארצה?
   - EN: Only one upscale model? Add parameters to fine-tune the upscale
81. [2026-10-21 21:16] (msg 117) Area: Gallery/Viewer (upscale)
   - HE: וגם למה לא רואים התפתחות של האפסקייל
   - EN: Show the upscale progress
82. [2026-10-21 21:19] (msg 118) Area: Models/Downloads
   - HE: תחפש פה לכל הדברים שיש להם פה אלטרנטיבןת נניח הסוואפ תכניס אם יש פה עוד סוואפים ראיתי שם אחד כמו כן האדיט של ק ...
   - EN: Search Hugging Face for alternatives to every app feature (more swaps, Krea identity)
   - EN: Fix the 'not supported' label for things the app has
   - EN: Look at everything, upscalers too, and add all that can be added
83. [2026-10-21 22:18] (msg 119) Area: Models/Downloads
   - HE: פוש ודאונלואד
   - EN: Push and download (Q6 enhancer files)
84. [2026-10-21 22:19] (msg 120) Area: Mobile
   - HE: תוודא שהאייפון מציג כמו שצריך
   - EN: Make sure the iPhone shows Swap correctly
85. [2026-10-21 22:21] (msg 121) Area: Settings, Swap
   - HE: תעשה את זה סליידרים בסטפס וסי אפ גי ותוודא שבסוואפ הסי אף גי 2 למעלה מהדיפולט של מודל
   - EN: Make Steps and CFG sliders
   - EN: In Swap, CFG 2 above the model default
86. [2026-10-21 22:23] (msg 123) Area: Speed/VRAM
   - HE: למה לעזאזל הריצה הנוכחית ככ איטית עם ה40 סטפס???? מה קורה שם
   - EN: Why is the current 40-step run so slow?
87. [2026-10-21 22:24] (msg 124) Area: Swap, Layout/UI
   - HE: ולמה זה תופס כככ הרבה מקום לעזאזל שזה יהיה קומפקטי ונוח למה צריך קוביה ענקית לכל אחד במקום בקטן אחד ליד השני
   - EN: Swap slots: compact, small, side by side
88. [2026-10-21 22:25] (msg 126) Area: Library/People
   - HE: ותוודא שבלייבררי אפשר לסמן ולמחוק אינפוטים שעלו כרגע אין שום כפלור כזה
   - EN: Library: select and delete uploaded inputs
89. [2026-10-21 22:26] (msg 128) Area: Speed/VRAM, Swap
   - HE: סתכל על האינפרנס עכשיו של הסוואפ למה לעזאזל הקוון סטנדרט נהיה כזה פאקינג איטי
   - EN: Why did Qwen Standard become so slow in Swap?
90. [2026-10-21 22:28] (msg 130) Area: Speed/VRAM
   - HE: מה ? למה שמודל הסטנדרט יקח כלכך הרבה זמן ככה זה היה תמיד????? נשמע לי הזוי2 דקות לתמונה??????? מה?
   - EN: Was Standard always this slow? 2 minutes per image?
91. [2026-10-21 22:34] (msg 132) Area: Swap
   - HE: כשאני עושה שחזור משימה על אאוטפוט של סוואפ זה לא מוביל אותי למשימה הנכונה בעמוד זוואפ
   - EN: Restore task on a Swap output does not open the correct task on the Swap page
92. [2026-10-21 22:34] (msg 134) Area: Layout/UI
   - HE: לא אבל עכשיו שוב עשית את הפרמטרים אחד אחרי השני וממלה את כל הדף בסקרול אינסופי אלוהים מה אתה לא יודע לעשות כלו ...
   - EN: Settings stacked one per row make an endless scroll; fix it
93. [2026-10-21 22:35] (msg 135) Area: Swap
   - HE: למה הלורות של הסוואפ לא זמינות בכל מקום גם באדיט נניח?
   - EN: Why are the Swap LoRAs not available everywhere, for example in Edit?
94. [2026-10-21 22:57] (msg 137) Area: Other
   - HE: תגיד אתה דפוק מה אתה מגנרט לי ילדים יא נרקומן פסיכופט תמחק את התמונות של הילדים
   - EN: Delete the generated test images of children
95. [2026-10-21 22:59] (msg 139) Area: Other
   - HE: הכל טוב אני אומר לך העברית כטקסט שנכתב זה הדבר האחרון כרגע תוודא לגבי השאר קודם וקדימה סיים
   - EN: Hebrew text inside pictures is the lowest priority; verify the rest first and finish
96. [2026-10-22 06:42] (msg 141) Area: Layout/UI
   - HE: תפריד בין כל הלורות בפנימיות ללורות הרגילות במסך האדיט שיהיה צקמארק דיפולט על אנצק של אינטרנל לורז״
   - EN: Edit screen: separate internal LoRAs from normal LoRAs, with an 'Internal LoRAs' checkbox on by default
97. [2026-10-22 09:42] (msg 142) Area: Other
   - HE: פוש
   - EN: Push
98. [2026-10-22 09:44] (msg 143) Area: Layout/UI, Mobile
   - HE: תס[Image #21] דר שם את כל הלייאאוט תפסיק עם הזבוז מקום ובלוקים ענקיים וטקסט מיותר בכל מקום ושיהיה פי אלף יותר  ...
   - EN: Fix the whole layout: no wasted space, huge blocks or extra text, comfortable on desktop and mobile, including settings panels, input pictures and all choices at the right size
99. [2026-10-22 09:49] (msg 145) Area: Enhance
   - HE: כמו כן משום מה האנהאנס של קריאה (ואלי גם של אחרים9 כל הזמן כותב קלוז אפ בפרומפטים אפילו אם אני כותב מדיום שוט
   - EN: Krea Enhance (maybe others) always writes close-up even when I write medium shot
100. [2026-10-22 09:57] (msg 147) Area: Speed/VRAM
   - HE: טוב שמע יש בעיה קשה לא יודע מה עשית אבל האינפרנסים איטיים לחלוטין קריאה כרגע עובד על מהירות הכי איטית בעולם הא ...
   - EN: Inference is extremely slow (Krea); check the GPU and that the optimizations are not broken
101. [2026-10-22 09:58] (msg 149) Area: Speed/VRAM
   - HE: המודי היה על 1k וזה על 2 k
   - EN: Context: the Moody job was at 1K, this one at 2K
102. [2026-10-22 10:02] (msg 151) Area: Krea
   - HE: שים לב בכללי בקריאה ה 1k יהיה קרופ לרוב אז צריך שיהיה האוטומטי לרוב על 3:4 או 4:3 אם זהה על 1k
   - EN: Krea at 1K crops; Auto at 1K should usually be 3:4 or 4:3
103. [2026-10-22 10:05] (msg 153) Area: Speed/VRAM
   - HE: הנה זה איטי לגמרי עכשיו
   - EN: It is completely slow now
104. [2026-10-22 10:05] (msg 155) Area: Speed/VRAM
   - HE: למה החלוקה בין השלבים לא מתבצעת בצורה מושלמת?????
   - EN: Why is the split between the stages (memory) not perfect?
105. [2026-10-22 10:06] (msg 157) Area: Speed/VRAM
   - HE: למה ש 4 גיגה מתוך 32 יאטו בצורה כזאת קיצונית
   - EN: Why does 4 GB of 32 GB (used elsewhere) slow it this much?
106. [2026-10-22 10:08] (msg 159) Area: Layout/UI, Speed/VRAM
   - HE: כמו כן למה אין כפתור לעצירה של האינפרנס הנוכחי וזה רק אם אני פאקינג פותח את הקיו? תוודא את זה רק אחרי שווידאת  ...
   - EN: A Stop button for the current job without opening the queue
   - EN: First verify every model is perfect and fast and does not hurt the PC
   - EN: Full research on what ComfyUI does and how to use it best
107. [2026-10-22 10:09] (msg 161) Area: Speed/VRAM
   - HE: [pasted text]
   - EN: (Pasted status) SNOFS Krea 1024 at 43.6 s, step 1 of 8
108. [2026-10-22 10:11] (msg 163) Area: Speed/VRAM, Krea
   - HE: עדיין איטי 10 שניות ל1 קיי זה גם איטי מבחינתי ולא היה ככה אתמול משהו שבור במה שעשית. כמו כן למה אין שום סליידר ...
   - EN: Still slow: 10 s for 1K is slow, not like yesterday, something broke
   - EN: Add a slider for Krea identity strength
109. [2026-10-22 10:15] (msg 164) Area: Gallery/Viewer
   - HE: כמו כן בתצוגה [Image #23] שים לב זמן האינפרנס תקקוע ומוחבא אחרי השם של המודל , שיהיה לצד התאריך והשעה
   - EN: Gallery: the inference time is hidden after the model name; put it beside the date and time
110. [2026-10-22 10:22] (msg 165) Area: Layout/UI
   - HE: [Image #24] שוב אני אומר לך אתה כנראה משחק אותה לא מבין אני אומר לך שזה מסורבל לחלוטין יש יותר מדי גאפ יותר מד ...
   - EN: Layout still cluttered: too much gap, too many space-taking things, no control
   - EN: CFG and Steps always available
   - EN: No scrolling over empty space; fix on all pages
111. [2026-10-22 10:23] (msg 167) Area: Layout/UI
   - HE: וגם פתיחת הגלריה בצד שבורה לגמרי בדסקטופ זה נפתח במקום אחר לגמרי מהחץ, הqueue והגלריה מתנגשים אחד בשני וזה הכל ...
   - EN: Desktop: the side gallery (Recent) opens far from its arrow; queue and gallery collide
112. [2026-10-22 10:29] (msg 168) Area: Swap
   - HE: [Image #25] האם זה פאקינג רלוונטי??? למה צריך את זה??? זה צריך לעשות פייס סוואפ שהפרצוף של הבנאדם יהיה בדיוק א ...
   - EN: Are Head pose and Color and style relevant? Face swap must keep the face exactly; is that what the internal prompt sends?
113. [2026-10-22 10:34] (msg 170) Area: Models/Downloads
   - HE: https://civitai.com/ למה לא הוספת גם את סיוויט איי איי הראשי? יש שם ארכיטקטורות ולורות ודברים שלא רק ברד ש!!!  ...
   - EN: Add the main civitai.com too: architectures, LoRAs, face swaps for Krea and others, Turbo models; add everything correctly
114. [2026-10-22 10:48] (msg 171) Area: Layout/UI
   - HE: [Image #26] תעשה באלה גם חצים פשוטים לצד המספר למעלה ולמטה לשליטה יותר ממוקדת
   - EN: Up and down arrows beside the numbers for finer control
115. [2026-10-22 10:52] (msg 172) Area: Layout/UI
   - HE: [Image #27] אתה רצני שזה ההסבר ששמת??? מה זה עוזר??!?!?!?!?!?!?!?!?! עשית הסבר על "חוזק" - "כמה חזק זה" ומה הד ...
   - EN: Hints must say in one line what higher and lower do, not 'strength: how strong'
116. [2026-10-22 10:55] (msg 173) Area: Swap
   - HE: [Image #28] מבדיקה שעשיתי כשזה פחות זה יותר דומה לפרצוף השני (לא המקור) אז תוודא שגם אפשר להוריד לפחות מ0.50
   - EN: Swap: lower is closer to the other face; allow values below 0.50
117. [2026-10-22 10:55] (msg 175) Area: Swap
   - HE: ותקרא לstrengh תקרא לזה keep original face או משהו כזה כדי שיהיה ברור
   - EN: Rename strength to 'Keep original face'
118. [2026-10-22 11:09] (msg 176) Area: Models/Downloads
   - HE: מה זה קוון אדיט 2511 זה יותר טוב מ2.1?
   - EN: What is Qwen Edit 2511? Is it better than 2.1?
119. [2026-10-22 11:09] (msg 177) Area: Swap
   - HE: למה "Keep original face", from 0.1 to 1.5 ולא עד 2?
   - EN: Why is Keep original face 0.1 to 1.5 and not up to 2?
120. [2026-10-22 11:11] (msg 179) Area: Swap
   - HE: וידאת שהפרימפט הפנימי של הסוואפ לורה משתמש בצורה נכונה לפי איך שזה מוגדר בהגדרות של מי שיצר את הסוואפ? וידאת ג ...
   - EN: Does the swap LoRA internal prompt follow its author's settings?
   - EN: Is there a swap for Krea 2 and others?
121. [2026-10-22 11:25] (msg 180) Area: Swap
   - HE: https://civitai.com/models/2780755/face-swap-sx-cu?modelVersionId=3131767 https://civitai.com/models/2780755/f ...
   - EN: Look at how face-swap-sx-cu uses Gemma 4
122. [2026-10-22 11:30] (msg 181) Area: Swap
   - HE: [Image #29] למה זה לא נותן לי לשנות את זה לעזאזל?!?!?! תגיד לי מה לעזאזל עשית עם התיגוים?!!??!?! מה קשור התיוג ...
   - EN: Why can I not change the swap prompt?
   - EN: What do the tags have to do with the internal prompt; it must work with any tag
   - EN: Are reference and head sent correctly in Swap?
123. [2026-10-22 11:34] (msg 183) Area: Swap, Settings
   - HE: אגב שים לב שבסוואפ צריך סי אף ג'י יותר גבוה אז אם אני וקובע דיפולט סי אף ג'י לסוואפ זה לא אומר שאותו מודל זה ה ...
   - EN: Swap needs a higher CFG; a Swap CFG default must not become the model's general default
124. [2026-10-22 11:44] (msg 184) Area: Swap
   - HE: [Image #30] ככה זה מומלץ מההגדרות של הלורה במקור?
   - EN: Is this prompt the LoRA author's recommendation?
125. [2026-10-22 11:47] (msg 185) Area: Swap
   - HE: https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/tree/main סתכל על הכל פה ותעשה שיהיה פייס סוואפ מושלם לכ ...
   - EN: Use everything in Alissonerdx/BFS-Best-Face-Swap: a perfect face swap for each model with the exact author settings
126. [2026-10-22 11:50] (msg 186) Area: Swap, Krea
   - HE: [Image #31] אני מעביר לקריאה וזה שולח את הסימוני פרצוף של פאקינג קוון תבדוק מה עשית שוב גרוע וגם למה זה לא מאפ ...
   - EN: Switching to Krea sends Qwen's face marks; why is it not allowed on Krea?
127. [2026-10-22 11:52] (msg 187) Area: Swap
   - HE: [Image #32] תבדוק למה זה זיהה שם פאקינג 2 , ותוודא שהבחיטרת פרצופים נכונה רק לתמונת המקור ולא לפרצו ףשמחליף (ת ...
   - EN: Why were 2 faces found? Face choice only in the source picture, never in the head picture (it has one face)
   - EN: 1 face and 1 head = automatic, no face choice
   - EN: To swap 2 faces I upload 2 face pictures
128. [2026-10-22 11:55] (msg 188) Area: Library/People, Swap
   - HE: [Image #33] ותוודא פה שפיפל זה הראשון- ואין כפתור "אפלואד" כי האפלואד נמצא כבר בלייבררי!!!! תסתכל פה [Image #3 ...
   - EN: Library: People first, no Upload button (upload is in the Library)
   - EN: Swap slots: no Replace and Library links; replace by clicking the picture like other pages
129. [2026-10-22 11:56] (msg 190) Area: Mobile
   - HE: כמו כן במובייל כשמורידים תמונה זה באג גדול פותח איזה עמוד מוזר במקום פשוט לפתוח את השייר או משהו כזה שיהיה סיי ...
   - EN: Mobile download opens a strange page; open the share sheet / easy Save Image
130. [2026-10-22 12:01] (msg 191) Area: Swap, Krea
   - HE: [Image #35] אין מצב שהקריאה 2 עובד כמו שצריך
   - EN: Krea 2 (swap) does not work correctly (screenshot)
131. [2026-10-22 12:02] (msg 192) Area: Gallery/Viewer
   - HE: ותבדוק אם אפשר בכל גינרוט להראות בצד איפה שהתמונה נוצרתץ (כל בלוק של תמונה שיהיה כבר מוכן עם התהליך שלו לא הפס ...
   - EN: Show each generation as an image tile with its progress, not the bars
   - EN: Each tile shows a developing diffusion preview like ComfyUI, for all image jobs
132. [2026-10-22 12:03] (msg 194) Area: Speed/VRAM
   - HE: ועכשיו מה שאתה מריץ איטי בצורהה קיצונית מה שוב הרסת!??!?!!
   - EN: The job you run now is extremely slow; what broke?
133. [2026-10-22 12:06] (msg 196) Area: Swap
   - HE: שים לב בהאגינג פייס שנתתי לך זה לא רק פייס סוואפ תפזר בצורה מושלמת בין כל הפיצ'רים כל אחד למקום המתאים שלו ואם ...
   - EN: The BFS repo has more than face swap: put each item in its correct feature, add features if needed, with full control, built-in settings per model and required inputs
134. [2026-10-22 12:13] (msg 197) Area: Gallery/Viewer
   - HE: [Image #37] תוודא שגם בתצוגה הזאת אפשר לדפדף בין התמונות ולא רק כשפותחים את התמונה ימזדין
   - EN: Allow stepping between images in this (Result) view too, not only in the opened image
135. [2026-10-22 12:19] (msg 198) Area: Swap, Speed/VRAM
   - HE: שוב הכל איטי ושוב מחליף מודל בסוואפ זה לא מתאים את ההגדרות כמו שצריך
   - EN: Everything slow again
   - EN: Changing the model in Swap does not adapt the settings
136. [2026-10-22 12:19] (msg 200) Area: Swap
   - HE: מחליף לסטנדרט נניח בקוון וזה עדיין על 8 סטפםס מה נסגר
   - EN: Switched to Qwen Standard and it still shows 8 steps
137. [2026-10-22 12:19] (msg 202) Area: Swap
   - HE: ולמה הסי אף ג'יז לא מותאמים ולמה ההגדרות לא נכונוות??
   - EN: CFGs are not adapted; the settings are wrong
138. [2026-10-22 12:24] (msg 203) Area: Swap, Library/People, Gallery/Viewer
   - HE: שהפריוויו יהיה ממש על התמונה הגדולה של הריזולט ההתפתחות עצמה. ותוריד כל דבר שצריך לא לשאול מעכישיו , ולא להוסי ...
   - EN: Live preview on the big Result picture itself
   - EN: From now on download what is needed without asking
   - EN: Do not add old models (FLUX.2 Klein, Qwen Edit 2511, LTX-2)
   - EN: Each person in People keeps a Female / Male label, not 'Person'
   - EN: The Scene slot is an input image, not 'Person'
   - EN: Match female and male automatically per face when there are many faces
   - EN: The person dropdown must not stretch down when a person has more photos
139. [2026-10-22 12:27] (msg 205) Area: Swap
   - HE: תעיף את הקטע של החלפת מספר פרצופים זה לא עובד. מה שתעשה זה רק בחירת איזה פרצוף בתמונת המקור- התמונה של הפרצוף  ...
   - EN: Remove multi-face swapping, it does not work: only choose which face in the source picture; the head picture is single and fixed
140. [2026-10-22 12:28] (msg 206) Area: Swap
   - HE: [Image #40] [Image #41] זה לא טוען את הסטפס הנכון או שזה הסטפס שעשית לסוואפ?
   - EN: Does it load the wrong steps, or are these the Swap steps?
141. [2026-10-22 12:28] (msg 207) Area: Swap
   - HE: [Image #42] [Image #43] בהגדרות כתוב 8 אבל זה עושה על 40 מה קורה
   - EN: Settings show 8 but it runs 40
142. [2026-10-22 12:31] (msg 209) Area: Swap, Speed/VRAM
   - HE: עכשיו התוצאה יצאה טובה אבל זה איטי לגמרי וההגדרות שנראת בעמוד הסוואפ לא תואמות למה שכתוב על ההאאוטפוט
   - EN: The result is good but very slow
   - EN: Swap page settings do not match what the output says
143. [2026-10-22 12:44] (msg 211) Area: Swap, Speed/VRAM
   - HE: Oct 22, 03:33 P זה ההכי טוב שיצא - פשוט איטי בטירוף . תבי למה זה יצא ככ טוב ותוודא שההגדרות כמו שצריך וזה לא א ...
   - EN: The Oct 22 03:33 PM result is the best: explain why it is so good, and make the settings correct and not slow
144. [2026-10-22 12:44] (msg 213) Area: Swap, Gallery/Viewer
   - HE: ושבתוצאה יהיה בקטן בסוואפ איזה תמונת אינפוט הייטב ואיזה תמונת פרצוף
   - EN: The Swap result shows small which input picture and which face picture were used
145. [2026-10-22 12:47] (msg 215) Area: Krea
   - HE: מדהי ם שהקריאה 2 למרות שאמרתי לך בכל לשון של בקשה עדיין עושה 1:1 כשזה על אאוטו לא יאומן אתה בכלל עושה משהו מהמ ...
   - EN: Krea 2 Auto still makes 1:1 despite repeated asks; you forget tasks; write everything down
146. [2026-10-22 12:47] (msg 217) Area: Gallery/Viewer
   - HE: ואמרתי לך שהפריוויו ייראה על גבי התמונה הגדולה שלהיזלוט!!!!!!!!!!!!!!! לא בקטן בצד
   - EN: The preview must show on the big Result picture, not small at the side
147. [2026-10-22 12:49] (msg 218) Area: Enhance
   - HE: [Image #44] [Image #45] וגם בישקתי ממך לעזאזל שיהיה אופציה לבחרו מודל אחר של טקסט שיעשה את זה נכון?? גם ג'מה ו ...
   - EN: An option to choose another text model for Enhance (Gemma and whatever is available)
148. [2026-10-22 12:50] (msg 220) Area: Gallery/Viewer
   - HE: הפרוויו עכשיו מציג את התמונה בענק קרופד חתיכת מזדיין
   - EN: The preview now shows the image huge and cropped
149. [2026-10-22 12:51] (msg 222) Area: Gallery/Viewer
   - HE: חתיכת מזדיין אם אני מדפדף בתמונות זה אמור להראות לי אותם גם בזמן גנרוט, רק כשמתחילים משימה אם אני לוחץ על התמו ...
   - EN: When I step through images during a job, show them; show the preview only at job start or when I click the job's tile
150. [2026-10-22 12:53] (msg 224) Area: Swap
   - HE: למה פיינדינג פייסס אין פיקטור לוקח שעות ולא עובד??
   - EN: Why does 'Finding faces' take forever and not work?
151. [2026-10-22 12:55] (msg 225) Area: Library/People
   - HE: [Image #46] ביקשתי ממך שהלייבררי יפתח על פאקינג פיפל נכון1?!?!??!?!?!?!??!? למה הוא נפתח על אפלואדז?!?!?!?!??! ...
   - EN: The Library must open on People, not Uploads
152. [2026-10-22 12:55] (msg 227) Area: Swap
   - HE: ולמה זה מתחיל לחפש פרצופים בתמונת סצינה רק אחרי שאני בוחר תמונת head לעזאזלץ.1.1ץ1ץ1ץ1ץץץץ21ץ.1ץ.1ץ.1ץ.1?<?!?! ...
   - EN: Why does the face search in the scene start only after I choose the head picture?
153. [2026-10-22 12:56] (msg 228) Area: Swap
   - HE: [Image #47] נו מה?!?!?! מה?!?!?
   - EN: (Screenshot) 'so what?!' complaint, no text
154. [2026-10-22 12:56] (msg 229) Area: Swap
   - HE: [Image #48] ואמרתי לך לסדר את התצוגה של הדרופדאון ואיפה מייל פימייל .1?!?!?!?!? ולמה לסצינה יש תיוג!??!?!?!?!? ...
   - EN: Fix the dropdown display; where are Male / Female?
   - EN: Why does the scene have a label?
155. [2026-10-22 12:59] (msg 230) Area: Swap, Other
   - HE: [Image #49] ואמרתי לך לסדר את איך שזה נראה תראה שה 1\2 מוריד את הפרסון למטה לנהןשגנוםפעש מףךיגןמ,םףצשגכשעשגד,ח ...
   - EN: The 1/2 arrows push the person down; fix how it looks
   - EN: Map every task you had, skip none
156. [2026-10-22 13:00] (msg 232) Area: Swap
   - HE: למה שזה לא ימצא תפרצופים בתמונה אלוהים ישמור זה עבד כל הזמן כל רגע מה יכול להיות שהרסת מה פרחקח לךףעקמףלךנ'כףמ ...
   - EN: Why does it not find faces in the picture? It always worked
157. [2026-10-22 13:02] (msg 234) Area: Enhance
   - HE: וגם למה לעזאזל יש ר ק 4 מודלים של אנהאנס וגם חתיכת נאצי מזדיין אתה לא שם לב שהתיוגים נאבדים וכל המהות שלהם אם  ...
   - EN: Why are there only 4 enhancer models?
   - EN: Labels / tags are lost when Enhance runs
158. [2026-10-22 13:08] (msg 236) Area: Swap, Krea
   - HE: הסוואפ של קריאה 2 כרגע לא עובד ___בכלל_________ חירבנת את זה לגמרי תמות
   - EN: Krea 2 swap does not work at all
159. [2026-10-22 13:10] (msg 237) Area: Swap, Other
   - HE: [Image #50] [Image #51] אמרתי לך שיהיה למטה מידע מי החליף את מה נכון1?!?!?!?!?!?!?!?!?!?! תקשיב הרסת הכל תעשה  ...
   - EN: Show under the result who swapped what
   - EN: Make a detailed plan with many agents covering everything from today, go item by item and do not stop until all is fixed; full map and a progress page before work
160. [2026-10-22 13:11] (msg N1) Area: Swap
   - HE: שים לב כל הסוואפים כרגע לא עובדים הרסת הכל לא ברור מה עשית שם בפרומפט הפני ולמה לעזאזל זה לא מוצא פרצופים
   - EN: All swaps do not work now: what changed in the internal prompt? Why does it not find faces? It does not change the face
161. [2026-10-22 13:12] (msg N2) Area: Swap, Library/People
   - HE: מה עשית לעזאזל?! מה זה מן וומן?! לא! זה פרסון קודם כל ואז מייל או פימייל וזה צריך להיות סט כבר על כל פרסון
   - EN: Not Man / Woman: label is 'Person' first, then Male or Female, preset on every person
   - EN: Find what broke in Swap since it worked
   - EN: Where is the face detection of the scene picture?
162. [2026-10-22 13:12] (msg N3) Area: Swap
   - HE: הנה עכשיו זה עבד מה הרסת שם?! תוודא שלכל מודל זה מושלם שמעת אותי?!
   - EN: Now it worked: what broke? Make sure it is perfect for every model
