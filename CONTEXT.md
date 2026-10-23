# Room 313 — سياق المشروع (علشان نكمّل في شات/جلسة جديدة)

## الصفحة
- الاسم: **غُرفة ٣١٣** — قصص جرائم غامضة وما وراء الطبيعة بالعربي (بديل صفحة Decode).
- الراوي: ElevenLabs Voice "Fathy Hammad - Strong and Deep" (eleven_v4)، بدون همس، Stability 45 / Style 40 / Speed 0.9، تشكيل كامل للنطق.
- الحلقة 1: **هِنترْكايفِك** (1922). الهوك: "عيلة كاملة اتقتلت في المزرعة… والقاتل واحد من البيت." الأصوات 1–3 في `projects/ep1_hinterkaifeck/input/`.
- باقي اسكربت الحلقة (الفقرات 4–11) موجود في الشات القديم: الخدّامة "البيت ده مسكون" / ماريا أول يوم شغل / الليلة (الزريبة) / الفاس / القاتل عاش 4 أيام في البيت / 4 أبريل / الرؤوس لميونخ / هدّ المزرعة والفاس في السقيفة.

## القرارات
- **تكلفة صفر** مطلقًا. ممنوع: Replicate/Kling/Runway/fal/ElevenLabs SFX/أي API مدفوع. ممنوع شخصيات SVG إجرائية أو سلايد شو كبديل.
- البنية: Claude = تخطيط + توليف + مونتاج (ffmpeg) · **GitHub Actions** = عامل التوليد المجاني (HF Spaces عبر gradio_client، نماذج مفتوحة على CPU، Kaggle API اختياري، Freesound CC0).
- أول Milestone: صورة شخصية 2D واحدة + لقطة متحركة 5 ثواني بنماذج مجانية — **قبل** الحلقة كاملة.

## المستودع
- `.github/workflows/free-audit.yml` (تشغيله أولًا)، `generate.yml` (resumable, budget=0)، `worker/generate.py`، `orchestrator/*`، `projects/ep1_hinterkaifeck/storyboard.json` (16 لقطة، 5 شخصيات، موسيقى + SFX).
- أسرار مجانية اختيارية في GitHub Secrets: HF_TOKEN, KAGGLE_USERNAME/KAGGLE_KEY, FREESOUND_API_KEY.
- الترتيب: dispatch connectivity/free-audit → راجع المخرجات → generate stage=character → image → video → audio → assemble.py → renders/final.mp4.

## تعليمات للجلسة الجديدة
ابدأ بـ: "هذا ريبو Sciltons/room313. اقرأ CONTEXT.md وREADME.md، ادفع الكود لو ناقص، وشغّل free-audit ثم الـ milestone الأول."
