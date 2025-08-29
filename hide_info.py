import re
import fitz  # PyMuPDF

# ====== Step 1. 定义要隐藏的敏感信息规则 ======
patterns = [
    r"\b1[3-9]\d{9}\b",                    # 手机号
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # 邮箱
    r"微信[:：]?\s*[a-zA-Z0-9_-]+",        # 微信号
]

def mask_sensitive_text(page, text_instances):
    """在 PDF 页面上遮罩敏感信息"""
    for inst in text_instances:
        rect = fitz.Rect(inst["rect"])  # 文本的矩形区域
        page.add_redact_annot(rect, fill=(0, 0, 0))  # 白底遮罩
        # 你也可以用黑条：fill=(0, 0, 0)
    page.apply_redactions()

# ====== Step 2. 处理 PDF ======
input_pdf = "ren.pdf"
output_pdf = "output_masked.pdf"

doc = fitz.open(input_pdf)

for page_num, page in enumerate(doc, start=1):
    text_instances = []
    blocks = page.get_text("dict")["blocks"]

    for b in blocks:
        if "lines" in b:
            for l in b["lines"]:
                for s in l["spans"]:
                    for pattern in patterns:
                        if re.search(pattern, s["text"]):
                            text_instances.append({"text": s["text"], "rect": s["bbox"]})

    # 遮罩敏感信息
    if text_instances:
        mask_sensitive_text(page, text_instances)

doc.save(output_pdf)
print(f"处理完成，已保存到: {output_pdf}")
