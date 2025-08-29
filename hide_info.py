import re
import fitz  # PyMuPDF

# ====== Step 1. 定义混淆函数 ======
def obfuscate_text(text: str) -> str:
    # 手机号：13812345678 -> 138*****5678
    if re.fullmatch(r"1[3-9]\d{9}", text):
        return text[:3] + "*****" + text[-4:]

    # 邮箱：feng.liu@example.com -> fe*****@example.com
    email_match = re.fullmatch(r"([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\.[A-Za-z]{2,})", text)
    if email_match:
        user, domain = email_match.groups()
        if len(user) > 2:
            return user[:2] + "*****@" + domain
        else:
            return user[0] + "*****@" + domain

    # 微信号：保留前 2 位
    if text.lower().startswith("微信") or len(text) > 6:
        return text[:2] + "*****"

    return text  # 不处理的直接返回

# ====== Step 2. 遍历 PDF，替换文本 ======
input_pdf = "ren.pdf"
output_pdf = "output_obfuscated.pdf"

doc = fitz.open(input_pdf)

for page_num, page in enumerate(doc, start=1):
    blocks = page.get_text("dict")["blocks"]

    for b in blocks:
        if "lines" in b:
            for l in b["lines"]:
                for s in l["spans"]:
                    original_text = s["text"]

                    # 手机号/邮箱正则匹配
                    if re.search(r"1[3-9]\d{9}", original_text) or \
                       re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", original_text):

                        new_text = obfuscate_text(original_text)
                        print(f"页面 {page_num}: 找到敏感信息 '{original_text}' -> '{new_text}'")
                        
                        # 在原位置覆盖文本
                        rect = fitz.Rect(s["bbox"])
                        print(f"  原始位置: {rect}")
                        
                        # 先添加遮罩
                        page.add_redact_annot(rect, fill=(1, 1, 1))  # 白底覆盖
                        page.apply_redactions()
                        
                        # 使用 insert_text 方法直接在指定位置插入文本
                        # 计算文本插入位置（左下角位置）
                        insert_point = fitz.Point(rect.x0, rect.y1 - 2)  # 稍微向上偏移
                        print(f"  插入位置: {insert_point}")
                        
                        try:
                            # 尝试使用 insert_text 方法
                            result = page.insert_text(insert_point, new_text, 
                                                    fontsize=s["size"], 
                                                    fontname="helv", 
                                                    color=(0, 0, 0))
                            print(f"  insert_text 结果: {result}")
                            
                        except Exception as e:
                            print(f"  insert_text 失败: {e}")
                            # 尝试更简单的方法
                            try:
                                result = page.insert_text(insert_point, new_text, 
                                                        fontsize=10, 
                                                        color=(0, 0, 0))
                                print(f"  简化 insert_text 结果: {result}")
                            except Exception as e2:
                                print(f"  所有插入方法都失败: {e2}")
                                # 最后尝试使用 add_text
                                try:
                                    page.add_text(insert_point, new_text)
                                    print("  使用 add_text 成功")
                                except Exception as e3:
                                    print(f"  add_text 也失败: {e3}")

doc.save(output_pdf)
print(f"处理完成，已保存到: {output_pdf}")
