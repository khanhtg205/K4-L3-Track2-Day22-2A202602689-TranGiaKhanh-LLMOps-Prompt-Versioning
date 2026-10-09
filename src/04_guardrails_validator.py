"""
Bước 4 — Guardrails AI Validators
====================================
NHIỆM VỤ:
  1. Xây dựng PIIDetector: phát hiện & redact email, số điện thoại, SSN, số thẻ tín dụng
  2. Xây dựng JSONFormatter: tự động sửa JSON lỗi
  3. Bọc mỗi validator trong Guard và test với các mẫu đầu vào
  4. Chạy demo với 6 trường hợp PII và 5 trường hợp JSON
  5. Lưu log minh chứng vào evidence/04_pii_demo_log.txt và evidence/04_json_demo_log.txt

DELIVERABLE: Tất cả test cases pass (PII bị redact, JSON được sửa thành công)
"""

import re
import json
from pathlib import Path

from guardrails import Guard
from guardrails.validators import Validator, register_validator, PassResult, FailResult

try:
    from guardrails.hub import OnFailAction
except ImportError:
    from guardrails.validator_base import OnFailAction


# ── 1. PII Detector Validator ──────────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Phát hiện và redact Personally Identifiable Information (PII).

    Các pattern được phát hiện:
      EMAIL       : xxx@xxx.xxx
      PHONE       : (123) 456-7890 hoặc 123-456-7890
      SSN         : 123-45-6789
      CREDIT_CARD : 1234 5678 9012 3456 (hoặc dấu gạch nối)
    """

    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE":       r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    }

    def validate(self, value: str, metadata: dict = None):
        """
        Tìm PII trong value; nếu phát hiện, redact và trả về FailResult với fix_value là text đã xử lý.
        """
        redacted_text = value
        found_pii     = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, value)
            for match in matches:
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append((pii_type, match))

        if found_pii:
            print(f"  ⚠️  Đã redact {len(found_pii)} PII: {[p[0] for p in found_pii]}")
            return FailResult(
                error_message=f"Phát hiện {len(found_pii)} thông tin PII",
                fix_value=redacted_text
            )

        return PassResult()


# ── 2. JSON Formatter Validator ────────────────────────────────────────────
@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validate và tự động sửa JSON lỗi.

    Các lỗi có thể sửa tự động:
      - Strip markdown code fences (``` hoặc ```json)
      - Thay single quotes → double quotes
      - Xóa trailing commas trước } hoặc ]
      - Re-serialize với json.dumps để định dạng chuẩn
    """

    @staticmethod
    def _repair(text: str) -> str:
        """
        Cố gắng sửa chuỗi JSON lỗi.
        """
        text = text.strip()

        # Xóa markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$',          '', text)
        text = text.strip()

        # Thay single quotes → double quotes
        text = text.replace("'", '"')

        # Xóa trailing commas trước } hoặc ]
        text = re.sub(r',\s*([}\]])', r'\1', text)

        return text

    def validate(self, value: str, metadata: dict = None):
        """
        Thử parse value thành JSON. Nếu thất bại, gọi _repair() rồi thử lại.
        """
        try:
            json.loads(value)
            return PassResult()
        except json.JSONDecodeError:
            pass

        try:
            repaired_text = self._repair(value)
            parsed        = json.loads(repaired_text)
            print(f"  🔧 JSON đã được sửa thành công")
            return FailResult(
                error_message="JSON lỗi, đã tự sửa",
                fix_value=json.dumps(parsed, indent=2, ensure_ascii=False)
            )
        except json.JSONDecodeError:
            fallback = json.dumps({"error": "Không thể phân tích JSON", "raw": value[:200]}, ensure_ascii=False)
            return FailResult(error_message="Không thể sửa JSON", fix_value=fallback)


# ── 3. Demo: PII Guard ─────────────────────────────────────────────────────
def demo_pii_guard() -> str:
    lines = []
    lines.append("=" * 55)
    lines.append("  Demo: PII Detection & Redaction")
    lines.append("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Email",        "Contact John at john.doe@example.com for details."),
        ("Phone",        "Call our support line at (555) 867-5309."),
        ("SSN",          "Patient SSN is 123-45-6789 on file."),
        ("Credit Card",  "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",    "Email: alice@example.com, Phone: 555-123-4567"),
        ("Clean",        "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        out_str = f"\n[{label}]\n  Input:  {text}\n  Output: {result.validated_output}"
        lines.append(out_str)
        print(out_str)

    output_text = "\n".join(lines)
    evidence_path = Path(__file__).parent.parent / "evidence" / "04_pii_demo_log.txt"
    evidence_path.write_text(output_text, encoding="utf-8")
    print(f"\n💾 Đã lưu log PII vào {evidence_path}")
    return output_text


# ── 4. Demo: JSON Guard ────────────────────────────────────────────────────
def demo_json_guard() -> str:
    lines = []
    lines.append("=" * 55)
    lines.append("  Demo: JSON Formatting & Repair")
    lines.append("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",       '{"name": "Alice", "age": 30}'),
        ("Markdown fences",  '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",    "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",   '{"key": "value",}'),
        ("Truly invalid",    "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)
        status = "✅ Pass" if result.validation_passed else "❌ Fail"
        out_str = f"\n[{label}] {status}\n  Input:  {text[:60]}\n  Output: {str(result.validated_output)[:60]}"
        lines.append(out_str)
        print(out_str)

    output_text = "\n".join(lines)
    evidence_path = Path(__file__).parent.parent / "evidence" / "04_json_demo_log.txt"
    evidence_path.write_text(output_text, encoding="utf-8")
    print(f"\n💾 Đã lưu log JSON vào {evidence_path}")
    return output_text


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Bước 4: Guardrails AI Validators")
    print("=" * 55)

    demo_pii_guard()
    demo_json_guard()

    print("\n✅ Bước 4 hoàn thành!")


if __name__ == "__main__":
    main()
