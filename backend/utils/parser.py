import re

GRADE_MAP = {
    "AA": 10, "AB": 9, "BB": 8, "BC": 7,
    "CC": 6, "CD": 5, "DD": 4, "FF": 0
}


def extract_sgpas(text):
    sgpas = []

    matches = re.findall(r"\bSGPA\s*[:\-]?\s*(\d+\.\d+)", text)

    if not matches:
        lines = text.split("\n")

        for i, line in enumerate(lines):
            if "SGPA" in line and "CREDIT" in line:
                
                for j in range(i + 1, min(i + 6, len(lines))):
                    nums = re.findall(r"\d+\.\d+", lines[j])

                    for val in nums:
                        f = float(val)

                        # stricter SGPA range
                        if 6.0 <= f <= 10.0:
                            sgpas.append(f)

                    if sgpas:
                        break

    else:
        sgpas = [float(x) for x in matches]

    seen = set()
    cleaned = []
    for x in sgpas:
        if x not in seen:
            cleaned.append(x)
            seen.add(x)

    return cleaned[:4]


def extract_subject_grades(text):
    lines = text.split("\n")

    math_scores = []
    hard_scores = []

    for line in lines:
        line = line.upper().strip()

        grades = re.findall(r"\b(AA|AB|BB|BC|CC|CD|DD|FF)\b", line)

        if not grades:
            continue

        grade = grades[-1]
        score = GRADE_MAP[grade]

        if re.search(r"\bMAL\d+\b", line):
            math_scores.append(score)

        if any(keyword in line for keyword in [
            "PHYSICS",
            "CHEMISTRY",
            "SIGNALS",
            "DIGITAL SIGNAL",
            "DSP"
        ]):
            hard_scores.append(score)

    return math_scores, hard_scores


def parse_text(text, tables=None):
    data = {}

    text = text.upper()

    sgpas = extract_sgpas(text)

    data['S1_SGPA'] = sgpas[0] if len(sgpas) > 0 else 0
    data['S2_SGPA'] = sgpas[1] if len(sgpas) > 1 else 0
    data['S3_SGPA'] = sgpas[2] if len(sgpas) > 2 else 0
    data['S4_SGPA'] = sgpas[3] if len(sgpas) > 3 else 0

    data['num_FF'] = len(re.findall(r"\bFF\b", text))
    data['num_AA'] = len(re.findall(r"\bAA\b", text))

    math_scores, hard_scores = extract_subject_grades(text)

    data['avg_grade_math'] = (
        sum(math_scores) / len(math_scores)
        if math_scores else 0
    )

    data['avg_grade_hard_subjects'] = (
        sum(hard_scores) / len(hard_scores)
        if hard_scores else 0
    )

    return data