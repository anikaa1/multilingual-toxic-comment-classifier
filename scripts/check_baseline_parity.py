#!/usr/bin/env python3
"""Parity check comparing Python baseline model and JavaScript docs/baseline.js predictions.

Requirement: Probabilities on 20 diverse sentences must match to 3 decimal places (|p_py - p_js| < 0.001).
"""

import json
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
from preprocess import clean_text

TEST_SENTENCES = [
    # 1. English overtly toxic
    "You are a complete idiot and should delete your account immediately.",
    # 2. English clean / constructive
    "This article provides very thoughtful and well-researched insights. Thank you.",
    # 3. Spanish toxic
    "Eres un imbécil total y no sabes nada de este tema, cállate la boca.",
    # 4. Spanish clean
    "Muchas gracias por la excelente explicación detallada y los ejemplos claros.",
    # 5. Turkish toxic
    "Sen tam bir aptalsın, hiçbir şeyden anlamıyorsun defol git buradan.",
    # 6. Turkish clean
    "Harika bir yazı olmuş, paylaştığınız değerli bilgiler için teşekkürler.",
    # 7. Russian toxic
    "Ты полный идиот и ничего не понимаешь в этом, закрой свой рот.",
    # 8. Russian clean
    "Большое спасибо за подробное и понятное объяснение темы.",
    # 9. French toxic
    "Tu es un idiot fini et tu ne comprends rien à ce sujet, tais-toi.",
    # 10. French clean
    "Merci beaucoup pour cette analyse approfondie et très pertinente.",
    # 11. Portuguese toxic
    "Você é um idiota completo e não entende nada sobre isso, cala a boca.",
    # 12. Portuguese clean
    "Parabéns pelo excelente trabalho e pela dedicação ao projeto.",
    # 13. HTML tags and entities
    "<b>Stop talking nonsense!</b> You are a fool &amp; a liar.",
    # 14. URLs and Mentions
    "Hey @admin_user check this link https://example.com/fake-news right now!",
    # 15. Aggressive punctuation & ALL-CAPS
    "GET OUT OF HERE RIGHT NOW YOU CLOWN!!!!!!",
    # 16. Emojis and sentiment
    "Great work team! Keep it up 👍 🎉 ✨",
    # 17. Mild toxic insult
    "Shut up and stop wasting everyone's time.",
    # 18. Polite disagreement
    "I respectfully disagree with your conclusion, although your premises are valid.",
    # 19. Excessive spacing and blank lines
    "Why are you   acting   so stupid?   \n\n\n\nNobody likes you.",
    # 20. Short colloquial toxic
    "Useless trash.",
]


def run_parity_check():
    model_path = Path("models/baseline_model.joblib")
    json_path = Path("docs/model/baseline.json")
    js_path = Path("docs/baseline.js")

    if not model_path.exists():
        sys.exit(f"Error: {model_path} not found.")
    if not json_path.exists():
        sys.exit(f"Error: {json_path} not found. Run scripts/export_baseline.py first.")
    if not js_path.exists():
        sys.exit(f"Error: {js_path} not found.")

    # 1. Compute Python predictions
    print(f"Loading Python baseline model from {model_path}...")
    pipeline = joblib.load(model_path)

    # Clean text identically to production API
    cleaned_texts = [clean_text(s) or s for s in TEST_SENTENCES]
    py_probs = pipeline.predict_proba(cleaned_texts)[:, 1].tolist()

    # 2. Compute JavaScript predictions via Node.js
    print(f"Executing JavaScript baseline in Node.js via {js_path}...")
    node_script = """
    const fs = require('fs');
    const { BaselineClassifier } = require('./docs/baseline.js');

    const modelData = JSON.parse(fs.readFileSync('./docs/model/baseline.json', 'utf8'));
    const classifier = new BaselineClassifier(modelData);

    const sentences = %s;
    const results = sentences.map(s => classifier.predictProba(s).probability);
    console.log(JSON.stringify(results));
    """ % json.dumps(TEST_SENTENCES)

    node_res = subprocess.run(
        ["node", "-e", node_script],
        capture_output=True,
        text=True,
        check=True,
    )
    js_probs = json.loads(node_res.stdout.strip())

    # 3. Compare parity
    print("\n" + "=" * 88)
    print(f"{'#':<3} | {'Python':<8} | {'JS':<8} | {'Diff':<8} | {'Match?':<6} | Sentence Preview")
    print("=" * 88)

    max_diff = 0.0
    all_passed = True

    for idx, (py_p, js_p, sentence) in enumerate(zip(py_probs, js_probs, TEST_SENTENCES), 1):
        diff = abs(py_p - js_p)
        max_diff = max(max_diff, diff)
        passed = diff < 0.001
        if not passed:
            all_passed = False
        status = "PASS" if passed else "FAIL"
        preview = (sentence[:40] + "...") if len(sentence) > 40 else sentence
        print(f"{idx:<3} | {py_p:<8.4f} | {js_p:<8.4f} | {diff:<8.6f} | {status:<6} | {preview}")

    print("=" * 88)
    print(f"Total Sentences Tested: {len(TEST_SENTENCES)}")
    print(f"Maximum Probability Difference: {max_diff:.6f}")
    print(f"Required Precision: < 0.001000 (3 decimal places)")

    if all_passed:
        print("\nSUCCESS: All 20 sentence predictions match within 3 decimal places!")
        return 0
    else:
        print("\nFAILURE: Some predictions differed by more than 0.001.")
        return 1


if __name__ == "__main__":
    sys.exit(run_parity_check())
