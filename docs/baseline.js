/**
 * In-browser Baseline Toxicity Classifier (TF-IDF + Logistic Regression).
 *
 * Implements the exact feature extraction, preprocessing, and inference pipeline
 * exported from scikit-learn (preprocess.py & src/baseline.py).
 */

class BaselineClassifier {
  constructor(modelData) {
    if (!modelData) {
      throw new Error('BaselineClassifier requires model data.');
    }
    this.model = modelData;

    this.wordVocab = modelData.word_vectorizer.vocabulary;
    this.wordIdf = modelData.word_vectorizer.idf;
    this.wordCoef = modelData.word_vectorizer.coef;

    this.charVocab = modelData.char_vectorizer.vocabulary;
    this.charIdf = modelData.char_vectorizer.idf;
    this.charCoef = modelData.char_vectorizer.coef;

    this.intercept = modelData.classifier.intercept;
    this.threshold = modelData.classifier.threshold || 0.5;
  }

  /**
   * Asynchronously load model JSON from relative path or URL.
   */
  static async load(jsonUrl = './model/baseline.json') {
    const res = await fetch(jsonUrl);
    if (!res.ok) {
      throw new Error(`Failed to load baseline model: HTTP ${res.status}`);
    }
    const data = await res.json();
    return new BaselineClassifier(data);
  }

  /**
   * HTML unescape matching Python html.unescape.
   */
  unescapeHtml(str) {
    if (!str) return '';
    return str
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&quot;/g, '"')
      .replace(/&#039;/g, "'")
      .replace(/&apos;/g, "'")
      .replace(/&#(\d+);/g, (_, code) => String.fromCharCode(parseInt(code, 10)))
      .replace(/&#x([0-9a-fA-F]+);/g, (_, hex) => String.fromCharCode(parseInt(hex, 16)));
  }

  /**
   * Clean text adhering to preprocess.py rules.
   */
  cleanText(text) {
    if (text == null) return '';
    let s = String(text);

    // 1. Unescape HTML entities
    s = this.unescapeHtml(s);

    // 2. Strip HTML tags
    s = s.replace(/<[^>]+>/g, ' ');

    // 3. Normalize URLs
    s = s.replace(/https?:\/\/\S+|www\.\S+/gi, '[URL]');

    // 4. Normalize user mentions
    s = s.replace(/@[\w\-]+/gi, '[USER]');

    // 5. Remove null bytes and non-printable control characters
    s = s.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, '');

    // 6. Normalize excessive blank lines and whitespace
    s = s.replace(/\n\s*\n+/g, '\n');
    s = s.replace(/[ \t]+/g, ' ');

    return s.trim();
  }

  /**
   * Compute toxicity probability and verdict for a comment string.
   */
  predictProba(rawText) {
    const text = this.cleanText(rawText);
    if (!text) {
      // Empty input probability is sigmoid(intercept)
      const emptyP = 1.0 / (1.0 + Math.exp(-this.intercept));
      return {
        probability: emptyP,
        verdict: emptyP >= this.threshold ? 'Toxic' : 'Clean',
        cleanedText: ''
      };
    }

    const lower = text.toLowerCase();

    // 1. Word TF-IDF (unigrams + bigrams using token_pattern: (?u)\b\w\w+\b)
    const tokens = lower.match(/[\p{L}\p{N}_]{2,}/gu) || [];
    const wordCounts = {};

    // Unigrams
    for (let i = 0; i < tokens.length; i++) {
      const t = tokens[i];
      wordCounts[t] = (wordCounts[t] || 0) + 1;
    }

    // Bigrams
    for (let i = 0; i < tokens.length - 1; i++) {
      const bg = `${tokens[i]} ${tokens[i + 1]}`;
      wordCounts[bg] = (wordCounts[bg] || 0) + 1;
    }

    let wordSqSum = 0.0;
    const wordFeatures = [];

    for (const term in wordCounts) {
      const idx = this.wordVocab[term];
      if (idx !== undefined) {
        const tf = 1.0 + Math.log(wordCounts[term]);
        const idf = this.wordIdf[idx];
        const val = tf * idf;
        wordFeatures.push({ idx, val });
        wordSqSum += val * val;
      }
    }

    const wordNorm = wordSqSum > 0 ? Math.sqrt(wordSqSum) : 1.0;
    let wordScore = 0.0;
    for (let i = 0; i < wordFeatures.length; i++) {
      const f = wordFeatures[i];
      const normVal = f.val / wordNorm;
      wordScore += normVal * this.wordCoef[f.idx];
    }

    // 2. Char TF-IDF (n-grams of length 2 to 5 chars)
    const charCounts = {};
    const textLen = lower.length;

    for (let n = 2; n <= 5; n++) {
      for (let i = 0; i <= textLen - n; i++) {
        const cng = lower.slice(i, i + n);
        charCounts[cng] = (charCounts[cng] || 0) + 1;
      }
    }

    let charSqSum = 0.0;
    const charFeatures = [];

    for (const term in charCounts) {
      const idx = this.charVocab[term];
      if (idx !== undefined) {
        const tf = 1.0 + Math.log(charCounts[term]);
        const idf = this.charIdf[idx];
        const val = tf * idf;
        charFeatures.push({ idx, val });
        charSqSum += val * val;
      }
    }

    const charNorm = charSqSum > 0 ? Math.sqrt(charSqSum) : 1.0;
    let charScore = 0.0;
    for (let i = 0; i < charFeatures.length; i++) {
      const f = charFeatures[i];
      const normVal = f.val / charNorm;
      charScore += normVal * this.charCoef[f.idx];
    }

    // 3. Logistic Regression Activation
    const z = this.intercept + wordScore + charScore;
    const prob = 1.0 / (1.0 + Math.exp(-z));

    return {
      probability: prob,
      verdict: prob >= this.threshold ? 'Toxic' : 'Clean',
      cleanedText: text
    };
  }
}

// Export for Node.js test environment or browser global
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { BaselineClassifier };
}
if (typeof window !== 'undefined') {
  window.BaselineClassifier = BaselineClassifier;
}
