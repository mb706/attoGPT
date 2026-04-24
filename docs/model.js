const SPECIAL_TOKENS = new Set(["<pad>", "<bos>", "<eos>", "<unk>"]);

export class WordTokenizer {
  constructor(meta) {
    this.kind = meta.kind;
    this.itos = meta.itos;
    this.stoi = new Map(this.itos.map((token, id) => [token, id]));
    this.padId = this.stoi.get("<pad>");
    this.bosId = this.stoi.get("<bos>");
    this.eosId = this.stoi.get("<eos>");
    this.unkId = this.stoi.get("<unk>");
    this.badIds = new Set([this.padId, this.bosId, this.unkId]);
  }

  normalize(text) {
    return text
      .toLowerCase()
      .replace(/[^a-z.,!?\n ]+/g, " ")
      .replace(/[ \t]+/g, " ")
      .replace(/ *\n+ */g, "\n")
      .trim();
  }

  tokenize(text) {
    return this.normalize(text).match(/[a-z]+|[.,!?]/g) ?? [];
  }

  encode(text, addBos = true) {
    const ids = this.tokenize(text).map((token) => this.stoi.get(token) ?? this.unkId);
    return addBos ? [this.bosId, ...ids] : ids;
  }

  decode(ids, skipSpecial = true) {
    let out = "";
    for (const id of ids) {
      const token = this.itos[id] ?? "<unk>";
      if (skipSpecial && SPECIAL_TOKENS.has(token)) continue;
      if ([".", ",", "!", "?"].includes(token)) {
        out = `${out.trimEnd()}${token} `;
      } else {
        out += `${token} `;
      }
    }
    return out.trim();
  }

  display(id) {
    return this.itos[id] ?? "<unk>";
  }

  isSpecial(id) {
    return SPECIAL_TOKENS.has(this.display(id));
  }
}

export class TinyGptWeb {
  constructor(manifest, weightsBuffer) {
    this.manifest = manifest;
    this.config = manifest.config;
    this.weightsRaw = new Float32Array(weightsBuffer);
    this.weights = {};
    for (const [name, spec] of Object.entries(manifest.tensors)) {
      this.weights[name] = this.weightsRaw.subarray(spec.offset, spec.offset + spec.length);
    }
    this.vocabSize = this.config.vocab_size;
    this.blockSize = this.config.block_size;
    this.nEmb = this.config.n_embd;
    this.nHead = this.config.n_head;
    this.headDim = this.nEmb / this.nHead;
    this.hidden = Math.floor(this.config.mlp_mult * this.nEmb);
    this.topkAttn = this.config.topk_attn;
  }

  rmsNorm(row, weight) {
    let meanSq = 0;
    for (let i = 0; i < row.length; i++) meanSq += row[i] * row[i];
    const scale = 1 / Math.sqrt(meanSq / row.length + 1e-6);
    const out = new Float32Array(row.length);
    for (let i = 0; i < row.length; i++) out[i] = row[i] * scale * weight[i];
    return out;
  }

  linear(input, weight, outDim, inDim) {
    const out = new Float32Array(outDim);
    for (let o = 0; o < outDim; o++) {
      const wBase = o * inDim;
      let sum = 0;
      for (let i = 0; i < inDim; i++) sum += input[i] * weight[wBase + i];
      out[o] = sum;
    }
    return out;
  }

  erf(x) {
    const sign = x < 0 ? -1 : 1;
    const ax = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * ax);
    const y =
      1 -
      (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t +
        0.254829592) *
        t *
        Math.exp(-ax * ax));
    return sign * y;
  }

  gelu(x) {
    return 0.5 * x * (1 + this.erf(x / Math.SQRT2));
  }

  forwardFocus(ids, focusIndex = ids.length - 1) {
    if (ids.length === 0) throw new Error("Cannot run model on an empty context.");
    const contextStart = Math.max(0, focusIndex - this.blockSize + 1);
    const contextIds = ids.slice(contextStart, focusIndex + 1);
    const T = contextIds.length;
    const C = this.nEmb;
    const H = this.nHead;
    const D = this.headDim;
    const focusLocal = T - 1;

    const tok = this.weights["tok_emb.weight"];
    const pos = this.weights["pos_emb.weight"];
    const ln1 = this.weights["block.ln1.weight"];
    const qkvW = this.weights["block.attn.qkv.weight"];
    const attProjW = this.weights["block.attn.proj.weight"];
    const ln2 = this.weights["block.ln2.weight"];
    const fcW = this.weights["block.mlp.fc.weight"];
    const mlpProjW = this.weights["block.mlp.proj.weight"];
    const lnF = this.weights["ln_f.weight"];

    const xRows = [];
    const kRows = new Float32Array(T * C);
    const vRows = new Float32Array(T * C);
    let qFocus = null;

    for (let t = 0; t < T; t++) {
      const id = contextIds[t];
      const x = new Float32Array(C);
      const tokBase = id * C;
      const posBase = t * C;
      for (let c = 0; c < C; c++) x[c] = tok[tokBase + c] + pos[posBase + c];
      xRows.push(x);

      const nx = this.rmsNorm(x, ln1);
      const qkv = this.linear(nx, qkvW, 3 * C, C);
      if (t === focusLocal) qFocus = qkv.subarray(0, C);
      for (let c = 0; c < C; c++) {
        kRows[t * C + c] = qkv[C + c];
        vRows[t * C + c] = qkv[2 * C + c];
      }
    }

    const attMix = new Float32Array(C);
    const attention = Array.from({ length: H }, () => new Float32Array(T));
    const scoreScale = 1 / Math.sqrt(D);
    const kKeep = Math.min(this.topkAttn || T, T);

    for (let h = 0; h < H; h++) {
      const topIdx = [];
      const topVal = [];
      const hBase = h * D;
      for (let t = 0; t < T; t++) {
        let score = 0;
        const kBase = t * C + hBase;
        for (let d = 0; d < D; d++) score += qFocus[hBase + d] * kRows[kBase + d];
        score *= scoreScale;
        insertTopK(topIdx, topVal, t, score, kKeep);
      }

      let maxScore = -Infinity;
      for (const value of topVal) if (value > maxScore) maxScore = value;
      let denom = 0;
      for (let i = 0; i < topVal.length; i++) {
        const p = Math.exp(topVal[i] - maxScore);
        topVal[i] = p;
        denom += p;
      }

      for (let i = 0; i < topVal.length; i++) {
        const source = topIdx[i];
        const prob = topVal[i] / denom;
        attention[h][source] = prob;
        const vBase = source * C + hBase;
        for (let d = 0; d < D; d++) attMix[hBase + d] += prob * vRows[vBase + d];
      }
    }

    const attProj = this.linear(attMix, attProjW, C, C);
    const residual = new Float32Array(C);
    const xFocus = xRows[focusLocal];
    for (let c = 0; c < C; c++) residual[c] = xFocus[c] + attProj[c];

    const nx2 = this.rmsNorm(residual, ln2);
    const hidden = this.linear(nx2, fcW, this.hidden, C);
    for (let i = 0; i < hidden.length; i++) hidden[i] = this.gelu(hidden[i]);
    const mlpOut = this.linear(hidden, mlpProjW, C, this.hidden);
    for (let c = 0; c < C; c++) residual[c] += mlpOut[c];

    const final = this.rmsNorm(residual, lnF);
    const logits = this.linear(final, tok, this.vocabSize, C);
    return { logits, attention, contextStart, contextIds };
  }
}

function insertTopK(topIdx, topVal, idx, value, k) {
  let pos = topVal.length;
  while (pos > 0 && value > topVal[pos - 1]) pos--;
  if (pos >= k) return;
  topIdx.splice(pos, 0, idx);
  topVal.splice(pos, 0, value);
  if (topIdx.length > k) {
    topIdx.pop();
    topVal.pop();
  }
}

export function topPredictions(logits, tokenizer, count = 12, temperature = 1) {
  const bad = tokenizer.badIds;
  const scaled = new Float32Array(logits.length);
  let maxLogit = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    const value = bad.has(i) ? -Infinity : logits[i] / Math.max(temperature, 1e-6);
    scaled[i] = value;
    if (value > maxLogit) maxLogit = value;
  }
  let denom = 0;
  for (let i = 0; i < scaled.length; i++) {
    if (!Number.isFinite(scaled[i])) continue;
    denom += Math.exp(scaled[i] - maxLogit);
  }
  const best = [];
  for (let i = 0; i < scaled.length; i++) {
    if (!Number.isFinite(scaled[i])) continue;
    const prob = Math.exp(scaled[i] - maxLogit) / denom;
    insertPrediction(best, { id: i, token: tokenizer.display(i), prob }, count);
  }
  return best;
}

export function sampleFromTop(predictions) {
  const total = predictions.reduce((sum, pred) => sum + pred.prob, 0);
  let r = Math.random() * total;
  for (const pred of predictions) {
    r -= pred.prob;
    if (r <= 0) return pred;
  }
  return predictions[predictions.length - 1];
}

function insertPrediction(best, item, count) {
  let pos = best.length;
  while (pos > 0 && item.prob > best[pos - 1].prob) pos--;
  if (pos >= count) return;
  best.splice(pos, 0, item);
  if (best.length > count) best.pop();
}
