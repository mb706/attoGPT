import { TinyGptWeb, WordTokenizer, sampleFromTop, topPredictions } from "./model.js";

const SPECIAL_TOKENS = new Set(["<pad>", "<bos>", "<eos>", "<unk>"]);

const PRESETS = [
  "once upon a time, there was a little girl named lily.",
  "the little dog wanted to play with his red ball.",
  "mia and ben found a big box in the garden.",
  "tom was sad, but his mom had a good idea.",
  "a tiny bird saw the rain and flew home.",
];

const state = {
  model: null,
  tokenizer: null,
  ids: [],
  focus: 0,
  selected: 0,
  predictions: [],
  attention: [],
  contextStart: 0,
  temperature: 0.8,
  predCount: 14,
};

const el = {
  status: document.querySelector("#status"),
  tokenCanvas: document.querySelector("#token-canvas"),
  decoded: document.querySelector("#decoded"),
  contextMeta: document.querySelector("#context-meta"),
  promptInput: document.querySelector("#prompt-input"),
  preset: document.querySelector("#preset"),
  tokenizeBtn: document.querySelector("#tokenize-btn"),
  resetBtn: document.querySelector("#reset-btn"),
  selectedToken: document.querySelector("#selected-token"),
  editInput: document.querySelector("#edit-input"),
  suggestions: document.querySelector("#suggestions"),
  insertBeforeBtn: document.querySelector("#insert-before-btn"),
  insertAfterBtn: document.querySelector("#insert-after-btn"),
  deleteBtn: document.querySelector("#delete-btn"),
  focusBtn: document.querySelector("#focus-btn"),
  predictions: document.querySelector("#predictions"),
  top1Btn: document.querySelector("#top1-btn"),
  sampleBtn: document.querySelector("#sample-btn"),
  temp: document.querySelector("#temp"),
  tempValue: document.querySelector("#temp-value"),
  predCount: document.querySelector("#pred-count"),
};

async function main() {
  setStatus("loading model bundle...");
  const [manifest, tokenizerMeta, weights] = await Promise.all([
    fetchJson("model/manifest.json"),
    fetchJson("model/tokenizer.json"),
    fetch("model/model.bin").then((r) => {
      if (!r.ok) throw new Error(`could not fetch model.bin: ${r.status}`);
      return r.arrayBuffer();
    }),
  ]);
  state.tokenizer = new WordTokenizer(tokenizerMeta);
  state.model = new TinyGptWeb(manifest, weights);
  setupControls();
  loadPrompt(PRESETS[0]);
  setStatus(`ready - ${manifest.checkpoint.params.toLocaleString()} parameters, top-k attention ${manifest.config.topk_attn}`);
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`could not fetch ${path}: ${response.status}`);
  return response.json();
}

function setupControls() {
  for (const preset of PRESETS) {
    const option = document.createElement("option");
    option.value = preset;
    option.textContent = preset;
    el.preset.append(option);
  }
  el.preset.addEventListener("change", () => loadPrompt(el.preset.value));
  el.tokenizeBtn.addEventListener("click", () => loadPrompt(el.promptInput.value));
  el.resetBtn.addEventListener("click", () => loadPrompt(PRESETS[0]));
  el.editInput.addEventListener("input", renderSuggestions);
  el.editInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      const first = el.suggestions.querySelector("button");
      if (first) replaceSelected(Number(first.dataset.id));
    }
  });
  el.insertBeforeBtn.addEventListener("click", () => insertToken(state.selected, tokenId("the")));
  el.insertAfterBtn.addEventListener("click", () => insertToken(state.selected + 1, tokenId("the")));
  el.deleteBtn.addEventListener("click", deleteSelected);
  el.focusBtn.addEventListener("click", () => setFocus(state.selected));
  el.top1Btn.addEventListener("click", () => {
    if (state.predictions[0]) insertPrediction(state.predictions[0].id);
  });
  el.sampleBtn.addEventListener("click", () => {
    if (state.predictions.length) insertPrediction(sampleFromTop(state.predictions).id);
  });
  el.temp.addEventListener("input", () => {
    state.temperature = Number(el.temp.value);
    el.tempValue.textContent = state.temperature.toFixed(2);
    runModel();
  });
  el.predCount.addEventListener("input", () => {
    state.predCount = Number(el.predCount.value);
    runModel();
  });
}

function loadPrompt(prompt) {
  el.promptInput.value = prompt;
  state.ids = state.tokenizer.encode(prompt, true).slice(0, state.model.blockSize);
  state.focus = state.ids.length - 1;
  state.selected = state.focus;
  runModel();
}

function runModel() {
  if (!state.model || !state.ids.length) return;
  state.focus = clamp(state.focus, 0, state.ids.length - 1);
  state.selected = clamp(state.selected, 0, state.ids.length - 1);
  const result = state.model.forwardFocus(state.ids, state.focus);
  state.attention = result.attention;
  state.contextStart = result.contextStart;
  state.predictions = topPredictions(
    result.logits,
    state.tokenizer,
    state.predCount,
    state.temperature,
  );
  renderAll();
}

function renderAll() {
  renderTokens();
  renderEditor();
  renderPredictions();
  el.decoded.textContent = state.tokenizer.decode(state.ids);
  el.contextMeta.textContent = `${state.ids.length}/${state.model.blockSize} tokens - predicting after token ${state.focus}`;
}

function renderTokens() {
  el.tokenCanvas.replaceChildren();
  const maxByHead = state.attention.map((row) => Math.max(0.000001, ...row));
  for (let i = 0; i < state.ids.length; i++) {
    const id = state.ids[i];
    const card = document.createElement("button");
    card.className = "token-card";
    if (i === state.selected) card.classList.add("selected");
    if (i === state.focus) card.classList.add("focus");
    if (state.tokenizer.isSpecial(id)) card.classList.add("special");
    card.title = `token ${i}: ${state.tokenizer.display(id)}`;
    card.addEventListener("click", () => {
      state.selected = i;
      state.focus = i;
      runModel();
    });

    const word = document.createElement("span");
    word.className = "word";
    word.textContent = state.tokenizer.display(id);
    card.append(word);

    for (let h = 0; h < state.model.nHead; h++) {
      const weight = i >= state.contextStart ? state.attention[h][i - state.contextStart] ?? 0 : 0;
      const heat = document.createElement("span");
      heat.className = `heat heat-${h}`;
      heat.style.setProperty("--alpha", String(Math.min(1, weight / maxByHead[h])));
      heat.title = `head ${h}: ${weight.toFixed(4)}`;
      card.append(heat);
    }
    el.tokenCanvas.append(card);
  }
}

function renderEditor() {
  const token = state.tokenizer.display(state.ids[state.selected]);
  el.selectedToken.textContent = `token ${state.selected}: ${token}`;
  if (document.activeElement !== el.editInput) el.editInput.value = token;
  el.deleteBtn.disabled = state.ids.length <= 1;
  renderSuggestions();
}

function renderSuggestions() {
  const query = el.editInput.value.trim().toLowerCase();
  const suggestions = suggestTokens(query, 10);
  el.suggestions.replaceChildren();
  for (const item of suggestions) {
    const button = document.createElement("button");
    button.dataset.id = String(item.id);
    button.innerHTML = `<strong>${escapeHtml(item.token)}</strong><span>#${item.id} · score ${item.score.toFixed(1)}</span>`;
    button.addEventListener("click", () => replaceSelected(item.id));
    el.suggestions.append(button);
  }
}

function renderPredictions() {
  el.predictions.replaceChildren();
  for (const pred of state.predictions) {
    const button = document.createElement("button");
    button.className = "prediction";
    button.addEventListener("click", () => insertPrediction(pred.id));
    button.innerHTML = `
      <span class="pred-token">${escapeHtml(pred.token)}</span>
      <span class="pred-prob">${(100 * pred.prob).toFixed(2)}%</span>
      <span class="pred-bar"><span style="width: ${Math.max(1, pred.prob * 100)}%"></span></span>
    `;
    el.predictions.append(button);
  }
}

function suggestTokens(query, limit) {
  const cleaned = query.match(/[a-z]+|[.,!?]/)?.[0] ?? "";
  const scored = [];
  for (let id = 0; id < state.tokenizer.itos.length; id++) {
    const token = state.tokenizer.itos[id];
    if (SPECIAL_TOKENS.has(token)) continue;
    let score;
    if (!cleaned) {
      score = id / 1000;
    } else {
      const dist = editDistance(cleaned, token);
      const prefixBonus = token.startsWith(cleaned) ? -0.75 : 0;
      const exactBonus = token === cleaned ? -4 : 0;
      const lengthPenalty = Math.abs(token.length - cleaned.length) * 0.15;
      const rankPenalty = Math.log2(id + 2) * 0.05;
      score = dist + prefixBonus + exactBonus + lengthPenalty + rankPenalty;
    }
    scored.push({ id, token, score });
  }
  scored.sort((a, b) => a.score - b.score || a.id - b.id);
  return scored.slice(0, limit);
}

function editDistance(a, b) {
  const prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  const cur = new Array(b.length + 1);
  for (let i = 1; i <= a.length; i++) {
    cur[0] = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
    }
    for (let j = 0; j <= b.length; j++) prev[j] = cur[j];
  }
  return prev[b.length];
}

function replaceSelected(id) {
  state.ids[state.selected] = id;
  runModel();
}

function insertPrediction(id) {
  insertToken(state.focus + 1, id);
}

function insertToken(index, id) {
  state.ids.splice(index, 0, id);
  if (state.ids.length > state.model.blockSize) {
    state.ids.shift();
    index -= 1;
  }
  state.selected = clamp(index, 0, state.ids.length - 1);
  state.focus = state.selected;
  runModel();
}

function deleteSelected() {
  if (state.ids.length <= 1) return;
  state.ids.splice(state.selected, 1);
  state.selected = clamp(state.selected, 0, state.ids.length - 1);
  state.focus = state.selected;
  runModel();
}

function setFocus(index) {
  state.focus = clamp(index, 0, state.ids.length - 1);
  runModel();
}

function tokenId(token) {
  return state.tokenizer.stoi.get(token) ?? state.tokenizer.unkId;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (ch) => {
    const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" };
    return map[ch];
  });
}

function setStatus(message) {
  el.status.textContent = message;
}

main().catch((error) => {
  console.error(error);
  setStatus(`error: ${error.message}`);
});
