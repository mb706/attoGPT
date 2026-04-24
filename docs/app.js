import { TinyGptWeb, WordTokenizer, sampleFromTop, topPredictions } from "./model.js?v=20260424-2";

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
  tutorialBtn: document.querySelector("#tutorial-btn"),
  explainer: document.querySelector(".explainer"),
  tourCard: document.querySelector("#tour-card"),
  tourStep: document.querySelector("#tour-step"),
  tourTitle: document.querySelector("#tour-title"),
  tourBody: document.querySelector("#tour-body"),
  tourPrev: document.querySelector("#tour-prev"),
  tourNext: document.querySelector("#tour-next"),
  tourClose: document.querySelector("#tour-close"),
};

const tour = {
  active: false,
  index: 0,
  target: null,
  steps: [
    {
      selector: ".explainer",
      title: "Start with the mental model",
      body:
        "This collapsed section explains the model in plain language: tokens become vectors, attention chooses previous tokens, the MLP transforms the result, and the output is a next-token probability list.",
      before() {
        el.explainer.open = true;
      },
    },
    {
      selector: "#prompt-input",
      title: "Choose or write a context",
      body:
        "The model only understands its 4096-token vocabulary. Write ordinary lowercase-ish English here, or pick a preset above. Tokenize text converts this box into model tokens.",
    },
    {
      selector: "#token-canvas",
      title: "Read the token strip",
      body:
        "Every rounded tile is one model token. The text wraps naturally instead of scrolling sideways. Click any token to ask: what would the model predict immediately after this token?",
    },
    {
      selector: ".legend",
      title: "Compare the two heads",
      body:
        "Each token tile has two tiny heat bars. Orange is attention head 0; teal is attention head 1. A longer bar means that head used that token more strongly.",
    },
    {
      selector: '[data-tour="editor"]',
      title: "Edit individual tokens",
      body:
        "The selected token appears here. Type a replacement and choose from suggestions ranked by edit distance and token frequency. This keeps edits inside the model vocabulary.",
    },
    {
      selector: "#suggestions",
      title: "Use vocabulary suggestions",
      body:
        "Suggestions update immediately, even for one-letter words like i. Click a suggestion or press Enter to replace the selected token.",
    },
    {
      selector: "#focus-btn",
      title: "Set the prediction point",
      body:
        "Predict here makes the selected token the current focus. The heat bars and next-token list then explain the model's prediction after that token.",
    },
    {
      selector: '[data-tour="next-token"]',
      title: "Inspect next-token probabilities",
      body:
        "This list shows the model's most likely next tokens and their probabilities. Click any row to append that token after the current focus.",
    },
    {
      selector: ".generation-controls",
      title: "Generate text interactively",
      body:
        "Use top 1 for the most likely token, or Sample for a random token weighted by probability. Temperature controls how adventurous sampling is.",
    },
    {
      selector: "#decoded",
      title: "Read the natural text",
      body:
        "This is the same token sequence decoded back into plain text. The model is tiny, so it will drift and repeat, but the mechanics are visible.",
    },
  ],
};

async function main() {
  setupTourControls();
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
  setupModelControls();
  loadPrompt(PRESETS[0]);
  setStatus(`ready - ${manifest.checkpoint.params.toLocaleString()} parameters, top-k attention ${manifest.config.topk_attn}`);
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`could not fetch ${path}: ${response.status}`);
  return response.json();
}

function setupModelControls() {
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

function setupTourControls() {
  el.tutorialBtn.addEventListener("click", startTour);
  el.tourPrev.addEventListener("click", () => showTourStep(tour.index - 1));
  el.tourNext.addEventListener("click", () => showTourStep(tour.index + 1));
  el.tourClose.addEventListener("click", endTour);
  window.addEventListener("keydown", (event) => {
    if (!tour.active) return;
    if (event.key === "Escape") endTour();
    if (event.key === "ArrowRight") showTourStep(tour.index + 1);
    if (event.key === "ArrowLeft") showTourStep(tour.index - 1);
  });
  window.addEventListener("resize", () => {
    if (tour.active) positionTourCard();
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

function startTour() {
  tour.active = true;
  el.tourCard.hidden = false;
  document.body.classList.add("tour-active");
  if (!state.model) setStatus("tutorial ready - model is still loading in the background...");
  showTourStep(0);
}

function showTourStep(index) {
  if (!tour.active) return;
  if (index < 0) index = 0;
  if (index >= tour.steps.length) {
    endTour();
    return;
  }
  clearTourTarget();
  tour.index = index;
  const step = tour.steps[index];
  if (step.before) step.before();
  const target = document.querySelector(step.selector);
  tour.target = target;
  if (target) {
    target.classList.add("tour-highlight");
    target.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
  }
  el.tourStep.textContent = `Step ${index + 1} of ${tour.steps.length}`;
  el.tourTitle.textContent = step.title;
  el.tourBody.textContent = step.body;
  el.tourPrev.disabled = index === 0;
  el.tourNext.textContent = index === tour.steps.length - 1 ? "Finish" : "Next";
  setTimeout(positionTourCard, 180);
}

function positionTourCard() {
  if (!tour.active) return;
  const card = el.tourCard;
  const target = tour.target;
  const margin = 18;
  const cardRect = card.getBoundingClientRect();
  let top = window.innerHeight / 2 - cardRect.height / 2;
  let left = window.innerWidth / 2 - cardRect.width / 2;

  if (target) {
    const rect = target.getBoundingClientRect();
    const preferRight = rect.left + rect.width / 2 < window.innerWidth / 2;
    left = preferRight ? rect.right + margin : rect.left - cardRect.width - margin;
    if (left < margin || left + cardRect.width > window.innerWidth - margin) {
      left = Math.min(
        window.innerWidth - cardRect.width - margin,
        Math.max(margin, rect.left + rect.width / 2 - cardRect.width / 2),
      );
      top = rect.bottom + margin;
      if (top + cardRect.height > window.innerHeight - margin) top = rect.top - cardRect.height - margin;
    } else {
      top = rect.top + rect.height / 2 - cardRect.height / 2;
    }
  }

  card.style.left = `${Math.max(margin, Math.min(left, window.innerWidth - cardRect.width - margin))}px`;
  card.style.top = `${Math.max(margin, Math.min(top, window.innerHeight - cardRect.height - margin))}px`;
}

function endTour() {
  clearTourTarget();
  tour.active = false;
  tour.index = 0;
  el.tourCard.hidden = true;
  document.body.classList.remove("tour-active");
}

function clearTourTarget() {
  if (tour.target) tour.target.classList.remove("tour-highlight");
  tour.target = null;
}

main().catch((error) => {
  console.error(error);
  setStatus(`error: ${error.message}`);
});
