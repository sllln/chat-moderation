"""
Step 10 - Real-time chat moderation app (requirements R1 + R5).

  - Chat-like interface: every message is labeled live (clean / offensive)
    with a confidence score; offensive messages are masked.
  - Side panel: choose the model (default: quantized BERTurk) and see the
    response time of every message against the 50 ms budget.
  - Tabs in the side panel:
      Log      - response time of every message
      Compare  - sends the last message to all five models
      Reading  - the cleaned text the models actually see, and lexicon matches

Run from the project folder:  python src/app.py
"""
import html
import time

import gradio as gr
import pandas as pd

from classifiers import LexiconClassifier, load_all_classifiers
from text_cleaning import clean_full, clean_light, to_ascii

DEFAULT_MODEL = "BERTurk (quantized)"
THRESHOLD = 0.5
LATENCY_BUDGET_MS = 50
LOG_COLUMNS = ["#", "model", "ms", "within 50 ms"]

EXAMPLES = [
    "bugün maç çok güzeldi, golü izlediniz mi?",
    "s3n t4m bir 4ptalsın",
    "bu film çok sıkıcı ama kahve güzeldi",
    "harbi cahil bu tabanın seçmeni ya",
    "a.p.t.a.l herif çoooook konuşuyorsun",
]

print("Loading models ...")
CLASSIFIERS = {model.name: model for model in load_all_classifiers()}
LEXICON = next(m for m in CLASSIFIERS.values() if isinstance(m, LexiconClassifier))
for model in CLASSIFIERS.values():          # warm-up, so the first message is not slow
    model.predict_proba("merhaba")
print("Ready.")


# --- Helpers ------------------------------------------------------------------

def classify(model_name: str, text: str) -> dict:
    model = CLASSIFIERS[model_name]
    start = time.perf_counter()
    probability = model.predict_proba(text)
    ms = (time.perf_counter() - start) * 1000
    offensive = probability >= THRESHOLD
    return {
        "model": model_name,
        "precision": model.precision_label,
        "offensive": offensive,
        "confidence": probability if offensive else 1 - probability,
        "ms": ms,
    }


def mask(text: str) -> str:
    """Replace every visible character with '*' (spaces are kept)."""
    return "".join(c if c.isspace() else "*" for c in text)


def render_chat(chat: list[dict]) -> str:
    if not chat:
        return ("<div class='chat'><div class='chat-empty'>🎮 The lobby is quiet. "
                "Send a message to see the moderation at work.</div></div>")
    bubbles = []
    for number, message in enumerate(chat, start=1):
        if message["offensive"]:
            text, badge, css = mask(message["text"]), "OFFENSIVE", "offensive"
        else:
            text, badge, css = message["text"], "CLEAN", "clean"
        confidence = message["confidence"]
        bubbles.append(
            f"<div class='bubble {css}'>"
            f"<div class='avatar'>P{number}</div>"
            f"<div class='bubble-body'>"
            f"<div class='bubble-head'><span class='nick'>player_{number}</span>"
            f"<span class='badge'>{badge}</span></div>"
            f"<div class='bubble-text'>{html.escape(text)}</div>"
            f"<div class='bubble-meta'>"
            f"<span class='conf'><span class='conf-fill' style='width:{confidence:.0%}'></span></span>"
            f"{confidence:.0%} sure · {message['ms']:.1f} ms · {html.escape(message['model'])}"
            f"</div></div></div>"
        )
    return "<div class='chat'>" + "".join(reversed(bubbles)) + "</div>"


def explain(text: str) -> str:
    matches = LEXICON.matcher.find_matches(text)
    matched = ", ".join(f"`{word}` → `{entry}`" for word, entry in matches) or "none"
    return (
        f"**BERTurk sees (light cleaning):** {clean_light(text) or '—'}\n\n"
        f"**Classical models see (full cleaning):** {clean_full(text) or '—'}\n\n"
        f"**Lexicon form (ASCII):** {to_ascii(clean_full(text)) or '—'}\n\n"
        f"**Lexicon matches:** {matched}"
    )


def latency_summary(log: list[list]) -> str:
    """A small gauge: last response time compared with the 50 ms budget."""
    if not log:
        last, average, over, count = 0.0, 0.0, 0, 0
    else:
        times = [row[2] for row in log]
        last, average, count = times[-1], sum(times) / len(times), len(times)
        over = sum(t > LATENCY_BUDGET_MS for t in times)
    fill = min(last / LATENCY_BUDGET_MS, 1.0)
    state = "over" if last > LATENCY_BUDGET_MS else "ok"
    return (
        f"<div class='meter'>"
        f"<div class='meter-top'><span>LAST RESPONSE</span><b>{last:.1f} ms</b></div>"
        f"<div class='meter-bar'><div class='meter-fill {state}' style='width:{fill:.0%}'></div></div>"
        f"<div class='meter-scale'><span>0</span><span>budget {LATENCY_BUDGET_MS} ms</span></div>"
        f"<div class='meter-stats'><div><b>{average:.1f}</b><span>avg ms</span></div>"
        f"<div><b>{count}</b><span>messages</span></div>"
        f"<div><b>{over}</b><span>over budget</span></div></div>"
        f"</div>"
    )


# --- Event handlers -----------------------------------------------------------

def send(text: str, model_name: str, chat: list, log: list):
    if not text or not text.strip():
        return text, render_chat(chat), pd.DataFrame(log, columns=LOG_COLUMNS), latency_summary(log), gr.update(), chat, log
    result = classify(model_name, text)
    chat = chat + [{"text": text, **result}]
    log = log + [[len(log) + 1, model_name, round(result["ms"], 2),
                  "✅" if result["ms"] <= LATENCY_BUDGET_MS else "⚠️"]]
    return ("", render_chat(chat), pd.DataFrame(log, columns=LOG_COLUMNS),
            latency_summary(log), explain(text), chat, log)


def compare_all(chat: list) -> str:
    """Send the last message to all five models and show the answers as a compact list."""
    if not chat:
        return "<div class='compare-empty'>Send a message first.</div>"
    text = chat[-1]["text"]
    rows = []
    for name in CLASSIFIERS:
        r = classify(name, text)
        css, label = ("offensive", "OFFENSIVE") if r["offensive"] else ("clean", "CLEAN")
        rows.append(
            f"<div class='cmp-row {css}'>"
            f"<div class='cmp-name'>{html.escape(name)}<span>{r['precision']}</span></div>"
            f"<span class='badge'>{label}</span>"
            f"<div class='cmp-num'>{r['confidence']:.0%}</div>"
            f"<div class='cmp-num'>{r['ms']:.1f} ms</div>"
            f"</div>"
        )
    return "<div class='cmp'>" + "".join(rows) + "</div>"


def clear():
    return (render_chat([]), pd.DataFrame(columns=LOG_COLUMNS), latency_summary([]),
            "Send a message to see how it was read.", compare_all([]), [], [])


# --- Look and layout ----------------------------------------------------------

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.rose,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Exo 2"), "Segoe UI", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"],
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill_dark="#0a0e1a",
    background_fill_primary_dark="#111827",
    background_fill_secondary_dark="#0d1322",
    block_background_fill_dark="#111827",
    block_border_color_dark="#1f2a44",
    block_label_background_fill_dark="#111827",
    input_background_fill_dark="#0d1322",
    border_color_primary_dark="#1f2a44",
    button_primary_background_fill_dark="#e11d48",
    button_primary_background_fill_hover_dark="#be123c",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill_dark="#1a2338",
    button_secondary_background_fill_hover_dark="#24304d",
    button_secondary_text_color_dark="#e2e8f0",
    checkbox_background_color_selected_dark="#e11d48",
)

# The app always opens in dark mode (game-chat look).
FORCE_DARK = "() => { document.body.classList.add('dark'); }"

CSS = """
.gradio-container { max-width: 1280px !important; }
.hero { display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
        padding: 18px 22px; border-radius: 14px; border: 1px solid #1f2a44;
        background: linear-gradient(120deg, #1a0f24 0%, #111827 45%, #0b2a2e 100%); }
.hero h1 { margin: 0; font-size: 24px; letter-spacing: 0.5px; color: #f8fafc; }
.hero p { margin: 4px 0 0; color: #94a3b8; font-size: 14px; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chip { font-size: 12px; padding: 5px 10px; border-radius: 999px; border: 1px solid #2c3a5c;
        color: #cbd5e1; background: rgba(15, 23, 42, 0.6); }
.chip.live::before { content: "●"; color: #2dd4bf; margin-right: 6px; }

.panel-title { font-size: 12px; font-weight: 700; letter-spacing: 1.5px;
               color: #8391a7; margin-bottom: 2px; }
.input-row { align-items: stretch; }
.chat { display: flex; flex-direction: column; gap: 10px; height: 420px; overflow-y: auto;
        padding: 12px; border-radius: 12px; background: #0d1322; border: 1px solid #1f2a44; }
.chat-empty { margin: auto; color: #64748b; }
.bubble { display: flex; gap: 10px; padding: 10px 12px; border-radius: 10px;
          background: #131c30; border-left: 4px solid; }
.bubble.clean { border-left-color: #2dd4bf; }
.bubble.offensive { border-left-color: #f43f5e; background: #1f1220; }
.avatar { flex: none; width: 36px; height: 36px; border-radius: 50%; display: grid; place-items: center;
          font-size: 12px; font-weight: 700; color: #0a0e1a; background: #94a3b8; }
.bubble.clean .avatar { background: #2dd4bf; }
.bubble.offensive .avatar { background: #f43f5e; }
.bubble-body { flex: 1; min-width: 0; }
.bubble-head { display: flex; align-items: center; gap: 8px; }
.nick { font-weight: 700; color: #e2e8f0; font-size: 13px; }
.badge { font-size: 10px; font-weight: 700; letter-spacing: 1px; padding: 2px 7px; border-radius: 999px; }
.bubble.clean .badge { color: #2dd4bf; border: 1px solid #2dd4bf; }
.bubble.offensive .badge { color: #fb7185; border: 1px solid #fb7185; }
.bubble-text { font-size: 15px; color: #f1f5f9; margin: 3px 0; overflow-wrap: anywhere; }
.bubble.offensive .bubble-text { font-family: "JetBrains Mono", monospace; color: #fb7185; letter-spacing: 1px; }
.bubble-meta { display: flex; align-items: center; gap: 8px; font-size: 12px; color: #8391a7; }
.conf { width: 60px; height: 5px; border-radius: 999px; background: #24304d; overflow: hidden; }
.conf-fill { display: block; height: 100%; background: currentColor; }
.bubble.clean .conf-fill { background: #2dd4bf; }
.bubble.offensive .conf-fill { background: #f43f5e; }

.cmp { display: flex; flex-direction: column; gap: 6px; }
.compare-empty { color: #64748b; padding: 8px 2px; }
.cmp-row { display: grid; grid-template-columns: 1fr auto 48px 64px; align-items: center; gap: 10px;
           padding: 8px 10px; border-radius: 8px; background: #131c30; border-left: 3px solid; }
.cmp-row.clean { border-left-color: #2dd4bf; }
.cmp-row.offensive { border-left-color: #f43f5e; }
.cmp-row.clean .badge { color: #2dd4bf; border: 1px solid #2dd4bf; }
.cmp-row.offensive .badge { color: #fb7185; border: 1px solid #fb7185; }
.cmp-name { font-size: 13px; color: #e2e8f0; }
.cmp-name span { display: block; font-size: 11px; color: #8391a7; }
.cmp-num { font-size: 13px; color: #cbd5e1; text-align: right; font-variant-numeric: tabular-nums; }
.meter { padding: 14px; border-radius: 12px; background: #0d1322; border: 1px solid #1f2a44; }
.meter-top { display: flex; justify-content: space-between; align-items: baseline;
             font-size: 11px; letter-spacing: 1.5px; color: #8391a7; }
.meter-top b { font-size: 22px; letter-spacing: 0; color: #f8fafc; }
.meter-bar { height: 10px; border-radius: 999px; background: #1f2a44; margin: 8px 0 4px; overflow: hidden; }
.meter-fill { height: 100%; border-radius: 999px; transition: width 0.4s; }
.meter-fill.ok { background: linear-gradient(90deg, #14b8a6, #2dd4bf); }
.meter-fill.over { background: linear-gradient(90deg, #e11d48, #fb7185); }
.meter-scale { display: flex; justify-content: space-between; font-size: 11px; color: #64748b; }
.meter-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; margin-top: 12px; text-align: center; }
.meter-stats b { display: block; font-size: 18px; color: #f8fafc; }
.meter-stats span { font-size: 11px; color: #8391a7; }
"""

HEADER = f"""
<div class='hero'>
  <div>
    <h1>🛡️ Real-Time Chat Moderation</h1>
    <p>Turkish game chat · offensive messages are masked instantly</p>
  </div>
  <div class='chips'>
    <span class='chip live'>live · CPU only</span>
    <span class='chip'>default: BERTurk INT8</span>
    <span class='chip'>budget {LATENCY_BUDGET_MS} ms</span>
  </div>
</div>
"""

with gr.Blocks(title="Chat Moderation") as demo:
    gr.HTML(HEADER)
    chat_state = gr.State([])
    log_state = gr.State([])

    with gr.Row(equal_height=False):
        # Left: the chat itself
        with gr.Column(scale=3):
            gr.HTML("<div class='panel-title'>💬 LIVE CHAT</div>")
            chat_view = gr.HTML(render_chat([]))
            with gr.Row(elem_classes="input-row"):
                message_box = gr.Textbox(placeholder="Type a message and press Enter ...",
                                         show_label=False, scale=6, container=False)
                send_button = gr.Button("Send ➤", variant="primary", scale=1, min_width=90)
                clear_button = gr.Button("Clear", scale=1, min_width=70)
            gr.Examples(EXAMPLES, inputs=message_box, label="Try an example")

        # Right: one control panel
        with gr.Column(scale=2):
            gr.HTML("<div class='panel-title'>⚙️ MODEL</div>")
            model_choice = gr.Radio(list(CLASSIFIERS), value=DEFAULT_MODEL, show_label=False)
            latency_info = gr.HTML(latency_summary([]))
            with gr.Tabs():
                with gr.Tab("⏱️ Log"):
                    latency_table = gr.Dataframe(headers=LOG_COLUMNS, show_label=False,
                                                 interactive=False, max_height=260)
                with gr.Tab("🔍 Compare"):
                    compare_button = gr.Button("Run all 5 models on the last message")
                    compare_table = gr.HTML(compare_all([]))
                with gr.Tab("🔬 Reading"):
                    explanation = gr.Markdown("Send a message to see how it was read.")

    outputs = [message_box, chat_view, latency_table, latency_info, explanation, chat_state, log_state]
    inputs = [message_box, model_choice, chat_state, log_state]
    send_button.click(send, inputs, outputs)
    message_box.submit(send, inputs, outputs)
    compare_button.click(compare_all, chat_state, compare_table)
    clear_button.click(clear, None, [chat_view, latency_table, latency_info, explanation,
                                     compare_table, chat_state, log_state])

if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=THEME, css=CSS, js=FORCE_DARK)
