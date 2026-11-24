"""수강생 연결: 강사 PC 의 가상 공장을 수강생 PC 의 브라우저에서 보고 조작하게 해 주는 작은 웹 서버.

- 진행 패널을 만드는 동안 버튼 · 슬라이더 · 체크박스를 자동으로 기록해 두었다가 웹 페이지에 같은 조작부를 만들어 줌.
- 웹에서 온 조작은 줄을 세워 두었다가 가상 공장의 주 루프(pump)에서 실행함. 강사가 "조별 조작 허용"을 켠 동안에만 받음.
- 가상 공장이 기록한 파일(data/session*/live)을 내려받는 주소도 제공함 → 수강생은 `./scripts/factory fetch <강사 PC 주소>`.
- 같은 내부망 안에서만 쓰는 용도임. 로그인 · 암호화가 없으므로 외부망에 열어 두지 않음.
"""
import json
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

SKIP_BUTTONS = ("교시", "카메라 창")          # 웹에 내보내지 않는 버튼 (강사 화면 구성용)


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))       # 실제로 보내지는 않음 · 바깥으로 나가는 주소만 알아냄
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class WebLink:
    def __init__(self, session, title, live_dir, port=8080):
        self.session, self.title, self.live_dir, self.port = session, title, Path(live_dir), port
        self.controls, self.labels = [], []
        self.group, self._last_label, self._wait_label = "", "", None
        self.queue, self.log = deque(), deque(maxlen=12)
        self.allow = False
        self._snap, self._n, self.server = b"{}", 0, None
        self._orig = {}
        self.url = f"http://{lan_ip()}:{port}"

    # ---- 패널을 만드는 동안 위젯을 기록 -------------------------------------------
    def record(self, ui):
        """with web.record(ui): 안에서 만든 버튼 · 슬라이더 · 체크박스 · 라벨을 기록함."""
        link = self

        class _Rec:
            def __enter__(self_):
                self_.orig = {n: getattr(ui, n) for n in ("Button", "FloatSlider", "CheckBox", "Label", "CollapsableFrame")}
                o = self_.orig
                link._orig = o

                def frame(title="", *a, **k):
                    link.group = title
                    return o["CollapsableFrame"](title, *a, **k)

                def label(text="", *a, **k):
                    w = o["Label"](text, *a, **k)
                    link._last_label = text
                    if link._wait_label is not None and text:
                        link._wait_label["label"], link._wait_label = text, None
                    link.labels.append(dict(w=w, first=text, group=link.group))
                    return w

                def button(text="", *a, **k):
                    w = o["Button"](text, *a, **k)
                    if k.get("clicked_fn") and not any(s in text for s in SKIP_BUTTONS):
                        link.controls.append(dict(kind="button", group=link.group, label=text, w=w, fn=k["clicked_fn"]))
                    return w

                def slider(*a, **k):
                    w = o["FloatSlider"](*a, **k)
                    link.controls.append(dict(kind="slider", group=link.group, label=link._last_label, w=w,
                                              lo=float(k.get("min", 0)), hi=float(k.get("max", 1)), step=float(k.get("step", 0.1))))
                    return w

                def check(*a, **k):
                    w = o["CheckBox"](*a, **k)
                    c = dict(kind="check", group=link.group, label="", w=w)
                    link.controls.append(c)
                    link._wait_label = c
                    return w

                for n, f in (("CollapsableFrame", frame), ("Label", label), ("Button", button), ("FloatSlider", slider), ("CheckBox", check)):
                    setattr(ui, n, f)

            def __exit__(self_, *exc):
                for n, f in self_.orig.items():
                    setattr(ui, n, f)
                link._orig = {}
                for i, c in enumerate(link.controls):
                    c["key"] = f"c{i}"
        return _Rec()

    def build_section(self, ui):
        """강사 패널에 넣는 '수강생 연결' 구역."""
        o = getattr(self, "_orig", {})                    # 이 구역의 위젯은 웹에 내보내지 않음
        Frame, Label, Check = (o.get(n, getattr(ui, n)) for n in ("CollapsableFrame", "Label", "CheckBox"))
        with Frame("수강생 연결: 브라우저 조작 · 데이터 받기", height=0, collapsed=False):
            with ui.VStack(spacing=3, height=0):
                self.url_lb = Label(f"주소  {self.url}", height=24, style={"font_size": 18, "color": 0xFFFFDD66})
                with ui.HStack(height=24, spacing=8):
                    cb = Check(width=22)
                    cb.model.add_value_changed_fn(lambda m: setattr(self, "allow", m.get_value_as_bool()))
                    Label("조별 조작 허용 (꺼 두면 보기 · 데이터 받기만 됨)")
                self.allow_cb = cb
                self.log_lb = Label("", height=58, word_wrap=True, style={"color": 0xFFAAAAAA, "font_size": 14})

    # ---- 주 루프에서 부름 ------------------------------------------------------
    def pump(self):
        while self.queue:
            key, value, who = self.queue.popleft()
            c = next((c for c in self.controls if c["key"] == key), None)
            if c is None:
                continue
            try:
                if c["kind"] == "button":
                    c["fn"]()
                    text = c["label"]
                elif c["kind"] == "slider":
                    v = min(max(float(value), c["lo"]), c["hi"])
                    c["w"].model.set_value(v)
                    text = f"{c['label']} → {v:g}"
                else:
                    c["w"].model.set_value(bool(value))
                    text = f"{c['label']} {'켬' if value else '끔'}"
                self.log.append(f"{time.strftime('%H:%M:%S')}  {who or '?'}: {text}")
            except Exception as e:  # noqa: BLE001  수업 중에 웹 조작 하나 때문에 공장이 멈추면 안 됨
                self.log.append(f"{time.strftime('%H:%M:%S')}  오류: {c['label']} ({e})")
        self._n += 1
        if self._n % 15 == 0:
            self._snapshot()

    def _snapshot(self):
        groups, order = {}, []

        def grp(name):
            if name not in groups:
                groups[name] = dict(title=name, controls=[], texts=[])
                order.append(name)
            return groups[name]

        for c in self.controls:
            if c["group"].startswith("화면") and "실행" not in c["label"]:     # 화면 전환 · 배속은 강사만
                continue
            d = dict(key=c["key"], kind=c["kind"], label=c["label"])
            if c["kind"] == "slider":
                d.update(lo=c["lo"], hi=c["hi"], step=c["step"], value=round(c["w"].model.get_value_as_float(), 3))
            elif c["kind"] == "check":
                d["value"] = bool(c["w"].model.get_value_as_bool())
            grp(c["group"])["controls"].append(d)
        for lb in self.labels:                        # 실행 중에 내용이 바뀌는 라벨만 (값 · 상태 표시)
            text = lb["w"].text
            if text and text != lb["first"] or (text and lb.get("dyn")):
                lb["dyn"] = True
                texts = grp(lb["group"])["texts"]
                if texts and len(text) < 9 and len(texts[-1]) < 60 and texts[-1][-1:].isdigit():   # 표의 숫자 칸은 한 줄로
                    texts[-1] += "   " + text
                else:
                    texts.append(text)
        self._snap = json.dumps(dict(session=self.session, title=self.title, allow=self.allow, log=list(self.log),
                                     groups=[groups[n] for n in order], files=self.files()), ensure_ascii=False).encode()
        if hasattr(self, "log_lb"):
            self.log_lb.text = "\n".join(list(self.log)[-3:]) or "아직 조작 없음"

    def files(self):
        if not self.live_dir.exists():
            return []
        fs = sorted((f for f in self.live_dir.iterdir() if f.is_file()), key=lambda f: f.stat().st_mtime, reverse=True)
        return [dict(name=f.name, kb=round(f.stat().st_size / 1024, 1), time=time.strftime("%H:%M:%S", time.localtime(f.stat().st_mtime)))
                for f in fs[:8]]

    # ---- 웹 서버 -------------------------------------------------------------
    def start(self):
        link = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(self, code, body, ctype="application/json; charset=utf-8", extra=()):
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for k, v in extra:
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = unquote(self.path.split("?")[0])
                if path == "/":
                    self.send(200, PAGE.encode(), "text/html; charset=utf-8")
                elif path == "/api/state":
                    self.send(200, link._snap)
                elif path.startswith("/live/"):
                    f = link.live_dir / Path(path[6:]).name          # 폴더 밖의 파일은 내주지 않음
                    if f.is_file():
                        self.send(200, f.read_bytes(), "text/csv; charset=utf-8",
                                  [("Content-Disposition", f'attachment; filename="{f.name}"')])
                    else:
                        self.send(404, b'{"error": "no file"}')
                else:
                    self.send(404, b'{"error": "not found"}')

            def do_POST(self):
                if self.path != "/api/cmd":
                    return self.send(404, b"{}")
                try:
                    d = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 4096)))
                except (ValueError, TypeError):
                    return self.send(400, b'{"error": "bad request"}')
                if not link.allow:
                    return self.send(403, json.dumps(dict(error="강사가 조작을 허용하지 않았습니다"), ensure_ascii=False).encode())
                if len(link.queue) < 50:
                    link.queue.append((str(d.get("key", "")), d.get("value"), str(d.get("who", ""))[:12]))
                self.send(200, b'{"ok": true}')

        try:
            self.server = ThreadingHTTPServer(("0.0.0.0", self.port), Handler)
        except OSError as e:
            self.url = f"웹 서버를 열지 못함 (포트 {self.port}: {e})"
            return False
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return True

    def close(self):
        if self.server:
            self.server.shutdown()


PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>가상 공장 조작</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--ink:#1d2330;--sub:#667085;--line:#dfe3ea;--accent:#1f6feb;--ok:#1a7f37;--warn:#b54708}
@media(prefers-color-scheme:dark){:root{--bg:#12151c;--card:#1b2029;--ink:#e8ebf1;--sub:#98a2b3;--line:#2c3340;--accent:#58a6ff;--ok:#3fb950;--warn:#f0883e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 "Noto Sans KR","Malgun Gothic",sans-serif}
header{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:12px 16px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:2}
h1{font-size:18px;margin:0;flex:1 1 220px}.pill{padding:4px 12px;border-radius:99px;font-size:14px;font-weight:700}
.on{background:color-mix(in srgb,var(--ok) 18%,transparent);color:var(--ok)}.off{background:color-mix(in srgb,var(--warn) 18%,transparent);color:var(--warn)}
#who{width:92px;padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);font:inherit}
main{padding:16px;display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));max-width:1500px;margin:0 auto}
section{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
h2{font-size:15px;margin:0 0 10px;color:var(--sub)}
.texts{font-size:14px;background:var(--bg);border-radius:8px;padding:8px 10px;margin-bottom:10px;white-space:pre-wrap;font-variant-numeric:tabular-nums}
.btns{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px}
button{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--bg);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--accent)}button:active{background:var(--accent);color:#fff}
.row{display:grid;grid-template-columns:minmax(110px,38%) 1fr 62px;gap:10px;align-items:center;margin:6px 0;font-size:15px}
.row output{text-align:right;font-variant-numeric:tabular-nums}input[type=range]{width:100%;accent-color:var(--accent)}
.chk{display:flex;gap:10px;align-items:center;margin:6px 0;font-size:15px}.chk input{width:20px;height:20px;accent-color:var(--accent)}
body.locked button,body.locked input[type=range],body.locked .chk input{opacity:.45;pointer-events:none}
#log,#files{font-size:14px;color:var(--sub);white-space:pre-wrap}a{color:var(--accent)}#msg{color:var(--warn);font-size:14px;min-height:20px}
</style></head><body class="locked">
<header><h1 id="title">가상 공장</h1><label>조 <input id="who" placeholder="예: 3조"></label><span id="allow" class="pill off">보기 전용</span></header>
<main id="main"></main>
<main><section><h2>최근 조작</h2><div id="log">-</div><div id="msg"></div></section>
<section><h2>가상 공장이 기록한 데이터</h2><div id="files">-</div></section></main>
<script>
const $=s=>document.querySelector(s), who=$("#who"); let built=false, busy={};
who.value=localStorage.getItem("who")||""; who.oninput=()=>localStorage.setItem("who",who.value);
async function send(key,value){const r=await fetch("/api/cmd",{method:"POST",body:JSON.stringify({key,value,who:who.value})});
  $("#msg").textContent=r.ok?"":(await r.json()).error||"보내지 못함";}
function build(s){const m=$("#main");m.innerHTML="";for(const g of s.groups){if(!g.controls.length&&!g.texts.length)continue;
  const sec=document.createElement("section");sec.innerHTML=`<h2></h2><div class="texts"></div><div class="btns"></div>`;sec.querySelector("h2").textContent=g.title;
  sec.dataset.g=g.title;const btns=sec.querySelector(".btns");
  for(const c of g.controls){if(c.kind=="button"){const b=document.createElement("button");b.textContent=c.label;b.onclick=()=>send(c.key);btns.append(b);}
    else if(c.kind=="slider"){const d=document.createElement("div");d.className="row";d.innerHTML=`<span></span><input type="range"><output></output>`;
      d.firstChild.textContent=c.label;const i=d.querySelector("input"),o=d.querySelector("output");i.min=c.lo;i.max=c.hi;i.step=c.step;i.value=c.value;o.textContent=c.value;i.id=c.key;
      i.oninput=()=>{o.textContent=i.value;busy[c.key]=Date.now()};i.onchange=()=>send(c.key,+i.value);sec.append(d);}
    else{const l=document.createElement("label");l.className="chk";l.innerHTML=`<input type="checkbox"><span></span>`;l.lastChild.textContent=c.label;
      const i=l.firstChild;i.id=c.key;i.checked=c.value;i.onchange=()=>send(c.key,i.checked);sec.append(l);}}
  m.append(sec);}built=true;}
async function tick(){try{const s=await(await fetch("/api/state")).json();if(!s.groups)return;
  $("#title").textContent=s.title;document.body.classList.toggle("locked",!s.allow);
  const a=$("#allow");a.textContent=s.allow?"조작 허용":"보기 전용";a.className="pill "+(s.allow?"on":"off");
  if(!built)build(s);
  for(const g of s.groups){const sec=[...document.querySelectorAll("section")].find(e=>e.dataset.g==g.title);if(!sec)continue;
    const t=sec.querySelector(".texts");t.textContent=g.texts.join("\\n");t.style.display=g.texts.length?"":"none";
    for(const c of g.controls){const i=document.getElementById(c.key);if(!i||Date.now()-(busy[c.key]||0)<1500)continue;
      if(c.kind=="slider"){i.value=c.value;i.parentNode.querySelector("output").textContent=c.value;}else if(c.kind=="check")i.checked=c.value;}}
  $("#log").textContent=s.log.length?s.log.slice().reverse().join("\\n"):"-";
  const f=$("#files");f.innerHTML="";if(!s.files.length)f.textContent="아직 없음";
  for(const x of s.files){const d=document.createElement("div");const l=document.createElement("a");l.href="/live/"+encodeURIComponent(x.name);l.textContent=x.name;
    d.append(l,`  ${x.kb} KB · ${x.time}`);f.append(d);}
}catch(e){$("#msg").textContent="강사 PC 에 연결되지 않음";}}
tick();setInterval(tick,1000);
</script></body></html>
"""
