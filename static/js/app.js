"use strict";

const PATHS = {
  home: "m3 10 9-7 9 7v10a1 1 0 0 1-1 1h-5v-7H9v7H4a1 1 0 0 1-1-1Z",
  grid: "M3 3h7v7H3zM14 3h7v7h-7zM3 14h7v7H3zM14 14h7v7h-7z",
  spark:
    "m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5ZM20 2v4m-2-2h4",
  graph: "M4 3v17h17M7 15c3 0 3-8 6-8s3 8 7 3",
  book: "M12 5C8 2 5 3 3 4v15c3-1 6-1 9 1 3-2 6-2 9-1V4c-2-1-5-2-9 1Zm0 0v15",
  formula: "M19 4H6l7 8-7 8h13M19 4v3m0 10v3",
  quiz: "m9 12 2 2 4-4M9 4h6l1 2h4v15H4V6h4z",
  theorem: "M12 3 2 21h20ZM12 3v18M7 12h10",
  people:
    "M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2M19 8a3 3 0 0 1 0 6m3 7v-2a4 4 0 0 0-3-4M13 7a4 4 0 1 1-8 0 4 4 0 0 1 8 0",
  projects: "M9 18h6M10 21h4M8 14a7 7 0 1 1 8 0c-1 1-1 2-1 2H9s0-1-1-2Z",
  cap: "m2 9 10-6 10 6-10 6ZM6 12v6c4 3 8 3 12 0v-6M22 9v8",
  bookmark: "M6 3h12v18l-6-4-6 4Z",
  "arrow-right": "M4 12h16m-6-6 6 6-6 6",
  "arrow-up-right": "M6 18 18 6M6 6h12v12",
  layers: "m12 3 10 6-10 6L2 9Zm-10 11 10 6 10-6M2 18l10 6 10-6",
  chevrons: "m9 8 3-3 3 3m-6 8 3 3 3-3",
  search: "M21 21l-5-5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0",
  settings:
    "m9 3-1 3-3 1-2 4 2 2v3l4 3 3-1 3 1 4-3v-3l2-2-2-4-3-1-1-3Zm6 9a3 3 0 1 1-6 0 3 3 0 0 1 6 0",
  help: "M9 9a3 3 0 0 1 6 0c0 2-3 2-3 5m0 3h.01M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0",
  close: "m6 6 12 12M6 18 18 6",
  menu: "M4 6h16M4 12h16M4 18h16",
  check: "m5 12 4 4L19 6",
  calendar: "M8 2v4m8-4v4M3 10h18M4 4h16v17H4Z",
  clock: "M12 7v5l3 2M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0",
  sun: "M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5M17 12a5 5 0 1 1-10 0 5 5 0 0 1 10 0",
  bolt: "m13 2-9 12h7l-1 8 10-13h-8Z",
  send: "m22 2-7 20-4-9-9-4Zm0 0L11 13",
  attach: "m8 13 6-6a3 3 0 0 1 4 4l-8 8a5 5 0 0 1-7-7l9-9a6 6 0 0 1 9 9l-9 9",
  plus: "M12 5v14M5 12h14",
  image: "M3 3h18v18H3ZM3 16l5-5 5 5 3-3 5 5M16 7h.01",
  trash: "M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7m4-7v7",
  download: "M12 3v12m-5-5 5 5 5-5M4 16v5h16v-5",
  copy: "M9 9h12v12H9ZM5 15H3V3h12v2",
  info: "M12 11v6m0-10h.01M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0",
  refresh: "M20 7a9 9 0 1 0 1 9M20 2v6h-6",
  leaf: "M20 3c-15-2-21 15-9 17 8 1 10-10 9-17ZM4 21 16 9",
  target:
    "M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Zm-4 0a5 5 0 1 1-10 0 5 5 0 0 1 10 0Zm-5 0h.01",
  history: "M3 4v5h5M3 9a9 9 0 1 1 0 7m9-9v5l4 2",
  zoom: "M21 21l-5-5M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0M6 10h8m-4-4v8",
};
const icon = (name, cls = "") =>
  `<svg class="icon ${cls}" viewBox="0 0 24 24" aria-hidden="true"><path d="${PATHS[name] || PATHS.spark}"/></svg>`;
const escapeHTML = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const uid = () =>
  globalThis.crypto?.randomUUID?.() ||
  Date.now().toString(36) + Math.random().toString(36).slice(2);
const TOOLS = {
  ask: {
    title: "AI Tutor",
    nav: "AI Tutor",
    icon: "spark",
    description: "A patient guide for every problem.",
    subtitle: "A fresh perspective, one question at a time.",
    color: "#9768d3",
    bg: "#f0e8fb",
    endpoint: "chat",
  },
  graph: {
    title: "Graph Plotter",
    nav: "Graph Plotter",
    icon: "graph",
    description: "See the shape behind the equation.",
    subtitle: "Bring your functions to life.",
    color: "#6c9f8f",
    bg: "#eaf4ef",
    endpoint: "graph",
  },
  pyqs: {
    title: "Question Bank",
    nav: "Question Bank",
    icon: "book",
    description: "Practice the questions that matter.",
    subtitle: "A little practice. A lot more confidence.",
    color: "#c59a59",
    bg: "#fbf2e2",
    endpoint: "pyq/load",
  },
  formula: {
    title: "Formula Sheets",
    nav: "Formula Sheets",
    icon: "formula",
    description: "Your essentials, all in one place.",
    subtitle: "The right formula, right when you need it.",
    color: "#769dbb",
    bg: "#eaf2f9",
    endpoint: "formula",
  },
  quiz: {
    title: "Quick Quiz",
    nav: "Quick Quiz",
    icon: "quiz",
    description: "Give your knowledge a little workout.",
    subtitle: "Make a little time for a new challenge.",
    color: "#c68e9d",
    bg: "#f9edf1",
    endpoint: "quiz/generate",
  },
  theorem: {
    title: "Theorem Explorer",
    nav: "Theorem Explorer",
    icon: "theorem",
    description: "Understand the why behind the what.",
    subtitle: "Follow an idea all the way to its proof.",
    color: "#a590cd",
    bg: "#f0ecf8",
    endpoint: "theorem/prove",
  },
  mathematician: {
    title: "Great Minds",
    nav: "Great Minds",
    icon: "people",
    description: "Meet the minds that changed mathematics.",
    subtitle: "Get to know the people behind the possibilities.",
    color: "#bca174",
    bg: "#f7f2e7",
    endpoint: "mathematician",
  },
  projects: {
    title: "Project Lab",
    nav: "Project Lab",
    icon: "projects",
    description: "Turn a spark of curiosity into a project.",
    subtitle: "Your next idea is waiting to happen.",
    color: "#6ea7a6",
    bg: "#eaf5f4",
    endpoint: "projects/generate",
  },
};
function readStorage(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem("mathsphere." + key)) ?? fallback;
  } catch {
    return fallback;
  }
}
let preferences = readStorage("preferences", { name: "", exam: "curiosity" });
let chats = readStorage("chats", []),
  saved = readStorage("saved", []),
  activity = readStorage("activity", []);
if (!Array.isArray(chats)) chats = [];
if (!Array.isArray(saved)) saved = [];
if (!Array.isArray(activity)) activity = [];
let route = "dashboard",
  health = null,
  healthStatus = "checking",
  activeChat = null,
  chatBusy = false,
  attachment = null,
  graphData = null,
  topicFilter = "All topics";
let preset = {};
const apiBaseUrl = String(window.MATHSPHERE_CONFIG?.apiBaseUrl || "").replace(
  /\/+$/,
  "",
);
const apiUrl = (path) => `${apiBaseUrl}/api/${path}`;

function offlineNotice() {
  return `<div class="ai-notice">${icon("info")}<span>Your learning services are currently unavailable. Your saved library and daily challenge are still here to explore.</span></div>`;
}
function persist(key, value) {
  try {
    localStorage.setItem("mathsphere." + key, JSON.stringify(value));
    return true;
  } catch {
    toast(
      "Your browser storage is full. Download your work to keep a copy.",
      "error",
    );
    return false;
  }
}
function hydrate(root = document) {
  $$("[data-icon]", root).forEach((el) => {
    el.innerHTML = icon(el.dataset.icon);
  });
}
function toolIcon(key) {
  const t = TOOLS[key];
  return `<span class="tool-icon" style="--tool-color:${t.color};--tool-bg:${t.bg}">${icon(t.icon)}</span>`;
}
function toast(message, type = "") {
  const el = document.createElement("div");
  el.className = "toast " + type;
  el.textContent = message;
  $("#toast-region").append(el);
  setTimeout(() => el.remove(), 4500);
}
function recordActivity(key, title) {
  activity = [
    { id: uid(), key, title, date: new Date().toISOString() },
    ...activity,
  ].slice(0, 20);
  persist("activity", activity);
}
function humanDate(value) {
  return new Date(value).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}
function navHTML(key, label, ico, tag = "") {
  return `<a href="#${key}" class="nav-item" data-nav="${key}">${icon(ico)}<span>${label}</span>${tag ? `<span class="nav-tag">${tag}</span>` : ""}</a>`;
}
function initNavigation() {
  $("#navigation").innerHTML =
    navHTML("dashboard", "Overview", "grid") +
    Object.entries(TOOLS)
      .map(([key, t]) => navHTML(key, t.nav, t.icon, key === "ask" ? "AI" : ""))
      .join("") +
    '<div class="nav-group-separator"></div><div class="nav-label">YOUR LEARNING</div>' +
    navHTML("exams", "Exam Preparation", "cap") +
    navHTML("library", "Saved Library", "bookmark");
  updateProfile();
}
function updateProfile() {
  const name = preferences.name?.trim();
  $("#profile-name").textContent = name || "Your learning space";
  $$(".profile-avatar,.top-avatar").forEach(
    (el) => (el.textContent = name ? name.charAt(0).toUpperCase() : "Y"),
  );
}
function setSidebar(open) {
  const mobile = window.matchMedia("(max-width:650px)").matches;
  const sidebar = $("#sidebar");
  sidebar.classList.toggle("open", open);
  $("#sidebar-backdrop").classList.toggle("open", open);
  $("#menu-button").setAttribute("aria-expanded", String(open));
  sidebar.inert = mobile && !open;
  document.body.style.overflow = mobile && open ? "hidden" : "";
  if (!open && mobile && sidebar.contains(document.activeElement))
    $("#menu-button").focus();
}
window
  .matchMedia("(max-width:650px)")
  .addEventListener("change", () => setSidebar(false));
function go(key, values = {}) {
  preset = values;
  if (location.hash === "#" + key) renderPage(key);
  else location.hash = key;
  setSidebar(false);
}
function renderPage(key) {
  const valid =
    TOOLS[key] || ["dashboard", "library", "exams", "toolkit"].includes(key);
  route = valid ? key : "dashboard";
  $$(".nav-item").forEach((el) => {
    const active = el.dataset.nav === route;
    el.classList.toggle("active", active);
    if (active) el.setAttribute("aria-current", "page");
    else el.removeAttribute("aria-current");
  });
  const title =
    TOOLS[route]?.title ||
    {
      dashboard: "Overview",
      library: "Saved Library",
      exams: "Exam Preparation",
      toolkit: "Your Toolkit",
    }[route];
  $("#breadcrumb-title").textContent = title;
  document.title = `${title} — MathSphere`;
  const main = $("#main");
  main.innerHTML = `<div class="page-view">${route === "dashboard" ? dashboard() : route === "ask" ? chatView() : route === "library" ? libraryView() : route === "exams" ? examsView() : route === "toolkit" ? toolkitView() : toolView(route)}</div>`;
  if (route === "graph") graphData = null;
  hydrate(main);
  renderMath(main);
  window.scrollTo({ top: 0, behavior: "instant" });
  setSidebar(false);
  if (route === "ask") setupChat();
  if (route === "exams")
    loadExam(preferences.exam === "curiosity" ? "jam" : preferences.exam);
  preset = {};
}
function heroArt() {
  return `<svg class="hero-art" viewBox="0 0 410 290" fill="none" aria-hidden="true"><defs><pattern id="hero-grid" width="21" height="21" patternUnits="userSpaceOnUse"><path d="M21 0H0v21" stroke="#d8ceec" stroke-width=".55"/></pattern><linearGradient id="sphere" x1="96" y1="70" x2="265" y2="230" gradientUnits="userSpaceOnUse"><stop stop-color="#f5f0ff"/><stop offset="1" stop-color="#d6c6ed"/></linearGradient><linearGradient id="orb" x1="95" y1="50" x2="300" y2="260"><stop stop-color="#ac8ada"/><stop offset="1" stop-color="#e2d5f4"/></linearGradient></defs><circle cx="223" cy="151" r="113" stroke="#e1d7ef" stroke-dasharray="3 6"/><circle cx="223" cy="151" r="91" fill="url(#sphere)" fill-opacity=".4" stroke="#cab4e7" stroke-width=".7"/><ellipse cx="223" cy="151" rx="91" ry="32" stroke="#c6b0e0" stroke-width=".65" transform="rotate(-28 223 151)"/><ellipse cx="223" cy="151" rx="39" ry="91" stroke="#c6b0e0" stroke-width=".65" transform="rotate(-28 223 151)"/><ellipse cx="223" cy="151" rx="76" ry="91" stroke="#c6b0e0" stroke-width=".65" transform="rotate(-28 223 151)"/><path d="m143 193 160-84m-150 9 137 82" stroke="#c6b0e0" stroke-width=".65"/><g transform="rotate(-8 204 144)"><rect x="123" y="79" width="172" height="136" rx="11" fill="#fff" fill-opacity=".91" stroke="#e3d8f1"/><rect x="134" y="91" width="150" height="112" rx="4" fill="url(#hero-grid)"/><path d="M145 164h128M184 98v96" stroke="#c2afd9" stroke-width=".8"/><path d="m269 161 4 3-4 3m-88-65 3-4 3 4" stroke="#c2afd9" stroke-width=".8"/><path d="M146 115c17 12 22 55 43 55s31-53 42-56 27 50 43 66" stroke="#9972cd" stroke-width="2.5" stroke-linecap="round"/><circle cx="230" cy="115" r="4" fill="#9972cd" stroke="white" stroke-width="2"/><text x="246" y="106" fill="#a58bbf" font-size="8" font-family="Georgia,serif">f(x)</text></g><g transform="rotate(11 305 75)"><rect x="281" y="49" width="48" height="49" rx="12" fill="#faf7ff" stroke="#e0d4ee"/><text x="295" y="82" fill="#9c7bc5" font-size="29" font-family="Georgia,serif">∫</text></g><g transform="rotate(-15 110 186)"><rect x="87" y="165" width="44" height="44" rx="11" fill="#f8f4fe" stroke="#dfd1ef"/><text x="98" y="195" fill="#9b7bc4" font-size="26" font-family="Georgia,serif">π</text></g><circle cx="171" cy="45" r="5" fill="#c5b1e0"/><circle cx="324" cy="201" r="4" fill="#bea2dc"/><circle cx="126" cy="107" r="2.5" fill="#bca2d8"/><path d="m333 132 2-7 2 7 7 2-7 2-2 7-2-7-7-2ZM155 235l1.5-5 1.5 5 5 1.5-5 1.5-1.5 5-1.5-5-5-1.5Z" fill="#b29ace"/><text x="258" y="247" fill="#b59bce" font-family="Georgia,serif" font-size="17" transform="rotate(9 258 247)">x² + y²</text><path d="M81 76h8m-4-4v8" stroke="#baa2d4"/><path d="M349 243h7m-3.5-3.5v7" stroke="#baa2d4"/></svg>`;
}
function toolCard(key) {
  const t = TOOLS[key];
  return `<button class="tool-card" data-route="${key}"><span class="tool-top">${toolIcon(key)}<span class="tool-arrow">${icon("arrow-up-right")}</span></span><h3>${t.title}</h3><p>${t.description}</p></button>`;
}
const LESSONS = [
  {
    title: "The beauty of derivatives",
    category: "Calculus",
    type: "Guided exploration",
    time: "5 min read",
    key: "ask",
    prompt:
      "Help me understand derivatives intuitively, using a simple real-world example and a short practice question.",
    art: "curve",
    bg: "#eeeaf8",
  },
  {
    title: "Matrices, made simple",
    category: "Linear Algebra",
    type: "Build your intuition",
    time: "6 min read",
    key: "ask",
    prompt:
      "Teach me how matrix multiplication works with a simple 2 by 2 example and a practice problem.",
    art: "matrix",
    bg: "#eaf2ee",
  },
  {
    title: "A closer look at infinity",
    category: "Analysis",
    type: "A different perspective",
    time: "4 min read",
    key: "theorem",
    prompt: "There are infinitely many prime numbers",
    art: "infinity",
    bg: "#f8f1e6",
  },
  {
    title: "See a sine wave come to life",
    category: "Calculus",
    type: "Interactive exploration",
    time: "Try it yourself",
    key: "graph",
    prompt: "sin(x)",
    art: "wave",
    bg: "#eaf2f9",
  },
  {
    title: "When vectors meet geometry",
    category: "Linear Algebra",
    type: "Proof, step by step",
    time: "Explore a theorem",
    key: "theorem",
    prompt: "Cauchy-Schwarz inequality",
    art: "matrix",
    bg: "#eeeaf8",
  },
  {
    title: "Limits without the mystery",
    category: "Analysis",
    type: "Guided exploration",
    time: "5 min read",
    key: "ask",
    prompt:
      "Explain the epsilon-delta definition of a limit with an intuitive example.",
    art: "curve",
    bg: "#f8f1e6",
  },
];
function lessonArt(type) {
  const paths = {
    curve:
      '<path d="M12 48V12m-4 32h48" stroke="#c2b2df"/><path d="M14 39c12 1 20-8 34-25" stroke="#ac90d0" stroke-width="1.5"/><path d="m23 40 28-25" stroke="#c9b8e0" stroke-width="1" stroke-dasharray="2 2"/><circle cx="33" cy="31" r="2" fill="#9e7bc6"/>',
    matrix:
      '<path d="M18 18h-4v28h4m30-28h4v28h-4" stroke="#83ab99" stroke-width="1.4"/><g fill="#8eac9d" font-family="Georgia" font-size="10"><text x="23" y="27">a</text><text x="39" y="27">b</text><text x="23" y="41">c</text><text x="39" y="41">d</text></g>',
    infinity:
      '<path d="M33 32c-9-17-25-12-23 0s16 13 23 0c9-17 25-12 23 0S40 45 33 32Z" stroke="#c3ac80" stroke-width="1.5"/>',
    wave: '<path d="M8 32h50M32 8v46" stroke="#a4bdd0" stroke-width=".6"/><path d="M8 32c7-25 15-25 23 0s18 25 27 0" stroke="#80a2be" stroke-width="1.5"/>',
  };
  return `<svg viewBox="0 0 66 64" fill="none" aria-hidden="true">${paths[type]}</svg>`;
}
function lessonsHTML() {
  return LESSONS.filter((l, i) =>
    topicFilter === "All topics" ? i < 3 : l.category === topicFilter,
  )
    .map(
      (l) =>
        `<button class="learning-card" data-lesson="${LESSONS.indexOf(l)}"><span class="lesson-art" style="background:${l.bg}">${lessonArt(l.art)}</span><span class="learning-copy"><span class="overline">${l.category}</span><h3>${l.title}</h3><span class="lesson-meta"><span>${icon("book")}${l.type}</span><span>·</span><span>${l.time}</span></span></span>${icon("arrow-right")}</button>`,
    )
    .join("");
}
function dashboard() {
  const name = preferences.name?.trim();
  return `<section class="page-intro"><div><h1>${name ? `Welcome back, ${escapeHTML(name)}` : "A good day to be curious."}<span class="greeting-symbol">☼</span></h1><p>Your next breakthrough starts with a little exploration.</p></div><span class="date-pill">${icon("calendar")}${new Date().toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric" })}</span></section><section class="hero"><div class="hero-copy"><span class="eyebrow">A LITTLE CURIOSITY. INFINITE POSSIBILITIES.</span><h2>Big ideas start with<br><span>a simple question.</span></h2><p>Untangle a tricky concept. Follow a new idea.<br>Make mathematics a little more yours.</p><div class="hero-cta-row"><button class="button primary" data-route="ask">Let's explore ${icon("arrow-right")}</button><span class="hero-note">${icon("spark")} A space to figure things out</span></div></div>${heroArt()}</section><section><div class="section-heading"><h2>What will you discover today?</h2><button class="text-button" data-route="toolkit">Explore all tools ${icon("arrow-right")}</button></div><div class="tools-grid">${["ask", "graph", "pyqs", "formula"].map(toolCard).join("")}</div></section><div class="dashboard-lower"><section><div class="section-heading"><div><h2>A little inspiration</h2><p>Fresh perspectives for your mathematical mind.</p></div><span class="small muted">Curated for you</span></div><div class="topic-tabs" role="group" aria-label="Filter inspiration topics">${["All topics", "Calculus", "Linear Algebra", "Analysis"].map((t) => `<button class="topic-tab ${topicFilter === t ? "active" : ""}" data-topic="${t}" aria-pressed="${topicFilter === t}">${t}</button>`).join("")}</div><div id="lessons-list">${lessonsHTML()}</div></section><aside><section class="challenge-card" id="daily-challenge">${challengeHTML()}</section><div class="study-note"><span class="note-icon">${icon("leaf")}</span><div><strong>Progress is a practice.</strong><p>Even ten curious minutes make a difference.</p></div></div></aside></div><section class="recent-section"><div class="section-heading"><h2>Pick up where you left off</h2><button class="text-button" data-route="library">Your saved library ${icon("arrow-right")}</button></div><div class="recent-list">${
    activity.length
      ? activity
          .slice(0, 3)
          .map(
            (a) =>
              `<div class="recent-item">${toolIcon(TOOLS[a.key] ? a.key : "ask")}<button data-route="${escapeHTML(a.key)}"><strong>${escapeHTML(a.title)}</strong><small>${TOOLS[a.key]?.title || "Learning session"}</small></button><time>${humanDate(a.date)}</time>${icon("arrow-up-right")}</div>`,
          )
          .join("")
      : `<div class="recent-empty">${icon("history")}<div><strong>A fresh page, full of possibilities.</strong><p>Your recent explorations will find a home right here.</p></div><button class="text-button" style="margin-left:auto" data-route="ask">Start exploring ${icon("arrow-right")}</button></div>`
  }</div></section>`;
}
const CHALLENGES = [
  {
    question: "A small question. A satisfying answer.",
    formula: "If f(x) = x², what is f′(3)?",
    options: ["3", "6", "9", "12"],
    answer: 1,
    explanation:
      "The derivative of x² is 2x. At x = 3, that gives us 2 × 3 = 6.",
  },
  {
    question: "Ready for a little number magic?",
    formula: "1 + 3 + 5 + ⋯ + 19 = ?",
    options: ["90", "99", "100", "110"],
    answer: 2,
    explanation:
      "The sum of the first n odd numbers is n². There are 10 terms, so the answer is 10² = 100.",
  },
  {
    question: "Give your intuition a little stretch.",
    formula: "What is the determinant of [[2, 1], [3, 4]]?",
    options: ["5", "8", "11", "2"],
    answer: 0,
    explanation:
      "For a 2 × 2 matrix, det = ad − bc. Here that is 2 × 4 − 1 × 3 = 5.",
  },
];
const dayKey = new Date().toLocaleDateString("en-CA");
const challenge =
  CHALLENGES[Math.floor(Date.now() / 86400000) % CHALLENGES.length];
let challengeState = readStorage("challenge", {});
function challengeHTML() {
  const answered = challengeState.day === dayKey;
  return `<div class="challenge-header">${icon("bolt")} THE DAILY SPARK <span class="challenge-tag">JUST FOR FUN</span></div><h3>${challenge.question}</h3><div class="challenge-formula">${challenge.formula}</div><div class="challenge-answers">${challenge.options.map((o, i) => `<button class="challenge-option ${answered ? (i === challenge.answer ? "correct" : challengeState.choice === i ? "incorrect" : "") : ""}" data-answer="${i}" ${answered ? "disabled" : ""}><span>${"ABCD"[i]}</span>${o}</button>`).join("")}</div>${answered ? `<p class="challenge-feedback" role="status">${challengeState.choice === challenge.answer ? "Nicely done! " : "A good chance to learn. "}${challenge.explanation}</p>` : ""}<div class="challenge-bottom"><p>${icon("clock")} A one-minute mind stretch</p><span class="text-button">${answered ? "See you tomorrow ✧" : "You’ve got this ✧"}</span></div>`;
}
function toolkitView() {
  return `<div class="page-intro"><div><h1>A tool for every possibility.</h1><p>Follow your curiosity. We’ll help with the rest.</p></div></div><div class="tools-grid all-tools-grid">${Object.keys(TOOLS).map(toolCard).join("")}</div>`;
}
function pageIntro(key) {
  const t = TOOLS[key];
  return `<div class="page-intro"><div class="page-title-row">${toolIcon(key)}<div><h1>${t.title}</h1><p>${t.subtitle}</p></div></div>${key === "ask" ? '<button class="button" id="new-chat">' + icon("plus") + " New conversation</button>" : ""}</div>`;
}
function emptyState(key, title, description) {
  return `<div class="empty-state"><div class="empty-orb">${icon(TOOLS[key]?.icon || "bookmark")}</div><h3>${title}</h3><p>${description}</p></div>`;
}
function aiNotice() {
  if (healthStatus === "offline") return offlineNotice();
  return health && !health.services.groq && !health.services.gemini
    ? `<div class="ai-notice">${icon("info")}<span>Your AI tools are waiting to be connected. You can still explore graphs, the question bank, and built-in mathematician profiles.</span></div>`
    : "";
}
const field = (name, label, value = "", placeholder = "", type = "text") =>
  `<label class="field-label" for="field-${name}">${label}</label><input class="field" id="field-${name}" name="${name}" type="${type}" value="${escapeHTML(value)}" placeholder="${escapeHTML(placeholder)}" required>`;
const select = (name, label, options, value) =>
  `<label class="field-label" for="field-${name}">${label}</label><select class="field" id="field-${name}" name="${name}">${options.map(([v, l]) => `<option value="${v}" ${v === value ? "selected" : ""}>${l}</option>`).join("")}</select>`;
const TOPICS = [
  ["Calculus", "Calculus"],
  ["Linear Algebra", "Linear Algebra"],
  ["Real Analysis", "Real Analysis"],
  ["Complex Analysis", "Complex Analysis"],
  ["Abstract Algebra", "Abstract Algebra"],
  ["Topology", "Topology"],
];
function toolView(key) {
  const t = TOOLS[key];
  let fields = "",
    button = "",
    hint = "",
    extras = "";
  if (key === "graph") {
    fields =
      field(
        "expression",
        "Your function, f(x)",
        preset.value || "sin(x)",
        "e.g. x^2, sin(x), exp(-x^2)",
      ) +
      '<div class="field-row"><div>' +
      field("x_min", "From", -10, "", "number") +
      "</div><div>" +
      field("x_max", "To", 10, "", "number") +
      "</div></div>" +
      select(
        "points",
        "Detail level",
        [
          ["200", "200 points · Quick sketch"],
          ["600", "600 points · Smooth curve"],
          ["1000", "1,000 points · Fine detail"],
        ],
        "600",
      );
    button = "Plot function";
    hint =
      "Use x as your variable. Try powers (^), trigonometric functions, logarithms, or square roots.";
    extras =
      '<div class="chips">' +
      ["sin(x)", "x^2", "exp(-x^2)", "log(x)"]
        .map(
          (v) =>
            `<button type="button" class="chip" data-example="${v}">${v}</button>`,
        )
        .join("") +
      "</div>";
  }
  if (key === "formula") {
    fields =
      field(
        "topic",
        "What are you studying?",
        preset.value || "Calculus",
        "A topic or concept",
      ) +
      select(
        "exam",
        "Study level",
        [
          ["General", "Explore the essentials"],
          ["IIT JAM", "IIT JAM"],
          ["GATE", "GATE"],
          ["CSIR NET", "CSIR NET"],
        ],
        "General",
      );
    button = "Create formula sheet";
    hint =
      "A focused reference to help you connect ideas and remember what matters.";
  }
  if (key === "pyqs") {
    fields =
      select(
        "exam",
        "Your exam",
        [
          ["jam", "IIT JAM"],
          ["gate", "GATE"],
          ["csir", "CSIR NET"],
        ],
        preset.exam ||
          (preferences.exam === "curiosity" ? "jam" : preferences.exam),
      ) +
      select(
        "difficulty",
        "Difficulty",
        [
          ["easy", "Easy · Build your foundation"],
          ["moderate", "Moderate · Find your rhythm"],
          ["difficult", "Difficult · Stretch yourself"],
        ],
        "moderate",
      ) +
      select(
        "count",
        "Number of questions",
        [
          ["5", "5 questions"],
          ["10", "10 questions"],
          ["20", "20 questions"],
        ],
        "5",
      ) +
      field("topic", "Topic (optional)", "", "All topics").replace(
        " required",
        "",
      ) +
      field("year", "Year (optional)", "", "Any year", "number").replace(
        " required",
        "",
      );
    button = "Find practice questions";
    hint =
      "Questions are drawn from your question bank. Reveal each solution when you’re ready.";
  }
  if (key === "quiz") {
    fields =
      field(
        "topic",
        "Pick a topic",
        preset.value || "Linear Algebra",
        "What would you like to practice?",
      ) +
      select(
        "count",
        "Questions",
        [
          ["3", "3 · A little warm-up"],
          ["5", "5 · A focused session"],
          ["10", "10 · A deeper dive"],
        ],
        "5",
      ) +
      select(
        "difficulty",
        "Challenge level",
        [
          ["easy", "Easy"],
          ["moderate", "Moderate"],
          ["difficult", "Difficult"],
        ],
        "moderate",
      );
    button = "Create my quiz";
    hint =
      "A fresh set of multiple-choice questions to help your knowledge stick.";
  }
  if (key === "theorem") {
    fields = field(
      "theorem",
      "An idea worth proving",
      preset.value || "Cauchy-Schwarz inequality",
      "Name a theorem or enter a statement",
    );
    button = "Explore the proof";
    hint =
      "Follow the reasoning, understand each step, and discover why it works.";
    extras =
      '<div class="chips">' +
      ["Pythagorean theorem", "Infinitely many primes", "Mean value theorem"]
        .map(
          (v) =>
            `<button type="button" class="chip" data-example="${v}">${v}</button>`,
        )
        .join("") +
      "</div>";
  }
  if (key === "mathematician") {
    fields = field(
      "name",
      "Who inspires you?",
      preset.value || "Ramanujan",
      "A mathematician’s name",
    );
    button = "Meet a great mind";
    hint =
      "Their ideas changed the way we see the world. Get to know the story behind them.";
    extras =
      '<div class="chips">' +
      ["Ramanujan", "Euler", "Gauss"]
        .map(
          (v) =>
            `<button type="button" class="chip" data-example="${v}">${v}</button>`,
        )
        .join("") +
      "</div>";
  }
  if (key === "projects") {
    fields = field(
      "topic",
      "Where does your curiosity lead?",
      preset.value || "Differential Equations",
      "A topic you would love to explore",
    );
    button = "Find my next idea";
    hint =
      "Discover five project ideas that turn abstract mathematics into something you can make.";
  }
  return (
    pageIntro(key) +
    (healthStatus === "offline"
      ? offlineNotice()
      : key !== "graph" && key !== "pyqs" && key !== "mathematician"
        ? aiNotice()
        : "") +
    `<div class="tool-layout"><form class="panel" id="tool-form" data-tool="${key}"><h2 class="form-title">${icon(t.icon)}${key === "graph" ? "Set the scene" : "Make it yours"}</h2>${fields}${extras}<button class="button primary form-submit" type="submit">${button}${icon("arrow-right")}</button><p class="form-hint">${hint}</p></form><section class="panel results-panel" id="tool-results" aria-live="polite">${emptyState(key, key === "graph" ? "Every function has a story." : key === "pyqs" ? "Find your next good challenge." : key === "mathematician" ? "A great mind. A bigger story." : "An idea is about to take shape.", key === "graph" ? "Enter a function and choose a range. Let’s see what it looks like." : "Choose your preferences on the left, and let your curiosity take it from here.")}</section></div>`
  );
}
async function api(path, payload) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 110000);
  try {
    const response = await fetch(apiUrl(path), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error(
        "Your learning services are currently unavailable. Please try again later. Your saved work is still here.",
      );
    }
    if (!response.ok || data.success === false)
      throw new Error(
        response.status === 429 && !data.code?.startsWith("ai_")
          ? "A little breather: please wait a moment before trying again."
          : data.error || "Something went wrong. Give it another try.",
      );
    return data;
  } catch (e) {
    if (e.name === "AbortError")
      throw new Error("This is taking longer than expected. Please try again.");
    if (e instanceof TypeError)
      throw new Error(
        "We couldn’t reach your learning services. Please try again shortly.",
      );
    throw e;
  } finally {
    clearTimeout(timeout);
  }
}
function loadingHTML() {
  return '<div class="loading-state"><span class="spinner"></span><span>Making a little room for discovery…</span></div>';
}
function renderMath(el) {
  if (typeof renderMathInElement === "function")
    renderMathInElement(el, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
      strict: false,
      trust: false,
    });
}
function richText(text) {
  // Keep multiline equations in one text node for KaTeX's delimiter parser.
  const mathBlocks = [];
  const lines = escapeHTML(text)
    .replace(/\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$|\\\([\s\S]*?\\\)/g, (block) => {
      mathBlocks.push(block);
      return `\u0000MATH${mathBlocks.length - 1}\u0000`;
    })
    .split("\n");
  let inCode = false;
  return (
    lines
      .map((line) => {
        if (line.startsWith("```")) {
          inCode = !inCode;
          return inCode ? "<pre><code>" : "</code></pre>";
        }
        if (inCode) return line + "\n";
        line = line
          .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
          .replace(/`([^`]+)`/g, "<code>$1</code>");
        if (/^### /.test(line)) return "<h3>" + line.slice(4) + "</h3>";
        if (/^##? /.test(line))
          return "<h2>" + line.replace(/^##? /, "") + "</h2>";
        if (/^[-*] /.test(line)) return "<p>• " + line.slice(2) + "</p>";
        return line ? "<p>" + line + "</p>" : "<br>";
      })
      .join("") + (inCode ? "</code></pre>" : "")
  ).replace(
    /\u0000MATH(\d+)\u0000/g,
    (_, index) => mathBlocks[Number(index)] ?? "",
  );
}
function requireAIText(text) {
  if (!text || String(text).includes("No AI service available"))
    throw new Error(
      "Your AI tutor isn’t connected yet. Connect an AI service to explore this feature. Your local learning tools are ready to use.",
    );
  return text;
}
function resultHeading(title, saveable = true) {
  return `<div class="result-topline"><span>${escapeHTML(title)}</span>${saveable ? `<button class="text-button" data-save-result>${icon("bookmark")} Save to library</button>` : ""}</div>`;
}
async function submitTool(form) {
  const key = form.dataset.tool;
  const output = $("#tool-results");
  const submit = $('[type="submit"]', form);
  if (submit.disabled) return;
  const payload = Object.fromEntries(new FormData(form));
  ["count", "points", "x_min", "x_max", "year"].forEach((k) => {
    if (payload[k] !== undefined) {
      if (payload[k] === "") delete payload[k];
      else payload[k] = Number(payload[k]);
    }
  });
  if (key === "graph" && payload.x_min >= payload.x_max) {
    toast("Choose a starting value smaller than your ending value.", "error");
    return;
  }
  submit.disabled = true;
  output.innerHTML = loadingHTML();
  output._result = null;
  try {
    const data = await api(TOOLS[key].endpoint, payload);
    let text = "",
      title =
        payload.topic ||
        payload.theorem ||
        payload.name ||
        payload.expression ||
        TOOLS[key].title;
    if (key === "graph") {
      graphData = { ...data, title };
      output.innerHTML =
        resultHeading("Your function, visualized", false) +
        `<div class="graph-viewport" id="graph-viewport"><div class="graph-hover" id="graph-hover"></div></div><div class="graph-caption"><span><strong>f(x) = ${escapeHTML(data.expression)}</strong> · ${data.point_count} points</span><button class="text-button" id="download-graph">${icon("download")} Export SVG</button></div><div class="graph-analysis">${data.analysis && !data.analysis.includes("No AI service available") ? richText(data.analysis) : "A new perspective on your function. Hover over the curve to explore its values."}</div>`;
      drawGraph(output, data);
    } else if (key === "pyqs") {
      const questions = data.questions || [];
      output.innerHTML =
        resultHeading(
          `${questions.length} question${questions.length === 1 ? "" : "s"} to explore`,
          false,
        ) +
        (questions.length
          ? questions
              .map(
                (q, i) =>
                  `<article class="pyq-card"><div class="pyq-meta"><span class="badge">QUESTION ${i + 1}</span><span class="badge">${escapeHTML(q.topic || payload.exam.toUpperCase())}</span><span class="badge">${escapeHTML(q.difficulty)}</span>${q.year ? '<span class="badge">' + escapeHTML(q.year) + "</span>" : ""}</div><div class="rich-text">${richText(q.question)}</div>${(Array.isArray(q.options) ? q.options : []).map((o) => '<div class="pyq-option">' + escapeHTML(o) + "</div>").join("")}<details><summary>Ready? Reveal the solution</summary><div class="rich-text">${richText("**Answer:** " + (q.answer || "Not supplied") + "\n" + (q.explanation || ""))}</div></details></article>`,
              )
              .join("")
          : emptyState(
              "pyqs",
              "A little quiet here, for now.",
              "There are no questions matching these filters. Try another topic or exam. If your bank is empty, try a quick quiz instead.",
            ) +
            '<div style="text-align:center"><button class="button secondary" data-route="quiz">Try a quick quiz ' +
            icon("arrow-right") +
            "</button></div>");
    } else {
      text = requireAIText(
        data.answer ||
          data.questions ||
          data.proof ||
          data.biography ||
          data.projects,
      );
      if (key === "mathematician") {
        text =
          "## " +
          data.name +
          "\n" +
          (data.birth
            ? data.birth + " – " + data.death + " · " + data.nationality + "\n"
            : "") +
          "\n" +
          (data.field ? "**Known for:** " + data.field + "\n\n" : "") +
          text;
      }
      const result = {
        id: uid(),
        key,
        title,
        text,
        date: new Date().toISOString(),
      };
      output._result = result;
      output.innerHTML =
        resultHeading(title) +
        '<div class="rich-text">' +
        richText(text) +
        "</div>";
      if (
        key === "mathematician" &&
        /^https:\/\/en\.wikipedia\.org\/wiki\//.test(data.wikipedia || "")
      )
        output.innerHTML += `<a class="button secondary" style="margin-top:20px" href="${escapeHTML(data.wikipedia)}" target="_blank" rel="noopener noreferrer">Keep exploring on Wikipedia ${icon("arrow-up-right")}</a>`;
    }
    recordActivity(key, title);
    renderMath(output);
  } catch (e) {
    output.innerHTML = `<div class="error-panel"><h3>Let’s try that again.</h3><p>${escapeHTML(e.message)}</p></div>`;
  } finally {
    submit.disabled = false;
  }
}
function drawGraph(container, data) {
  const viewport = $("#graph-viewport", container);
  const points = data.points;
  const finite = points.filter((p) => Number.isFinite(p.y));
  if (!finite.length) {
    viewport.innerHTML = emptyState(
      "graph",
      "No real values in this range.",
      "Try a different range or a different function.",
    );
    return;
  }
  const W = 700,
    H = 340,
    pad = { l: 47, r: 20, t: 18, b: 35 };
  let xmin = data.x_range[0],
    xmax = data.x_range[1];
  const sorted = finite.map((p) => p.y).sort((a, b) => a - b);
  let ymin = sorted[Math.floor(sorted.length * 0.02)],
    ymax =
      sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.98))];
  if (ymin === ymax) {
    ymin -= 1;
    ymax += 1;
  }
  const margin = (ymax - ymin) * 0.13;
  ymin -= margin;
  ymax += margin;
  const sx = (x) => pad.l + ((x - xmin) / (xmax - xmin)) * (W - pad.l - pad.r),
    sy = (y) => H - pad.b - ((y - ymin) / (ymax - ymin)) * (H - pad.t - pad.b);
  let grid = "",
    labels = "";
  for (let i = 0; i <= 6; i++) {
    let x = xmin + ((xmax - xmin) * i) / 6;
    grid += `<path d="M${sx(x)} ${pad.t}V${H - pad.b}"/>`;
    labels += `<text x="${sx(x)}" y="${H - 12}" text-anchor="middle">${Number(x.toFixed(2))}</text>`;
  }
  for (let i = 0; i <= 5; i++) {
    let y = ymin + ((ymax - ymin) * i) / 5;
    grid += `<path d="M${pad.l} ${sy(y)}H${W - pad.r}"/>`;
    labels += `<text x="${pad.l - 9}" y="${sy(y) + 3}" text-anchor="end">${Number(y.toPrecision(3))}</text>`;
  }
  let path = "",
    previous = null;
  points.forEach((p) => {
    if (!Number.isFinite(p.y) || p.y < ymin || p.y > ymax) {
      previous = null;
      return;
    }
    const jump = previous && Math.abs(p.y - previous.y) > (ymax - ymin) * 0.4;
    path +=
      (previous && !jump ? "L" : "M") +
      sx(p.x).toFixed(2) +
      " " +
      sy(p.y).toFixed(2);
    previous = p;
  });
  let axis = "";
  if (ymin <= 0 && ymax >= 0)
    axis += `<path d="M${pad.l} ${sy(0)}H${W - pad.r}"/>`;
  if (xmin <= 0 && xmax >= 0)
    axis += `<path d="M${sx(0)} ${pad.t}V${H - pad.b}"/>`;
  viewport.insertAdjacentHTML(
    "afterbegin",
    `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Graph of ${escapeHTML(data.expression)} from ${xmin} to ${xmax}" xmlns="http://www.w3.org/2000/svg"><rect width="700" height="340" fill="white"/><g stroke="#f0ebf5" stroke-width="1">${grid}</g><g stroke="#d7c9e7" stroke-width="1">${axis}</g><g fill="#af9cbd" font-family="sans-serif" font-size="10">${labels}</g><path d="${path}" fill="none" stroke="#9570ce" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/><circle id="graph-pointer" r="4.5" fill="#9570ce" stroke="white" stroke-width="2" opacity="0"/></svg>`,
  );
  viewport.addEventListener("pointermove", (e) => {
    const rect = viewport.getBoundingClientRect();
    const rel = (e.clientX - rect.left) / rect.width;
    const x =
      xmin +
      Math.max(0, Math.min(1, (rel * W - pad.l) / (W - pad.l - pad.r))) *
        (xmax - xmin);
    const p = points.reduce(
      (best, p) => (Math.abs(p.x - x) < Math.abs(best.x - x) ? p : best),
      points[0],
    );
    const marker = $("#graph-pointer", viewport),
      tip = $("#graph-hover", viewport);
    if (!Number.isFinite(p.y) || p.y < ymin || p.y > ymax) return;
    marker.setAttribute("cx", sx(p.x));
    marker.setAttribute("cy", sy(p.y));
    marker.setAttribute("opacity", "1");
    tip.style.display = "block";
    tip.style.left =
      Math.min(rect.width - 135, Math.max(5, e.clientX - rect.left + 10)) +
      "px";
    tip.style.top = Math.max(5, e.clientY - rect.top - 35) + "px";
    tip.textContent = `x: ${p.x.toFixed(3)} · y: ${p.y.toFixed(3)}`;
  });
  viewport.addEventListener("pointerleave", () => {
    $("#graph-pointer", viewport)?.setAttribute("opacity", "0");
    $("#graph-hover", viewport).style.display = "none";
  });
}
function downloadFile(filename, text, type = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function chatView() {
  if (!activeChat)
    activeChat = {
      id: uid(),
      title: "A new conversation",
      date: new Date().toISOString(),
      messages: [],
    };
  return (
    pageIntro("ask") +
    `<section class="conversation-shell">${aiNotice()}<div id="chat-welcome" ${activeChat.messages.length ? "hidden" : ""}><div class="chat-welcome"><div class="empty-orb">${icon("spark")}</div><span class="eyebrow">A LITTLE HELP. A WHOLE NEW PERSPECTIVE.</span><h2>What’s on your mind?</h2><p>No question is too small. Let’s work through it together.</p></div><div class="prompt-grid">${[
      {
        i: "formula",
        t: "Make derivatives feel intuitive",
        p: "Explain derivatives intuitively with a simple real-world example.",
      },
      {
        i: "theorem",
        t: "Walk me through a proof",
        p: "Walk me through a proof of the Pythagorean theorem, explaining every step.",
      },
      {
        i: "graph",
        t: "Help me understand eigenvalues",
        p: "What are eigenvalues? Please start with the intuition and show me a simple example.",
      },
      {
        i: "book",
        t: "Prepare for my next big exam",
        p: "Help me plan a focused one-week mathematics study session for IIT JAM.",
      },
    ]
      .map(
        (p) =>
          `<button class="prompt-card" data-prompt="${escapeHTML(p.p)}">${icon(p.i)}${p.t}${icon("arrow-up-right")}</button>`,
      )
      .join(
        "",
      )}</div></div><div class="messages" id="messages" role="log" aria-label="Your conversation">${activeChat.messages.map(messageHTML).join("")}${chatBusy ? '<div class="loading-state" style="min-height:90px"><span class="spinner"></span><span>Your tutor is thinking…</span></div>' : ""}</div><form class="composer" id="chat-form"><div id="attachment-preview"></div><label class="skip-link" for="chat-input">Your question</label><textarea id="chat-input" rows="2" maxlength="4000" placeholder="Ask a question. Follow a thought. Start anywhere." required></textarea><div class="composer-actions"><div class="composer-left"><button type="button" class="icon-button" id="attach-button" aria-label="Attach a mathematics problem image">${icon("attach")}</button><span>Bring a problem, or just your curiosity.</span><input type="file" accept="image/jpeg,image/png,image/webp" id="image-input" hidden></div><button type="submit" class="button primary composer-send" id="chat-send" ${chatBusy ? "disabled" : ""}>Send ${icon("arrow-right")}</button></div></form><p class="composer-hint">A thoughtful companion, not a final authority. Always check your work.</p></section>`
  );
}
function messageHTML(m) {
  return `<article class="message ${m.role === "user" ? "user" : "assistant"}"><span class="message-avatar">${m.role === "user" ? escapeHTML(preferences.name?.charAt(0).toUpperCase() || "Y") : "∑"}</span><div class="message-content"><span class="message-label">${m.role === "user" ? "YOU" : "MATHSPHERE"}</span><div class="rich-text">${richText(m.content)}</div>${m.role === "assistant" ? mathEvidenceHTML(m) : ""}${m.image ? '<img src="' + escapeHTML(m.image) + '" alt="Your uploaded mathematics problem">' : ""}</div></article>`;
}
function mathEvidenceHTML(message) {
  const calculations = Array.isArray(message.verification?.calculations)
    ? message.verification.calculations
        .filter((c) => c?.status === "computed")
        .slice(0, 3)
    : [];
  const references = Array.isArray(message.references)
    ? message.references.slice(0, 3)
    : [];
  let html = "";
  if (calculations.length) {
    html += `<details class="math-evidence"><summary>${icon("check")} Symbolic calculations <span>${calculations.length} completed</span></summary><p class="evidence-notice">These check the displayed calculations, not the interpretation of your question or every claim in the explanation.</p>${calculations.map((c) => `<div class="calculation-result"><strong>${escapeHTML(c.operation || "Calculation")}</strong><div class="evidence-expression">\\[${escapeHTML(c.input_latex || "")}\\]</div><div class="evidence-answer">\\[${escapeHTML(c.latex || c.result || "")}${c.constant_of_integration ? "+ C" : ""}\\]</div><p>${escapeHTML(c.check || "")}</p><p>${escapeHTML(c.assumptions || "")}</p>${c.input_domain ? "<p><strong>Original domain:</strong> " + escapeHTML(c.input_domain) + "</p>" : ""}${c.conditions ? "<p>" + escapeHTML(c.conditions) + "</p>" : ""}${Array.isArray(c.restrictions) ? "<p>" + c.restrictions.map(escapeHTML).join("; ") + "</p>" : ""}</div>`).join("")}</details>`;
  } else if (message.verification) {
    html +=
      '<p class="evidence-unchecked">No completed symbolic check for this explanation.</p>';
  }
  if (references.length) {
    html += `<details class="math-references"><summary>Reference notes <span>${references.length} relevant</span></summary><p class="evidence-notice">Locally authored notes supplied to the tutor. Further-reading links are background resources.</p>${references
      .map((r) => {
        let link = "";
        try {
          const url = new URL(r.further_reading?.url);
          if (url.protocol === "https:" && !url.username && !url.password)
            link = `<a href="${escapeHTML(url.href)}" target="_blank" rel="noopener noreferrer">Further reading: ${escapeHTML(r.further_reading.title)}</a>`;
        } catch {}
        return `<div class="reference-note"><strong>${escapeHTML(r.title || "")}</strong><small>${escapeHTML(r.id || "")} · ${escapeHTML(r.topic || "")}</small>${link}</div>`;
      })
      .join("")}</details>`;
  }
  return html;
}
function setupChat() {
  const toolbar = document.createElement("div");
  toolbar.className = "tutor-options";
  toolbar.innerHTML = `<span>${icon("book")} Understand the idea. Check the mathematics.</span><label for="explanation-style">Explain it <select id="explanation-style"><option value="step_by_step">Step by step</option><option value="concise">Concisely</option><option value="exam">For an exam</option></select></label>`;
  $(".conversation-shell").prepend(toolbar);
  $("#explanation-style").value =
    preferences.explanation_style || "step_by_step";
  $("#explanation-style").addEventListener("change", (e) => {
    preferences.explanation_style = e.target.value;
    persist("preferences", preferences);
  });
  const input = $("#chat-input");
  if (preset.value) input.value = preset.value;
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      $("#chat-form").requestSubmit();
    }
  });
  input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 170) + "px";
  });
  $("#image-input").addEventListener("change", (e) =>
    attachImage(e.target.files[0]),
  );
  if (attachment) renderAttachment();
}
function renderAttachment() {
  const el = $("#attachment-preview");
  if (!el) return;
  el.innerHTML = attachment
    ? `<div class="attachment-preview"><img src="${attachment.data}" alt="Attached problem preview"><span>${escapeHTML(attachment.name)}</span><button type="button" class="icon-button" id="remove-attachment" aria-label="Remove attachment">${icon("close")}</button></div>`
    : "";
  const input = $("#chat-input");
  if (input) input.required = !attachment;
}
async function attachImage(file) {
  if (!file) return;
  if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
    toast("Choose a JPG, PNG, or WebP image.", "error");
    return;
  }
  if (file.size > 8 * 1024 * 1024) {
    toast("Please choose an image smaller than 8 MB.", "error");
    return;
  }
  try {
    const data = await new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
    attachment = { data, type: file.type, name: file.name };
    renderAttachment();
  } catch {
    toast("This image couldn’t be attached. Please try another file.", "error");
  }
}
function storeChat(conversation) {
  const safe = {
    ...conversation,
    messages: conversation.messages.map(
      ({ role, content, verification, references }) => ({
        role,
        content,
        ...(role === "assistant" ? { verification, references } : {}),
      }),
    ),
  };
  chats = [safe, ...chats.filter((c) => c.id !== conversation.id)].slice(0, 40);
  persist("chats", chats);
}
async function sendChat() {
  if (chatBusy) return;
  const input = $("#chat-input");
  const text =
    input.value.trim() ||
    (attachment ? "Please solve the mathematics problem in this image." : "");
  if (!text) return;
  const conversation = activeChat;
  const image = attachment;
  const userMessage = {
    role: "user",
    content: text,
    ...(image ? { image: image.data } : {}),
  };
  const context = conversation.messages.map(({ role, content }) => ({
    role,
    content,
  }));
  conversation.messages.push(userMessage);
  if (conversation.messages.length === 1)
    conversation.title = text.slice(0, 75);
  conversation.date = new Date().toISOString();
  storeChat(conversation);
  input.value = "";
  input.style.height = "auto";
  attachment = null;
  renderAttachment();
  $("#chat-welcome").hidden = true;
  $("#messages").insertAdjacentHTML(
    "beforeend",
    messageHTML(userMessage) +
      '<div id="chat-thinking" class="loading-state" style="min-height:90px"><span class="spinner"></span><span>Your tutor is thinking…</span></div>',
  );
  $("#chat-thinking").scrollIntoView({ behavior: "smooth", block: "nearest" });
  chatBusy = true;
  $("#chat-send").disabled = true;
  try {
    const payload = {
      messages: [...context, { role: "user", content: text }],
      explanation_style: $("#explanation-style")?.value || "step_by_step",
    };
    if (image) {
      payload.image_b64 = image.data.split(",")[1];
      payload.image_type = image.type;
    }
    const data = await api("chat", payload);
    const answer = requireAIText(data.answer);
    const assistantMessage = {
      role: "assistant",
      content: answer,
      verification: data.verification,
      references: data.references,
    };
    conversation.messages.push(assistantMessage);
    storeChat(conversation);
    recordActivity("ask", conversation.title);
    if (route === "ask" && activeChat?.id === conversation.id) {
      $("#chat-thinking")?.remove();
      $("#messages").insertAdjacentHTML(
        "beforeend",
        messageHTML(assistantMessage),
      );
      renderMath($("#messages"));
      $(".message:last-child").scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    }
  } catch (e) {
    if (route === "ask" && activeChat?.id === conversation.id) {
      $("#chat-thinking")?.remove();
      $("#messages").insertAdjacentHTML(
        "beforeend",
        `<div class="error-panel"><h3>Your thought is saved.</h3><p>${escapeHTML(e.message)}</p><button class="text-button" id="retry-chat" style="margin-top:10px">Try this question again ${icon("refresh")}</button></div>`,
      );
    } else
      toast(
        "Your question was saved, but the tutor couldn’t respond.",
        "error",
      );
  } finally {
    chatBusy = false;
    if ($("#chat-send")) $("#chat-send").disabled = false;
  }
}
function libraryView() {
  const allCount = chats.length + saved.length;
  return `<div class="page-intro"><div><h1>Your little library of discoveries.</h1><p>Good ideas deserve a place to come back to.</p></div><span class="badge">${allCount} ${allCount === 1 ? "discovery" : "discoveries"}</span></div><div class="library-tools"><label class="skip-link" for="library-search">Search saved discoveries</label><input class="field" id="library-search" placeholder="Find a conversation or saved idea…"><button class="button" id="export-library" ${allCount ? "" : "disabled"}>${icon("download")} Export library</button></div><div id="library-content">${libraryCards("")}</div>`;
}
function libraryCards(query) {
  const items = [
    ...chats.map((c) => ({
      ...c,
      key: "ask",
      isChat: true,
      text: c.messages.map((m) => m.content).join("\n\n"),
    })),
    ...saved,
  ]
    .filter((i) => i.title.toLowerCase().includes(query.toLowerCase()))
    .sort((a, b) => new Date(b.date) - new Date(a.date));
  return items.length
    ? '<div class="library-grid">' +
        items
          .map(
            (item) =>
              `<article class="saved-card">${toolIcon(item.key)}<h3>${escapeHTML(item.title)}</h3><p>${escapeHTML(item.text?.slice(0, 160))}</p><button class="text-button" ${item.isChat ? "data-chat" : "data-open-saved"}="${escapeHTML(item.id)}">${item.isChat ? "Continue the conversation" : "Revisit this idea"} ${icon("arrow-right")}</button><div class="saved-card-footer"><span>${humanDate(item.date)} · ${item.isChat ? item.messages.length + " messages" : TOOLS[item.key].title}</span><button class="icon-button danger-button" data-delete="${escapeHTML(item.id)}" data-kind="${item.isChat ? "chat" : "saved"}" aria-label="Delete ${escapeHTML(item.title)}">${icon("trash")}</button></div></article>`,
          )
          .join("") +
        "</div>"
    : `<div class="panel">${emptyState("library", query ? "No discoveries by that name." : "Your discoveries belong here.", query ? "Try a different word or a shorter search." : "Conversations are saved as you go. Save a formula, a proof, or a project idea to start your own collection.")}<div style="text-align:center"><button class="button secondary" data-route="ask">Follow your first thought ${icon("arrow-right")}</button></div></div>`;
}
function examsView() {
  return `<div class="page-intro"><div><h1>A little closer to your goal.</h1><p>Make a plan. Build your confidence. Take the next step.</p></div></div><div class="exam-grid">${[
    ["jam", "IIT JAM", "A foundation for what comes next."],
    ["gate", "GATE Mathematics", "Turn preparation into possibility."],
    ["csir", "CSIR NET", "Deep understanding. New horizons."],
  ]
    .map(
      ([key, title, desc]) =>
        `<button class="exam-card" data-exam="${key}"><span class="tool-icon">${icon("cap")}</span><h3>${title}</h3><p>${desc}</p></button>`,
    )
    .join(
      "",
    )}</div><section class="panel exam-details" id="exam-details" aria-live="polite">${loadingHTML()}</section>`;
}
let examRequest = 0;
async function loadExam(exam) {
  const requestNumber = ++examRequest;
  const panel = $("#exam-details");
  if (!panel) return;
  $$("[data-exam]").forEach((el) =>
    el.classList.toggle("selected", el.dataset.exam === exam),
  );
  panel.innerHTML = loadingHTML();
  try {
    const data = await api("exam/info", { exam });
    if (requestNumber !== examRequest || !panel.isConnected) return;
    const d = data.details;
    panel.innerHTML =
      '<span class="eyebrow">YOUR NEXT CHAPTER</span><h2>' +
      escapeHTML(d.title) +
      '</h2><dl class="exam-facts"><div><dt>TIME TO MAKE IT COUNT</dt><dd>' +
      escapeHTML(d.duration) +
      "</dd></div><div><dt>THE CHALLENGE AHEAD</dt><dd>" +
      escapeHTML(d.questions) +
      '</dd></div></dl><h3 style="font-size:12px">A world of ideas to explore</h3><p class="exam-subjects">' +
      escapeHTML(d.subjects) +
      '</p><div class="exam-actions"><button class="button primary" data-exam-practice="' +
      exam +
      '">Find practice questions ' +
      icon("arrow-right") +
      '</button><button class="button" data-prompt="Help me create a structured four-week study plan for ' +
      escapeHTML(d.title) +
      '.">Build a study plan ' +
      icon("spark") +
      "</button></div>";
  } catch (e) {
    if (requestNumber === examRequest)
      panel.innerHTML =
        '<div class="error-panel">' + escapeHTML(e.message) + "</div>";
  }
}
function openDialog(id) {
  const dialog = $("#" + id);
  if (!dialog.open) dialog.showModal();
}
function openSearch() {
  openDialog("search-dialog");
  $("#global-search").value = "";
  searchTools("");
  setTimeout(() => $("#global-search").focus(), 50);
}
function searchTools(query) {
  const q = query.toLowerCase().trim();
  const items = [
    ...Object.entries(TOOLS).map(([key, t]) => ({
      key,
      title: t.title,
      description: t.description,
      icon: t.icon,
    })),
    {
      key: "exams",
      title: "Exam Preparation",
      description: "IIT JAM, GATE, and CSIR NET",
      icon: "cap",
    },
    {
      key: "library",
      title: "Saved Library",
      description: "Your saved conversations and discoveries",
      icon: "bookmark",
    },
    ...LESSONS.map((l, i) => ({
      key: l.key,
      title: l.title,
      description: l.category,
      icon: TOOLS[l.key].icon,
      lesson: i,
    })),
  ].filter((i) => (i.title + " " + i.description).toLowerCase().includes(q));
  $("#search-results").innerHTML = items.length
    ? items
        .map(
          (i) =>
            `<button class="search-result" ${i.lesson !== undefined ? 'data-lesson="' + i.lesson + '"' : 'data-route="' + i.key + '"'}><span class="tool-icon">${icon(i.icon)}</span><span><strong>${i.title}</strong><small>${i.description}</small></span>${icon("arrow-right")}</button>`,
        )
        .join("")
    : '<div class="empty-mini">No tools found. Try “graph”, “quiz”, or “calculus”.</div>';
}
function openSettings() {
  const form = $("#settings-form");
  form.elements.name.value = preferences.name || "";
  $("#learner-exam").value = preferences.exam || "curiosity";
  openDialog("settings-dialog");
}
function saveResult(output, button) {
  const result = output?._result;
  if (!result) {
    toast("There’s no result to save yet.");
    return;
  }
  if (saved.some((s) => s.id === result.id)) {
    toast("This idea already has a home in your library.");
    return;
  }
  saved = [result, ...saved].slice(0, 100);
  persist("saved", saved);
  button.innerHTML = icon("check") + " Saved to library";
  toast("A good idea, safely tucked away.");
}
function openSaved(id) {
  const item = saved.find((i) => i.id === id);
  if (!item) return;
  $("#main").innerHTML =
    `<div class="page-view"><div class="page-intro"><div><h1>${escapeHTML(item.title)}</h1><p>${TOOLS[item.key].title} · Saved ${humanDate(item.date)}</p></div><button class="button" data-route="library">Back to library</button></div><section class="panel"><div class="result-topline"><span>Your saved discovery</span><button class="text-button" id="download-saved" data-id="${escapeHTML(id)}">${icon("download")} Download</button></div><div class="rich-text">${richText(item.text)}</div></section></div>`;
  renderMath($("#main"));
  window.scrollTo({ top: 0, behavior: "instant" });
}

document.addEventListener("click", (e) => {
  const target = e.target.closest("button,a");
  if (!target) return;
  if (target.dataset.route) {
    e.preventDefault();
    $$("dialog[open]").forEach((d) => d.close());
    go(target.dataset.route);
  }
  if (target.dataset.closeDialog) $("#" + target.dataset.closeDialog).close();
  if (target.dataset.lesson !== undefined) {
    $$("dialog[open]").forEach((d) => d.close());
    const l = LESSONS[Number(target.dataset.lesson)];
    go(l.key, { value: l.prompt });
  }
  if (target.dataset.topic) {
    topicFilter = target.dataset.topic;
    $$("[data-topic]").forEach((t) => {
      t.classList.toggle("active", t.dataset.topic === topicFilter);
      t.setAttribute("aria-pressed", String(t.dataset.topic === topicFilter));
    });
    $("#lessons-list").innerHTML = lessonsHTML();
  }
  if (target.dataset.answer !== undefined) {
    if (challengeState.day === dayKey) return;
    challengeState = { day: dayKey, choice: Number(target.dataset.answer) };
    persist("challenge", challengeState);
    $("#daily-challenge").innerHTML = challengeHTML();
    recordActivity("quiz", "A daily spark of curiosity");
  }
  if (target.dataset.example) {
    const form = $("#tool-form");
    const input = $("input", form);
    input.value = target.dataset.example;
    input.focus();
  }
  if (target.hasAttribute("data-save-result"))
    saveResult(target.closest("#tool-results"), target);
  if (target.dataset.prompt) {
    if (chatBusy) {
      toast("Your tutor is finishing a thought. Please wait a moment.");
      return;
    }
    go("ask", { value: target.dataset.prompt });
    setTimeout(() => $("#chat-input")?.focus(), 50);
  }
  if (target.dataset.exam) loadExam(target.dataset.exam);
  if (target.dataset.examPractice)
    go("pyqs", { exam: target.dataset.examPractice });
  if (target.dataset.chat) {
    if (chatBusy) {
      toast("Your tutor is finishing a thought. Please wait a moment.");
      return;
    }
    activeChat = chats.find((c) => c.id === target.dataset.chat);
    go("ask");
  }
  if (target.dataset.openSaved) openSaved(target.dataset.openSaved);
  if (target.dataset.delete) {
    const id = target.dataset.delete;
    if (target.dataset.kind === "chat") {
      chats = chats.filter((c) => c.id !== id);
      persist("chats", chats);
      if (activeChat?.id === id) activeChat = null;
    } else {
      saved = saved.filter((s) => s.id !== id);
      persist("saved", saved);
    }
    $("#library-content").innerHTML = libraryCards($("#library-search").value);
    toast("Removed from your library.");
  }
  if (target.id === "new-chat") {
    if (chatBusy) {
      toast("Your tutor is finishing a thought. Please wait a moment.");
      return;
    }
    activeChat = null;
    attachment = null;
    renderPage("ask");
  }
  if (target.id === "attach-button") $("#image-input").click();
  if (target.id === "remove-attachment") {
    attachment = null;
    $("#image-input").value = "";
    renderAttachment();
  }
  if (target.id === "retry-chat") {
    const user = activeChat.messages.findLast((m) => m.role === "user");
    $("#chat-input").value = user?.content || "";
    $("#chat-input").focus();
    target.closest(".error-panel").remove();
  }
  if (target.id === "download-graph") {
    const svg = $("#graph-viewport svg");
    if (svg)
      downloadFile(
        "mathsphere-graph.svg",
        new XMLSerializer().serializeToString(svg),
        "image/svg+xml",
      );
  }
  if (target.id === "export-library") {
    downloadFile(
      "mathsphere-library.json",
      JSON.stringify(
        {
          exportedAt: new Date().toISOString(),
          conversations: chats,
          discoveries: saved,
        },
        null,
        2,
      ),
      "application/json",
    );
    toast("Your discoveries, ready to take with you.");
  }
  if (target.id === "download-saved") {
    const item = saved.find((i) => i.id === target.dataset.id);
    if (item)
      downloadFile(
        "mathsphere-discovery.md",
        "# " + item.title + "\n\n" + item.text,
      );
  }
});
document.addEventListener("submit", (e) => {
  if (e.target.id === "tool-form") {
    e.preventDefault();
    submitTool(e.target);
  }
  if (e.target.id === "chat-form") {
    e.preventDefault();
    sendChat();
  }
});
$("#settings-form").addEventListener("submit", (e) => {
  e.preventDefault();
  preferences = {
    ...preferences,
    name: e.target.elements.name.value.trim(),
    exam: $("#learner-exam").value,
  };
  persist("preferences", preferences);
  updateProfile();
  $("#settings-dialog").close();
  if (route === "dashboard") renderPage("dashboard");
  toast("A little more you. Preferences saved.");
});
$("#menu-button").addEventListener("click", () =>
  setSidebar(!$("#sidebar").classList.contains("open")),
);
$("#sidebar-backdrop").addEventListener("click", () => setSidebar(false));
$("#search-trigger").addEventListener("click", openSearch);
$("#help-button").addEventListener("click", () => openDialog("help-dialog"));
["profile-button", "top-avatar", "workspace-switch"].forEach((id) =>
  $("#" + id).addEventListener("click", openSettings),
);
$("#global-search").addEventListener("input", (e) =>
  searchTools(e.target.value),
);
document.addEventListener("input", (e) => {
  if (e.target.id === "library-search")
    $("#library-content").innerHTML = libraryCards(e.target.value);
});
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    openSearch();
  }
  if (e.key === "Escape") setSidebar(false);
});
$$("dialog").forEach((dialog) =>
  dialog.addEventListener("click", (e) => {
    if (e.target === dialog) {
      const rect = dialog.getBoundingClientRect();
      if (
        e.clientX < rect.left ||
        e.clientX > rect.right ||
        e.clientY < rect.top ||
        e.clientY > rect.bottom
      )
        dialog.close();
    }
  }),
);
window.addEventListener("hashchange", () => renderPage(location.hash.slice(1)));
initNavigation();
hydrate();
renderPage(location.hash.slice(1) || "dashboard");
fetch(apiUrl("health"))
  .then((r) => {
    if (!r.ok) throw new Error("unavailable");
    return r.json();
  })
  .then((data) => {
    health = data;
    healthStatus = "ready";
    $("#service-info").textContent =
      data.services.groq || data.services.gemini
        ? "Your AI learning tools are connected. Graphs and the question bank are also available."
        : "Your local math tools are ready. AI tools need a connected service before they can respond.";
  })
  .catch(() => {
    healthStatus = "offline";
    $("#service-info").textContent =
      "We couldn’t reach your learning services. Please try refreshing in a moment.";
  });
