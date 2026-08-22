// Deploy Checklist -- a lab that changes on its own while the agent works.
//
// The checklist is the task. The activity feed is the experiment: it appends an
// entry every INTERVAL_MS regardless of what the agent is doing, and records the
// wall-clock time of every append.
//
// That log is ground truth for `external_change`. A detector claiming to
// separate "the action caused this" from "something else caused this" can be
// scored against it: for each agent step, an injection either did or did not
// land inside the step's before/after window, and the detector either did or did
// not raise the flag. Without a log like this the signal can only be judged by
// eye, which is how it shipped -- provisional and untested.
const ITEMS = [
  { id: "migrations", label: "Run database migrations" },
  { id: "cache",      label: "Warm the cache" },
  { id: "smoke",      label: "Run smoke tests" },
  { id: "announce",   label: "Announce the release" },
];

// Slow enough that most agent steps see no injection, frequent enough that some
// do. A detector that fires on everything is as useless as one that never fires,
// and a mixed corpus is the only way to tell them apart.
const INTERVAL_MS = 7000;

const MESSAGES = [
  "deploy-bot: build #418 finished",
  "deploy-bot: staging health check passed",
  "alice: pushed a hotfix to release/2.4",
  "deploy-bot: image pushed to registry",
  "bob: commented on the release ticket",
  "deploy-bot: canary at 10% traffic",
];

const state = {
  checked: {},
  // [{ at: epoch_ms, text }] -- every change this lab made on its own.
  injections: [],
};
ITEMS.forEach((i) => { state.checked[i.id] = false; });

const list = document.getElementById("list");
const feed = document.getElementById("feed");

// The runner reads state by CALLING this hook -- `window.__QWEN_CUA_STATE__()`.
// Assigning an object instead of a function fails the whole run with
// "is not a function", after the work is already done.
function publish() {
  const snapshot = {
    checked: { ...state.checked },
    allChecked: ITEMS.every((i) => state.checked[i.id]),
    injections: state.injections.map((x) => ({ ...x })),
    injectionCount: state.injections.length,
  };
  window.__QWEN_CUA_STATE__ = () => snapshot;
}

function renderList() {
  list.innerHTML = "";
  ITEMS.forEach((item) => {
    const li = document.createElement("li");
    li.className = "item";
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.id = `check-${item.id}`;
    box.checked = state.checked[item.id];
    box.addEventListener("change", () => {
      state.checked[item.id] = box.checked;
      renderList();
    });
    const text = document.createElement("span");
    text.textContent = item.label;
    label.append(box, text);
    li.append(label);
    list.append(li);
  });
  document.getElementById("done").hidden = !ITEMS.every((i) => state.checked[i.id]);
  publish();
}

let n = 0;
function inject() {
  const text = `${MESSAGES[n % MESSAGES.length]}`;
  n += 1;
  const li = document.createElement("li");
  li.textContent = text;
  feed.prepend(li);
  // Keep the feed bounded so the page does not grow without limit; the change is
  // meant to be local, not a full-page reflow.
  while (feed.children.length > 6) feed.removeChild(feed.lastChild);
  state.injections.push({ at: Date.now(), text });
  publish();
}

// `?quiet=1` runs the identical page with the timer off. It is the control for
// "does agent-independent change degrade the AGENT, not just the verifier?" --
// same layout, same task, same pixels at rest, only the interruptions removed.
const QUIET = new URLSearchParams(location.search).get("quiet") === "1";

renderList();
if (!QUIET) setInterval(inject, INTERVAL_MS);
