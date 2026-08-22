// Release Checklist -- a lab whose targets move while a plan is being executed.
//
// Each item, once ticked, inserts a confirmation note beneath itself. Every item
// below shifts down by the note's height. The displacement is uniform, known,
// and reported in the verification state, so an experiment can compare a
// tracker's estimate against ground truth rather than against a guess.
const ITEMS = [
  { id: "changelog", label: "Update the changelog" },
  { id: "version",   label: "Bump the version number" },
  { id: "tag",       label: "Tag the release" },
];

const state = {
  checked: {},
  // Vertical offset, in CSS pixels, that each item has accumulated since first
  // render. Ground truth for re-grounding.
  shifted: {},
  initialTop: {},
};
ITEMS.forEach((i) => { state.checked[i.id] = false; state.shifted[i.id] = 0; });

const list = document.getElementById("list");

function render() {
  list.innerHTML = "";
  ITEMS.forEach((item) => {
    const li = document.createElement("li");
    li.className = "item";
    li.id = `item-${item.id}`;
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.id = `check-${item.id}`;
    box.checked = state.checked[item.id];
    box.addEventListener("change", () => {
      state.checked[item.id] = box.checked;
      render();
    });
    const text = document.createElement("span");
    text.textContent = item.label;
    label.append(box, text);
    li.append(label);
    list.append(li);

    if (state.checked[item.id]) {
      const note = document.createElement("p");
      note.className = "note";
      note.textContent = `${item.label} — confirmed.`;
      list.append(note);
    }
  });

  document.getElementById("done").hidden = !ITEMS.every((i) => state.checked[i.id]);
  measure();
}

// Record where each item actually sits, and how far it has travelled from its
// first observed position.
function measure() {
  ITEMS.forEach((item) => {
    const el = document.getElementById(`item-${item.id}`);
    if (!el) return;
    const top = Math.round(el.getBoundingClientRect().top + window.scrollY);
    if (state.initialTop[item.id] === undefined) state.initialTop[item.id] = top;
    state.shifted[item.id] = top - state.initialTop[item.id];
  });
  // The runner CALLS this hook; it must be a function, not an object.
  const snapshot = {
    checked: { ...state.checked },
    shifted: { ...state.shifted },
    allChecked: ITEMS.every((i) => state.checked[i.id]),
  };
  window.__QWEN_CUA_STATE__ = () => snapshot;
}

render();
