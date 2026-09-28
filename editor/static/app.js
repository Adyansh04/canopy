// canopy map editor: draws the saved world and turns clicks and drags into /api/edit calls.
// Everything the page shows comes from /api/world; every change goes through the API, so a
// different client (another UI, an agent) sees and makes the same edits.
"use strict";

const canvas = document.getElementById("map");
const ctx = canvas.getContext("2d");
const hint = document.getElementById("hint");
const statusBar = document.getElementById("status");
const selectionPanel = document.getElementById("selection");

let world = null;
const mapImage = new Image();
const view = { scale: 1, x: 0, y: 0 };
let selected = null; // { kind: "object" | "room", id }
let drag = null; // The gesture in progress; see onDown.
let mode = null; // null, "add", "split" or "merge": what the next gesture on the map means.
let preview = null; // A box being moved, turned or resized, before the edit is sent.
let drawn = null; // A rectangle drawn for "add" or "split": { yaw, corners }.
const HANDLE_PX = 7;
const ROTATE_PX = 26;

// --- the API --------------------------------------------------------------------------------

async function call(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const answer = await response.json();
  if (!response.ok) {
    const error = new Error(answer.error || response.statusText);
    error.status = response.status;
    throw error;
  }
  return answer;
}

async function edit(op, message) {
  try {
    const answer = await call("/api/edit", op);
    show(answer.world);
    say(message || `${op.op} done`);
    return answer.created || [];
  } catch (error) {
    say(error.message, true);
    return null;
  }
}

async function command(path, message) {
  try {
    const answer = await call(path, {});
    show(answer.world);
    say(answer.message || message);
  } catch (error) {
    say(error.message, true);
  }
}

async function save() {
  try {
    const answer = await call("/api/save", {});
    show(answer.world);
    say(answer.message);
  } catch (error) {
    // canopy saved since the page read the world: this session's edits can be made again on it.
    if (error.status === 409 && window.confirm(`${error.message}\n\nRebase your ${world.unsaved} edits now?`)) {
      return command("/api/rebase", "made again");
    }
    say(error.message, true);
  }
}

function say(text, isError = false) {
  statusBar.textContent = text;
  statusBar.classList.toggle("error", isError);
}

// --- geometry -------------------------------------------------------------------------------

function toPixel(x, y) {
  const m = world.map;
  return [(x - m.origin[0]) / m.resolution, m.height - (y - m.origin[1]) / m.resolution];
}

function toScreen(x, y) {
  const [px, py] = toPixel(x, y);
  return [px * view.scale + view.x, py * view.scale + view.y];
}

function toWorld(sx, sy) {
  const m = world.map;
  const px = (sx - view.x) / view.scale;
  const py = (sy - view.y) / view.scale;
  return [m.origin[0] + px * m.resolution, m.origin[1] + (m.height - py) * m.resolution];
}

function local(box, x, y) {
  const c = Math.cos(box.yaw);
  const s = Math.sin(box.yaw);
  const dx = x - box.centre[0];
  const dy = y - box.centre[1];
  return [c * dx + s * dy, -s * dx + c * dy];
}

function fromLocal(box, u, v) {
  const c = Math.cos(box.yaw);
  const s = Math.sin(box.yaw);
  return [box.centre[0] + c * u - s * v, box.centre[1] + s * u + c * v];
}

function corners(box) {
  const [hu, hv] = [box.size[0] / 2, box.size[1] / 2];
  return [[hu, hv], [-hu, hv], [-hu, -hv], [hu, -hv]].map(([u, v]) => fromLocal(box, u, v));
}

function inBox(box, x, y) {
  const [u, v] = local(box, x, y);
  return Math.abs(u) <= box.size[0] / 2 && Math.abs(v) <= box.size[1] / 2;
}

function inPolygon(polygon, x, y) {
  let hit = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if ((yi > y) !== (yj > y) && x < xi + ((y - yi) * (xj - xi)) / (yj - yi)) {
      hit = !hit;
    }
  }
  return hit;
}

// The walls' angle, which canopy lays boxes along: the most common yaw, a quarter turn apart.
function wallYaw() {
  const counts = new Map();
  for (const object of world.objects) {
    const quarter = Math.PI / 2;
    const yaw = Math.round((((object.yaw % quarter) + quarter) % quarter) * 50) / 50;
    counts.set(yaw, (counts.get(yaw) || 0) + 1);
  }
  let best = 0;
  let most = 0;
  for (const [yaw, count] of counts) {
    if (count > most) {
      best = yaw;
      most = count;
    }
  }
  return best;
}

function roomCentre(room) {
  if (room.outline && room.outline.length) {
    const n = room.outline.length;
    return [
      room.outline.reduce((sum, p) => sum + p[0], 0) / n,
      room.outline.reduce((sum, p) => sum + p[1], 0) / n,
    ];
  }
  return [room.x, room.y];
}

// --- what is where --------------------------------------------------------------------------

function objectById(id) {
  return world.objects.find((object) => object.id === id);
}

function roomById(id) {
  return world.rooms.find((room) => room.id === id);
}

function suspicion(kind, id) {
  return world.suggestions.find((s) => s.kind === kind && s.id === id);
}

function objectAt(x, y) {
  const hits = world.objects.filter((object) => object.shown && inBox(boxOf(object), x, y));
  hits.sort((a, b) => a.size[0] * a.size[1] - b.size[0] * b.size[1]);
  return hits[0] || null;
}

function roomAt(x, y) {
  const outlined = world.rooms.find((room) => room.outline.length > 2 && inPolygon(room.outline, x, y));
  if (outlined) {
    return outlined;
  }
  // Worlds saved before rooms carried outlines: the room whose point is nearest.
  let best = null;
  let nearest = 1.5;
  for (const room of world.rooms) {
    const d = Math.hypot(room.x - x, room.y - y);
    if (d < nearest) {
      best = room;
      nearest = d;
    }
  }
  return best;
}

function boxOf(object) {
  return preview && preview.id === object.id ? preview : object;
}

// Which part of the selected box the pointer is on: a corner, the turning handle, or inside.
function handleAt(sx, sy) {
  if (!selected || selected.kind !== "object") {
    return null;
  }
  const object = objectById(selected.id);
  if (!object) {
    return null;
  }
  const box = boxOf(object);
  const points = corners(box).map(([x, y]) => toScreen(x, y));
  for (let i = 0; i < points.length; i++) {
    if (Math.hypot(points[i][0] - sx, points[i][1] - sy) <= HANDLE_PX + 2) {
      return { kind: "corner", index: i };
    }
  }
  const knob = rotateKnob(box);
  if (Math.hypot(knob[0] - sx, knob[1] - sy) <= HANDLE_PX + 2) {
    return { kind: "rotate" };
  }
  const [x, y] = toWorld(sx, sy);
  return inBox(box, x, y) ? { kind: "move" } : null;
}

function rotateKnob(box) {
  const [ex, ey] = toScreen(...fromLocal(box, 0, box.size[1] / 2));
  const [cx, cy] = toScreen(...box.centre);
  const length = Math.hypot(ex - cx, ey - cy) || 1;
  return [ex + ((ex - cx) / length) * ROTATE_PX, ey + ((ey - cy) / length) * ROTATE_PX];
}

// --- drawing --------------------------------------------------------------------------------

function draw() {
  const ratio = window.devicePixelRatio || 1;
  const { clientWidth: width, clientHeight: height } = canvas;
  if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
    canvas.width = width * ratio;
    canvas.height = height * ratio;
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  if (!world) {
    return;
  }
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(mapImage, view.x, view.y, world.map.width * view.scale, world.map.height * view.scale);

  const style = getComputedStyle(document.documentElement);
  const colour = (name) => style.getPropertyValue(name).trim();

  world.rooms.forEach((room, index) => {
    const hue = (index * 67) % 360;
    if (room.outline.length > 2) {
      ctx.beginPath();
      room.outline.forEach(([x, y], i) => {
        const [sx, sy] = toScreen(x, y);
        i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy);
      });
      ctx.closePath();
      ctx.fillStyle = `hsla(${hue}, 70%, 55%, 0.16)`;
      ctx.fill();
      const chosen = selected && selected.kind === "room" && selected.id === room.id;
      ctx.lineWidth = chosen ? 3 : 1;
      ctx.strokeStyle = `hsla(${hue}, 70%, 40%, ${chosen ? 0.9 : 0.5})`;
      ctx.stroke();
    }
    const [sx, sy] = toScreen(...roomCentre(room));
    const text = `${room.id} ${room.type || room.name}`;
    ctx.font = "bold 15px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillStyle = `hsl(${hue}, 60%, 32%)`;
    ctx.fillText(text, sx, sy);
    if (suspicion("room", room.id)) {
      ctx.fillStyle = colour("--suspect");
      ctx.fillText("?", sx + ctx.measureText(text).width / 2 + 8, sy);
    }
  });

  const isChosen = (object) => selected && selected.kind === "object" && selected.id === object.id;
  // A removed object only while it is selected, from the Removed list.
  for (const object of world.objects.filter((o) => o.shown || isChosen(o))) {
    const box = boxOf(object);
    const chosen = isChosen(object);
    let stroke = colour("--object");
    if (!object.shown) {
      stroke = colour("--danger");
    } else if (object.checked) {
      stroke = colour("--checked");
    } else if (suspicion("object", object.id)) {
      stroke = colour("--suspect");
    }
    const points = corners(box).map(([x, y]) => toScreen(x, y));
    ctx.beginPath();
    points.forEach(([sx, sy], i) => (i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy)));
    ctx.closePath();
    ctx.setLineDash(object.box_pinned ? [5, 3] : []);
    ctx.lineWidth = chosen ? 3 : 1.5;
    ctx.strokeStyle = stroke;
    ctx.stroke();
    ctx.setLineDash([]);
    if (chosen) {
      ctx.fillStyle = "rgba(47, 111, 221, 0.12)";
      ctx.fill();
      ctx.fillStyle = stroke;
      for (const [sx, sy] of points) {
        ctx.fillRect(sx - HANDLE_PX / 2, sy - HANDLE_PX / 2, HANDLE_PX, HANDLE_PX);
      }
      const knob = rotateKnob(box);
      const [ex, ey] = toScreen(...fromLocal(box, 0, box.size[1] / 2));
      ctx.beginPath();
      ctx.moveTo(ex, ey);
      ctx.lineTo(knob[0], knob[1]);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(knob[0], knob[1], HANDLE_PX / 1.4, 0, 2 * Math.PI);
      ctx.fill();
    }
    if (view.scale >= 0.9 || chosen) {
      const [sx, sy] = toScreen(...box.centre);
      ctx.font = "12px system-ui, sans-serif";
      ctx.textAlign = "center";
      const label = object.label;
      const width = ctx.measureText(label).width + 6;
      ctx.fillStyle = "rgba(255, 255, 255, 0.78)";
      ctx.fillRect(sx - width / 2, sy - 8, width, 15);
      ctx.fillStyle = stroke;
      ctx.fillText(label, sx, sy + 4);
    }
  }

  if (drawn) {
    ctx.beginPath();
    drawn.corners.map(([x, y]) => toScreen(x, y)).forEach(([sx, sy], i) => (i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy)));
    ctx.closePath();
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 2;
    ctx.strokeStyle = colour("--suspect");
    ctx.stroke();
    ctx.setLineDash([]);
  }
}

function redraw() {
  requestAnimationFrame(draw);
}

function fit() {
  const { clientWidth: width, clientHeight: height } = canvas;
  view.scale = Math.min(width / world.map.width, height / world.map.height) * 0.95;
  view.x = (width - world.map.width * view.scale) / 2;
  view.y = (height - world.map.height * view.scale) / 2;
}

function centreOn(x, y) {
  const [sx, sy] = toScreen(x, y);
  view.x += canvas.clientWidth / 2 - sx;
  view.y += canvas.clientHeight / 2 - sy;
}

// --- the side panel -------------------------------------------------------------------------

// Builds an element; text is always set as text, never parsed: names come from a model.
function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (key.startsWith("on")) {
      node.addEventListener(key.slice(2), value);
    } else if (value !== undefined && value !== null && value !== false) {
      node.setAttribute(key, value === true ? "" : value);
    }
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined && child !== false) {
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
  }
  return node;
}

// Appends to the side panel, skipping the parts a condition left out.
function put(...parts) {
  selectionPanel.append(...parts.filter((part) => part !== null && part !== undefined && part !== false));
}

function field(title, input) {
  return el("label", { class: "field" }, el("span", {}, title), input);
}

function renderPanel() {
  selectionPanel.replaceChildren();
  if (mode === "add" && drawn) {
    return renderNewObject();
  }
  if (mode === "split" && drawn) {
    return renderSplit();
  }
  if (!selected) {
    selectionPanel.append(el("p", { class: "muted" }, "Click an object or a room on the map."));
    return;
  }
  if (selected.kind === "object") {
    const object = objectById(selected.id);
    return object ? renderObject(object) : clearSelection();
  }
  const room = roomById(selected.id);
  return room ? renderRoom(room) : clearSelection();
}

function renderObject(object) {
  const reasons = suspicion("object", object.id);
  const label = el("input", { list: "labels", value: object.label });
  const name = el("input", { value: object.name, placeholder: "no name yet" });
  const votes = Object.entries(object.votes)
    .slice(0, 4)
    .map(([word, weight]) => `${word} ${weight.toFixed(1)}`)
    .join(" · ");
  put(
    el(
      "h3",
      {},
      `O${object.id} ${object.label} `,
      object.checked ? el("span", { class: "badge checked" }, "checked") : null,
      " ",
      object.box_pinned ? el("span", { class: "badge" }, "box set by hand") : null,
      " ",
      object.state !== "active" ? el("span", { class: "badge" }, object.state) : null,
    ),
    object.crop ? el("img", { class: "crop", src: `/api/crops/O${object.id}.jpg`, alt: `what the camera saw of O${object.id}` }) : null,
    reasons ? el("ul", { class: "reasons" }, reasons.reasons.map((r) => el("li", {}, r))) : null,
    object.removed_by ? el("ul", { class: "reasons" }, el("li", {}, object.removed_by)) : null,
    field("What it is", label),
    el("div", { class: "row" },
      el("button", { onclick: () => edit({ op: "label", id: object.id, label: label.value }, `O${object.id} is now ${label.value}`) }, "Set label"),
      object.operator_label ? el("button", { onclick: () => edit({ op: "label", id: object.id, label: "" }, `O${object.id} goes back to the detector's label`) }, "Use the votes") : null,
    ),
    field("Name", name),
    el("div", { class: "row" },
      el("button", { onclick: () => edit({ op: "name", id: object.id, name: name.value }, `O${object.id} named`) }, "Set name"),
    ),
    el("dl", { class: "facts" },
      el("dt", {}, "votes"), el("dd", {}, votes || "none"),
      el("dt", {}, "describer"), el("dd", {}, object.caption || "no caption"),
      el("dt", {}, "sightings"), el("dd", {}, `${object.observations}, vote weight ${object.weight.toFixed(1)}`),
      el("dt", {}, "size"), el("dd", {}, `${object.size[0].toFixed(2)} × ${object.size[1].toFixed(2)} m, ${object.z_min.toFixed(2)}–${object.z_max.toFixed(2)} m high`),
      el("dt", {}, "voxels"), el("dd", {}, object.voxels),
    ),
    el("div", { class: "row" },
      object.state === "removed" ? el("button", { class: "primary", onclick: () => edit({ op: "restore", ids: [object.id] }, `O${object.id} restored, and checked`) }, "Restore") : null,
      el("button", { onclick: () => edit({ op: "check", id: object.id, checked: !object.checked }, object.checked ? `O${object.id} unchecked` : `O${object.id} checked`) }, object.checked ? "Uncheck" : "Mark checked (C)"),
      el("button", { onclick: () => startMode("merge") }, "Merge with…"),
      el("button", { onclick: () => startMode("split") }, "Split off…"),
      el("button", { class: "danger", onclick: () => deleteSelected() }, "Delete"),
    ),
  );
}

function renderRoom(room) {
  const reasons = suspicion("room", room.id);
  const type = el("input", { list: "room-types", value: room.type });
  const name = el("input", { value: room.name });
  put(
    el("h3", {}, `${room.id} ${room.type || room.name} `, room.checked ? el("span", { class: "badge checked" }, "checked") : null),
    reasons ? el("ul", { class: "reasons" }, reasons.reasons.map((r) => el("li", {}, r))) : null,
    field("Type", type),
    el("div", { class: "row" },
      el("button", { onclick: () => edit({ op: "room_type", room: room.id, type: type.value }, `${room.id} is a ${type.value}`) }, "Set type"),
    ),
    field("Name", name),
    el("div", { class: "row" },
      el("button", { onclick: () => edit({ op: "room_name", room: room.id, name: name.value }, `${room.id} renamed`) }, "Set name"),
    ),
    el("dl", { class: "facts" },
      el("dt", {}, "typed by"), el("dd", {}, room.type_source ? `${room.type_source}, ${room.type_confidence.toFixed(2)}` : "nothing yet"),
    ),
    el("div", { class: "row" },
      el("button", { onclick: () => edit({ op: "room_check", room: room.id, checked: !room.checked }, room.checked ? `${room.id} unchecked` : `${room.id} checked`) }, room.checked ? "Uncheck" : "Mark checked (C)"),
    ),
  );
}

function renderNewObject() {
  const label = el("input", { list: "labels", placeholder: "wardrobe" });
  const name = el("input", { placeholder: "optional" });
  const low = el("input", { type: "number", step: "0.05", value: "0" });
  const high = el("input", { type: "number", step: "0.05", value: "0.8" });
  put(
    el("h3", {}, "New object"),
    el("p", { class: "muted small" }, "It is saved as an object the operator saw, and canopy keeps its box."),
    field("What it is", label),
    field("Name", name),
    field("Bottom, m above the floor", low),
    field("Top, m above the floor", high),
    el("div", { class: "row" },
      el("button", {
        class: "primary",
        onclick: async () => {
          const box = drawnBox();
          const created = await edit({ op: "add", label: label.value, name: name.value, centre: box.centre, size: box.size, yaw: box.yaw, z_min: Number(low.value), z_max: Number(high.value) }, `added a ${label.value}`);
          if (created && created.length) {
            stopMode();
            select({ kind: "object", id: created[0] });
          }
        },
      }, "Add"),
      el("button", { onclick: () => stopMode() }, "Cancel"),
    ),
  );
  label.focus();
}

function renderSplit() {
  const label = el("input", { list: "labels", placeholder: "chair" });
  const name = el("input", { placeholder: "optional" });
  put(
    el("h3", {}, `Split off part of O${selected.id}`),
    el("p", { class: "muted small" }, "The voxels inside the drawn box become a new object."),
    field("What the part is", label),
    field("Name", name),
    el("div", { class: "row" },
      el("button", {
        class: "primary",
        onclick: async () => {
          const created = await edit({ op: "split", id: selected.id, polygon: drawn.corners, label: label.value, name: name.value }, `split a ${label.value} off O${selected.id}`);
          if (created && created.length) {
            stopMode();
            select({ kind: "object", id: created[0] });
          }
        },
      }, "Split"),
      el("button", { onclick: () => stopMode() }, "Cancel"),
    ),
  );
  label.focus();
}

function renderReview() {
  const list = document.getElementById("review");
  list.replaceChildren(
    ...world.suggestions.map((item) => {
      const thing = item.kind === "object" ? objectById(item.id) : roomById(item.id);
      const title = item.kind === "object" ? `O${item.id} ${thing ? thing.label : ""}` : `${item.id} ${thing ? thing.type : ""}`;
      return el("li", { onclick: () => goTo(item) }, el("strong", {}, title), " — ", item.reasons[0]);
    }),
  );
  document.getElementById("review-count").textContent = world.suggestions.length;
  const phantoms = world.suggestions.filter((s) => s.phantom).length;
  const button = document.getElementById("remove-phantoms");
  button.hidden = !phantoms;
  button.textContent = `Remove the ${phantoms} likely phantoms…`;
}

function renderRemoved() {
  const removed = world.objects.filter((object) => object.state === "removed" && object.removed_by);
  document.getElementById("removed-section").hidden = !removed.length;
  document.getElementById("removed-count").textContent = removed.length;
  document.getElementById("removed").replaceChildren(
    ...removed.map((object) =>
      el("li", { onclick: () => goTo({ kind: "object", id: object.id }) }, el("strong", {}, `O${object.id} ${object.label}`), " — ", object.removed_by),
    ),
  );
}

function removePhantoms() {
  const phantoms = world.suggestions.filter((s) => s.phantom);
  const names = phantoms.map((s) => `O${s.id} ${objectById(s.id).label}`).join(", ");
  const question = `Remove ${phantoms.length} objects that look like no object at all?\n\n${names}\n\nThey stay in the file, listed under Removed, and each can be restored.`;
  if (window.confirm(question)) {
    edit({ op: "remove", ids: phantoms.map((s) => s.id), reason: "removed by hand: a likely phantom" }, `${phantoms.length} likely phantoms removed`);
  }
}

function goTo(item) {
  select({ kind: item.kind, id: item.id });
  const thing = item.kind === "object" ? objectById(item.id) : roomById(item.id);
  if (thing) {
    centreOn(...(item.kind === "object" ? thing.centre : roomCentre(thing)));
  }
  redraw();
}

function show(next) {
  world = next;
  document.getElementById("path").textContent = world.directory;
  document.getElementById("undo").disabled = !world.can_undo;
  document.getElementById("redo").disabled = !world.can_redo;
  document.getElementById("save").textContent = world.unsaved ? `Save (${world.unsaved})` : "Save";
  document.getElementById("labels").replaceChildren(...world.labels.map((word) => el("option", { value: word })));
  document.getElementById("room-types").replaceChildren(...world.room_types.map((word) => el("option", { value: word })));
  if (selected && !(selected.kind === "object" ? objectById(selected.id) : roomById(selected.id))) {
    selected = null;
  }
  renderPanel();
  renderReview();
  renderRemoved();
  redraw();
}

function select(next) {
  selected = next;
  preview = null;
  renderPanel();
  redraw();
}

function clearSelection() {
  select(null);
}

// --- modes and gestures ---------------------------------------------------------------------

function startMode(next) {
  if ((next === "merge" || next === "split") && (!selected || selected.kind !== "object")) {
    return say("select an object first", true);
  }
  mode = next;
  drawn = null;
  canvas.classList.toggle("drawing", next === "add" || next === "split");
  const words = {
    add: "Drag a box over the object the models missed.",
    split: `Drag a box over the part of O${selected && selected.id} to split off.`,
    merge: `Click the object to merge into O${selected && selected.id}.`,
  };
  hint.textContent = `${words[next]} Esc cancels.`;
  hint.classList.add("shown");
  renderPanel();
}

function stopMode() {
  mode = null;
  drawn = null;
  canvas.classList.remove("drawing");
  hint.classList.remove("shown");
  renderPanel();
  redraw();
}

function drawnBox() {
  const [a, , c] = drawn.corners;
  const box = { yaw: drawn.yaw, centre: [(a[0] + c[0]) / 2, (a[1] + c[1]) / 2] };
  const [u, v] = local({ ...box, size: [0, 0] }, a[0], a[1]);
  box.size = [Math.abs(2 * u), Math.abs(2 * v)];
  return box;
}

function deleteSelected() {
  if (!selected || selected.kind !== "object") {
    return;
  }
  const object = objectById(selected.id);
  if (object && window.confirm(`Delete O${object.id} (${object.label})?`)) {
    edit({ op: "delete", id: object.id }, `O${object.id} deleted`);
  }
}

function pointer(event) {
  const rect = canvas.getBoundingClientRect();
  return [event.clientX - rect.left, event.clientY - rect.top];
}

canvas.addEventListener("mousedown", (event) => {
  if (!world) {
    return;
  }
  const [sx, sy] = pointer(event);
  const [x, y] = toWorld(sx, sy);
  if (mode === "add" || mode === "split") {
    const frame = mode === "split" ? objectById(selected.id).yaw : wallYaw();
    drag = { kind: "draw", start: [x, y], yaw: frame };
    return;
  }
  const handle = event.button === 0 ? handleAt(sx, sy) : null;
  if (handle && !mode) {
    const object = objectById(selected.id);
    preview = { id: object.id, centre: [...object.centre], size: [...object.size], yaw: object.yaw };
    drag = { kind: handle.kind, index: handle.index, from: [x, y], at: [sx, sy], moved: false, origin: { ...preview, centre: [...preview.centre], size: [...preview.size] } };
    return;
  }
  drag = { kind: "pan", from: [sx, sy], view: { ...view }, moved: false };
});

canvas.addEventListener("mousemove", (event) => {
  if (!drag || !world) {
    return;
  }
  const [sx, sy] = pointer(event);
  const [x, y] = toWorld(sx, sy);
  if (drag.kind === "pan") {
    view.x = drag.view.x + sx - drag.from[0];
    view.y = drag.view.y + sy - drag.from[1];
    drag.moved = drag.moved || Math.hypot(sx - drag.from[0], sy - drag.from[1]) > 3;
  } else if (drag.kind === "draw") {
    const frame = { centre: drag.start, yaw: drag.yaw };
    const [u, v] = local(frame, x, y);
    drawn = { yaw: drag.yaw, corners: [[0, 0], [u, 0], [u, v], [0, v]].map(([a, b]) => fromLocal(frame, a, b)) };
  } else if (drag.kind === "move") {
    drag.moved = drag.moved || Math.hypot(sx - drag.at[0], sy - drag.at[1]) > 3;
    preview.centre = [drag.origin.centre[0] + x - drag.from[0], drag.origin.centre[1] + y - drag.from[1]];
  } else if (drag.kind === "rotate") {
    drag.moved = drag.moved || Math.hypot(sx - drag.at[0], sy - drag.at[1]) > 3;
    preview.yaw = Math.atan2(y - drag.origin.centre[1], x - drag.origin.centre[0]) - Math.PI / 2;
  } else if (drag.kind === "corner") {
    drag.moved = drag.moved || Math.hypot(sx - drag.at[0], sy - drag.at[1]) > 3;
    // The opposite corner stays where it is.
    const signs = [[1, 1], [-1, 1], [-1, -1], [1, -1]][drag.index];
    const fixed = fromLocal(drag.origin, (-signs[0] * drag.origin.size[0]) / 2, (-signs[1] * drag.origin.size[1]) / 2);
    const [u, v] = local({ centre: fixed, yaw: drag.origin.yaw }, x, y);
    preview.size = [Math.max(Math.abs(u), 0.05), Math.max(Math.abs(v), 0.05)];
    preview.centre = fromLocal({ centre: fixed, yaw: drag.origin.yaw }, u / 2, v / 2);
  }
  redraw();
});

window.addEventListener("mouseup", async (event) => {
  if (!drag || !world) {
    return;
  }
  const finished = drag;
  drag = null;
  if (finished.kind === "draw") {
    if (drawn) {
      const box = drawnBox();
      if (box.size[0] < 0.05 || box.size[1] < 0.05) {
        drawn = null;
      }
    }
    renderPanel();
    redraw();
    return;
  }
  if (finished.kind === "move" || finished.kind === "rotate" || finished.kind === "corner") {
    const box = preview;
    // A click on the box is no edit: only a drag pins it.
    if (finished.moved) {
      await edit({ op: "box", id: box.id, centre: box.centre, size: box.size, yaw: box.yaw }, `O${box.id}'s box set by hand`);
    }
    preview = null;
    redraw();
    return;
  }
  if (finished.kind === "pan" && !finished.moved && event.target === canvas) {
    const [x, y] = toWorld(...pointer(event));
    const object = objectAt(x, y);
    if (mode === "merge") {
      if (object && object.id !== selected.id && window.confirm(`Merge O${object.id} (${object.label}) into O${selected.id}?`)) {
        const into = selected.id;
        stopMode();
        await edit({ op: "merge", ids: [into, object.id], into }, `O${object.id} merged into O${into}`);
      }
      return;
    }
    if (object) {
      select({ kind: "object", id: object.id });
    } else {
      const room = roomAt(x, y);
      select(room ? { kind: "room", id: room.id } : null);
    }
  }
  redraw();
});

canvas.addEventListener("wheel", (event) => {
  if (!world) {
    return;
  }
  event.preventDefault();
  const [sx, sy] = pointer(event);
  const factor = Math.exp(-event.deltaY * 0.0015);
  const scale = Math.min(Math.max(view.scale * factor, 0.2), 40);
  view.x = sx - ((sx - view.x) * scale) / view.scale;
  view.y = sy - ((sy - view.y) * scale) / view.scale;
  view.scale = scale;
  redraw();
}, { passive: false });

window.addEventListener("keydown", (event) => {
  const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName);
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    return save();
  }
  if (typing) {
    if (event.key === "Enter") {
      const button = document.activeElement.closest("label")?.nextElementSibling?.querySelector("button");
      button?.click();
    }
    return;
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
    event.preventDefault();
    return command(event.shiftKey ? "/api/redo" : "/api/undo", event.shiftKey ? "redone" : "undone");
  }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
    event.preventDefault();
    return command("/api/redo", "redone");
  }
  if (event.key === "Escape") {
    return mode ? stopMode() : clearSelection();
  }
  if ((event.key === "Delete" || event.key === "Backspace") && selected) {
    return deleteSelected();
  }
  if (event.key.toLowerCase() === "c" && selected) {
    const thing = selected.kind === "object" ? objectById(selected.id) : roomById(selected.id);
    return selected.kind === "object"
      ? edit({ op: "check", id: thing.id, checked: !thing.checked }, `O${thing.id} ${thing.checked ? "unchecked" : "checked"}`)
      : edit({ op: "room_check", room: thing.id, checked: !thing.checked }, `${thing.id} ${thing.checked ? "unchecked" : "checked"}`);
  }
  if (event.key.toLowerCase() === "n" && world.suggestions.length) {
    const at = world.suggestions.findIndex((s) => selected && s.kind === selected.kind && s.id === selected.id);
    return goTo(world.suggestions[(at + 1) % world.suggestions.length]);
  }
});

document.getElementById("undo").addEventListener("click", () => command("/api/undo", "undone"));
document.getElementById("redo").addEventListener("click", () => command("/api/redo", "redone"));
document.getElementById("add").addEventListener("click", () => startMode("add"));
document.getElementById("save").addEventListener("click", save);
document.getElementById("remove-phantoms").addEventListener("click", removePhantoms);
document.getElementById("reload").addEventListener("click", () => {
  if (!world.unsaved || window.confirm(`Discard ${world.unsaved} unsaved edits?`)) {
    command("/api/reload", "read the world from disk again");
  }
});
window.addEventListener("resize", redraw);
window.addEventListener("beforeunload", (event) => {
  if (world && world.unsaved) {
    event.preventDefault();
    event.returnValue = "";
  }
});

(async () => {
  try {
    const answer = await call("/api/world");
    mapImage.onload = () => {
      fit();
      redraw();
    };
    mapImage.src = "/api/map.png";
    show(answer);
    const shown = answer.objects.filter((o) => o.shown).length;
    say(`${shown} objects in ${answer.rooms.length} rooms; ${answer.suggestions.length} to review`);
  } catch (error) {
    say(error.message, true);
  }
})();
