/* Seattle flat routes -- the route page.
 *
 * One card: where from, where to, and a slider from the shortest route to
 * the flattest. Everything runs in the page: the graph and the cost model
 * come from engine.js, and place search is an offline index built from the
 * graph's own street names plus Overture places and addresses packed into
 * the bundle (see sf_flat_routes/places.py).
 *
 * The slider is a family of routes, not one route: the whole frontier of
 * distance against climbing between the two points, every route that no
 * other route beats on both counts, sorted from shortest to flattest. A
 * weighted sum (length + alpha * climbing, swept over alpha) finds only the
 * frontier's convex hull and jumps straight across its dents, which on
 * some trips is most of the interesting routes; the frontier is found by a
 * bi-objective search instead (Graph.pareto in engine.js). Along it,
 * distance only ever grows and climbing only ever falls, so the slider does
 * exactly what its ends say. The shortest route appears at once and the
 * rest fills in over the next second or so; dragging is then instant, and
 * the map crossfades between neighbouring members.
 */
"use strict";

(function () {
  const MI = 1609.344, FT = 3.28084;
  const $ = (id) => document.getElementById(id);
  const DATA = window.DATA;

  /* lambda sweep for the slider; the min-climb objective is appended as
   * the last stop so the right-hand end is literally "fewest feet climbed" */
  /* The flat end of the frontier is where a metre of climb is worth
   * ALPHA_MAX metres of walking (the analysis's minimum-climb weight is
   * 120). Past about 200 the router starts walking miles to save a few feet
   * (7.5 miles instead of 4.6 to save 55 ft, on the default trip), which
   * nobody would call a route, so the frontier is cut there. */
  const ALPHA_MAX = 200;
  // furthest a pin may move to reach a street corner before a path will do
  const CORNER_MAX_M = 300;
  /* frontier points closer than this in climbing are merged */
  const EPS_GAIN_CM = 50;
  /* the same tolerance inside the search, at intermediate nodes */
  const EPS_NODE_CM = 10;
  /* at most this many routes on the slider, spread evenly along the frontier */
  const MAX_ROUTES = 30;

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const ease = (t) => 1 - Math.pow(1 - t, 3);

  function hexToRgb(h) {
    h = h.replace("#", "");
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const n = parseInt(h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  /* three-stop ramp, shortest -> middle -> flattest, matching the slider */
  function ramp(t) {
    const stops = [hexToRgb(css("--short")), hexToRgb(css("--mid")), hexToRgb(css("--route"))];
    const x = clamp(t, 0, 1) * 2, i = Math.min(1, Math.floor(x)), k = x - i;
    return "rgb(" + stops[i].map((v, c) => Math.round(lerp(v, stops[i + 1][c], k))).join(",") + ")";
  }

  const fmtMi = (m) => (m / MI < 10 ? (m / MI).toFixed(1) : Math.round(m / MI)) + "<small>mi</small>";
  const fmtFt = (m) => Math.round(m * FT).toLocaleString() + "<small>ft</small>";
  const fmtPct = (g) => (g * 100).toFixed(g * 100 < 10 ? 1 : 0) + "<small>%</small>";

  /* -------------------------------------------------------- text matching */
  /* Street-type words collapse to their abbreviations on both the index and
   * the query, so "Geary Blvd", "Geary Boulevard" and "geary" all match. */
  const ABBREV = { street: "st", avenue: "ave", boulevard: "blvd", drive: "dr", road: "rd",
    court: "ct", place: "pl", lane: "ln", terrace: "ter", highway: "hwy", parkway: "pkwy",
    circle: "cir", alley: "aly", square: "sq", stairway: "stwy", stairs: "stwy", way: "wy",
    north: "n", south: "s", east: "e", west: "w", saint: "st", mount: "mt" };
  const norm = (s) => s.toLowerCase()
    .replace(/[’']/g, "")
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .replace(/\s+/g, " ").trim()
    .split(" ").map((w) => ABBREV[w] || w).join(" ");

  /* 0 exact, 1 starts with, 2 every token is a word prefix, 3 substring, -1 none */
  function matchScore(nn, q, toks) {
    if (nn === q) return 0;
    if (nn.startsWith(q)) return 1;
    let all = true;
    for (const t of toks) { if (!(" " + nn).includes(" " + t)) { all = false; break; } }
    if (all) return 2;
    if (q.length >= 3 && nn.includes(q)) return 3;
    return -1;
  }

  /* -------------------------------------------------------- search index */
  class Index {
    constructor(graph, geom, bundle) {
      this.graph = graph; this.geom = geom;
      this.buildIntersections();
      this.places = [];
      if (DATA.manifest.strings.places) {
        const p = JSON.parse(bundle.text("places"));
        for (let i = 0; i < p.names.length; i++) {
          this.places.push({ name: p.names[i], nn: norm(p.names[i]), kind: p.groups[p.group[i]],
            lon: p.lon[i], lat: p.lat[i] });
        }
      }
      this.addr = null;
      if (DATA.addr && DATA.manifest.strings.addr_streets) {
        const streets = JSON.parse(bundle.text("addr_streets"));
        const st = bundle.array("addr_street");
        const start = new Int32Array(streets.length + 1);
        for (let i = 0; i < st.length; i++) start[st[i] + 1]++;
        for (let i = 0; i < streets.length; i++) start[i + 1] += start[i];
        this.addr = {
          streets, nn: streets.map(norm), start,
          number: bundle.array("addr_number"),
          lon: bundle.array("addr_lon"), lat: bundle.array("addr_lat"),
          origin: DATA.addr.origin, step: DATA.addr.step,
        };
      }
    }

    /* "24th St & Mission St": a node with two or more distinct street names */
    buildIntersections() {
      const g = this.graph, geom = this.geom;
      const per = new Array(g.n);
      const add = (node, nameIdx) => {
        if (!nameIdx) return;
        let s = per[node];
        if (!s) { s = per[node] = []; }
        if (s.indexOf(nameIdx) < 0) s.push(nameIdx);
      };
      for (let u = 0; u < g.n; u++) {
        for (let a = g.indptr[u]; a < g.indptr[u + 1]; a++) {
          const ni = geom.name[g.arcEdge[a]];
          add(u, ni); add(g.head[a], ni);
        }
      }
      this.nodeNames = per;
      const seen = new Map();
      const items = [];
      for (let u = 0; u < g.n; u++) {
        const s = per[u];
        if (!s || s.length < 2) continue;
        const names = s.map((i) => geom.names[i - 1]).sort();
        const key = names.join("|");
        if (seen.has(key)) continue;
        seen.set(key, items.length);
        items.push({ name: names.slice(0, 3).join(" & "), parts: names.map(norm), node: u,
          lon: g.nodeLon(u), lat: g.nodeLat(u), kind: "intersection" });
      }
      this.intersections = items;
    }

    /* nearest named corner, for labelling a dropped pin */
    describe(node, grid) {
      const g = this.graph;
      const corner = (s) => s.slice(0, 2).map((i) => this.geom.names[i - 1]).join(" & ");
      const s = this.nodeNames[node];
      if (s && s.length >= 2) return corner(s);
      const near = grid ? grid.nearest(g.nodeLon(node), g.nodeLat(node),
        (i) => this.nodeNames[i] && this.nodeNames[i].length >= 2) : -1;
      if (near >= 0) return corner(this.nodeNames[near]);
      if (s && s.length === 1) return this.geom.names[s[0] - 1];
      return g.nodeLat(node).toFixed(4) + ", " + g.nodeLon(node).toFixed(4);
    }

    search(raw, limit = 8) {
      const q = norm(raw);
      if (!q) return [];
      const toks = q.split(" ");
      const qq = q.replace(/ /g, "");
      const out = [];

      // "1234 Valencia" -- a street address
      const am = /^(\d+)\s+(\D.*)$/.exec(q);
      if (am && this.addr) {
        const want = +am[1], sq = am[2], stoks = sq.split(" ");
        const hits = [];
        for (let i = 0; i < this.addr.streets.length; i++) {
          const sc = matchScore(this.addr.nn[i], sq, stoks);
          if (sc >= 0 && sc <= 2) hits.push([sc, i]);
        }
        hits.sort((a, b) => a[0] - b[0] || this.addr.streets[a[1]].length - this.addr.streets[b[1]].length);
        for (const [sc, si] of hits.slice(0, 4)) {
          const a = this.addr, lo = a.start[si], hi = a.start[si + 1];
          // numbers are sorted within a street: binary search for the nearest
          let l = lo, h = hi - 1;
          while (l < h) { const m = (l + h) >> 1; if (a.number[m] < want) l = m + 1; else h = m; }
          let best = l;
          if (l > lo && Math.abs(a.number[l - 1] - want) < Math.abs(a.number[l] - want)) best = l - 1;
          const num = a.number[best];
          const exact = num === want;
          out.push({ score: exact ? -1 : sc, name: num + " " + a.streets[si],
            kind: exact ? "address" : "nearest address", rank: 0,
            lon: a.origin[0] + a.lon[best] * a.step, lat: a.origin[1] + a.lat[best] * a.step });
        }
      }

      // "24th & mission" -- an intersection
      const parts = raw.toLowerCase().split(/\s+(?:and|at)\s+|\s*[&\/@+]\s*/).map(norm).filter(Boolean);
      if (parts.length === 2) {
        for (const it of this.intersections) {
          let ok = 0;
          for (const p of parts) { if (it.parts.some((n) => n.startsWith(p) || (" " + n).includes(" " + p))) ok++; }
          if (ok === 2) out.push({ score: 1, rank: 1, ...it });
        }
      } else if (parts.length === 1) {
        for (const it of this.intersections) {
          if (it.parts.some((n) => n.startsWith(q))) out.push({ score: 2, rank: 3, ...it });
        }
      }

      // places
      // mapped features and landmarks first, then everyday places
      const KIND_RANK = { landmark: 1, transit: 1, civic: 1, shop: 2, food: 2, lodging: 2 };
      for (const p of this.places) {
        let sc = matchScore(p.nn, q, toks);
        if (sc < 0 && qq.length >= 4 && p.nn.replace(/ /g, "").startsWith(qq)) sc = 2;
        if (sc >= 0) out.push({ score: sc, rank: 1 + (KIND_RANK[p.kind] || 0), ...p });
      }

      out.sort((a, b) => a.score - b.score || a.rank - b.rank || a.name.length - b.name.length);
      // one intersection per pair of streets is already guaranteed; dedupe places by name
      const seen = new Set(), res = [];
      for (const r of out) {
        const k = r.kind + "|" + r.name;
        if (seen.has(k)) continue;
        seen.add(k); res.push(r);
        if (res.length >= limit) break;
      }
      return res;
    }
  }

  /* --------------------------------------------- faint street canvas layer */
  const StreetLayer = L.Layer.extend({
    initialize(geom) { this.geom = geom; },
    onAdd(map) {
      this._map = map;
      this._canvas = L.DomUtil.create("canvas", "leaflet-zoom-animated");
      map.getPanes().overlayPane.appendChild(this._canvas);
      map.on("moveend zoomend resize", this._redraw, this);
      map.on("zoomanim", this._animate, this);
      this._redraw();
    },
    onRemove(map) {
      L.DomUtil.remove(this._canvas);
      map.off("moveend zoomend resize", this._redraw, this);
      map.off("zoomanim", this._animate, this);
    },
    _animate(e) {
      const scale = this._map.getZoomScale(e.zoom);
      const offset = this._map._latLngToNewLayerPoint(
        this._map.getBounds().getNorthWest(), e.zoom, e.center);
      L.DomUtil.setTransform(this._canvas, offset, scale);
    },
    _redraw() {
      const map = this._map; if (!map) return;
      const size = map.getSize(), dpr = window.devicePixelRatio || 1;
      if (this._canvas.width !== size.x * dpr || this._canvas.height !== size.y * dpr) {
        this._canvas.width = size.x * dpr; this._canvas.height = size.y * dpr;
        this._canvas.style.width = size.x + "px"; this._canvas.style.height = size.y + "px";
      }
      const nw = map.getBounds().getNorthWest();
      L.DomUtil.setTransform(this._canvas, map.latLngToLayerPoint(nw), 1);
      const ctx = this._canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, size.x, size.y);
      const z = map.getZoom();
      if (z < 11) return;
      const b = map.getBounds().pad(0.05);
      const west = b.getWest(), east = b.getEast(), south = b.getSouth(), north = b.getNorth();
      const minLen = z >= 15 ? 0 : z >= 14 ? 10 : z >= 13 ? 20 : 40;
      const g = this.geom, origin = map.latLngToLayerPoint(nw);
      ctx.strokeStyle = css("--street") || "#c9ccc9";
      ctx.lineWidth = z >= 16 ? 1.6 : z >= 14 ? 1.1 : 0.8;
      ctx.lineCap = "round";
      ctx.beginPath();
      for (let i = 0; i < g.nEdges; i++) {
        const o = i * 4;
        if (g.bbox[o + 2] < west || g.bbox[o] > east || g.bbox[o + 3] < south || g.bbox[o + 1] > north) continue;
        if (minLen && g.len[i] / g.DM < minLen) continue;
        const s = g.starts[i], e = g.starts[i + 1];
        for (let k = s; k < e; k += 2) {
          const p = map.latLngToLayerPoint([g.coords[k + 1], g.coords[k]]);
          if (k === s) ctx.moveTo(p.x - origin.x, p.y - origin.y); else ctx.lineTo(p.x - origin.x, p.y - origin.y);
        }
      }
      ctx.stroke();
    },
  });

  /* -------------------------------------------------------------- the app */
  const App = {
    state: { mode: "walk", from: null, to: null, t: 1, focus: "from", calm: true },
    family: null, shown: null, fading: null,

    async start() {
      $("status").textContent = "Loading the street network…";
      const buf = await loadBundle(DATA, (got, total) => {
        $("status").textContent = "Loading the street network… "
          + (total ? Math.round(100 * got / total) + "%" : (got / 1e6).toFixed(1) + " MB");
      });
      const bundle = new Bundle(buf, DATA.manifest);
      this.graph = new Graph(bundle, DATA.meta);
      this.geom = new Geometry(bundle, DATA.meta);
      DATA.bundle = null;
      const g = this.graph;
      const lonF = new Float32Array(g.n), latF = new Float32Array(g.n);
      for (let i = 0; i < g.n; i++) { lonF[i] = g.nodeLon(i); latF[i] = g.nodeLat(i); }
      this.nodeGrid = new Grid(lonF, latF, 0.004);
      this.index = new Index(g, this.geom, bundle);

      this.buildMap();
      this.buildUI();
      $("status").textContent = "";
      if (!this.readHash() && DATA.default && DATA.default.length === 2) {
        this.setPoint("from", this.placeToPoint(DATA.default[0]), false);
        this.setPoint("to", this.placeToPoint(DATA.default[1]), false);
      }
      this.recompute(true);
    },

    /* ---------------------------------------------------------------- map */
    buildMap() {
      const map = L.map("map", {
        zoomControl: false, attributionControl: true, preferCanvas: true,
        center: [47.615, -122.335], zoom: 12, minZoom: 10.5, maxZoom: 18, zoomSnap: 0.5,
      });
      map.attributionControl.setPrefix("");
      map.attributionControl.addAttribution(
        "Streets © <a href='https://overturemaps.org'>Overture</a> / <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> · Elevation USGS 3DEP");
      L.control.zoom({ position: "bottomright" }).addTo(map);
      this.map = map;

      if (DATA.hillshade) {
        const hs = L.imageOverlay(DATA.hillshade.url || DATA.hillshade.data_uri, DATA.hillshade.bounds,
          { opacity: 1, interactive: false, className: "hillshade" }).addTo(map);
        hs.on("error", () => map.getContainer().classList.add("no-shade"));
      } else {
        map.getContainer().classList.add("no-shade");
      }
      this.streets = new StreetLayer(this.geom).addTo(map);

      // sparse neighborhood names, hidden when zoomed far out or far in
      const labels = L.layerGroup().addTo(map);
      for (const l of DATA.labels || []) {
        L.marker([l.lat, l.lon], { interactive: false, keyboard: false,
          icon: L.divIcon({ className: "nblabel", html: l.n, iconSize: null }) }).addTo(labels);
      }
      const zoomClass = () => {
        const z = map.getZoom();
        map.getContainer().classList.toggle("z-low", z < 12);
        map.getContainer().classList.toggle("z-high", z >= 15.5);
      };
      map.on("zoomend", zoomClass); zoomClass();
      map.on("zoomend", () => { if (this._labelled) this.labelRoute(this._labelled); });

      this.familyLayer = L.layerGroup().addTo(map);
      this.routeLayer = L.layerGroup().addTo(map);
      this.markers = L.layerGroup().addTo(map);
      this.labelLayer = L.layerGroup().addTo(map);

      map.on("click", (e) => {
        const which = !this.state.from ? "from" : (!this.state.to ? "to" : this.state.focus);
        this.setPoint(which, this.pointAt(e.latlng.lng, e.latlng.lat), true);
        this.recompute("auto");
      });
    },

    /* ---------------------------------------------------------- endpoints */
    /* Snap to the nearest street corner (node flag bit 3); only where there
     * is none within CORNER_MAX_M, deep in a park, to the nearest path node. */
    nearestNode(lon, lat) {
      const g = this.graph, bit = g.modeBit(this.state.mode);
      const corner = this.nodeGrid.nearest(lon, lat,
        (i) => (g.nodeFlags[i] & bit) !== 0 && (g.nodeFlags[i] & 8) !== 0);
      if (corner >= 0) {
        const kx = 111320 * Math.cos(lat * Math.PI / 180);
        const d = Math.hypot((g.nodeLon(corner) - lon) * kx, (g.nodeLat(corner) - lat) * 111320);
        if (d <= CORNER_MAX_M) return corner;
      }
      return this.nodeGrid.nearest(lon, lat, (i) => (g.nodeFlags[i] & bit) !== 0);
    },
    /* A point is always a routable node: whatever was clicked or searched
     * snaps to one, so the pin sits where the route actually starts rather
     * than in the bay or the middle of a building. */
    pointAt(lon, lat, label) {
      const node = this.nearestNode(lon, lat);
      if (node < 0) return null;
      const g = this.graph;
      return { lon: g.nodeLon(node), lat: g.nodeLat(node), node,
        label: label || ("near " + this.index.describe(node, this.nodeGrid)) };
    },
    placeToPoint(p) { return this.pointAt(p.lon, p.lat, p.label || p.name); },

    setPoint(which, pt, typed) {
      this.state[which] = pt;
      const input = $(which);
      input.value = pt ? pt.label : "";
      input.dataset.set = pt ? "1" : "";
      this.hideSuggest(which);
      this.drawMarkers();
      if (typed && !this.state[which === "from" ? "to" : "from"]) {
        $(which === "from" ? "to" : "from").focus();
      }
    },

    drawMarkers() {
      this.markers.clearLayers();
      for (const which of ["from", "to"]) {
        const p = this.state[which]; if (!p) continue;
        const m = L.marker([p.lat, p.lon], {
          draggable: true, keyboard: false, title: which === "from" ? "Start" : "Destination",
          icon: L.divIcon({ className: "pin-icon " + which, iconSize: [18, 18], iconAnchor: [9, 9] }),
        }).addTo(this.markers);
        m.on("dragend", () => {
          const ll = m.getLatLng();
          this.setPoint(which, this.pointAt(ll.lng, ll.lat), false);
          this.recompute(false);
        });
      }
    },

    /* ------------------------------------------------------------- search */
    buildUI() {
      for (const which of ["from", "to"]) {
        const input = $(which), list = $(which + "_s");
        let sel = -1, items = [];
        const render = () => {
          list.innerHTML = "";
          items.forEach((it, i) => {
            const li = document.createElement("li");
            li.setAttribute("role", "option");
            li.setAttribute("aria-selected", i === sel ? "true" : "false");
            li.innerHTML = "<span class='n'></span><span class='k'></span>";
            li.firstChild.textContent = it.name;
            li.lastChild.textContent = it.kind;
            li.addEventListener("mousedown", (e) => { e.preventDefault(); pick(i); });
            list.appendChild(li);
          });
          list.hidden = items.length === 0;
        };
        const pick = (i) => {
          const it = items[i]; if (!it) return;
          const pt = it.node !== undefined ? { lon: it.lon, lat: it.lat, node: it.node, label: it.name }
            : this.pointAt(it.lon, it.lat, it.name);
          items = []; render();
          this.setPoint(which, pt, true);
          this.recompute("auto");
        };
        input.addEventListener("focus", () => {
          this.state.focus = which; $("card").classList.add("typing");
          if (input.dataset.set) input.select();
        });
        input.addEventListener("input", () => {
          input.dataset.set = "";
          items = this.index.search(input.value); sel = items.length ? 0 : -1; render();
        });
        input.addEventListener("keydown", (e) => {
          if (e.key === "ArrowDown" && items.length) { sel = (sel + 1) % items.length; render(); e.preventDefault(); }
          else if (e.key === "ArrowUp" && items.length) { sel = (sel - 1 + items.length) % items.length; render(); e.preventDefault(); }
          else if (e.key === "Enter") { if (sel >= 0) pick(sel); e.preventDefault(); }
          else if (e.key === "Escape") { items = []; render(); input.blur(); }
        });
        input.addEventListener("blur", () => {
          setTimeout(() => {
            items = []; render();
            if (!input.dataset.set && this.state[which]) input.value = this.state[which].label;
            if (document.activeElement !== $("from") && document.activeElement !== $("to")) $("card").classList.remove("typing");
          }, 120);
        });
        this["hide_" + which] = () => { items = []; render(); };
      }

      $("swap").addEventListener("click", () => {
        const a = this.state.from, b = this.state.to;
        this.setPoint("from", b, false); this.setPoint("to", a, false);
        this.recompute("auto");
      });
      for (const btn of $("mode").querySelectorAll("button")) {
        btn.addEventListener("click", () => {
          if (this.state.mode === btn.dataset.v) return;
          this.state.mode = btn.dataset.v;
          for (const b of $("mode").querySelectorAll("button")) b.setAttribute("aria-pressed", b === btn ? "true" : "false");
          $("calmrow").hidden = this.state.mode !== "bike";
          // endpoints may sit on stairs or a footpath that a bike cannot use
          for (const w of ["from", "to"]) {
            const p = this.state[w]; if (!p) continue;
            this.state[w] = Object.assign({}, p, { node: this.nearestNode(p.lon, p.lat) });
          }
          this.recompute(false);
        });
      }
      $("calm").addEventListener("change", () => {
        this.state.calm = $("calm").checked;
        this.recompute(false);
      });
      const sl = $("sl");
      sl.addEventListener("input", () => { this.state.t = +sl.value; this.show(); this.writeHash(); });
      $("share").addEventListener("click", () => {
        const url = this.shareUrl(), box = $("sharebox"), btn = $("share");
        const done = () => { btn.textContent = "Link copied"; setTimeout(() => { btn.textContent = "Copy link"; }, 1800); };
        const fallback = () => { box.value = url; box.hidden = false; box.focus(); box.select(); };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(url).then(done, fallback);
        } else fallback();
      });
      window.addEventListener("resize", () => { if (this.shown) this.drawProfile(this.shown, this.shown, 1); });
    },
    hideSuggest(which) { if (this["hide_" + which]) this["hide_" + which](); },

    /* ------------------------------------------------------------ routing */
    /* length + alpha * gain, on real length (no comfort multipliers), so the
     * family is a true distance-versus-climbing trade-off. On a bike with
     * calm streets on, length is comfort-weighted instead (engine.js
     * arcLenStress): a protected lane counts shorter, a busy arterial
     * longer, and the climbing axis is untouched. */
    calm() { return this.state.mode === "bike" && this.state.calm; },
    lenKey() { return this.calm() ? "stress_m" : "distance_m"; },
    weights(alpha) {
      return { alpha, beta: 0, gamma: 0, penalties: [0, 0, 0, 0, 0], extreme: 0,
        use_class_multiplier: false, stress: this.calm() };
    },

    recompute(fit) {
      const { from, to, mode } = this.state;
      this.family = null;
      const gen = ++this._gen;
      if (!from || !to) {
        this.clearRoute();
        $("status").textContent = !from && !to ? "Type two places, or click the map twice."
          : (!from ? "Where are you starting from?" : "Where to?");
        return;
      }
      if (from.node === to.node) { this.clearRoute(); $("status").textContent = "Those are the same corner."; return; }
      const g = this.graph;
      const shortest = g.route(from.node, to.node, mode, this.weights(0));
      if (!shortest) {
        this.clearRoute();
        $("status").textContent = mode === "bike" ? "No bikeable route between those points."
          : "No route between those points.";
        return;
      }
      // the shortest and the flattest routes go up at once, so the map can
      // be fitted to the whole family; the frontier fills in between them
      const first = this.member(shortest.arcs);
      const flat = g.route(from.node, to.node, mode, this.weights(ALPHA_MAX)) || shortest;
      const fm = this.member(flat.arcs);
      const flatStats = fm.stats;
      const sameEnds = fm.arcs.length === first.arcs.length && fm.arcs.every((a, i) => a === first.arcs[i]);
      this.family = { unique: sameEnds ? [first] : [first, fm], shortest: first, partial: true };
      this.family.unique.forEach((m, i) => { m.id = i; });
      this.shown = null;
      $("status").textContent = "Finding every route between shortest and flattest…";
      $("sl").disabled = true;
      this.drawFamily();
      this.show(true);
      if (fit === true || (fit === "auto" && !this.inView())) this.fit();
      this.writeHash();
      $("share").hidden = false; $("sharebox").hidden = true;

      const search = g.pareto(from.node, to.node, mode, {
        eps: EPS_GAIN_CM, epsNode: EPS_NODE_CM, stress: this.calm(),
        dCap: Math.round(flatStats[this.lenKey()] * g.DM) + 1,
        gCap: Math.round(first.stats.elev_gain_m * g.CM) + 1,
      });
      const run = () => {
        if (gen !== this._gen) return;          // the trip changed underneath us
        if (!search.step(30)) {
          $("status").textContent = "Finding every route between shortest and flattest… "
            + search.solutions.length;
          setTimeout(run, 0);
          return;
        }
        this.finishFamily(search, first, fm, fit);
      };
      setTimeout(run, 0);
    },

    member(arcs) {
      const s = this.graph.summarise(arcs);
      return { arcs, stats: s, latlngs: this.graph.geometry(arcs, this.geom),
        profile: resample(s.profile, 160) };
    },

    /* the frontier is in, sorted shortest to flattest: pick the routes the
     * slider will step through */
    finishFamily(search, first, fm, fit) {
      let members = search.solutions.map((r) => this.member(r.arcs));
      if (!members.length) members = [first];
      const key = this.lenKey();
      members.sort((a, b) => a.stats[key] - b.stats[key]);
      // The weighted flattest route is a frontier point by construction.
      // The search's tolerances can leave it out by a few feet, and if the
      // search was cut short it is missing altogether, so it closes the
      // family whenever it beats what the search found.
      const last = members[members.length - 1];
      if (fm.stats.elev_gain_m < last.stats.elev_gain_m - 1e-6) members.push(fm);
      this._search = search;
      members = thinFrontier(members, MAX_ROUTES);
      members.forEach((m, i) => { m.id = i; });
      this.family = { unique: members, shortest: members[0], partial: false };
      $("sl").disabled = false;
      $("status").textContent = members.length === 1
        ? "One route: the shortest is already the flattest."
        : members.length + " distinct routes, from shortest to flattest.";
      this.drawFamily();
      this.show(false);
      if (fit && !this.inView()) this.fit();
    },

    clearRoute() {
      this.familyLayer.clearLayers(); this.routeLayer.clearLayers(); this.labelLayer.clearLayers();
      $("turns").hidden = true;
      this.shown = null;
      $("result").hidden = true; $("prof").hidden = true; $("delta").textContent = ""; $("slpos").textContent = "";
      $("share").hidden = true; $("sharebox").hidden = true;
      $("sl").disabled = false;
    },

    /* slider position -> index into the family, evenly over its members */
    stepAt(t) {
      const n = this.family.unique.length;
      return clamp(Math.round(t * (n - 1)), 0, n - 1);
    },

    drawFamily() {
      this.familyLayer.clearLayers();
      for (const u of this.family.unique) {
        L.polyline(u.latlngs, { color: css("--family"), weight: 2, opacity: 0.45, interactive: false,
          lineJoin: "round", lineCap: "round" }).addTo(this.familyLayer);
      }
    },

    /* show the family member for the current slider position */
    show(immediate) {
      if (!this.family) return;
      const t = this.state.t, step = this.stepAt(t), n = this.family.unique.length;
      const u = this.family.unique[step];
      const colour = ramp(t);
      $("slpos").textContent = this.family.partial ? "" : (n === 1 ? "" : (step + 1) + " of " + n);
      const prev = this.shown;
      if (prev === u) { this.tintRoute(colour); return; }
      this.shown = u;
      this.drawRoute(u, colour, prev && !immediate ? prev : null);
      this.drawStats(u, prev && !immediate ? prev : u);
      this.animateProfile(prev && !immediate ? prev : u, u);
    },

    tintRoute(colour) {
      if (this._line) this._line.setStyle({ color: colour });
      $("prof").dataset.colour = colour;
      if (this.shown) this.drawProfile(this.shown, this.shown, 1);
    },

    drawRoute(u, colour, prev) {
      // crossfade: the old line fades out while the new one fades in
      const casing = css("--route-casing");
      if (this._line && prev) {
        const oldCase = this._casing, oldLine = this._line;
        fadeOut([oldCase, oldLine], 260, () => { this.routeLayer.removeLayer(oldCase); this.routeLayer.removeLayer(oldLine); });
      } else {
        this.routeLayer.clearLayers();
      }
      this._casing = L.polyline(u.latlngs, { color: casing, weight: 10, opacity: prev ? 0 : 0.9, interactive: false,
        lineJoin: "round", lineCap: "round" }).addTo(this.routeLayer);
      this._line = L.polyline(u.latlngs, { color: colour, weight: 5, opacity: prev ? 0 : 1, interactive: false,
        lineJoin: "round", lineCap: "round" }).addTo(this.routeLayer);
      if (prev) fadeIn([[this._casing, 0.9], [this._line, 1]], 260);
      this._casing.bringToFront(); this._line.bringToFront();
      $("prof").dataset.colour = colour;
      this.labelRoute(u);
    },

    /* The streets a route follows, as runs of consecutive arcs sharing a
     * name: [{name, arcs, length_m, start (index into latlngs)}]. Unnamed
     * stubs and runs under minRun metres are dropped from the list. */
    runs(u, minRun = 40) {
      const g = this.graph, geom = this.geom, out = [];
      let cur = null;
      for (const a of u.arcs) {
        const n = geom.name[g.arcEdge[a]], name = n ? geom.names[n - 1] : null;
        const L = g.arcLen[a] / g.DM;
        if (cur && cur.name === name) { cur.arcs.push(a); cur.length_m += L; }
        else { cur = { name, arcs: [a], length_m: L }; out.push(cur); }
      }
      // drop unnamed stubs and short runs, then re-merge what that joins
      // up ("18th St, 18th St" either side of a nameless crossing)
      const kept = out.filter((r) => r.name && r.length_m >= minRun), merged = [];
      for (const r of kept) {
        const last = merged[merged.length - 1];
        if (last && last.name === r.name) { last.arcs = last.arcs.concat(r.arcs); last.length_m += r.length_m; }
        else merged.push({ name: r.name, arcs: r.arcs.slice(), length_m: r.length_m });
      }
      return merged;
    },

    /* Street names drawn along the highlighted route: one per named run,
     * rotated to the line's bearing, only where the run is long enough on
     * screen to carry its text, never overlapping another label. Redrawn
     * on zoom, since what fits changes. */
    labelRoute(u) {
      this.labelLayer.clearLayers();
      this._labelled = u;
      if (!u) return;
      const g = this.graph, geom = this.geom, map = this.map;
      const runs = this.runs(u, 80).sort((a, b) => b.length_m - a.length_m);
      const placed = [];
      for (const r of runs) {
        if (placed.length >= 10) break;
        const text = shortStreet(r.name);
        const pts = [];
        for (const a of r.arcs) for (const ll of g.geometry([a], geom)) {
          const last = pts[pts.length - 1];
          if (!last || last[0] !== ll[0] || last[1] !== ll[1]) pts.push(ll);
        }
        if (pts.length < 2) continue;
        // pixel length along the run at this zoom, and its midpoint
        const px = pts.map((ll) => map.latLngToContainerPoint(ll));
        const seg = [];
        let total = 0;
        for (let i = 1; i < px.length; i++) { const d = px[i].distanceTo(px[i - 1]); seg.push(d); total += d; }
        const need = text.length * 6.6 + 28;
        if (total < need) continue;
        let acc = 0, i = 1;
        while (i < pts.length - 1 && acc + seg[i - 1] < total / 2) { acc += seg[i - 1]; i++; }
        const f = seg[i - 1] ? (total / 2 - acc) / seg[i - 1] : 0;
        const mid = px[i - 1].add(px[i].subtract(px[i - 1]).multiplyBy(f));
        if (placed.some((q) => q.distanceTo(mid) < need * 0.6)) continue;
        let deg = Math.atan2(px[i].y - px[i - 1].y, px[i].x - px[i - 1].x) * 180 / Math.PI;
        if (deg > 90) deg -= 180; else if (deg < -90) deg += 180;
        const m = L.marker(map.containerPointToLatLng(mid), { interactive: false, keyboard: false,
          icon: L.divIcon({ className: "rtlabel", iconSize: null,
            html: `<span style="--rot:${deg.toFixed(1)}deg"></span>` }) }).addTo(this.labelLayer);
        m.getElement().firstChild.textContent = text;
        placed.push(mid);
      }
    },

    drawStats(u, prevU) {
      $("result").hidden = false;
      const runs = this.runs(u, 60);
      const box = $("turns");
      box.innerHTML = "";
      const render = (all) => {
        box.innerHTML = "";
        const show = all || runs.length <= 8 ? runs : runs.slice(0, 7);
        show.forEach((r, i) => {
          if (i) { const v = document.createElement("span"); v.className = "via"; v.textContent = "→"; box.appendChild(v); }
          box.appendChild(document.createTextNode(shortStreet(r.name)));
        });
        if (show.length < runs.length) {
          const more = document.createElement("button");
          more.type = "button"; more.className = "link more";
          more.textContent = "+" + (runs.length - show.length) + " more";
          more.addEventListener("click", () => render(true));
          box.appendChild(more);
        }
      };
      render(false);
      box.hidden = runs.length === 0;
      const s = u.stats, p = prevU.stats;
      tween(260, (k) => {
        $("v_dist").innerHTML = fmtMi(lerp(p.distance_m, s.distance_m, k));
        $("v_climb").innerHTML = fmtFt(lerp(p.elev_gain_m, s.elev_gain_m, k));
        $("v_grade").innerHTML = fmtPct(lerp(p.max_grade, s.max_grade, k));
      });
      const sh = this.family.shortest.stats;
      if (u === this.family.shortest) {
        $("delta").innerHTML = this.family.unique.length > 1
          ? (this.calm() ? "The shortest route on calm streets. Slide right to trade distance for less climbing."
            : "The shortest route. Slide right to trade distance for less climbing.")
          : "Shortest and flattest at once.";
      } else {
        const dd = s.distance_m - sh.distance_m, dc = sh.elev_gain_m - s.elev_gain_m;
        const pd = sh.distance_m ? Math.round(100 * dd / sh.distance_m) : 0;
        const pc = sh.elev_gain_m ? Math.round(100 * dc / sh.elev_gain_m) : 0;
        const longer = dd < 80 ? "about the same distance"
          : "<b class='up'>+" + (dd / MI).toFixed(1) + " mi</b> (" + pd + "% longer)";
        const less = dc <= 0 ? "no less climbing"
          : "<b class='down'>−" + Math.round(dc * FT).toLocaleString() + " ft</b> of climbing (" + pc + "% less)";
        $("delta").innerHTML = "vs. shortest: " + longer + ", " + less;
      }
    },

    /* ------------------------------------------------------------ profile */
    animateProfile(a, b) {
      $("prof").hidden = false;
      if (this._profAnim) cancelAnimationFrame(this._profAnim);
      const t0 = performance.now();
      const frame = (now) => {
        const k = clamp((now - t0) / 300, 0, 1);
        this.drawProfile(a, b, ease(k));
        if (k < 1) this._profAnim = requestAnimationFrame(frame);
      };
      this._profAnim = requestAnimationFrame(frame);
    },

    drawProfile(a, b, k) {
      const cv = $("prof"), dpr = window.devicePixelRatio || 1;
      const W = cv.clientWidth || 360, H = cv.clientHeight || 92;
      if (cv.width !== W * dpr || cv.height !== H * dpr) { cv.width = W * dpr; cv.height = H * dpr; }
      const ctx = cv.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);
      const n = b.profile.z.length;
      const z = new Float64Array(n);
      for (let i = 0; i < n; i++) z[i] = lerp(a.profile.z[i], b.profile.z[i], k);
      const dist = lerp(a.stats.distance_m, b.stats.distance_m, k);
      // a fixed vertical scale across the family keeps the hills comparable
      let zmin = Infinity, zmax = -Infinity;
      for (const u of this.family.unique) for (const v of u.profile.z) { if (v < zmin) zmin = v; if (v > zmax) zmax = v; }
      const span = Math.max(zmax - zmin, 15);
      zmin -= span * 0.08; zmax = zmin + span * 1.2;
      const padL = 6, padR = 6, top = 8, bottom = 18;
      const X = (i) => padL + (W - padL - padR) * i / (n - 1);
      const Y = (v) => top + (H - top - bottom) * (1 - (v - zmin) / (zmax - zmin));
      const colour = cv.dataset.colour || css("--route");
      ctx.beginPath();
      ctx.moveTo(X(0), Y(zmin));
      for (let i = 0; i < n; i++) ctx.lineTo(X(i), Y(z[i]));
      ctx.lineTo(X(n - 1), Y(zmin)); ctx.closePath();
      ctx.fillStyle = colour; ctx.globalAlpha = 0.18; ctx.fill(); ctx.globalAlpha = 1;
      ctx.beginPath();
      for (let i = 0; i < n; i++) { if (i) ctx.lineTo(X(i), Y(z[i])); else ctx.moveTo(X(i), Y(z[i])); }
      ctx.strokeStyle = colour; ctx.lineWidth = 2; ctx.lineJoin = "round"; ctx.stroke();
      // labels: start and end elevation, the high point, the distance scale
      ctx.fillStyle = css("--muted"); ctx.font = "500 10px " + css("--mono");
      ctx.textBaseline = "alphabetic";
      let hi = 0; for (let i = 1; i < n; i++) if (z[i] > z[hi]) hi = i;
      const lab = (v) => Math.round(v * FT) + " ft";
      ctx.textAlign = "left"; ctx.fillText(lab(z[0]), padL, H - 5);
      ctx.textAlign = "right"; ctx.fillText(lab(z[n - 1]), W - padR, H - 5);
      ctx.textAlign = "center"; ctx.fillText((dist / MI).toFixed(1) + " mi", W / 2, H - 5);
      if (hi > n * 0.06 && hi < n * 0.94 && z[hi] - Math.min(z[0], z[n - 1]) > 6) {
        ctx.textAlign = X(hi) < 40 ? "left" : X(hi) > W - 40 ? "right" : "center";
        ctx.fillStyle = css("--ink");
        ctx.fillText(lab(z[hi]), X(hi), Math.max(10, Y(z[hi]) - 5));
      }
    },

    familyBounds() {
      const b = L.latLngBounds([]);
      for (const u of this.family.unique) for (const ll of u.latlngs) b.extend(ll);
      return b;
    },
    /* is the whole family inside the part of the map the card does not cover? */
    inView() {
      if (!this.family) return true;
      const map = this.map, size = map.getSize(), card = $("card").getBoundingClientRect();
      const b = this.familyBounds();
      const sw = map.latLngToContainerPoint(b.getSouthWest()), ne = map.latLngToContainerPoint(b.getNorthEast());
      const wide = size.x > 640;
      const x0 = wide ? card.right + 16 : 16, y1 = wide ? size.y - 16 : card.top - 16;
      return sw.x >= x0 && ne.x <= size.x - 16 && ne.y >= 16 && sw.y <= y1;
    },
    fit() {
      if (!this.family) return;
      const b = this.familyBounds();
      const size = this.map.getSize();
      const wide = size.x > 640;
      const card = $("card").getBoundingClientRect();
      this.map.fitBounds(b, wide
        ? { paddingTopLeft: [card.right + 24, 24], paddingBottomRight: [40, 40], maxZoom: 15 }
        : { paddingTopLeft: [16, 16], paddingBottomRight: [16, card.height + 16], maxZoom: 15 });
    },

    /* ------------------------------------------------------------ sharing */
    /* The trip travels in the URL fragment as one bare token of letters,
     * digits and . _ ~ - (labels hex-escaped), which survives any host,
     * link shortener or chat client that mangles key=value fragments. */
    token() {
      const { from, to, mode, t } = this.state;
      const c = (p) => p.lon.toFixed(5) + "~" + p.lat.toFixed(5);
      return ["t", c(from), c(to), mode === "bike" ? (this.state.calm ? "b" : "bx") : "w", t.toFixed(3),
        encLabel(from.label), encLabel(to.label)].join("~");
    },
    writeHash() {
      const { from, to } = this.state;
      if (!from || !to) return;
      try { history.replaceState(null, "", "#" + this.token()); } catch (e) { /* sandboxed */ }
    },
    shareUrl() {
      let base = "";
      try { base = location.href.split("#")[0]; } catch (e) { /* sandboxed */ }
      return base + "#" + this.token();
    },
    readHash() {
      let h = "";
      try { h = location.hash; } catch (e) { return false; }
      if (!h || h.length < 2) return false;
      const parts = h.slice(1).split("~");
      if (parts[0] !== "t" || parts.length < 8) return false;
      const nums = parts.slice(1, 5).map(Number);
      if (nums.some((v) => !Number.isFinite(v))) return false;
      if (parts[5] === "b" || parts[5] === "bx") {
        this.state.mode = "bike";
        this.state.calm = parts[5] === "b";
        $("calm").checked = this.state.calm;
        $("calmrow").hidden = false;
        for (const b of $("mode").querySelectorAll("button")) b.setAttribute("aria-pressed", b.dataset.v === "bike" ? "true" : "false");
      }
      const tt = parseFloat(parts[6]);
      if (Number.isFinite(tt)) { this.state.t = clamp(tt, 0, 1); $("sl").value = this.state.t; }
      this.setPoint("from", this.pointAt(nums[0], nums[1], decLabel(parts[7]) || undefined), false);
      this.setPoint("to", this.pointAt(nums[2], nums[3], decLabel(parts[8] || "") || undefined), false);
      return true;
    },
  };

  function encLabel(s) {
    let out = "";
    for (const ch of String(s || "")) {
      if (/[A-Za-z0-9.\-]/.test(ch)) out += ch;
      else { const c = ch.codePointAt(0); out += c < 256 ? "_" + c.toString(16).padStart(2, "0") : "_u" + c.toString(16).padStart(4, "0"); }
    }
    return out;
  }
  function decLabel(s) {
    return String(s || "").replace(/_u([0-9a-f]{4})|_([0-9a-f]{2})/gi, (m, u, b) => String.fromCodePoint(parseInt(u || b, 16)));
  }

  /* ------------------------------------------------------------- helpers */
  const STREET_SHORT = { Street: "St", Avenue: "Ave", Boulevard: "Blvd", Drive: "Dr", Road: "Rd",
    Terrace: "Ter", Place: "Pl", Court: "Ct", Lane: "Ln", Highway: "Hwy", Parkway: "Pkwy" };
  function shortStreet(name) {
    return String(name).split(" ").map((w) => STREET_SHORT[w] || w).join(" ");
  }

  /* Keep at most k members, spread evenly along the frontier's length in
   * normalised (distance, climbing) space, always keeping both ends. */
  function thinFrontier(members, k) {
    const n = members.length;
    if (n <= k) return members;
    const d = members.map((m) => m.stats.distance_m), c = members.map((m) => m.stats.elev_gain_m);
    const dr = Math.max(1e-9, d[n - 1] - d[0]), cr = Math.max(1e-9, c[0] - c[n - 1]);
    const cum = [0];
    for (let i = 1; i < n; i++) {
      cum.push(cum[i - 1] + Math.hypot((d[i] - d[i - 1]) / dr, (c[i] - c[i - 1]) / cr));
    }
    const total = cum[n - 1], out = [], used = new Set();
    for (let j = 0; j < k; j++) {
      const target = total * j / (k - 1);
      let best = -1, bestErr = Infinity;
      for (let i = 0; i < n; i++) {
        if (used.has(i)) continue;
        const err = Math.abs(cum[i] - target);
        if (err < bestErr) { bestErr = err; best = i; }
      }
      used.add(best); out.push(members[best]);
    }
    return out.sort((a, b) => a.stats.distance_m - b.stats.distance_m);
  }

  function resample(prof, n) {
    const { d, z } = prof;
    const out = new Float64Array(n);
    if (!d.length) return { z: out };
    const total = d[d.length - 1] || 1;
    let j = 0;
    for (let i = 0; i < n; i++) {
      const x = total * i / (n - 1);
      while (j < d.length - 2 && d[j + 1] < x) j++;
      const span = d[j + 1] - d[j];
      out[i] = span > 0 ? lerp(z[j], z[j + 1], (x - d[j]) / span) : z[j];
    }
    return { z: out };
  }

  function tween(ms, fn) {
    const t0 = performance.now();
    const frame = (now) => { const k = clamp((now - t0) / ms, 0, 1); fn(ease(k)); if (k < 1) requestAnimationFrame(frame); };
    requestAnimationFrame(frame);
  }
  function fadeOut(layers, ms, done) {
    const start = layers.map((l) => l.options.opacity);
    tween(ms, (k) => { layers.forEach((l, i) => l.setStyle({ opacity: start[i] * (1 - k) })); if (k >= 1) done(); });
  }
  function fadeIn(pairs, ms) {
    tween(ms, (k) => { pairs.forEach(([l, o]) => l.setStyle({ opacity: o * k })); });
  }

  App.ALPHA_MAX = ALPHA_MAX; App._gen = 0;
  window.App = App;
  App.start().catch((err) => {
    console.error(err);
    $("status").textContent = "Could not start: " + err.message;
  });
})();
