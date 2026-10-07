/* San Francisco flat routes -- the explorer page (everything on screen).
 * Depends on engine.js. */
"use strict";

/* --------------------------------------------- street network canvas layer */
const BUCKETS = [
  { colour: "#2c7bb6", label: "under 3% - flat" },
  { colour: "#7fcdbb", label: "3-5% - gentle" },
  { colour: "#fed976", label: "5-8% - noticeable" },
  { colour: "#fd8d3c", label: "8-10% - hard" },
  { colour: "#e3492e", label: "10-15% - very hard" },
  { colour: "#7f1d1d", label: "over 15% - extreme" },
];

/* 88,000 polylines is far too many for individual Leaflet layers, so the
 * network is painted straight onto a canvas from the packed arrays, with
 * viewport culling and a zoom-dependent minimum edge length. */
const NetworkLayer = L.Layer.extend({
  initialize(geom, opts) { this.geom = geom; L.setOptions(this, opts || {}); },

  onAdd(map) {
    this._map = map;
    this._canvas = L.DomUtil.create("canvas", "leaflet-zoom-animated");
    const size = map.getSize();
    this._canvas.width = size.x; this._canvas.height = size.y;
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
    const map = this._map;
    if (!map) return;
    const size = map.getSize();
    if (this._canvas.width !== size.x || this._canvas.height !== size.y) {
      this._canvas.width = size.x; this._canvas.height = size.y;
    }
    const nw = map.getBounds().getNorthWest();
    L.DomUtil.setTransform(this._canvas, map.latLngToLayerPoint(nw), 1);

    const ctx = this._canvas.getContext("2d");
    ctx.clearRect(0, 0, size.x, size.y);

    const z = map.getZoom();
    const b = map.getBounds().pad(0.08);
    const west = b.getWest(), east = b.getEast();
    const south = b.getSouth(), north = b.getNorth();
    // at low zoom, drop the shortest edges: they are invisible anyway and
    // they dominate the vertex count
    const minLen = z >= 15 ? 0 : (z >= 14 ? 8 : (z >= 13 ? 16 : 26));
    const g = this.geom;
    const origin = map.latLngToLayerPoint(nw);
    // under a warp the vertices move; culling by the geographic bbox would
    // then be wrong, so cull only when drawing the real city
    const coords = this.altCoords || g.coords;
    const cull = !this.altCoords;

    const byBucket = [[], [], [], [], [], []];
    for (let i = 0; i < g.nEdges; i++) {
      const o = i * 4;
      if (cull && (g.bbox[o + 2] < west || g.bbox[o] > east
        || g.bbox[o + 3] < south || g.bbox[o + 1] > north)) continue;
      if (minLen && g.len[i] / g.DM < minLen) continue;
      byBucket[g.bucket[i]].push(i);
    }

    // flat first, steep last, so the walls read on top
    for (let bk = 0; bk < 6; bk++) {
      const list = byBucket[bk];
      if (!list.length) continue;
      ctx.strokeStyle = BUCKETS[bk].colour;
      // The hills should read, but this is a map about flat streets, so the
      // flat classes are not allowed to disappear underneath them.
      ctx.globalAlpha = bk >= 3 ? 0.82 : 0.95;
      ctx.lineWidth = bk >= 3
        ? (z >= 15 ? 2.3 : z >= 13 ? 1.5 : 1.0)
        : (z >= 15 ? 2.0 : z >= 13 ? 1.35 : 1.0);
      ctx.beginPath();
      for (const i of list) {
        const s = g.starts[i], e = g.starts[i + 1];
        for (let k = s; k < e; k += 2) {
          const p = map.latLngToLayerPoint([coords[k + 1], coords[k]]);
          const x = p.x - origin.x, y = p.y - origin.y;
          if (k === s) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  },
});

/* ------------------------------------------------------------- formatting */
const MI = 1609.344, FT = 3.28084;
function fmt(v, how) {
  if (v === null || v === undefined || v === "" || Number.isNaN(v)) return "—";
  switch (how) {
    case "pct": return (v * 100).toFixed(1) + "%";
    case "m0": return Math.round(v).toLocaleString() + " m";
    case "m1": return (+v).toFixed(1) + " m";
    case "km": return (+v).toFixed(2) + " km";
    case "ft": return Math.round(v).toLocaleString() + " ft";
    case "int": return (+v).toLocaleString();
    case "km2": return (+v).toFixed(2) + " km²";
    case "bool": return v ? "yes" : "no";
    default: return String(v);
  }
}

/* --------------------------------------------------------------- the app */
const App = {
  state: {
    mode: "walk", profile: "balanced",
    src: null, dst: null, picking: "src",
    alpha: null, beta: null, gamma: null, custom: false,
  },

  async start(DATA) {
    this.DATA = DATA;
    const buf = await inflate(DATA.bundle);
    const bundle = new Bundle(buf, DATA.manifest);
    this.meta = DATA.meta;
    this.graph = new Graph(bundle, DATA.meta);
    this.geom = new Geometry(bundle, DATA.meta);
    DATA.layers = JSON.parse(bundle.text("layers"));
    DATA.bundle = null;          // release the base64 string

    const g = this.graph;
    const lonF = new Float32Array(g.n), latF = new Float32Array(g.n);
    for (let i = 0; i < g.n; i++) { lonF[i] = g.nodeLon(i); latF[i] = g.nodeLat(i); }
    this.nodeGrid = new Grid(lonF, latF, 0.004);

    // edge midpoints, for click-to-inspect on the network layer
    const mx = new Float32Array(this.geom.nEdges), my = new Float32Array(this.geom.nEdges);
    for (let i = 0; i < this.geom.nEdges; i++) {
      const o = i * 4;
      mx[i] = (this.geom.bbox[o] + this.geom.bbox[o + 2]) / 2;
      my[i] = (this.geom.bbox[o + 1] + this.geom.bbox[o + 3]) / 2;
    }
    this.edgeGrid = new Grid(mx, my, 0.004);

    this.buildMap();
    this.buildUI();
    this.setDefaultPair();
    this.recompute();
  },

  /* ---------------------------------------------------------------- map */
  buildMap() {
    const map = L.map("map", {
      preferCanvas: true, center: [37.762, -122.437], zoom: 12,
      minZoom: 10, maxZoom: 18, zoomControl: true,
    });
    this._fitCity = () => {
      const nb = this.DATA.layers.neighborhoods;
      if (!nb) return;
      const b = L.geoJSON(nb).getBounds();
      map.fitBounds(b, { padding: [14, 14] });
    };
    this.map = map;
    L.control.scale({ imperial: true, metric: true }).addTo(map);
    if (this.DATA.basemap !== false) this.tiles = L.tileLayer(
      "https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}@2x.png", {
      subdomains: "abc", maxZoom: 19, opacity: 0.5, crossOrigin: true,
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        + ' contributors, &copy; <a href="https://carto.com/attributions">CARTO</a>'
        + ' &middot; elevation USGS 3DEP &middot; streets Overture Maps',
    }).addTo(map);

    this.overlays = {};
    this.network = new NetworkLayer(this.geom).addTo(map);
    this.overlays.network = this.network;

    const defs = [
      ["neighborhoods", "Neighborhood boundaries", true,
        () => ({ color: "#8492a0", weight: 1.1, opacity: 0.8, fill: false, dashArray: "4,3" })],
      ["basins", "Lowland basins (street below 15 m)", false,
        () => ({ color: "#3f9b6d", weight: 1.2, opacity: 0.55 })],
      ["bike_network", "Bicycle facilities (OSM-derived)", false,
        () => ({ color: "#39d98a", weight: 2.0, opacity: 0.9 })],
      ["low_stress", "Car-free / living streets", false,
        () => ({ color: "#c792ea", weight: 2.6, opacity: 0.95 })],
      ["barriers", "Steep barriers (unavoidable climbs)", false,
        () => ({ color: "#ff5f4d", weight: 3.0, opacity: 0.85 })],
      ["corridors", "Flat corridors (discovered)", true,
        f => ({
          color: f.properties.mode === "bike" ? "#9be7ff" : "#2c7bb6",
          weight: 4.2 + 3.4 * Math.min((f.properties.total_score || 0) / 900, 1),
          opacity: 0.95, lineCap: "round",
        })],
    ];
    this.layerDefs = [{ key: "network", label: "Streets coloured by gradient", on: true }];
    for (const [key, label, on, style] of defs) {
      const gj = this.DATA.layers[key];
      if (!gj) continue;
      const lyr = L.geoJSON(gj, {
        style,
        onEachFeature: (f, l) => l.bindPopup(this.popupHTML(key, f.properties)),
      });
      this.overlays[key] = lyr;
      if (on) lyr.addTo(map);
      this.layerDefs.push({ key, label, on });
    }
    if (this.DATA.layers.passes) {
      this.overlays.passes = L.layerGroup(
        this.DATA.layers.passes.features.map(f => {
          const c = f.geometry.coordinates;
          const flat = Array.isArray(c[0][0]) ? c[0] : c;
          const mid = flat[Math.floor(flat.length / 2)];
          return L.circleMarker([mid[1], mid[0]], {
            radius: 6, color: "#4a3b10", weight: 1.4,
            fillColor: "#ffd166", fillOpacity: 0.95,
          }).bindPopup(this.popupHTML("passes", f.properties));
        })).addTo(map);
      this.layerDefs.push({ key: "passes", label: "Critical passes / saddles", on: true });
    }

    this.cmpLine = L.polyline([], {
      color: "#c3ced9", weight: 3.4, opacity: 0.95, dashArray: "7,6",
    }).addTo(map);
    this.routeHalo = L.polyline([], { color: "#01080d", weight: 10, opacity: 0.85 }).addTo(map);
    this.routeLine = L.polyline([], { color: "#ffffff", weight: 4.4, opacity: 1 }).addTo(map);
    this.marks = L.layerGroup().addTo(map);

    map.on("click", e => this.onMapClick(e));
    this._fitCity();

    const legend = L.control({ position: "bottomleft" });
    legend.onAdd = () => {
      const div = L.DomUtil.create("div", "legend");
      div.innerHTML =
        '<div class="key"><i style="background:#ffffff;height:4px"></i> your route</div>'
        + '<div class="key"><i style="background:#8a97a4"></i> shortest route (comparison)</div>'
        + '<div class="key"><i style="background:#2c7bb6;height:5px"></i> flat corridor</div>'
        + '<div class="key"><i style="background:#ff5f4d"></i> steep barrier</div>'
        + '<div class="key"><i style="background:#ffd166;height:9px;width:9px;border-radius:50%"></i> critical pass</div>';
      return div;
    };
    legend.addTo(map);
  },

  popupHTML(kind, props) {
    const spec = {
      corridors: [["corridor_name", "Corridor"], ["mode", "Mode"],
        ["length_km", "Length", "km"], ["mean_abs_grade", "Mean gradient", "pct"],
        ["max_grade", "Max gradient", "pct"], ["gain_per_km", "Climb per km", "m1"],
        ["pair_count_max", "Neighborhood pairs served", "int"],
        ["neighborhood_span", "Neighborhoods spanned", "int"],
        ["climb_saved_m", "Climbing avoided (total)", "m0"],
        ["elev_min_m", "Lowest point", "m0"], ["elev_max_m", "Highest point", "m0"],
        ["neighborhoods", "Passes through"], ["street_names", "Streets"]],
      passes: [["name", "Street"], ["neighborhood", "Neighborhood"],
        ["pass_elev_ft", "Lowest possible crossing", "ft"],
        ["pairs_served", "Neighborhood pairs forced over it", "int"],
        ["max_abs_grade", "Max gradient", "pct"],
        ["neighborhoods_separated", "Separates"]],
      barriers: [["name", "Street"], ["neighborhood", "Neighborhood"],
        ["max_abs_grade", "Max gradient", "pct"], ["length_m", "Length", "m0"],
        ["shortest_use", "Pairs via shortest route", "int"],
        ["flat_use_per_objective", "... still via the flat route", "int"],
        ["unavoidability", "Unavoidable share", "pct"]],
      bike_network: [["name", "Street"], ["bike_facility", "Facility"],
        ["cls", "Road class"], ["max_abs_grade", "Max gradient", "pct"],
        ["length_m", "Length", "m0"]],
      low_stress: [["name", "Street"], ["cls", "Class"],
        ["bike_facility", "Facility"], ["length_m", "Length", "m0"]],
      neighborhoods: [["neighborhood", "Neighborhood"], ["area_km2", "Area", "km2"]],
      basins: [["basin_label", "Lowland basin"], ["length_km", "Street below 15 m", "km"]],
    }[kind] || [];
    let h = "";
    for (const [k, lab, how] of spec) {
      if (!(k in props) || props[k] === null || props[k] === "") continue;
      h += `<div><b>${lab}:</b> ${fmt(props[k], how)}</div>`;
    }
    return h || "<div>(no attributes)</div>";
  },

  /* --------------------------------------------------------------- input */
  onMapClick(e) {
    if (this.warpOn) return;            // the warped page is not a place to click
    const { lat, lng } = e.latlng;
    const bit = this.graph.modeBit(this.state.mode);
    const node = this.nodeGrid.nearest(lng, lat,
      i => (this.graph.nodeFlags[i] & bit) !== 0);
    if (node < 0) return;

    // a click well away from any street is an inspection, not a routing pick
    const dLon = (this.graph.nodeLon(node) - lng) * Math.cos(lat * Math.PI / 180);
    const dLat = this.graph.nodeLat(node) - lat;
    const metres = Math.hypot(dLon, dLat) * 111320;
    if (metres > 260) { this.inspect(e.latlng); return; }

    if (this.state.picking === "src") {
      this.state.src = node; this.state.picking = "dst";
    } else {
      this.state.dst = node; this.state.picking = "src";
    }
    this.syncPickButtons();
    this.recompute();
  },

  inspect(latlng) {
    const row = this.edgeGrid.nearest(latlng.lng, latlng.lat);
    if (row < 0) return;
    const info = this.geom.edgeInfo(row);
    const html = `<div><b>Street:</b> ${info.name || "(unnamed)"}</div>`
      + `<div><b>Gradient class:</b> ${BUCKETS[info.bucket].label}</div>`
      + `<div><b>Max gradient:</b> ${fmt(info.max_grade, "pct")}</div>`
      + `<div><b>Mean gradient:</b> ${fmt(info.avg_grade, "pct")}</div>`
      + `<div><b>Climb per km:</b> ${fmt(info.gain_per_km, "m1")}</div>`
      + `<div><b>Length:</b> ${fmt(info.length_m, "m0")}</div>`
      + `<div><b>Road class:</b> ${info.cls}</div>`;
    L.popup({ maxWidth: 300 }).setLatLng(latlng).setContent(html).openOn(this.map);
  },

  setDefaultPair() {
    const pts = this.DATA.points[this.state.mode] || {};
    const pick = name => {
      const p = pts[name];
      if (!p) return -1;
      const bit = this.graph.modeBit(this.state.mode);
      return this.nodeGrid.nearest(p[0], p[1],
        i => (this.graph.nodeFlags[i] & bit) !== 0);
    };
    this.state.src = pick("Mission");
    this.state.dst = pick("Outer Sunset");
  },

  weights() {
    const base = this.meta.profiles[this.state.profile];
    if (!this.state.custom) return base;
    return Object.assign({}, base, {
      alpha: this.state.alpha, beta: this.state.beta, gamma: this.state.gamma,
    });
  },

  /* ----------------------------------------------------------- recompute */
  recompute() {
    const s = this.state;
    const out = document.getElementById("res");
    if (s.src == null || s.dst == null || s.src < 0 || s.dst < 0 || s.src === s.dst) {
      out.innerHTML = '<p class="sub">Click the map to set an origin, then a '
        + 'destination. Or pick neighborhoods from the menus.</p>';
      this.routeLine.setLatLngs([]); this.routeHalo.setLatLngs([]);
      this.cmpLine.setLatLngs([]); this.marks.clearLayers();
      this.drawProfile(null);
      return;
    }
    const t0 = performance.now();
    const w = this.weights();
    const r = this.graph.route(s.src, s.dst, s.mode, w);
    const sh = this.graph.route(s.src, s.dst, s.mode,
      this.meta.profiles.shortest);
    const ms = performance.now() - t0;

    if (!r) {
      out.innerHTML = '<p class="sub">No route found between those two points '
        + 'in this travel mode.</p>';
      this.routeLine.setLatLngs([]); this.routeHalo.setLatLngs([]);
      this.drawProfile(null);
      return;
    }

    const sum = this.graph.summarise(r.arcs);
    const shSum = sh ? this.graph.summarise(sh.arcs) : null;
    this.lastRoute = { line: this.graph.geometry(r.arcs, this.geom),
                       cmp: sh && this.state.profile !== "shortest" && !this.state.custom
                         ? this.graph.geometry(sh.arcs, this.geom) : [] };
    this.drawRouteLines();

    this.marks.clearLayers();
    const mk = (node, colour, label) => L.circleMarker(
      this.place(this.graph.nodeLat(node), this.graph.nodeLon(node)),
      { radius: 7, color: "#04121a", weight: 2, fillColor: colour, fillOpacity: 1 })
      .bindPopup(`<b>${label}</b><br>${this.graph.nodeZ(node).toFixed(1)} m `
        + `(${Math.round(this.graph.nodeZ(node) * FT)} ft)`);
    mk(s.src, "#7fcdbb", "Origin").addTo(this.marks);
    mk(s.dst, "#e3492e", "Destination").addTo(this.marks);

    this.renderResult(sum, shSum, ms, r.settled);
    this.drawProfile(sum);
    if (this.warpOn) this.map.fitBounds(L.latLngBounds(this.routeLine.getLatLngs()).pad(0.08));
  },

  /* a lat/lon as currently displayed: warped and morphed, or as it is */
  place(lat, lon) {
    if (!this.warpOn || !this.warp) return [lat, lon];
    const [wlon, wlat] = this.warp.transform(lon, lat);
    const t = this.warpT;
    return [lat + (wlat - lat) * t, lon + (wlon - lon) * t];
  },

  drawRouteLines() {
    if (!this.lastRoute) return;
    const w = pts => pts.map(ll => this.place(ll[0], ll[1]));
    this.routeHalo.setLatLngs(w(this.lastRoute.line));
    this.routeLine.setLatLngs(w(this.lastRoute.line));
    this.cmpLine.setLatLngs(w(this.lastRoute.cmp));
  },

  renderResult(sum, sh, ms, settled) {
    const mi = sum.distance_m / MI, ft = sum.elev_gain_m * FT;
    let h = `<div class="big">${mi.toFixed(2)} mi &middot; `
      + `${Math.round(ft).toLocaleString()} ft climb</div><table>`;
    h += `<tr><td class="k">Distance</td><td class="v">${mi.toFixed(2)} mi / `
      + `${(sum.distance_m / 1000).toFixed(2)} km</td></tr>`;
    h += `<tr><td class="k">Elevation gain</td><td class="v">${Math.round(ft)} ft / `
      + `${sum.elev_gain_m.toFixed(0)} m</td></tr>`;
    h += `<tr><td class="k">Elevation loss</td><td class="v">`
      + `${Math.round(sum.elev_loss_m * FT)} ft / ${sum.elev_loss_m.toFixed(0)} m</td></tr>`;
    h += `<tr><td class="k">Steepest climb</td><td class="v">`
      + `${(sum.max_grade * 100).toFixed(1)}%</td></tr>`;
    h += `<tr><td class="k">Mean gradient</td><td class="v">`
      + `${(sum.avg_abs_grade * 100).toFixed(1)}%</td></tr>`;
    this.meta.thresholds.forEach((t, i) => {
      h += `<tr><td class="k">Distance climbing over ${t}%</td><td class="v">`
        + `${Math.round(sum.thresholds[i]).toLocaleString()} m</td></tr>`;
    });
    h += "</table>";
    if (sh && sh.distance_m > 0 && (this.state.profile !== "shortest" || this.state.custom)) {
      const dMi = (sum.distance_m - sh.distance_m) / MI;
      const dFt = (sh.elev_gain_m - sum.elev_gain_m) * FT;
      h += `<div class="cmp">
        <div class="card"><div class="t">Extra distance</div><div class="n">`
        + `${dMi >= 0 ? "+" : ""}${dMi.toFixed(2)} mi `
        + `(${(100 * (sum.distance_m / sh.distance_m - 1)).toFixed(0)}%)</div></div>
        <div class="card"><div class="t">Climbing saved</div>`
        + `<div class="n ${dFt > 0 ? "good" : "bad"}">`
        + `${dFt >= 0 ? "" : "+"}${Math.abs(Math.round(dFt))} ft</div></div></div>`;
      h += `<div class="note">Shortest route, shown dashed: `
        + `${(sh.distance_m / MI).toFixed(2)} mi with `
        + `${Math.round(sh.elev_gain_m * FT)} ft of climbing, steepest `
        + `${(sh.max_grade * 100).toFixed(0)}%.</div>`;
    }
    h += `<div class="note">Routed in the browser in ${ms.toFixed(0)} ms `
      + `(${settled.toLocaleString()} nodes settled, ${sum.n_edges} street `
      + `segments).</div>`;
    document.getElementById("res").innerHTML = h;
  },

  drawProfile(sum) {
    const cv = document.getElementById("prof");
    const ctx = cv.getContext("2d");
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    const note = document.getElementById("profnote");
    if (!sum || !sum.profile.d.length) { note.textContent = ""; return; }
    const d = sum.profile.d, e = sum.profile.z, n = d.length;
    const emin = Math.min(...e), emax = Math.max(...e);
    const span = Math.max(emax - emin, 8);
    const padL = 48, padR = 16, padT = 14, padB = 30;
    const x = i => padL + (d[i] / d[n - 1]) * (W - padL - padR);
    const y = v => padT + (1 - (v - emin) / span) * (H - padT - padB);

    for (let i = 1; i < n; i++) {
      const dz = e[i] - e[i - 1], dd = Math.max(d[i] - d[i - 1], 1e-6);
      const g = Math.abs(dz / dd);
      ctx.fillStyle = g < 0.03 ? "#2c7bb6" : g < 0.05 ? "#7fcdbb" : g < 0.08 ? "#fed976"
        : g < 0.10 ? "#fd8d3c" : g < 0.15 ? "#e3492e" : "#7f1d1d";
      ctx.globalAlpha = 0.8;
      ctx.beginPath();
      ctx.moveTo(x(i - 1), y(e[i - 1])); ctx.lineTo(x(i), y(e[i]));
      ctx.lineTo(x(i), H - padB); ctx.lineTo(x(i - 1), H - padB);
      ctx.closePath(); ctx.fill();
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = "#e9eef3"; ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < n; i++) { i ? ctx.lineTo(x(i), y(e[i])) : ctx.moveTo(x(i), y(e[i])); }
    ctx.stroke();
    ctx.strokeStyle = "#39434e"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(padL, H - padB); ctx.lineTo(W - padR, H - padB); ctx.stroke();
    ctx.fillStyle = "#93a1af"; ctx.font = "11px sans-serif"; ctx.textAlign = "right";
    ctx.fillText(Math.round(emax * FT) + " ft", padL - 5, padT + 9);
    ctx.fillText(Math.round(emin * FT) + " ft", padL - 5, H - padB - 1);
    ctx.fillText((d[n - 1] / MI).toFixed(2) + " mi", W - padR, H - 9);
    ctx.textAlign = "left"; ctx.fillText("0", padL, H - 9);
    note.textContent = `Elevation ${Math.round(emin * FT)}–${Math.round(emax * FT)} ft, `
      + `sampled at every intersection. Shading shows gradient.`;
  },
};

/* ------------------------------------------------------------ UI wiring */
Object.assign(App, {
  buildUI() {
    const names = this.DATA.neighborhood_names;
    const oSel = document.getElementById("o"), dSel = document.getElementById("d");
    oSel.appendChild(new Option("(map point)", ""));
    dSel.appendChild(new Option("(map point)", ""));
    for (const n of names) {
      oSel.appendChild(new Option(n, n));
      dSel.appendChild(new Option(n, n));
    }
    oSel.value = "Mission"; dSel.value = "Outer Sunset";
    const setFrom = (sel, which) => {
      const name = sel.value;
      if (!name) return;
      const p = (this.DATA.points[this.state.mode] || {})[name];
      if (!p) return;
      const bit = this.graph.modeBit(this.state.mode);
      const node = this.nodeGrid.nearest(p[0], p[1],
        i => (this.graph.nodeFlags[i] & bit) !== 0);
      this.state[which] = node;
      this.recompute();
    };
    oSel.onchange = () => setFrom(oSel, "src");
    dSel.onchange = () => setFrom(dSel, "dst");

    document.getElementById("swap").onclick = () => {
      const t = this.state.src; this.state.src = this.state.dst; this.state.dst = t;
      const to = oSel.value; oSel.value = dSel.value; dSel.value = to;
      this.recompute();
    };
    document.getElementById("clear").onclick = () => {
      this.state.src = null; this.state.dst = null; this.state.picking = "src";
      oSel.value = ""; dSel.value = "";
      this.syncPickButtons(); this.recompute();
    };

    document.querySelectorAll("#pickrow button").forEach(b => {
      b.onclick = () => { this.state.picking = b.dataset.pick; this.syncPickButtons(); };
    });
    this.syncPickButtons();

    const seg = (id, field, after) => {
      const box = document.getElementById(id);
      box.querySelectorAll("button").forEach(b => {
        b.onclick = () => {
          box.querySelectorAll("button").forEach(x => x.setAttribute("aria-pressed", "false"));
          b.setAttribute("aria-pressed", "true");
          this.state[field] = b.dataset.v;
          if (after) after();
          this.recompute();
        };
      });
    };
    seg("mode", "mode", () => {
      // a mode change can invalidate a picked node (stairs are not rideable)
      const bit = this.graph.modeBit(this.state.mode);
      for (const key of ["src", "dst"]) {
        const nd = this.state[key];
        if (nd != null && nd >= 0 && (this.graph.nodeFlags[nd] & bit) === 0) {
          this.state[key] = this.nodeGrid.nearest(
            this.graph.nodeLon(nd), this.graph.nodeLat(nd),
            i => (this.graph.nodeFlags[i] & bit) !== 0);
        }
      }
      this.network._redraw();
    });
    seg("pref", "profile", () => { this.state.custom = false; this.syncSliders(); });

    // weight sliders
    const defs = [
      ["alpha", "Climbing weight α", 0, 250, 1,
       "equivalent metres of detour accepted per metre climbed"],
      ["beta", "Steep-grade penalty β", 0, 30, 0.5,
       "multiplier on the cumulative grade-threshold penalties"],
      ["gamma", "Extreme-grade penalty γ", 0, 20, 0.5,
       "multiplier on the penalty above the highest threshold"],
    ];
    const box = document.getElementById("sliders");
    for (const [key, label, min, max, step, help] of defs) {
      const wrap = document.createElement("div");
      wrap.className = "slider";
      wrap.innerHTML = `<div class="row"><span>${label}</span><b id="lab_${key}"></b></div>`
        + `<input type="range" id="sl_${key}" min="${min}" max="${max}" step="${step}">`;
      wrap.title = help;
      box.appendChild(wrap);
      const input = wrap.querySelector("input");
      input.oninput = () => {
        this.state[key] = parseFloat(input.value);
        this.state.custom = true;
        document.getElementById("lab_" + key).textContent = input.value;
        document.getElementById("customnote").style.display = "block";
        this.recompute();
      };
    }
    document.getElementById("resetw").onclick = () => {
      this.state.custom = false; this.syncSliders(); this.recompute();
    };
    this.syncSliders();

    // guided examples, so the map demonstrates the findings unprompted
    const ebox = document.getElementById("examples");
    for (const ex of (this.DATA.examples || [])) {
      const btn = document.createElement("button");
      btn.className = "act";
      btn.innerHTML = `<b>${ex.o} &rarr; ${ex.d}</b><br><span class="exnote">`
        + `${ex.note}</span>`;
      btn.onclick = () => {
        this.state.mode = ex.mode || "walk";
        this.state.profile = ex.profile || "min_climb";
        this.state.custom = false;
        document.querySelectorAll("#mode button").forEach(b =>
          b.setAttribute("aria-pressed", String(b.dataset.v === this.state.mode)));
        document.querySelectorAll("#pref button").forEach(b =>
          b.setAttribute("aria-pressed", String(b.dataset.v === this.state.profile)));
        const pts = this.DATA.points[this.state.mode] || {};
        const bit = this.graph.modeBit(this.state.mode);
        const f = i => (this.graph.nodeFlags[i] & bit) !== 0;
        const snap = name => {
          const pt = pts[name];
          return pt ? this.nodeGrid.nearest(pt[0], pt[1], f) : null;
        };
        this.state.src = snap(ex.o);
        this.state.dst = snap(ex.d);
        document.getElementById("o").value = ex.o;
        document.getElementById("d").value = ex.d;
        this.syncSliders();
        this.recompute();
      };
      ebox.appendChild(btn);
    }

    // layers
    const lbox = document.getElementById("layers");
    for (const def of this.layerDefs) {
      const wrap = document.createElement("label");
      wrap.innerHTML = `<input type="checkbox" id="chk_${def.key}" `
        + `${def.on ? "checked" : ""}> ${def.label}`;
      lbox.appendChild(wrap);
      wrap.querySelector("input").onchange = ev => {
        const lyr = this.overlays[def.key];
        if (!lyr) return;
        if (ev.target.checked) { lyr.addTo(this.map); this.raise(); }
        else this.map.removeLayer(lyr);
      };
    }
    const kbox = document.getElementById("key");
    BUCKETS.forEach(b => {
      const el = document.createElement("div");
      el.className = "key";
      el.innerHTML = `<i style="background:${b.colour}"></i> ${b.label}`;
      kbox.appendChild(el);
    });
    this.raise();
  },

  syncPickButtons() {
    document.querySelectorAll("#pickrow button").forEach(b => {
      b.setAttribute("aria-pressed", String(b.dataset.pick === this.state.picking));
    });
    document.getElementById("pickhint").textContent =
      this.state.picking === "src"
        ? "Click the map to set the ORIGIN."
        : "Click the map to set the DESTINATION.";
  },

  syncSliders() {
    const base = this.meta.profiles[this.state.profile];
    if (!this.state.custom) {
      this.state.alpha = base.alpha; this.state.beta = base.beta;
      this.state.gamma = base.gamma;
      document.getElementById("customnote").style.display = "none";
    }
    for (const key of ["alpha", "beta", "gamma"]) {
      const sl = document.getElementById("sl_" + key);
      if (!sl) continue;
      sl.value = this.state[key];
      document.getElementById("lab_" + key).textContent = String(this.state[key]);
    }
  },

  raise() {
    [this.cmpLine, this.routeHalo, this.routeLine].forEach(l => l.bringToFront());
    this.marks.eachLayer(l => l.bringToFront());
  },
});

// exposed so the page can be driven from the console and from tests
window.App = App;

window.addEventListener("DOMContentLoaded", () => {
  const res = document.getElementById("res");
  res.innerHTML = '<p class="sub">Unpacking the street graph\u2026</p>';
  App.start(window.DATA).catch(err => {
    res.innerHTML = '<p class="sub">Could not start: ' + String(err.message || err)
      + "</p>";
    throw err;
  });
});

/* ------------------------------------------------------------ the warp */
Object.assign(App, {
  warpOn: false, warpT: 1.0, warpLambda: 1.0, warp: null, warpCache: {},

  warpWeights() {
    const base = this.meta.profiles.balanced;
    const lam = this.warpLambda;
    return Object.assign({}, base, {
      alpha: base.alpha * lam, beta: base.beta * lam, gamma: base.gamma * lam,
      use_class_multiplier: lam > 0,
    });
  },

  ensureWarp() {
    const key = `${this.state.mode}|${this.warpLambda}`;
    if (!this.warpCache[key]) {
      this.warpCache[key] = Warp.build(this, this.state.mode, this.warpWeights());
      const w = this.warpCache[key];
      w.coords = Warp.warpCoords(this.geom.coords, w.transform);
      w.neighborhoods = Warp.warpGeoJSON(this.DATA.layers.neighborhoods, w.transform);
      w.corridors = this.DATA.layers.corridors
        ? Warp.warpGeoJSON(this.DATA.layers.corridors, w.transform) : null;
    }
    this.warp = this.warpCache[key];
    return this.warp;
  },

  /* lerp between the real and the warped city */
  warpedCoords() {
    const w = this.warp, t = this.warpT, src = this.geom.coords;
    if (t >= 1) return w.coords;
    if (!this._lerp || this._lerp.length !== src.length) this._lerp = new Float32Array(src.length);
    const out = this._lerp;
    for (let k = 0; k < src.length; k++) out[k] = src[k] + (w.coords[k] - src[k]) * t;
    return out;
  },

  applyWarpView() {
    const on = this.warpOn;
    document.getElementById("warpstate").hidden = !on;
    if (on) {
      const busy = document.getElementById("busy");
      busy.classList.add("on");
      // let the busy notice paint before the (synchronous) build
      setTimeout(() => {
        try {
          this.ensureWarp();
          this.network.altCoords = this.warpedCoords();
          this.network._redraw();
          this.redrawWarpedOverlays();
          this.drawRouteLines();
          this.renderWarpReadout();
          if (this.tiles) this.tiles.setOpacity(0);
          for (const key of ["passes", "barriers", "basins", "bike_network", "low_stress"]) {
            if (this.overlays[key] && this.map.hasLayer(this.overlays[key])) {
              this.map.removeLayer(this.overlays[key]);
              const chk = document.getElementById("chk_" + key);
              if (chk) chk.checked = false;
            }
          }
          if (this.overlays.neighborhoods && this.map.hasLayer(this.overlays.neighborhoods)) {
            this.map.removeLayer(this.overlays.neighborhoods);
          }
          if (this.overlays.corridors && this.map.hasLayer(this.overlays.corridors)) {
            this.map.removeLayer(this.overlays.corridors);
          }
        } finally {
          busy.classList.remove("on");
        }
      }, 30);
    } else {
      this.network.altCoords = null;
      this.network._redraw();
      if (this.warpLayers) { this.warpLayers.forEach(l => this.map.removeLayer(l)); this.warpLayers = null; }
      if (this.tiles) this.tiles.setOpacity(0.5);
      const chkN = document.getElementById("chk_neighborhoods");
      if (this.overlays.neighborhoods && chkN && chkN.checked) this.overlays.neighborhoods.addTo(this.map);
      const chkC = document.getElementById("chk_corridors");
      if (this.overlays.corridors && chkC && chkC.checked) this.overlays.corridors.addTo(this.map);
      this.drawRouteLines();
      this.raise();
    }
    this.syncPickButtons();
  },

  redrawWarpedOverlays() {
    if (this.warpLayers) this.warpLayers.forEach(l => this.map.removeLayer(l));
    const lerpGJ = gj => Warp.warpGeoJSON(gj, (lon, lat) => {
      const p = this.place(lat, lon); return [p[1], p[0]];
    });
    const layers = [];
    layers.push(L.geoJSON(lerpGJ(this.DATA.layers.neighborhoods), {
      style: () => ({ color: "#aab6c2", weight: 1.3, opacity: 0.85, fill: false, dashArray: "4,3" }),
      onEachFeature: (f, l) => l.bindPopup(this.popupHTML("neighborhoods", f.properties)),
    }).addTo(this.map));
    if (this.DATA.layers.corridors) {
      layers.push(L.geoJSON(lerpGJ(this.DATA.layers.corridors), {
        style: f => ({ color: f.properties.mode === "bike" ? "#9be7ff" : "#2c7bb6",
                       weight: 3.6, opacity: 0.9, lineCap: "round" }),
        onEachFeature: (f, l) => l.bindPopup(this.popupHTML("corridors", f.properties)),
      }).addTo(this.map));
    }
    // neighborhood names, so the deformation can be read
    const pts = this.DATA.points[this.state.mode] || {};
    const labels = L.layerGroup();
    for (const name of Object.keys(pts)) {
      const [lon, lat] = pts[name];
      const ll = this.place(lat, lon);
      L.marker(ll, { icon: L.divIcon({ className: "nblabel", html: name, iconSize: null }),
                     interactive: false }).addTo(labels);
    }
    layers.push(labels.addTo(this.map));
    this.warpLayers = layers;
    this.raise();
  },

  renderWarpReadout() {
    const w = this.warp;
    if (!w) return;
    const a = this.meta.profiles.balanced.alpha * this.warpLambda;
    document.getElementById("warpinfo").innerHTML =
      `<div><b>${a.toFixed(0)} m</b> of walking per metre of climb &middot; `
      + `${w.anchors} anchors &middot; stress ${(100 * w.stress).toFixed(1)}% &middot; `
      + `mean shift ${Math.round(w.meanShiftM).toLocaleString()} m &middot; `
      + `${w.ms.toFixed(0)} ms</div>`;
  },

  bindWarpUI() {
    const toggle = document.getElementById("warptoggle");
    toggle.onclick = () => {
      this.warpOn = !this.warpOn;
      toggle.setAttribute("aria-pressed", String(this.warpOn));
      toggle.textContent = this.warpOn ? "Show the real city" : "Show the warped city";
      this.applyWarpView();
    };
    const lam = document.getElementById("sl_lambda");
    const lamLab = document.getElementById("lab_lambda");
    lam.value = this.warpLambda; lamLab.textContent = this.warpLambda.toFixed(1);
    lam.oninput = () => { lamLab.textContent = (+lam.value).toFixed(1); };
    lam.onchange = () => {
      this.warpLambda = parseFloat(lam.value);
      if (this.warpOn) this.applyWarpView();
    };
    const morph = document.getElementById("sl_morph");
    const morphLab = document.getElementById("lab_morph");
    morph.value = this.warpT; morphLab.textContent = Math.round(this.warpT * 100) + "%";
    morph.oninput = () => {
      this.warpT = parseFloat(morph.value);
      morphLab.textContent = Math.round(this.warpT * 100) + "%";
      if (!this.warpOn || !this.warp) return;
      this.network.altCoords = this.warpedCoords();
      this.network._redraw();
      this.redrawWarpedOverlays();
      this.drawRouteLines();
    };
  },
});

// the warp controls exist only once the page has its graph
const _origStart = App.start;
App.start = async function (DATA) {
  await _origStart.call(this, DATA);
  this.bindWarpUI();
  const prevSync = this.syncPickButtons.bind(this);
  this.syncPickButtons = () => {
    prevSync();
    if (this.warpOn) {
      document.getElementById("pickhint").textContent =
        "Map clicks are off while the city is warped. Choose neighborhoods from the menus.";
    }
  };
};
