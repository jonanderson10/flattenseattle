/* The warped city.
 *
 * Geography answers "how far", and in San Francisco that is the wrong
 * question. Here the city is redrawn so that distance on the page means
 * *climbing cost*: the equivalent-metre cost the router charges, in which a
 * metre of ascent counts as some multiple of a metre of walking. Places
 * separated by a ridge move apart; places joined by a flat corridor pull
 * together. The multiple is the viewer's to set, and the same weights drive
 * the routes, so the picture and the directions always agree.
 *
 * Method, in brief: ~100 anchor intersections (every neighborhood's access
 * point plus a lattice over the city) get a full cost matrix from the
 * in-page router; stress majorisation (SMACOF, unit weights) lays them out
 * so page distance matches cost, starting from their true positions so the
 * result is the least deformation that fits; a Procrustes fit turns and
 * scales the layout back onto geography so north stays up; and a thin-plate
 * spline through the anchors' displacements carries every street vertex,
 * neighborhood outline and route along for the ride.
 */
"use strict";

const Warp = {
  /* --------------------------------------------------------- local frame */
  frame(lat0, lon0) {
    const kx = 111320 * Math.cos(lat0 * Math.PI / 180), ky = 110540;
    return {
      toXY: (lon, lat) => [(lon - lon0) * kx, (lat - lat0) * ky],
      toLL: (x, y) => [lon0 + x / kx, lat0 + y / ky],
    };
  },

  /* ----------------------------------------------------- anchor selection */
  /* Neighborhood access points plus a lattice of routable intersections. */
  anchors(app, mode, cellM = 1000) {
    const g = app.graph, bit = g.modeBit(mode);
    const ok = i => (g.nodeFlags[i] & bit) !== 0;
    const nodes = new Set();
    const pts = app.DATA.points[mode] || {};
    for (const name of Object.keys(pts)) {
      const p = pts[name];
      const n = app.nodeGrid.nearest(p[0], p[1], ok);
      if (n >= 0) nodes.add(n);
    }
    const nb = app.DATA.layers.neighborhoods;
    const b = L.geoJSON(nb).getBounds();
    const f = this.frame(b.getCenter().lat, b.getCenter().lng);
    const [x0, y0] = f.toXY(b.getWest(), b.getSouth());
    const [x1, y1] = f.toXY(b.getEast(), b.getNorth());
    for (let y = y0 + cellM / 2; y < y1; y += cellM) {
      for (let x = x0 + cellM / 2; x < x1; x += cellM) {
        const [lon, lat] = f.toLL(x, y);
        const n = app.nodeGrid.nearest(lon, lat, ok);
        if (n < 0) continue;
        const [nx, ny] = f.toXY(g.nodeLon(n), g.nodeLat(n));
        if (Math.hypot(nx - x, ny - y) > cellM * 0.6) continue;   // off the grid: water, park
        nodes.add(n);
      }
    }
    return { nodes: Array.from(nodes), frame: f };
  },

  /* ------------------------------------------------------- cost matrix */
  costMatrix(graph, nodes, mode, w) {
    const K = nodes.length, D = new Float64Array(K * K);
    const index = new Map(nodes.map((n, i) => [n, i]));
    for (let i = 0; i < K; i++) {
      const d = graph.distances(nodes[i], mode, w, nodes);
      for (let j = 0; j < K; j++) D[i * K + j] = d[j];
    }
    // symmetrise: the cost of a round trip, halved
    for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) {
      let a = D[i * K + j], b = D[j * K + i];
      if (!isFinite(a)) a = b; if (!isFinite(b)) b = a;
      const m = isFinite(a) ? (a + b) / 2 : 0;
      D[i * K + j] = m; D[j * K + i] = m;
    }
    return D;
  },

  /* -------------------------------------------------- stress majorisation */
  smacof(D, X0, iters = 150) {
    const K = X0.length;
    let X = X0.map(p => [p[0], p[1]]);
    const stress = X => {
      let s = 0, t = 0;
      for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) {
        const d = D[i * K + j];
        const e = Math.hypot(X[i][0] - X[j][0], X[i][1] - X[j][1]);
        s += (e - d) * (e - d); t += d * d;
      }
      return Math.sqrt(s / t);
    };
    let prev = stress(X);
    for (let it = 0; it < iters; it++) {
      // Guttman transform with unit weights: X' = (1/K) B(X) X
      const Y = X.map(() => [0, 0]);
      for (let i = 0; i < K; i++) {
        let bii = 0;
        for (let j = 0; j < K; j++) {
          if (i === j) continue;
          const dx = X[i][0] - X[j][0], dy = X[i][1] - X[j][1];
          const e = Math.hypot(dx, dy);
          const bij = e > 1e-9 ? -D[i * K + j] / e : 0;
          Y[i][0] += bij * X[j][0]; Y[i][1] += bij * X[j][1];
          bii -= bij;
        }
        Y[i][0] += bii * X[i][0]; Y[i][1] += bii * X[i][1];
        Y[i][0] /= K; Y[i][1] /= K;
      }
      X = Y;
      const s = stress(X);
      if (prev - s < 1e-7) { prev = s; break; }
      prev = s;
    }
    return { X, stress: prev };
  },

  /* ------------------------------------------- Procrustes onto geography */
  /* Rotate, reflect and scale X so it sits over G as closely as possible. */
  procrustes(X, G) {
    const K = X.length;
    const mean = P => P.reduce((a, p) => [a[0] + p[0] / K, a[1] + p[1] / K], [0, 0]);
    const mx = mean(X), mg = mean(G);
    const Xc = X.map(p => [p[0] - mx[0], p[1] - mx[1]]);
    const Gc = G.map(p => [p[0] - mg[0], p[1] - mg[1]]);
    const fit = reflect => {
      const Xr = reflect ? Xc.map(p => [-p[0], p[1]]) : Xc;
      let a = 0, b = 0, c = 0, d = 0, nx = 0;
      for (let i = 0; i < K; i++) {
        a += Xr[i][0] * Gc[i][0]; b += Xr[i][0] * Gc[i][1];
        c += Xr[i][1] * Gc[i][0]; d += Xr[i][1] * Gc[i][1];
        nx += Xr[i][0] ** 2 + Xr[i][1] ** 2;
      }
      const th = Math.atan2(b - c, a + d);
      const cs = Math.cos(th), sn = Math.sin(th);
      const s = ((a + d) * cs + (b - c) * sn) / nx;
      const out = Xr.map(p => [mg[0] + s * (cs * p[0] - sn * p[1]),
                               mg[1] + s * (sn * p[0] + cs * p[1])]);
      let err = 0;
      for (let i = 0; i < K; i++) err += (out[i][0] - G[i][0]) ** 2 + (out[i][1] - G[i][1]) ** 2;
      return { out, err };
    };
    const r0 = fit(false), r1 = fit(true);
    return r0.err <= r1.err ? r0.out : r1.out;
  },

  /* ---------------------------------------------------- thin-plate spline */
  /* Interpolates the anchors' displacements to any point. Coordinates are
   * in kilometres to keep the system well conditioned; `smooth` relaxes the
   * exact fit a little so no single anchor can tear the map. */
  tps(P, V, smooth = 0.05) {
    const K = P.length, n = K + 3;
    const phi = r => (r > 1e-12 ? r * r * Math.log(r) : 0);
    const solve = rhs => {
      const A = Array.from({ length: n }, () => new Float64Array(n));
      const bvec = new Float64Array(n);
      for (let i = 0; i < K; i++) {
        for (let j = 0; j < K; j++) {
          A[i][j] = phi(Math.hypot(P[i][0] - P[j][0], P[i][1] - P[j][1]));
        }
        A[i][i] += smooth;
        A[i][K] = 1; A[i][K + 1] = P[i][0]; A[i][K + 2] = P[i][1];
        A[K][i] = 1; A[K + 1][i] = P[i][0]; A[K + 2][i] = P[i][1];
        bvec[i] = rhs[i];
      }
      // Gaussian elimination with partial pivoting
      for (let c = 0; c < n; c++) {
        let piv = c;
        for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
        if (piv !== c) { [A[c], A[piv]] = [A[piv], A[c]]; const t = bvec[c]; bvec[c] = bvec[piv]; bvec[piv] = t; }
        const p = A[c][c] || 1e-12;
        for (let r = c + 1; r < n; r++) {
          const f = A[r][c] / p;
          if (!f) continue;
          for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
          bvec[r] -= f * bvec[c];
        }
      }
      const x = new Float64Array(n);
      for (let r = n - 1; r >= 0; r--) {
        let s = bvec[r];
        for (let k = r + 1; k < n; k++) s -= A[r][k] * x[k];
        x[r] = s / (A[r][r] || 1e-12);
      }
      return x;
    };
    const wx = solve(V.map(v => v[0])), wy = solve(V.map(v => v[1]));
    const px = Float64Array.from(P.map(p => p[0])), py = Float64Array.from(P.map(p => p[1]));
    return (x, y) => {
      let ux = wx[K] + wx[K + 1] * x + wx[K + 2] * y;
      let uy = wy[K] + wy[K + 1] * x + wy[K + 2] * y;
      for (let k = 0; k < K; k++) {
        const r = Math.hypot(x - px[k], y - py[k]);
        if (r > 1e-12) { const f = r * r * Math.log(r); ux += wx[k] * f; uy += wy[k] * f; }
      }
      return [ux, uy];
    };
  },

  /* ------------------------------------------------------------ driver */
  /* Build the full warp for a mode and weighting. Returns a transform
   * lon,lat -> warped lon,lat plus diagnostics. */
  build(app, mode, w) {
    const t0 = performance.now();
    const { nodes, frame } = this.anchors(app, mode);
    const g = app.graph;
    const G = nodes.map(n => frame.toXY(g.nodeLon(n), g.nodeLat(n)));
    const D = this.costMatrix(g, nodes, mode, w);
    // start from geography scaled to the cost scale, so the result is the
    // smallest deformation that fits
    let sumD = 0, sumG = 0, cnt = 0;
    const K = nodes.length;
    for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) {
      sumD += D[i * K + j]; sumG += Math.hypot(G[i][0] - G[j][0], G[i][1] - G[j][1]); cnt++;
    }
    const scale = sumG > 0 ? sumD / sumG : 1;
    const X0 = G.map(p => [p[0] * scale, p[1] * scale]);
    const { X, stress } = this.smacof(D, X0);
    const Xa = this.procrustes(X, G);
    const Pkm = G.map(p => [p[0] / 1000, p[1] / 1000]);
    const Vkm = Xa.map((p, i) => [(p[0] - G[i][0]) / 1000, (p[1] - G[i][1]) / 1000]);
    const disp = this.tps(Pkm, Vkm);
    const transform = (lon, lat) => {
      const [x, y] = frame.toXY(lon, lat);
      const [ux, uy] = disp(x / 1000, y / 1000);
      return frame.toLL(x + ux * 1000, y + uy * 1000);
    };
    // how much the city stretched, as the mean anchor displacement
    let moved = 0;
    for (let i = 0; i < K; i++) moved += Math.hypot(Xa[i][0] - G[i][0], Xa[i][1] - G[i][1]);
    return {
      transform, stress, anchors: K, meanShiftM: moved / K,
      ms: performance.now() - t0,
      anchorLL: nodes.map((n, i) => ({ from: [g.nodeLat(n), g.nodeLon(n)],
                                       to: frame.toLL(Xa[i][0], Xa[i][1]).reverse() })),
    };
  },

  /* Apply a transform to a flat [lon,lat,...] Float32Array. */
  warpCoords(coords, transform) {
    const out = new Float32Array(coords.length);
    for (let k = 0; k < coords.length; k += 2) {
      const [lon, lat] = transform(coords[k], coords[k + 1]);
      out[k] = lon; out[k + 1] = lat;
    }
    return out;
  },

  warpGeoJSON(gj, transform) {
    const walk = c => (typeof c[0] === "number")
      ? transform(c[0], c[1]) : c.map(walk);
    return {
      type: "FeatureCollection",
      features: gj.features.map(f => ({
        type: "Feature", properties: f.properties,
        geometry: { type: f.geometry.type, coordinates: walk(f.geometry.coordinates) },
      })),
    };
  },
};
window.Warp = Warp;
