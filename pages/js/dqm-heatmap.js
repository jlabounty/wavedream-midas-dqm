//
// dqm-heatmap.js -- a 2D histogram drawn as an image, for the accumulated-plot pages.
//
// Why not mplot's colormap: it draws every bin with its own fillRect and a
// freshly formatted 'hsl(...)' fillStyle, twice per update (setData and redraw
// each schedule a draw), whether or not the histogram changed. A 128 x 130 map
// is 16,600 draw calls per draw; on a GPU-backed canvas (a desktop Chrome)
// every one of them goes through the command buffer, and the ToT tab stalled
// the page for hundreds of milliseconds per refresh.
//
// Here the bins become one ImageData of nx x ny pixels (a colour from a
// precomputed table per bin), put on an offscreen canvas, and drawn scaled onto
// the visible one with smoothing off: one drawImage per update. Axes, ticks,
// labels and the colour bar live on a second canvas underneath, redrawn only
// when the layout changes (size, pixel ratio, ranges, titles, z scale). An
// update whose bins are the same as last time draws nothing at all.
//
// The colours are mplot's (hue 240 -> 0 from the bottom of the z scale to the
// top, bins below 0.5 white), so the page keeps its look. Under- and overflow
// bins are not drawn; the page states their content in the footnote.
//

(function (root) {
"use strict";

const FONT = "12px sans-serif";
const AXIS = "#808080";
const LABEL = "#404040";
const ZERO = [255, 255, 255, 255];        // bins below 0.5 (mplot's zeroColor "white")
const NAN = [128, 128, 128, 255];

/** RGB of hsl(h, 100 %, 50 %), h in degrees. */
function hslRgb(h) {
  const f = (n) => {
    const k = (n + h / 30) % 12;
    return 0.5 - 0.5 * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return [f(0), f(8), f(4)].map((x) => Math.round(x * 255));
}

/**
 * The colour table: 256 RGBA entries, index 0 the bottom of the z scale (blue,
 * hue 240), 255 the top (red, hue 0) -- mplot's floor((1 - v) * 240).
 */
function makeLut() {
  const lut = new Uint8ClampedArray(256 * 4);
  for (let k = 0; k < 256; k++) {
    const [r, g, b] = hslRgb(Math.floor((1 - k / 255) * 240));
    lut[4 * k] = r; lut[4 * k + 1] = g; lut[4 * k + 2] = b; lut[4 * k + 3] = 255;
  }
  return lut;
}
const LUT = makeLut();

/**
 * The z scale of a histogram's in-range bins. Log: from 0.5 (one count sits
 * just above the bottom) to the largest bin. Linear: from min(0, smallest) to
 * the largest. Returns {log, min, max, outside}; `outside` is the content of
 * the under/overflow bins, which are not drawn.
 */
function zScale(data, nx, ny, log) {
  let lo = Infinity, hi = -Infinity, all = 0, inside = 0;
  const w = nx + 2;
  for (let i = 0; i < data.length; i++) { const v = data[i]; if (v === v) all += v; }
  for (let j = 0; j < ny; j++) {
    const row = (j + 1) * w + 1;
    for (let i = 0; i < nx; i++) {
      const v = data[row + i];
      if (v !== v) continue;                 // NaN
      inside += v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (hi === -Infinity) { lo = 0; hi = 1; }
  let min, max;
  if (log) {
    min = 0.5;
    max = Math.max(hi, 1);
  } else {
    min = Math.min(0, lo);
    max = hi > min ? hi : min + 1;
  }
  return { log: !!log, min, max, outside: all - inside };
}

/**
 * Bins -> RGBA pixels. `px` is nx*ny*4 bytes, row 0 at the top, so bin row
 * j (y increasing upwards) goes to pixel row ny - 1 - j. Under/overflow bins
 * (index 0 and n+1 of each axis in `data`) are skipped.
 */
function fillPixels(px, data, nx, ny, scale, lut) {
  lut = lut || LUT;
  const w = nx + 2;
  const log = scale.log;
  const a = log ? Math.log(scale.min) : scale.min;
  const span = (log ? Math.log(scale.max) : scale.max) - a;
  const k = span > 0 ? 255 / span : 0;
  for (let j = 0; j < ny; j++) {
    const src = (j + 1) * w + 1;
    let dst = (ny - 1 - j) * nx * 4;
    for (let i = 0; i < nx; i++, dst += 4) {
      const z = data[src + i];
      let c;
      if (z !== z) c = NAN;
      else if (z < 0.5) c = ZERO;
      else {
        let v = ((log ? Math.log(z) : z) - a) * k;
        v = v < 0 ? 0 : v > 255 ? 255 : Math.round(v);
        const o = v * 4;
        px[dst] = lut[o]; px[dst + 1] = lut[o + 1]; px[dst + 2] = lut[o + 2]; px[dst + 3] = 255;
        continue;
      }
      px[dst] = c[0]; px[dst + 1] = c[1]; px[dst + 2] = c[2]; px[dst + 3] = c[3];
    }
  }
  return px;
}

/** A cheap fingerprint of the bins (FNV-1a over both 32-bit halves of each value). */
function checksum(data) {
  let h = 0x811c9dc5 | 0;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    h = Math.imul(h ^ (v | 0), 16777619);
    h = Math.imul(h ^ ((v / 4294967296) | 0), 16777619);
  }
  return h >>> 0;
}

/** Tick values for a linear axis: steps of 1, 2 or 5 x 10^k, at most `max` of them. */
function niceTicks(lo, hi, max) {
  if (!(hi > lo)) return [lo];
  const raw = (hi - lo) / Math.max(1, max);
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  let step = p;
  for (const m of [1, 2, 5, 10]) { step = m * p; if (step >= raw) break; }
  const out = [];
  for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + step * 1e-9; k++) {
    out.push(Number((k * step).toPrecision(12)));
  }
  return out;
}

/** Tick values for a log axis: decades, with 2 and 5 when there are few decades. */
function logTicks(lo, hi) {
  const out = [];
  const d0 = Math.floor(Math.log10(lo)), d1 = Math.ceil(Math.log10(hi));
  const mults = Math.log10(hi / lo) <= 2 ? [1, 2, 5] : [1];
  for (let d = d0; d <= d1; d++) {
    for (const m of mults) {
      const v = m * Math.pow(10, d);
      if (v >= lo * (1 - 1e-9) && v <= hi * (1 + 1e-9)) out.push(v);
    }
  }
  return out;
}

/** A tick label: short, "1e+6" style for large or tiny numbers (as mplot). */
function fmt(v) {
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e4 || a < 1e-3) {
    const [m, e] = v.toExponential().split("e");
    return `${Number(Number(m).toPrecision(3))}e${e}`;
  }
  return String(Number(v.toPrecision(6)));
}

/** A bin content for the hover readout. */
function fmtCount(v) {
  if (v !== v) return "NaN";
  return Number.isInteger(v) ? v.toLocaleString() : String(Number(v.toPrecision(6)));
}

function dpr() {
  const r = root.devicePixelRatio;
  return typeof r === "number" && r > 0 ? r : 1;
}

class DQMHeatmap {
  /**
   * @param {HTMLElement} div  the plot box; its CSS size is the plot's size
   * @param {object} [opts]    {width, height}: a size for when the div has none (tests)
   */
  constructor(div, opts) {
    this.div = div;
    this.opts = opts || {};
    const doc = root.document;
    div.style.position = "relative";
    this.axes = doc.createElement("canvas");    // frame, ticks, labels, colour bar
    this.plot = doc.createElement("canvas");    // the image, and nothing else
    for (const c of [this.axes, this.plot]) {
      c.style.position = "absolute";
      c.style.left = "0px";
      c.style.top = "0px";
      div.appendChild(c);
    }
    this.image = doc.createElement("canvas");   // offscreen, nx x ny
    this.tip = doc.createElement("div");
    this.tip.className = "dqm-heatmap-tip";
    // Styled here rather than in a stylesheet, so the renderer is one file.
    Object.assign(this.tip.style, {
      display: "none", position: "absolute", pointerEvents: "none", whiteSpace: "nowrap",
      background: "rgba(255, 255, 255, 0.92)", border: "1px solid #808080", borderRadius: "3px",
      padding: "1px 5px", font: FONT, color: "#000",
    });
    div.appendChild(this.tip);

    this.hist = null;       // the last histogram drawn
    this.scale = null;      // its z scale
    this.sig = null;        // what the image was made from
    this.layoutKey = null;  // what the axes canvas was drawn for
    this.box = null;        // plot rectangle in CSS px: {x1, y1 (bottom), x2, y2 (top)}
    this.xTitle = "";
    this.yTitle = "";
    this.logZ = true;
    this.draws = 0;         // image draws (for tests and profiling)
    this.axisDraws = 0;

    const self = this;
    this.plot.addEventListener("mousemove", function (e) { self.hover(e.offsetX, e.offsetY); });
    this.plot.addEventListener("mouseleave", function () { self.hover(null, null); });
    // A window resize (or a browser zoom, which changes the pixel ratio) is a
    // new layout; the next draw notices by itself, this only triggers it.
    if (root.window && root.window.addEventListener) {
      root.window.addEventListener("resize", function () {
        self.cssSize = null;
        if (self.hist) self.draw();
      });
    }
  }

  /**
   * Show a decoded 2D histogram. Returns true if anything was drawn: an
   * update with the same bins, scale and titles as the last one is a no-op.
   *
   * @param {object} hist  BRPC.histogram()'s result (dimensions 2)
   * @param {object} opt   {logZ, xTitle, yTitle}
   */
  setData(hist, opt) {
    opt = opt || {};
    this.xTitle = opt.xTitle || "";
    this.yTitle = opt.yTitle || "";
    this.logZ = opt.logZ !== undefined ? !!opt.logZ : this.logZ;
    const sig = [checksum(hist.data), hist.entries, hist.nBins.join(","), hist.lowEdge.join(","),
                 hist.highEdge.join(","), this.logZ].join("|");
    if (sig === this.sig) return this.draw(false);
    this.sig = sig;
    this.hist = hist;
    const nx = hist.nBins[0], ny = hist.nBins[1];
    this.scale = zScale(hist.data, nx, ny, this.logZ);
    if (this.image.width !== nx || this.image.height !== ny || !this.pixels) {
      this.image.width = nx;
      this.image.height = ny;
      this.ictx = this.image.getContext("2d");
      this.pixels = this.ictx.createImageData(nx, ny);
    }
    fillPixels(this.pixels.data, hist.data, nx, ny, this.scale, LUT);
    this.ictx.putImageData(this.pixels, 0, 0);
    const drawn = this.draw(true);
    // The readout under a resting pointer follows the new contents.
    if (this.pointer && this.tip.style.display !== "none") this.hover(this.pointer[0], this.pointer[1]);
    return drawn;
  }

  /**
   * CSS size of the box. Measured once and again only after a resize: reading
   * clientWidth right after the page changed some text forces a layout, on
   * every update.
   */
  size() {
    if (this.cssSize) return this.cssSize;
    const cw = this.div.clientWidth, ch = this.div.clientHeight;
    const size = [cw || this.opts.width || 560, ch || this.opts.height || 330];
    if (cw && ch) this.cssSize = size;          // a hidden box (0 x 0) is measured again
    return size;
  }

  /**
   * Draw: the axes canvas when the layout changed, the image when it or the
   * layout changed. Returns true if the image was drawn.
   */
  draw(imageChanged) {
    if (!this.hist) return false;
    const [w, h] = this.size();
    const r = dpr();
    const s = this.scale;
    const key = [w, h, r, this.hist.nBins, this.hist.lowEdge, this.hist.highEdge,
                 this.xTitle, this.yTitle, s.log, s.min, s.max].join("|");
    const relayout = key !== this.layoutKey;
    if (relayout) {
      for (const c of [this.axes, this.plot]) {
        c.width = Math.round(w * r);
        c.height = Math.round(h * r);
        c.style.width = `${w}px`;
        c.style.height = `${h}px`;
      }
      this.layoutKey = key;
      this.drawAxes(w, h, r);
    }
    if (!relayout && imageChanged === false) return false;
    const ctx = this.plot.getContext("2d");
    const b = this.box;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.plot.width, this.plot.height);
    ctx.imageSmoothingEnabled = false;
    // Device pixels, snapped, so bin edges are sharp at any pixel ratio.
    const X1 = Math.round(b.x1 * r), X2 = Math.round(b.x2 * r);
    const Y2 = Math.round(b.y2 * r), Y1 = Math.round(b.y1 * r);
    ctx.drawImage(this.image, 0, 0, this.image.width, this.image.height, X1, Y2, X2 - X1, Y1 - Y2);
    this.draws++;
    return true;
  }

  /** Frame, ticks, tick labels, axis titles and the colour bar. */
  drawAxes(w, h, r) {
    const ctx = this.axes.getContext("2d");
    const hist = this.hist, s = this.scale;
    ctx.setTransform(r, 0, 0, r, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#FFFFFF";
    ctx.fillRect(0, 0, w, h);
    ctx.font = FONT;

    const xlo = hist.lowEdge[0], xhi = hist.highEdge[0];
    const ylo = hist.lowEdge[1], yhi = hist.highEdge[1];
    // Counts: the log scale starts at 0.5, but a label there reads as half a count.
    const zt = s.log ? logTicks(Math.max(s.min, Math.min(1, s.max)), s.max) : niceTicks(s.min, s.max, 8);
    const zw = Math.max(0, ...zt.map((v) => ctx.measureText(fmt(v)).width));

    // Right to left: colour bar labels, the bar, the gap; bottom to top: the
    // x title, the x tick labels, the ticks.
    const barW = 10;
    const x2 = w - (6 + zw + 6 + barW + 10);
    const y2 = 8;
    const y1 = h - (this.xTitle ? 38 : 22);
    const approxY = niceTicks(ylo, yhi, Math.max(2, Math.floor((y1 - y2) / 28)));
    const yw = Math.max(0, ...approxY.map((v) => ctx.measureText(fmt(v)).width));
    const x1 = (this.yTitle ? 20 : 4) + yw + 8;
    this.box = { x1, y1, x2, y2 };

    const X = (v) => x1 + (v - xlo) / (xhi - xlo) * (x2 - x1);
    const Y = (v) => y1 - (v - ylo) / (yhi - ylo) * (y1 - y2);

    ctx.strokeStyle = AXIS;
    ctx.fillStyle = LABEL;
    ctx.lineWidth = 1;
    ctx.beginPath();
    const xt = niceTicks(xlo, xhi, Math.max(2, Math.floor((x2 - x1) / 55)));
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const v of xt) {
      const x = Math.round(X(v)) + 0.5;
      ctx.moveTo(x, y1); ctx.lineTo(x, y1 + 5);
      ctx.fillText(fmt(v), x, y1 + 7);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const v of approxY) {
      const y = Math.round(Y(v)) + 0.5;
      ctx.moveTo(x1 - 5, y); ctx.lineTo(x1, y);
      ctx.fillText(fmt(v), x1 - 7, y);
    }
    ctx.stroke();
    ctx.strokeRect(Math.round(x1) - 0.5, Math.round(y2) - 0.5, Math.round(x2 - x1) + 1, Math.round(y1 - y2) + 1);

    if (this.xTitle) {
      ctx.textAlign = "center";
      ctx.textBaseline = "bottom";
      ctx.fillText(this.xTitle, (x1 + x2) / 2, h - 2);
    }
    if (this.yTitle) {
      ctx.save();
      ctx.translate(2, (y1 + y2) / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(this.yTitle, 0, 0);
      ctx.restore();
    }

    // The colour bar: the table itself, scaled, and its ticks.
    const bx = x2 + 10;
    if (!this.bar) {
      this.bar = root.document.createElement("canvas");
      this.bar.width = 1;
      this.bar.height = 256;
      const bctx = this.bar.getContext("2d");
      const img = bctx.createImageData(1, 256);
      for (let k = 0; k < 256; k++) img.data.set(LUT.subarray(4 * (255 - k), 4 * (256 - k)), 4 * k);
      bctx.putImageData(img, 0, 0);
    }
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(this.bar, 0, 0, 1, 256, bx, y2, barW, y1 - y2);
    ctx.strokeRect(bx - 0.5, Math.round(y2) - 0.5, barW + 1, Math.round(y1 - y2) + 1);
    const Z = s.log
      ? (v) => y1 - (Math.log(v) - Math.log(s.min)) / (Math.log(s.max) - Math.log(s.min)) * (y1 - y2)
      : (v) => y1 - (v - s.min) / (s.max - s.min) * (y1 - y2);
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.beginPath();
    for (const v of zt) {
      const y = Math.round(Z(v)) + 0.5;
      ctx.moveTo(bx + barW, y); ctx.lineTo(bx + barW + 4, y);
      ctx.fillText(fmt(v), bx + barW + 6, y);
    }
    ctx.stroke();
    this.axisDraws++;
  }

  /**
   * The bin under a point (CSS px relative to the box), or null outside the
   * plot area: {i, j} (0-based, in-range bins), the bin's centre {x, y} and
   * its content z.
   */
  binAt(px, py) {
    const b = this.box, hist = this.hist;
    if (!b || !hist || px === null || px < b.x1 || px >= b.x2 || py <= b.y2 || py > b.y1) return null;
    const nx = hist.nBins[0], ny = hist.nBins[1];
    const i = Math.min(nx - 1, Math.floor((px - b.x1) / (b.x2 - b.x1) * nx));
    const j = Math.min(ny - 1, Math.floor((b.y1 - py) / (b.y1 - b.y2) * ny));
    const wx = (hist.highEdge[0] - hist.lowEdge[0]) / nx;
    const wy = (hist.highEdge[1] - hist.lowEdge[1]) / ny;
    return {
      i, j,
      x: hist.lowEdge[0] + (i + 0.5) * wx,
      y: hist.lowEdge[1] + (j + 0.5) * wy,
      z: hist.data[(i + 1) + (j + 1) * (nx + 2)],
    };
  }

  /** The hover readout: bin centre x, y and content, next to the pointer. */
  hover(px, py) {
    this.pointer = px === null ? null : [px, py];
    const bin = this.binAt(px, py);
    if (!bin) { this.tip.style.display = "none"; return null; }
    const name = (t, d) => (t ? t.replace(/\s*\(.*\)\s*$/, "") : d);
    const text = `${name(this.xTitle, "x")} = ${fmt(Number(bin.x.toPrecision(6)))} · ` +
                 `${name(this.yTitle, "y")} = ${fmt(Number(bin.y.toPrecision(6)))} · ${fmtCount(bin.z)}`;
    if (this.tip.textContent !== text) this.tip.textContent = text;
    const [w] = this.size();
    this.tip.style.display = "";
    // Right of the pointer, or left of it near the right edge.
    if (px > w / 2) { this.tip.style.left = ""; this.tip.style.right = `${Math.round(w - px + 12)}px`; }
    else { this.tip.style.right = ""; this.tip.style.left = `${Math.round(px + 12)}px`; }
    this.tip.style.top = `${Math.round(Math.max(0, py - 26))}px`;
    return bin;
  }
}

DQMHeatmap.LUT = LUT;
DQMHeatmap.zScale = zScale;
DQMHeatmap.fillPixels = fillPixels;
DQMHeatmap.checksum = checksum;
DQMHeatmap.niceTicks = niceTicks;
DQMHeatmap.logTicks = logTicks;
DQMHeatmap.fmt = fmt;

root.DQMHeatmap = DQMHeatmap;
if (typeof module !== "undefined" && module.exports) module.exports = DQMHeatmap;

})(typeof globalThis !== "undefined" ? globalThis : this);
