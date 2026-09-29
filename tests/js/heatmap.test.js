//
// DQMHeatmap (pages/js/dqm-heatmap.js): bins -> pixels, colours, orientation,
// hover, redraw only on change, pixel ratio. Canvases are domstub's recording
// ones, so "what was drawn" is the list of context calls.
//

const test = require("node:test");
const assert = require("node:assert");
const path = require("node:path");

const { El, makeDocument } = require("./domstub.js");

const H = require(path.join(__dirname, "..", "..", "pages", "js", "dqm-heatmap.js"));

function withDom(fn, ratio) {
  const g = globalThis;
  const saved = { document: g.document, window: g.window, devicePixelRatio: g.devicePixelRatio };
  g.document = makeDocument();
  const listeners = {};
  g.window = { addEventListener: (n, f) => (listeners[n] = listeners[n] || []).push(f) };
  g.devicePixelRatio = ratio;
  try { return fn(listeners); } finally { Object.assign(g, saved); }
}

/** A 2D histogram with nx x ny in-range bins; f(i, j) the content, `flow` in every outer bin. */
function hist(nx, ny, f, flow = 0, edges = [[0, nx], [0, ny]]) {
  const data = new Float64Array((nx + 2) * (ny + 2)).fill(flow);
  let entries = 0;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const v = f(i, j);
      data[(i + 1) + (j + 1) * (nx + 2)] = v;
      entries += v;
    }
  }
  return { dimensions: 2, nBins: [nx, ny], lowEdge: [edges[0][0], edges[1][0]],
           highEdge: [edges[0][1], edges[1][1]], entries: entries + flow * 2 * (nx + ny + 2), data };
}

const rgba = (px, nx, col, row) => Array.from(px.slice((row * nx + col) * 4, (row * nx + col) * 4 + 4));
const RED = [255, 0, 0, 255], BLUE = [0, 0, 255, 255], GREEN = [0, 255, 0, 255], WHITE = [255, 255, 255, 255];

test("the colour table runs from blue (bottom) through green to red (top), as mplot's hues", () => {
  const L = H.LUT;
  assert.deepStrictEqual(Array.from(L.slice(0, 4)), BLUE);
  assert.deepStrictEqual(Array.from(L.slice(255 * 4, 256 * 4)), RED);
  // v = 0.5 -> floor(0.5 * 240) = hue 120, pure green
  const mid = Math.round(127.5);
  const [r, g, b] = Array.from(L.slice(mid * 4, mid * 4 + 3));
  assert.ok(g === 255 && r < 10 && b < 10, `${r},${g},${b}`);
});

test("bins map to pixels with y up, and under/overflow are neither drawn nor scaled", () => {
  // 3 x 2 bins, bin (i, j) = 1 + i + 3 j; the outer bins hold 1e9, which must
  // not set the top of the z scale.
  const h = hist(3, 2, (i, j) => 1 + i + 3 * j, 1e9);
  const s = H.zScale(h.data, 3, 2, false);
  assert.strictEqual(s.max, 6, "the largest in-range bin, not the overflow");
  assert.strictEqual(s.min, 0);
  assert.strictEqual(s.outside, 1e9 * 2 * (3 + 2 + 2));
  const px = H.fillPixels(new Uint8ClampedArray(3 * 2 * 4), h.data, 3, 2, s);
  assert.strictEqual(px.length, 24, "one pixel per in-range bin");
  // Bin (2, 1) = 6 is the maximum: top row (pixel row 0), right column.
  assert.deepStrictEqual(rgba(px, 3, 2, 0), RED);
  // Bin (0, 0) = 1 is the bottom-left pixel, near the bottom of the scale.
  const bl = rgba(px, 3, 0, 1);
  assert.ok(bl[2] > 200 && bl[0] === 0, `bottom-left ${bl}`);
});

test("log z: 0.5 is the bottom, the largest bin the top, empty bins white", () => {
  const h = hist(4, 1, (i) => [0, 1, 100, 10000][i]);
  const s = H.zScale(h.data, 4, 1, true);
  assert.deepStrictEqual([s.min, s.max], [0.5, 10000]);
  const px = H.fillPixels(new Uint8ClampedArray(16), h.data, 4, 1, s);
  assert.deepStrictEqual(rgba(px, 4, 0, 0), WHITE, "an empty bin");
  assert.deepStrictEqual(rgba(px, 4, 3, 0), RED);
  // log10(100/0.5) / log10(10000/0.5) = 0.535 of the way up: green-ish, hue ~111.
  const [r, g, b] = rgba(px, 4, 2, 0);
  assert.ok(g === 255 && b === 0 && r > 0 && r < 80, `${r},${g},${b}`);
  // The same bins on a linear scale: 100 of 10000 is nearly the bottom (blue).
  const lin = H.fillPixels(new Uint8ClampedArray(16), h.data, 4, 1, H.zScale(h.data, 4, 1, false));
  const c = rgba(lin, 4, 2, 0);
  assert.ok(c[2] === 255 && c[0] === 0, `linear ${c}`);
});

test("ticks: 1-2-5 steps inside the range, log decades", () => {
  assert.deepStrictEqual(H.niceTicks(-0.5, 17.5, 5), [0, 5, 10, 15]);
  assert.deepStrictEqual(H.niceTicks(0, 1, 5), [0, 0.2, 0.4, 0.6, 0.8, 1]);
  assert.deepStrictEqual(H.logTicks(0.5, 12000), [1, 10, 100, 1000, 10000]);
  assert.deepStrictEqual(H.logTicks(1, 30), [1, 2, 5, 10, 20]);
  assert.strictEqual(H.fmt(1e6), "1e+6");
  assert.strictEqual(H.fmt(2500), "2500");
  assert.strictEqual(H.fmt(10000), "1e+4", "as mplot labels its colour bar");
});

test("one image per update: putImageData + a single scaled drawImage, smoothing off", () => withDom(() => {
  const div = new El("div");
  const hm = new H(div);
  const h = hist(18, 256, (i, j) => (i * j) % 7);
  assert.strictEqual(hm.setData(h, { logZ: true, xTitle: "SMA channel", yTitle: "ToT code" }), true);
  assert.strictEqual(hm.image.width, 18);
  assert.strictEqual(hm.image.height, 256);
  const ictx = hm.image.getContext("2d");
  assert.strictEqual(ictx.count("putImageData"), 1);
  assert.strictEqual(ictx.count("fillRect"), 0, "no per-bin rectangles");
  const pctx = hm.plot.getContext("2d");
  assert.strictEqual(pctx.count("drawImage"), 1);
  assert.strictEqual(pctx.count("fillRect"), 0);
  assert.strictEqual(pctx.imageSmoothingEnabled, false);
  const args = pctx.ops.find((o) => o[0] === "drawImage")[1];
  assert.deepStrictEqual(args.slice(1, 5), [0, 0, 18, 256], "the whole image");
  const b = hm.box;
  assert.deepStrictEqual(args.slice(5), [Math.round(b.x1), Math.round(b.y2),
                                         Math.round(b.x2) - Math.round(b.x1), Math.round(b.y1) - Math.round(b.y2)]);
  // The axes canvas carries the titles and the tick labels.
  const texts = hm.axes.getContext("2d").texts();
  assert.ok(texts.includes("SMA channel") && texts.includes("ToT code"), texts.join(","));
  assert.ok(texts.includes("0") && texts.includes("250"), texts.join(","));
}));

test("no redraw when the histogram has not changed; a change or a scale toggle redraws", () => withDom(() => {
  const hm = new H(new El("div"));
  const h = hist(5, 4, (i, j) => i + j);
  const opt = { logZ: true };
  assert.strictEqual(hm.setData(h, opt), true);
  const axesDraws = hm.axisDraws;
  // A fresh decode of the same bytes: a new object with the same contents.
  const same = Object.assign({}, h, { data: new Float64Array(h.data) });
  assert.strictEqual(hm.setData(same, opt), false);
  assert.strictEqual(hm.draws, 1);
  assert.strictEqual(hm.axisDraws, axesDraws, "the axes are not redrawn either");
  assert.strictEqual(hm.image.getContext("2d").count("putImageData"), 1);

  const more = hist(5, 4, (i, j) => i + j + (i === 2 && j === 2 ? 1 : 0));
  assert.strictEqual(hm.setData(more, opt), true, "one bin changed");
  assert.strictEqual(hm.draws, 2);
  assert.strictEqual(hm.setData(more, { logZ: false }), true, "log z toggled");
  assert.strictEqual(hm.draws, 3);
}));

test("the axes are drawn once per layout, not per update", () => withDom(() => {
  const hm = new H(new El("div"));
  hm.setData(hist(5, 4, () => 3), { logZ: true });
  assert.strictEqual(hm.axisDraws, 1);
  // Same z range (max still 3), different bins: image only.
  hm.setData(hist(5, 4, (i) => (i === 0 ? 3 : 2)), { logZ: true });
  assert.strictEqual(hm.draws, 2);
  assert.strictEqual(hm.axisDraws, 1);
  // A new maximum changes the colour-bar labels: a new layout.
  hm.setData(hist(5, 4, () => 30), { logZ: true });
  assert.strictEqual(hm.axisDraws, 2);
}));

test("hover: the pointer's bin, with y up, and nothing outside the plot", () => withDom(() => {
  const hm = new H(new El("div"));
  // x: 4 bins on [-0.5, 3.5) (channels 0..3); y: 2 bins on [0, 100).
  const h = hist(4, 2, (i, j) => 10 * i + j, 7, [[-0.5, 3.5], [0, 100]]);
  hm.setData(h, { logZ: false, xTitle: "SMA channel", yTitle: "ToT code" });
  const b = hm.box;
  const at = (fx, fy) => [b.x1 + fx * (b.x2 - b.x1), b.y1 - fy * (b.y1 - b.y2)];
  let bin = hm.binAt(...at(0.9, 0.9));                  // top right
  assert.deepStrictEqual([bin.i, bin.j, bin.x, bin.y, bin.z], [3, 1, 3, 75, 31]);
  bin = hm.binAt(...at(0.1, 0.1));                      // bottom left
  assert.deepStrictEqual([bin.i, bin.j, bin.z], [0, 0, 0]);
  assert.strictEqual(hm.binAt(b.x1 - 3, b.y1 - 5), null, "left of the frame");
  assert.strictEqual(hm.binAt(b.x1 + 5, b.y1 + 3), null, "below the frame (the x labels)");

  // The readout follows the pointer, and goes away with it.
  hm.plot.dispatch("mousemove", { offsetX: at(0.6, 0.9)[0], offsetY: at(0.6, 0.9)[1] });
  assert.strictEqual(hm.tip.style.display, "");
  assert.strictEqual(hm.tip.textContent, "SMA channel = 2 · ToT code = 75 · 21");
  hm.plot.dispatch("mouseleave", {});
  assert.strictEqual(hm.tip.style.display, "none");
}));

test("device pixel ratio: backing store scaled, drawing in CSS pixels", () => withDom((listeners) => {
  const div = new El("div");
  div.clientWidth = 560; div.clientHeight = 330;
  const hm = new H(div);
  hm.setData(hist(5, 4, (i) => i), { logZ: true });
  assert.deepStrictEqual([hm.axes.width, hm.axes.height], [1120, 660]);
  assert.deepStrictEqual([hm.plot.width, hm.plot.height], [1120, 660]);
  assert.strictEqual(hm.plot.style.width, "560px");
  const t = hm.axes.getContext("2d").ops.find((o) => o[0] === "setTransform")[1];
  assert.deepStrictEqual(t, [2, 0, 0, 2, 0, 0]);
  // The image goes to device pixels: the frame's CSS box times two.
  const args = hm.plot.getContext("2d").ops.find((o) => o[0] === "drawImage")[1];
  assert.strictEqual(args[5], Math.round(hm.box.x1 * 2));
  assert.strictEqual(args[7], Math.round(hm.box.x2 * 2) - Math.round(hm.box.x1 * 2));

  // A resize (or a zoom, which fires one) with the same data: new layout, redrawn.
  div.clientWidth = 400;
  listeners.resize.forEach((f) => f());
  assert.strictEqual(hm.axes.width, 800);
  assert.strictEqual(hm.draws, 2);
}, 2));
