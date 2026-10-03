//
// dqm-common.js -- discovery, ODB shape handling and small RPC wrappers.
//
// Generic: nothing in this file knows about WaveDream, or about any particular
// equipment, bank or channel. The page-specific files build on it.
//
// Everything hangs off a single global `DQM`, the way every other MIDAS custom
// page does it -- no build step, no bundler, and the same `<script src>` loading
// the stock resources use. ES modules were the alternative and were rejected
// for one concrete reason: later stages load tab fragments through musip's
// self-deleting-iframe trick, which works precisely because fragments share the
// parent's global scope. Modules deliberately do not.
//
// The pure functions are also exported for `node --test`; the shim at the
// bottom is a no-op in a browser.
//

(function (root) {
"use strict";

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------
//
// This object is what runs when /DQM/Scalars does not exist, which is the
// normal state of an experiment nobody has set up yet -- so the page works on
// first open and says it is using built-ins.
//
// It is deliberately plain JSON with no comments inside: tests/test_manifest.py
// parses it and asserts it agrees key-for-key with the Python copy in
// mdqm/install/config_defaults.py, which is what gets seeded into the ODB.
// Explanations for each key live in that file.
//
const DEFAULTS = {
  "Equipment": [""],
  "Bank Pattern": "^([STXD])(\\d{3})$",
  "Role Rates": "S",
  "Role Timestamp": "T",
  "Role Temperature": "X",
  "Role Threshold": "D",
  "Trigger Scaler Names": ["ptrn_trg", "ext_trg"],
  "Clock Scaler Name": "ext_clk",
  "Ticks Per Second": 80e6,
  "Disabled Value": -1,
  "Stale Seconds": 10.0,
  "Rate Warn Hz": 0.0,
  "Rate Alarm Hz": 0.0,
  "Temp Warn C": 60.0,
  "Temp Alarm C": 70.0,
  "History Timescale": "10m",
  "Health Subtrees": ["Variables/Thread"],
  "Refresh ms": 1000
};

const CONFIG_ROOT = "/DQM/Scalars";

// ---------------------------------------------------------------------------
// ODB shape handling
// ---------------------------------------------------------------------------

/**
 * Coerce an ODB value to an array of length `n`.
 *
 * A MIDAS array of length one is indistinguishable from a scalar in the JSON
 * encoding, so `X036` arrives as a bare float and `Names X036` as a bare
 * string. `num_values` from db_ls is the authority, and it is *absent* rather
 * than 1 in that case -- hence the explicit default at every call site.
 *
 * Calling .map() or .length on a raw ODB value without going through here is a
 * latent crash that surfaces the day somebody configures a one-element bank.
 */
function asArray(v, n) {
  if (v === undefined || v === null) return [];
  const a = Array.isArray(v) ? v : [v];
  if (n && a.length < n) return a.concat(new Array(n - a.length).fill(null));
  return a;
}

/**
 * Coerce an ODB numeric to a JS number.
 *
 * TID_DWORD and TID_BOOL come back as hex *strings*: a timestamp bank reads
 * ["0x1dd3d038", "0x00000002", "0x00000000"]. Reaching for parseInt without a
 * radix would read "0x..." correctly by accident but "010" as octal in older
 * engines, so the radix is explicit.
 */
function asUInt(v) {
  if (typeof v === "number") return v;
  if (typeof v === "boolean") return v ? 1 : 0;
  if (typeof v !== "string") return NaN;
  const s = v.trim();
  if (s === "") return NaN;
  return /^0[xX]/.test(s) ? parseInt(s, 16) : parseInt(s, 10);
}

/** Join the two halves of a 64-bit value split across two DWORDs. */
function asUInt64(lsb, msb) {
  const lo = asUInt(lsb), hi = asUInt(msb);
  if (Number.isNaN(lo) || Number.isNaN(hi)) return NaN;
  return hi * 4294967296 + lo;
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/**
 * Match one Variables key against the configured bank pattern.
 *
 * Returns {bank, role, board} or null. Capture 1 is the role letter, capture 2
 * the board id -- that pairing is the entire contract between this page and a
 * frontend's bank naming, and it lives in the ODB so another experiment is a
 * config change rather than a code change.
 */
function parseBank(name, pattern) {
  const re = pattern instanceof RegExp ? pattern : new RegExp(pattern);
  const m = re.exec(name);
  if (!m) return null;
  return { bank: name, role: m[1], board: m[2] };
}

/**
 * Turn a db_ls of several /Equipment/<eq>/Variables into a board model.
 *
 * `lsByEq` is {equipmentName: {key: value, "key/key": {num_values, type, ...}}},
 * i.e. exactly what mjsonrpc_db_ls returns. Boards with no rates bank are
 * dropped -- there is nothing to show for them -- but a board missing any of
 * the timestamp, temperature or threshold banks is kept, because those are
 * genuinely optional and the page degrades per-panel.
 */
function groupBoards(lsByEq, cfg) {
  const roleOf = {};
  roleOf[cfg["Role Rates"]] = "rates";
  roleOf[cfg["Role Timestamp"]] = "timestamp";
  roleOf[cfg["Role Temperature"]] = "temperature";
  roleOf[cfg["Role Threshold"]] = "threshold";

  const boards = [];
  for (const eq of Object.keys(lsByEq || {}).sort()) {
    const ls = lsByEq[eq] || {};
    const byBoard = {};
    for (const key of Object.keys(ls)) {
      if (key.endsWith("/key")) continue;
      const hit = parseBank(key, cfg["Bank Pattern"]);
      if (!hit) continue;
      const role = roleOf[hit.role];
      if (!role) continue;
      const meta = ls[key + "/key"] || {};
      (byBoard[hit.board] = byBoard[hit.board] || { equipment: eq, board: hit.board, banks: {} })
        .banks[role] = {
          name: key,
          numValues: meta.num_values || 1,
          type: meta.type,
          // Unix seconds, from db_ls. This is how the page knows on its very
          // first paint whether the values it is about to show are current --
          // without it, a page opened onto a frontend that died last week reads
          // as live until enough time passes to notice nothing is changing.
          lastWritten: meta.last_written,
        };
    }
    for (const id of Object.keys(byBoard).sort()) {
      if (byBoard[id].banks.rates) boards.push(byBoard[id]);
    }
  }
  return boards;
}

/**
 * The `EVENT:TAG` strings MhistoryGraph wants for a bank.
 *
 * Prefer the tags mlogger actually recorded over Settings/Names: the on-disk
 * schema is what the history reader will match, and it can lag a channel
 * rename by one mlogger restart. Names is the fallback, and `bank[i]` the
 * fallback's fallback, so a graph always has *some* label.
 */
function historyVarString(event, tags, names, n) {
  let labels = (tags && tags.length) ? tags : asArray(names, n);
  if (!labels.length) {
    labels = [];
    for (let i = 0; i < (n || 0); i++) labels.push(String(i));
  }
  return labels.filter((t) => t !== null && t !== undefined && t !== "")
               .map((t) => `${event}:${t}`);
}

/**
 * Human labels for one bank, with the same three-step fallback.
 */
function bankLabels(bankName, names, n) {
  const out = [];
  const given = asArray(names, n);
  for (let i = 0; i < n; i++) {
    const v = given[i];
    out.push(v === null || v === undefined || v === "" ? `${bankName}[${i}]` : String(v));
  }
  return out;
}

// ---------------------------------------------------------------------------
// RPC wrappers
// ---------------------------------------------------------------------------
// Modelled on musip's quads.js:12-56, which is the cleanest small pair in that
// tree. Every failure goes through mjsonrpc_error_alert so a broken page says
// so rather than sitting silently stale.

async function getODB(paths) {
  if (!Array.isArray(paths)) {
    const rpc = await mjsonrpc_db_get_value(paths);
    return rpc.result.data[0];
  }
  const rpc = await mjsonrpc_db_get_values(paths);
  return rpc.result.data;
}

async function setODB(paths, values, errorText) {
  try {
    if (!Array.isArray(paths)) return await mjsonrpc_db_paste([paths], [values]);
    return await mjsonrpc_db_paste(paths, values);
  } catch (error) {
    mjsonrpc_error_alert(errorText || `Could not set ${paths}: ${error}`);
    return null;
  }
}

/**
 * db_ls several paths at once.
 *
 * Discovery uses db_ls throughout rather than db_get_values for two reasons:
 * db_get_values lower-cases key names ("names s036" for "Names S036"), and
 * db_ls is the only call that reports num_values, without which a one-element
 * array cannot be told from a scalar.
 */
async function lsODB(paths) {
  const rpc = await mjsonrpc_call("db_ls", { paths: paths });
  return rpc.result.data;
}

/** Load the page config, falling back to built-ins. Never throws. */
async function loadConfig(root) {
  const cfg = Object.assign({}, DEFAULTS);
  cfg._seeded = false;
  try {
    const rpc = await mjsonrpc_db_get_values([root || CONFIG_ROOT]);
    const got = rpc.result.data[0];
    if (got && rpc.result.status[0] === 1) {
      for (const key of Object.keys(DEFAULTS)) {
        // db_get_values lower-cases; match case-insensitively and keep our
        // canonical key so the rest of the page can use one spelling.
        for (const k of Object.keys(got)) {
          if (k.endsWith("/key")) continue;
          if (k.toLowerCase() === key.toLowerCase()) cfg[key] = got[k];
        }
      }
      cfg._seeded = true;
    }
  } catch (e) {
    // A missing config subtree is the expected state on a fresh experiment,
    // not an error worth interrupting anyone about.
  }
  return cfg;
}


// ---------------------------------------------------------------------------
// Colour scale
// ---------------------------------------------------------------------------
//
// viridis, the one sequential scale the pages use for "more is brighter": it
// reads in greyscale and for the common colour-vision deficiencies, which the
// hue rainbow (blue -> green -> red) does not. matplotlib's 256-entry table,
// index 0 dark purple to 255 yellow, as one hex string so the file stays short.

const VIRIDIS_HEX =
  "44015444025645045745055946075a46085c460a5d460b5e470d60470e61471063471164471365481467481668481769" +
  "48186a481a6c481b6d481c6e481d6f481f70482071482173482374482475482576482677482878482979472a7a472c7a" +
  "472d7b472e7c472f7d46307e46327e46337f463480453581453781453882443983443a83443b84433d84433e85423f85" +
  "4240864241864142874144874045884046883f47883f48893e49893e4a893e4c8a3d4d8a3d4e8a3c4f8a3c508b3b518b" +
  "3b528b3a538b3a548c39558c39568c38588c38598c375a8c375b8d365c8d365d8d355e8d355f8d34608d34618d33628d" +
  "33638d32648e32658e31668e31678e31688e30698e306a8e2f6b8e2f6c8e2e6d8e2e6e8e2e6f8e2d708e2d718e2c718e" +
  "2c728e2c738e2b748e2b758e2a768e2a778e2a788e29798e297a8e297b8e287c8e287d8e277e8e277f8e27808e26818e" +
  "26828e26828e25838e25848e25858e24868e24878e23888e23898e238a8d228b8d228c8d228d8d218e8d218f8d21908d" +
  "21918c20928c20928c20938c1f948c1f958b1f968b1f978b1f988b1f998a1f9a8a1e9b8a1e9c891e9d891f9e891f9f88" +
  "1fa0881fa1881fa1871fa28720a38620a48621a58521a68522a78522a88423a98324aa8325ab8225ac8226ad8127ad81" +
  "28ae8029af7f2ab07f2cb17e2db27d2eb37c2fb47c31b57b32b67a34b67935b77937b87838b9773aba763bbb753dbc74" +
  "3fbc7340bd7242be7144bf7046c06f48c16e4ac16d4cc26c4ec36b50c46a52c56954c56856c66758c7655ac8645cc863" +
  "5ec96260ca6063cb5f65cb5e67cc5c69cd5b6ccd5a6ece5870cf5773d05675d05477d1537ad1517cd2507fd34e81d34d" +
  "84d44b86d54989d5488bd6468ed64590d74393d74195d84098d83e9bd93c9dd93ba0da39a2da37a5db36a8db34aadc32" +
  "addc30b0dd2fb2dd2db5de2bb8de29bade28bddf26c0df25c2df23c5e021c8e020cae11fcde11dd0e11cd2e21bd5e21a" +
  "d8e219dae319dde318dfe318e2e418e5e419e7e419eae51aece51befe51cf1e51df4e61ef6e620f8e621fbe723fde725";

const VIRIDIS = new Uint8Array(256 * 3);
for (let k = 0; k < 256 * 3; k++) VIRIDIS[k] = parseInt(VIRIDIS_HEX.substr(2 * k, 2), 16);

/** viridis at v in [0, 1] (clamped) as a CSS "rgb(r, g, b)". */
function viridis(v) {
  const k = 3 * Math.round(255 * Math.min(1, Math.max(0, v)));
  return `rgb(${VIRIDIS[k]}, ${VIRIDIS[k + 1]}, ${VIRIDIS[k + 2]})`;
}


// ---------------------------------------------------------------------------
// Publish. `DQM` in a browser, module.exports under node --test.
// ---------------------------------------------------------------------------
const DQM = { CONFIG_ROOT, DEFAULTS, VIRIDIS, asArray, asUInt, asUInt64, bankLabels, getODB, groupBoards, historyVarString, loadConfig, lsODB, parseBank, setODB, viridis };
root.DQM = DQM;
if (typeof module !== "undefined" && module.exports) module.exports = DQM;

})(typeof globalThis !== "undefined" ? globalThis : this);
