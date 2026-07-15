/* TSDB — in-browser port of the Node Float64 block store (TSDB.md v1.3).
 *
 * Faithful to the invariant: dumb, compact, append-oriented, Float64-based,
 * high-level agnostic. The ONLY thing swapped out is the physical layer —
 * where the Node build appends bytes to `<id>.fabf` files via `fs`, this port
 * appends into per-file Uint8Array buffers held in memory. Everything above
 * that (manifest, files, keys, km schemas, blocks, sorted-insert bound(),
 * serialize/parse, flush→snapshot) is the same shape and the same public API,
 * so the page can drive it exactly like the real thing.
 *
 * Not backed by a disk, so `flush()` snapshots the manifest into `.saved`
 * instead of writing a `<name>.json`, and `close()` is a no-op keeper.
 */
(function (root) {
  'use strict';

  class TSDBMem {
    constructor(name = 'tsdb') {
      this.name = name || 'tsdb';
      this.dirty = 0;
      /** physical store: id -> Uint8Array (the '<id>.fabf' bytes) */
      this.blobs = {};
      /** logical name -> { id } */
      this.files = {};
      this.fids = {};
      this.saved = null;               // last flushed manifest snapshot (JSON)
      this.meta = { name: this.name, version: 0, files: {} };
      this.dirty = 1;
    }

    /** @param {string} name */
    create(name) {
      const root = this.meta.files;
      if (!root[name]) {
        const id = this.id();
        root[name] = { id, size: 0, version: 1, waste: 0, keys: {} };
        this.files[name] = { id };
        this.fids[id] = name;
        this.dirty = 1;
      }
      return this;
    }

    /** @param {string} name @param {string} key @param {[Object,number]} km */
    createKey(name, key, km, forceNew = false) {
      this.create(name);
      const fm = this.meta.files[name];
      const box = fm.keys;

      if (forceNew && box[key]) {
        const old = box[key];
        fm.waste += old.blocks.reduce((p, b) => p + b.count * old.size, 0);
        delete box[key];
        this.dirty = 1;
      }
      if (!box[key]) {
        box[key] = { km, size: km[1] * 8, blocks: [] };
        this.dirty = 1;
      }
      return this;
    }

    /** Build a block descriptor for a bulk append (time bounds inferred). */
    createBlock(km, u8, info = null, begin = null, ends = null) {
      const map = km[0];
      const len = km[1];
      const count = u8.byteLength / (len * 8);

      if (count && (begin == null || ends == null)) {
        const fa = new Float64Array(u8.buffer, u8.byteOffset, u8.byteLength / 8);
        if (begin == null) begin = fa[map.time];
        if (ends == null) ends = fa[(count - 1) * len + map.time];
      }
      return {
        info: info || {},
        begin: begin ?? 0,
        ends: ends ?? 0,
        count,
        offset: 0,
      };
    }

    /** Rows -> Float64 bytes in schema order (sorted on `time` by default). */
    serialize(km, rows, u8 = null, default_value = 0, isSorted = false) {
      const map = km[0];
      const len = km[1];
      if (!isSorted) rows.sort((a, b) => a.time - b.time);

      const bytes = rows.length * len * 8;
      const out = u8 && u8.byteLength === bytes ? u8 : new Uint8Array(bytes);
      const fa = new Float64Array(out.buffer, out.byteOffset, bytes / 8);

      for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        const p = i * len;
        for (const k in map) fa[p + map[k]] = row[k] ?? default_value;
      }
      return out;
    }

    /** Float64 bytes -> rows (inverse of serialize). */
    parse(km, u8, assign = null, bytes = u8.byteLength) {
      const map = km[0];
      const len = km[1];
      const count = bytes / (len * 8);
      const fa = new Float64Array(u8.buffer, u8.byteOffset, bytes / 8);
      const out = new Array(count);

      for (let i = 0; i < count; i++) {
        const row = assign ? { ...assign } : {};
        const p = i * len;
        for (const k in map) row[k] = fa[p + map[k]];
        out[i] = row;
      }
      return out;
    }

    /** Append a bulk block of bytes at the file's current end offset. */
    append(name, key, block, u8) {
      const fm = this.meta.files[name];
      const km = fm.keys[key];
      const f = this.files[name];

      const bytes = u8.byteLength;
      const offset = fm.size;

      if (bytes) {
        const cur = this.blobs[f.id] || new Uint8Array(0);
        const grown = new Uint8Array(offset + bytes);
        grown.set(cur.subarray(0, offset), 0);
        grown.set(u8, offset);
        this.blobs[f.id] = grown;
      }

      block.offset = offset;
      block.count = bytes / km.size;

      km.blocks.splice(this.bound(name, key, block.begin, 1), 0, block);

      fm.size += bytes;
      fm.version++;
      this.meta.version++;
      this.dirty = 1;
      return block;
    }

    /** Binary search for sorted block insert / range lookup by begin-time. */
    bound(name, key, time, upper = 0, start = 0) {
      const a = this.meta.files[name].keys[key].blocks;
      let lo = start | 0;
      let hi = a.length;
      while (lo < hi) {
        const m = (lo + hi) >>> 1;
        if (upper ? a[m].begin <= time : a[m].begin < time) lo = m + 1;
        else hi = m;
      }
      return lo;
    }

    getBlocks(name, key, filter = null, start = 0, end = null) {
      let a = this.meta.files[name].keys[key].blocks;
      if (start | (end || 0)) a = a.slice(start | 0, end == null ? undefined : end);
      return typeof filter === 'function' ? a.filter(filter) : a;
    }

    /** Read just this block's bytes back out of the physical buffer. */
    bf(name, key, block) {
      const f = this.files[name];
      const bytes = block.count * this.meta.files[name].keys[key].size;
      const src = this.blobs[f.id] || new Uint8Array(0);
      // copy so callers get an independent, 8-byte-aligned buffer
      return src.slice(block.offset, block.offset + bytes);
    }

    flush() {
      if (!this.dirty) return this;
      this.saved = JSON.stringify(this.meta);
      this.dirty = 0;
      return this;
    }

    close() { return this; }

    /** Random small id, unique among live physical buffers. */
    id() {
      while (1) {
        const id = ((Math.random() * 0xffe) >>> 0) + 1;
        if (id && this.blobs[id] === undefined && !this.fids[id]) {
          this.blobs[id] = new Uint8Array(0);
          return id;
        }
      }
    }

    /** @param {string} str -> [ {field:index}, rowLength ] */
    keyMap(str) {
      const entries = str.split(',').map((e, i) => [e.trim(), i]);
      return [Object.fromEntries(entries), entries.length];
    }

    /** Total physical bytes across all `.fabf` buffers. */
    physicalBytes() {
      let n = 0;
      for (const id in this.blobs) n += this.blobs[id].byteLength;
      return n;
    }
  }

  root.TSDBMem = TSDBMem;
})(typeof window !== 'undefined' ? window : this);
