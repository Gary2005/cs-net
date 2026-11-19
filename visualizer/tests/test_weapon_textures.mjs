// Verify weapon GLB textures end to end in Node -- no browser needed.
//
// The browser self-test already proved three things (WebGL 2.0 context, official
// icon decode, all four sprite sequences decode). The one thing it never got to
// was the weapon GLBs, and that was purely the harness: `--virtual-time-budget`
// jumps the clock forward, so a 3 MB GLB plus nine 1024^2 textures gets cut off
// mid-flight. It was not a functional failure.
//
// Node cannot decode images on its own (no createImageBitmap), which is exactly
// why the earlier run reported "Couldn't load texture". This shims
// createImageBitmap with Pillow so the real decode path runs, then asserts that
// every mesh material ends up with a map that actually has pixels.
//
// Usage: node visualizer/tests/test_weapon_textures.mjs <server-url> <glb-dir>...
import { execFileSync } from 'node:child_process';
import { readFileSync, readdirSync, statSync, mkdirSync, writeFileSync } from 'node:fs';
import { join, resolve, dirname, basename, extname } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createServer } from 'node:http';

const base = process.argv[2] || 'http://127.0.0.1:5099';
const dirs = process.argv.slice(3);
if (!dirs.length) {
  console.error('usage: node test_weapon_textures.mjs <server-url> <glb-dir>...');
  process.exit(2);
}

const PY = process.env.PYTHON || 'python';
const cache = '.build/three-node';
mkdirSync(cache, { recursive: true });

// ── browser shims ──────────────────────────────────────────────────────────
globalThis.self = globalThis;
if (typeof globalThis.ProgressEvent === 'undefined') {
  globalThis.ProgressEvent = class ProgressEvent extends Event {
    constructor(type, init = {}) {
      super(type);
      this.lengthComputable = !!init.lengthComputable;
      this.loaded = init.loaded || 0;
      this.total = init.total || 0;
    }
  };
}

let decoded = 0, decodeFailed = 0;
const decodeErrors = [];
globalThis.createImageBitmap = async (blob, opts) => {
  const buf = Buffer.from(await blob.arrayBuffer());
  try {
    const out = execFileSync(PY, ['-c', [
      'import sys, json, io',
      'from PIL import Image',
      'raw = sys.stdin.buffer.read()',
      "im = Image.open(io.BytesIO(raw)).convert('RGBA')",
      "sys.stdout.buffer.write(json.dumps({'w': im.width, 'h': im.height}).encode())",
      "sys.stdout.buffer.write(b'\\n')",
      'sys.stdout.buffer.write(im.tobytes())',
    ].join('\n')], { input: buf, maxBuffer: 1 << 28 });
    const nl = out.indexOf(0x0a);
    const meta = JSON.parse(out.subarray(0, nl).toString());
    decoded++;
    return { width: meta.w, height: meta.h,
             data: new Uint8ClampedArray(out.subarray(nl + 1)), close() {} };
  } catch (e) {
    decodeFailed++;
    if (decodeErrors.length < 5) {
      decodeErrors.push(`blob.type=${JSON.stringify(blob.type)} bytes=${buf.length} `
                        + `opts=${JSON.stringify(opts)} -> ${e.message}`);
    }
    throw e;
  }
};

// ── load three + GLTFLoader over HTTP (same bytes the browser gets) ────────
async function grab(url, name, subs) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}`);
  let src = await r.text();
  src = src.replace(/from\s+['"]three['"]/g,
                    `from '${pathToFileURL(`${cache}/three.module.js`).href}'`);
  for (const [pat, file] of subs) {
    src = src.replace(pat, `from '${pathToFileURL(`${cache}/${file}`).href}'`);
  }
  writeFileSync(`${cache}/${name}`, src);
}
writeFileSync(`${cache}/three.module.js`,
              await (await fetch(`${base}/static/three/build/three.module.js`)).text());
await grab(`${base}/static/three/examples/jsm/utils/BufferGeometryUtils.js`,
           'BufferGeometryUtils.js', []);
await grab(`${base}/static/three/examples/jsm/loaders/GLTFLoader.js`,
           'GLTFLoader.js',
           [[/from\s+['"]\.\.\/utils\/BufferGeometryUtils\.js['"]/g, 'BufferGeometryUtils.js']]);

const THREE = await import(pathToFileURL(`${cache}/three.module.js`).href);
const { GLTFLoader } = await import(pathToFileURL(`${cache}/GLTFLoader.js`).href);
console.log(`three r${THREE.REVISION} + GLTFLoader loaded in Node`);

// ── serve each GLB directory so its relative texture uris resolve ──────────
// Content-Type matters: three's ImageBitmapLoader does res.blob() and hands the
// blob to createImageBitmap. A blob with an empty type makes the real
// createImageBitmap throw, so a server that omits the header produces a failure
// that looks like a texture bug but is really a harness bug.
const MIME = {
  '.glb': 'model/gltf-binary', '.webp': 'image/webp', '.png': 'image/png',
  '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.tga': 'image/x-tga',
};
let port = 54000;
const mounts = new Map();
for (const d of dirs) {
  const abs = resolve(d);
  const srv = createServer((req, res) => {
    try {
      const p = join(abs, decodeURIComponent(req.url.split('?')[0]));
      if (!statSync(p).isFile()) { res.writeHead(404).end(); return; }
      const ct = MIME[extname(p).toLowerCase()] || 'application/octet-stream';
      res.writeHead(200, { 'Content-Type': ct }).end(readFileSync(p));
    } catch { res.writeHead(404).end(); }
  });
  await new Promise((r) => srv.listen(port, '127.0.0.1', r));
  mounts.set(abs, `http://127.0.0.1:${port}/`);
  port++;
}
console.log(`serving ${mounts.size} model dir(s)\n`);

const loader = new GLTFLoader();
let bad = 0;
for (const d of dirs) {
  const abs = resolve(d);
  const glbs = readdirSync(abs).filter((f) => extname(f) === '.glb' && !f.includes('physics'));
  if (!glbs.length) { console.log(`!! no GLB in ${abs}`); bad++; continue; }
  const glb = glbs[0];
  const before = decoded;

  let gltf;
  try {
    gltf = await loader.loadAsync(mounts.get(abs) + encodeURIComponent(glb));
  } catch (e) {
    console.log(`${basename(abs)}: LOAD FAILED ${e.message}`);
    bad++;
    continue;
  }

  let meshes = 0, tris = 0, withMap = 0, withPixels = 0, noMap = [];
  gltf.scene.traverse((o) => {
    if (!o.isMesh) return;
    meshes++;
    const g = o.geometry;
    tris += (g.index ? g.index.count : g.attributes.position.count) / 3;
    const mats = (Array.isArray(o.material) ? o.material : [o.material]).filter(Boolean);
    const maps = mats.map((m) => m.map).filter(Boolean);
    if (!maps.length) { noMap.push(o.name || '(unnamed)'); return; }
    withMap++;
    if (maps.some((t) => t.image && t.image.width > 0)) withPixels++;
  });

  const box = new THREE.Box3().setFromObject(gltf.scene);
  const size = new THREE.Vector3(); box.getSize(size);
  const dims = `${size.x.toFixed(2)}x${size.y.toFixed(2)}x${size.z.toFixed(2)}`;
  const ok = meshes > 0 && withPixels === meshes && decoded > before;
  if (!ok) bad++;

  console.log(`${ok ? 'OK  ' : 'BAD '} ${basename(abs)}  (${glb})`);
  console.log(`       ${meshes} meshes, ${Math.round(tris).toLocaleString()} tris, `
              + `bbox ${dims} m`);
  console.log(`       materials with a map: ${withMap}/${meshes}; `
              + `map has decoded pixels: ${withPixels}/${meshes}`);
  console.log(`       images decoded during this model: ${decoded - before}`);
  if (noMap.length) console.log(`       NO MAP on: ${noMap.join(', ')}`);
}

console.log(`\n  images decoded: ${decoded}, decode failures: ${decodeFailed}`);
if (decodeErrors.length) {
  console.log('  first decode errors:');
  for (const e of decodeErrors) console.log(`    ${e}`);
}
console.log(`  ${dirs.length - bad}/${dirs.length} models fully textured`);
process.exit(bad ? 1 : 0);
