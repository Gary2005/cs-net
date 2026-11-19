// Measure the staged weapon GLBs: bounding box, so the renderer knows what
// scale and offset each model needs.
//
// Each weapon model is authored in its own local space, and VRF does not
// normalise them -- the AK, the karambit and the C4 all have different origins
// and extents. Guessing a uniform scale would leave some weapons floating and
// others buried in the player's chest, so this measures them instead.
//
// Usage: node visualizer/tests/measure_weapons.mjs <server-url> <glb> [glb...]
import { writeFileSync } from 'node:fs';

const base = process.argv[2] || 'http://127.0.0.1:5099';
const glbs = process.argv.slice(3);
if (!glbs.length) {
  console.error('usage: node measure_weapons.mjs <server-url> <glb> [glb...]');
  process.exit(2);
}

globalThis.self = globalThis;

// three's FileLoader dispatches a ProgressEvent, which Node does not define.
// loadAsync() goes through FileLoader (the other test uses loader.parse() and
// so never hit this).
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

// three needs a canvas-ish environment for texture decode; the geometry we
// measure does not depend on textures, so let texture loads fail quietly.
const { execFileSync } = await import('node:child_process');
const PY = process.env.PYTHON || 'python';
globalThis.createImageBitmap = async (blob) => {
  const buf = Buffer.from(await blob.arrayBuffer());
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
  return { width: meta.w, height: meta.h,
           data: new Uint8ClampedArray(out.subarray(nl + 1)), close() {} };
};

// Pull the loader bytes over HTTP from the running server, so the code under
// test is byte-identical to what the browser gets.
const { mkdirSync, writeFileSync: wf } = await import('node:fs');
const { pathToFileURL } = await import('node:url');
const cache = '.build/three-node';
mkdirSync(cache, { recursive: true });

async function grab(url, name) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}`);
  let src = await r.text();
  src = src.replace(/from\s+['"]three['"]/g, `from '${pathToFileURL(`${cache}/three.module.js`).href}'`);
  src = src.replace(/from\s+['"]\.\.\/utils\/BufferGeometryUtils\.js['"]/g,
                    `from '${pathToFileURL(`${cache}/BufferGeometryUtils.js`).href}'`);
  wf(`${cache}/${name}`, src);
  return src;
}
const threeSrc = await (await fetch(`${base}/static/three/build/three.module.js`)).text();
wf(`${cache}/three.module.js`, threeSrc);
await grab(`${base}/static/three/examples/jsm/utils/BufferGeometryUtils.js`, 'BufferGeometryUtils.js');
await grab(`${base}/static/three/examples/jsm/loaders/GLTFLoader.js`, 'GLTFLoader.js');
const THREE = await import(pathToFileURL(`${cache}/three.module.js`).href);
const { GLTFLoader } = await import(pathToFileURL(`${cache}/GLTFLoader.js`).href);

// Serve each GLB's own directory so relative texture uris resolve.
const { createServer } = await import('node:http');
const { readFileSync, statSync } = await import('node:fs');
const { join, dirname, resolve } = await import('node:path');
const { extname } = await import('node:path');

const dirs = new Map();
for (const g of glbs) {
  const abs = resolve(g);
  dirs.set(dirname(abs), null);
}
let port = 53000;
for (const [dir] of dirs) {
  const srv = createServer((req, res) => {
    try {
      const p = join(dir, decodeURIComponent(req.url.split('?')[0]));
      if (!statSync(p).isFile()) { res.writeHead(404).end(); return; }
      res.writeHead(200).end(readFileSync(p));
    } catch { res.writeHead(404).end(); }
  });
  await new Promise((r) => srv.listen(port, '127.0.0.1', r));
  dirs.set(dir, `http://127.0.0.1:${port}/`);
  port++;
}

const loader = new GLTFLoader();
const out = [];
for (const g of glbs) {
  const abs = resolve(g);
  const baseUrl = dirs.get(dirname(abs));
  const gltf = await loader.loadAsync(baseUrl + encodeURIComponent(abs.split(/[\\/]/).pop()));
  const box = new THREE.Box3().setFromObject(gltf.scene);
  const size = new THREE.Vector3(); box.getSize(size);
  const centre = new THREE.Vector3(); box.getCenter(centre);
  let tris = 0, meshes = 0;
  gltf.scene.traverse((o) => {
    if (o.isMesh && o.geometry?.index) { tris += o.geometry.index.count / 3; meshes++; }
    else if (o.isMesh && o.geometry?.attributes?.position) {
      tris += o.geometry.attributes.position.count / 3; meshes++;
    }
  });
  // glTF is metres-ish here but VRF writes Source units straight through, so
  // report raw numbers and let the caller decide the scale.
  const rec = {
    name: abs.split(/[\\/]/).pop().replace(/\.glb$/, ''),
    size: [+size.x.toFixed(2), +size.y.toFixed(2), +size.z.toFixed(2)],
    centre: [+centre.x.toFixed(2), +centre.y.toFixed(2), +centre.z.toFixed(2)],
    min: [+box.min.x.toFixed(2), +box.min.y.toFixed(2), +box.min.z.toFixed(2)],
    max: [+box.max.x.toFixed(2), +box.max.y.toFixed(2), +box.max.z.toFixed(2)],
    longest_axis: size.x >= size.y && size.x >= size.z ? 'x'
                : size.y >= size.z ? 'y' : 'z',
    meshes, triangles: Math.round(tris),
  };
  out.push(rec);
  console.log(`${rec.name}`);
  console.log(`  size   ${rec.size.join(' x ')}   longest=${rec.longest_axis}`);
  console.log(`  centre ${rec.centre.join(', ')}`);
  console.log(`  min    ${rec.min.join(', ')}`);
  console.log(`  max    ${rec.max.join(', ')}`);
  console.log(`  ${rec.meshes} meshes, ${rec.triangles.toLocaleString()} tris`);
}
// .build is git-ignored and always writable; create it rather than failing on ENOENT.
const outPath = process.env.WEAPON_MEASURE_OUT
              || '.build/weapon_measure.json';
mkdirSync(dirname(outPath), { recursive: true });
writeFileSync(outPath, JSON.stringify(out, null, 2));
console.log(`\nwrote ${outPath}`);
process.exit(0);
