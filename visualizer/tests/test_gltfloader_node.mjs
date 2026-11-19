// Parse a GLB with the real three.js GLTFLoader inside Node.
//
// Why this exists: the extracted weapon GLBs carry JOINTS_0/WEIGHTS_0 on their
// primitives but declare no `skins` array. Whether three.js tolerates that is
// not something to reason about from the spec -- it depends on which loader
// branch builds the object. So run the actual loader and inspect what it made.
//
// The loader sources are fetched over HTTP from the running visualizer (so the
// bytes tested are exactly the bytes the browser gets), then their bare/relative
// imports are rewritten to file paths, because Node has no import map.
//
// Usage:
//   node visualizer/tests/test_gltfloader_node.mjs <base-url> <file.glb> [more.glb ...]

import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, createReadStream, statSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname, extname, basename } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createServer } from 'node:http';

const base = process.argv[2];
const glbs = process.argv.slice(3);
if (!base || glbs.length === 0) {
  console.error('usage: node test_gltfloader_node.mjs <base-url> <file.glb> [...]');
  process.exit(2);
}

const work = mkdtempSync(join(tmpdir(), 'gltf-'));
const threeDir = join(work, 'three');
const jsmDir = join(work, 'jsm');
mkdirSync(join(jsmDir, 'utils'), { recursive: true });
mkdirSync(join(jsmDir, 'loaders'), { recursive: true });
mkdirSync(threeDir, { recursive: true });

// three.module.js touches browser globals at module scope. Node has no `self`,
// so the import throws "self is not defined" before a single line of ours runs.
globalThis.self = globalThis;

// three decodes textures through createImageBitmap, which Node lacks -- so a
// perfectly good fetch still reports "Couldn't load texture". Shimming it via
// the Python interpreter (Pillow is available) lets the test distinguish a real
// 404 from a decode step that only exists in a browser.
let decodeFail = 0;
if (typeof globalThis.createImageBitmap !== 'function') {
  const { execFileSync } = await import('node:child_process');
  const PY = process.env.PYTHON || 'python';
  globalThis.createImageBitmap = async (blob) => {
    const buf = Buffer.from(await blob.arrayBuffer());
    // Hand the bytes to Pillow and get back raw RGBA + dimensions.
    const out = execFileSync(PY, ['-c', `
import sys, json, io
from PIL import Image
raw = sys.stdin.buffer.read()
im = Image.open(io.BytesIO(raw)).convert('RGBA')
sys.stdout.buffer.write(json.dumps({'w': im.width, 'h': im.height}).encode())
sys.stdout.buffer.write(b'\\n')
sys.stdout.buffer.write(im.tobytes())
`], { input: buf, maxBuffer: 1 << 28 });
    const nl = out.indexOf(0x0a);
    const meta = JSON.parse(out.subarray(0, nl).toString());
    const data = new Uint8ClampedArray(out.subarray(nl + 1));
    return { width: meta.w, height: meta.h, data, close() {} };
  };
}

const get = async (path) => {
  const r = await fetch(new URL(path, base));
  if (!r.ok) throw new Error(`${path} -> HTTP ${r.status}`);
  return r.text();
};

// three.module.js is fetched so the loader imports the same revision.
const threePath = join(threeDir, 'three.module.js');
writeFileSync(threePath, await get('/static/three/build/three.module.js'));
const threeUrl = pathToFileURL(threePath).href;

const rewrite = (src, selfDir) =>
  src
    .replace(/from\s+['"]three['"]/g, `from '${threeUrl}'`)
    .replace(/from\s+['"]\.\.\/utils\/BufferGeometryUtils\.js['"]/g,
             `from '${pathToFileURL(join(jsmDir, 'utils', 'BufferGeometryUtils.js')).href}'`)
    .replace(/from\s+['"]three\/addons\//g, `from '${pathToFileURL(join(jsmDir, '')).href}/`);

const bgu = await get('/static/three/examples/jsm/utils/BufferGeometryUtils.js');
// BufferGeometryUtils imports 'three' too -- rewrite it or Node cannot resolve
// the bare specifier (there is no import map outside the browser).
writeFileSync(join(jsmDir, 'utils', 'BufferGeometryUtils.js'), rewrite(bgu));
const loaderSrc = await get('/static/three/examples/jsm/loaders/GLTFLoader.js');
const loaderPath = join(jsmDir, 'loaders', 'GLTFLoader.js');
writeFileSync(loaderPath, rewrite(loaderSrc, dirname(loaderPath)));

const { GLTFLoader } = await import(pathToFileURL(loaderPath).href);
const THREE = await import(threeUrl);
console.log(`three r${THREE.REVISION} + GLTFLoader loaded in Node\n`);

// three's FileLoader fetches through the global fetch, and Node's fetch refuses
// file:// URLs -- so textures would never resolve from disk. Serving the GLB
// directories over HTTP exercises the same code path the browser uses.
const MIME = {
  '.glb': 'model/gltf-binary', '.gltf': 'model/gltf+json', '.png': 'image/png',
  '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.webp': 'image/webp',
  '.bin': 'application/octet-stream',
};
const mounts = new Map();  // url prefix -> filesystem dir
for (const g of glbs) {
  const d = dirname(g);
  mounts.set('/m' + mounts.size + '/', d);
}
const server = createServer((req, res) => {
  const url = decodeURIComponent(req.url.split('?')[0]);
  for (const [prefix, dir] of mounts) {
    if (!url.startsWith(prefix)) continue;
    const rel = url.slice(prefix.length);
    // no path traversal
    if (rel.includes('..')) { res.writeHead(403).end(); return; }
    const file = join(dir, rel);
    try {
      if (!statSync(file).isFile()) throw new Error('not a file');
    } catch {
      res.writeHead(404).end();
      return;
    }
    res.writeHead(200, { 'Content-Type': MIME[extname(file).toLowerCase()] || 'application/octet-stream' });
    createReadStream(file).pipe(res);
    return;
  }
  res.writeHead(404).end();
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const origin = `http://127.0.0.1:${server.address().port}`;
console.log(`serving ${mounts.size} model dir(s) at ${origin}\n`);

for (const [i, glb] of glbs.entries()) {
  const name = basename(glb);
  console.log(`=== ${name} ===`);
  let buf;
  try {
    buf = readFileSync(glb);
  } catch (e) {
    console.log(`  cannot read: ${e.message}\n`);
    continue;
  }
  const loader = new GLTFLoader();
  // Image decoding goes through createImageBitmap, which Node lacks. Texture
  // *fetch* is still exercised, so a 404 shows up as a load error while a
  // successful fetch that merely fails to decode shows up differently.
  let texErr = 0;
  loader.manager.onError = () => { texErr += 1; };

  // Resource path = the GLB's own directory, served over HTTP.
  const dir = `${origin}/m${i}/`;

  await new Promise((resolve) => {
    loader.parse(
      buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
      dir,
      (gltf) => {
        let meshes = 0, skinned = 0, withBones = 0, plain = 0, tris = 0;
        gltf.scene.traverse((o) => {
          if (!o.isMesh) return;
          meshes += 1;
          if (o.isSkinnedMesh) {
            skinned += 1;
            if (o.skeleton && o.skeleton.bones && o.skeleton.bones.length) withBones += 1;
          } else {
            plain += 1;
          }
          const g = o.geometry;
          if (g && g.index) tris += g.index.count / 3;
          else if (g && g.attributes.position) tris += g.attributes.position.count / 3;
        });
        const anims = gltf.animations ? gltf.animations.length : 0;
        console.log(`  parse OK`);
        console.log(`  meshes=${meshes}  SkinnedMesh=${skinned} (withBones=${withBones})  plainMesh=${plain}`);
        console.log(`  triangles=${Math.round(tris).toLocaleString()}  animations=${anims}`);
        // The decisive question: did the skin attributes produce a skeleton?
        if (skinned > 0 && withBones === 0) {
          console.log(`  !! SkinnedMesh built with ZERO bones -- skinning has no skeleton`);
        } else if (plain > 0 && skinned === 0) {
          console.log(`  -> skin attributes were dropped; meshes are rigid (bind pose)`);
        }
        if (texErr) console.log(`  texture errors: ${texErr}`);
        console.log('');
        resolve();
      },
      (err) => {
        console.log(`  !! PARSE FAILED: ${err && err.message ? err.message : err}\n`);
        resolve();
      },
    );
  });
}

server.close();
