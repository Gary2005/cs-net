/**
 * 读服务端那条 NDJSON 流（`?progress=1`）—— 回放页和玩家页共用。
 *
 * 服务端的长任务（一个回合的 AI 预测、一整场的 AI 统计、一个选手的跨场统计）
 * 都是这个形状：响应立刻开始，每 ~80 ms 推一行
 *
 *     {"type":"progress","pct":0.42,"stage":"…","elapsed":3.1}
 *     {"type":"result","payload":{…},"elapsed":23.9}
 *     {"type":"error","error":"…","kind":"models"}
 *
 * 用 `fetch` + 流式读取而不是 `EventSource`：这里是 GET、只要一条一次性的流，
 * 而 `EventSource` 自带重连语义 —— 一次二十多秒的 GPU 计算重连没有意义。
 * 拿不到流（服务端是旧版本 / 浏览器不给 `res.body`）就退回一次性 JSON：同一个
 * 路由不带 `progress` 参数就是纯 JSON，行为完全一样。
 *
 * 两个页面各写一遍这段解析，就会出现"一边认 `error.kind`、一边不认"的漂移，
 * 所以放在这里。文案（读流彻底断了 / 服务端回了错误）由调用方给，因为它们
 * 属于各自的页面词表。
 */

/**
 * @param {string} url                 带 `progress=1` 的地址
 * @param {(ev: object) => void} onEvent  每个事件回调（progress / error 都走这里）
 * @param {object} [opts]
 * @param {AbortSignal} [opts.signal]  取消
 * @param {string} [opts.brokenText]   流结束了却没收到 result 时的文案
 * @returns {Promise<object>}          `result` 事件里的 payload
 */
export async function streamJson(url, onEvent, opts = {}) {
  const { signal, brokenText = 'stream ended without a result' } = opts;
  const res = await fetch(url, signal ? { signal } : undefined);
  const type = res.headers.get('content-type') || '';
  if (!res.ok || !type.includes('ndjson') || !res.body) {
    const json = await res.json().catch(() => ({}));
    if (!res.ok || json.error) {
      const err = new Error(json.error || `HTTP ${res.status}`);
      err.kind = json.kind || '';
      throw err;
    }
    return json;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  let payload = null;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    for (let nl = buf.indexOf('\n'); nl >= 0; nl = buf.indexOf('\n')) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      let ev;
      try { ev = JSON.parse(line); } catch { continue; }
      if (ev.type === 'result') payload = ev.payload;
      else if (ev.type === 'error') {
        const err = new Error(ev.error || brokenText);
        err.kind = ev.kind || '';
        throw err;
      }
      if (onEvent) onEvent(ev);
    }
  }
  if (!payload) throw new Error(brokenText);
  return payload;
}
