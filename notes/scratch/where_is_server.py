"""Report where the visualizer server actually is reachable.

`Get-NetTCPConnection` / `Get-NetIPAddress` return nothing under this sandbox, so
this uses the stdlib instead. Also probes each candidate address, because a
server bound to 127.0.0.1 is NOT reachable from a phone on the same LAN -- that
difference is the whole point of the question.
"""

import socket
import sys
import urllib.request


def local_ips():
    """Every non-loopback IPv4 this machine has, without needing a route out."""
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    # The UDP-connect trick: no packet is sent, but the OS picks the interface
    # it *would* use, which is the address other machines can reach.
    for probe in ("10.255.255.255", "8.8.8.8"):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect((probe, 1))
            ips.add(s.getsockname()[0])
        except OSError:
            pass
        finally:
            s.close()
    return sorted(i for i in ips if not i.startswith("127."))


def probe(host, port, timeout=2.0):
    url = f"http://{host}:{port}/static/assets/manifest.json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, len(r.read())
    except Exception as e:  # noqa: BLE001
        return None, type(e).__name__


port = int(sys.argv[1]) if len(sys.argv) > 1 else 5099

print(f"本机非回环 IPv4: {local_ips() or '（一个都没拿到）'}")
print()
print(f"探测 {port} 端口（GET /static/assets/manifest.json）：")
for host in ["127.0.0.1", "localhost", "0.0.0.0"] + local_ips():
    status, extra = probe(host, port)
    mark = "OK  " if status == 200 else "--  "
    detail = f"{status}, {extra:,} B" if status == 200 else str(extra)
    print(f"  {mark} {host:<16}:{port}  {detail}")

print()
print("如果 127.0.0.1 通、局域网 IP 不通，说明服务器绑的是 --host 127.0.0.1，")
print("只有本机能访问；要让手机/别的机器也能开，得用 --host 0.0.0.0 重启。")
