// The HTTP server behind `spectacular render` and `spectacular serve`. It binds loopback
// only. The built viewer is public; the spec's directory is reachable only under
// /s/<random token>/, so another local user who finds the port cannot read the data.
import { createServer } from "node:http";
import { randomBytes } from "node:crypto";
import { existsSync, watch } from "node:fs";
import { readFile } from "node:fs/promises";
import { basename, dirname, extname, relative, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { localData } from "./check.mjs";

export const VIEWER = resolve(dirname(fileURLToPath(import.meta.url)), "../packages/viz/dist");

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".css": "text/css",
  ".wasm": "application/wasm",
  ".json": "application/json",
  ".jsonld": "application/ld+json",
  ".csv": "text/csv; charset=utf-8",
  ".tsv": "text/tab-separated-values; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
};

/** `file` inside `root`, or undefined when the path would leave it. */
function inside(root, path) {
  const file = resolve(root, "." + sep + decodeURIComponent(path));
  return file === root || file.startsWith(root + sep) ? file : undefined;
}

/**
 * Serves `specFile` and the files beside it. `specText` replaces the file's contents (a
 * spec read from stdin is served as if it were a file named `specFile`). Calls
 * `onSelection(body)` for each selection a page posts, and with `watch`, tells pages to
 * reload and calls `onReload(file)` when the spec or its local data file changes.
 */
export async function startServer({ specFile, specText, port = 0, watch: watching = false, onSelection, onReload }) {
  if (!existsSync(resolve(VIEWER, "index.html")))
    throw Object.assign(new Error(`the viewer is not built (no ${VIEWER}); run \`npm run build\` in the spectacular repo`), { exit: 2 });
  const root = dirname(resolve(specFile));
  const specName = basename(specFile);
  const token = randomBytes(16).toString("hex");
  const prefix = `/s/${token}/`;
  const clients = new Set();

  const send = (res, status, body, type = "text/plain; charset=utf-8") => {
    res.writeHead(status, { "content-type": type, "cache-control": "no-store" });
    res.end(body);
  };
  const sendFile = async (res, file) => {
    try {
      send(res, 200, await readFile(file), TYPES[extname(file).toLowerCase()] ?? "application/octet-stream");
    } catch {
      send(res, 404, "not found");
    }
  };

  const server = createServer(async (req, res) => {
    const path = new URL(req.url, "http://localhost").pathname;
    if (!path.startsWith(prefix)) {
      if (path.startsWith("/s/")) return send(res, 404, "not found");
      const file = inside(VIEWER, path === "/" ? "index.html" : path.slice(1));
      return file ? sendFile(res, file) : send(res, 404, "not found");
    }
    const rest = path.slice(prefix.length);
    if (rest === "__spectacular/events") {
      res.writeHead(200, { "content-type": "text/event-stream", "cache-control": "no-store", connection: "keep-alive" });
      res.write(": connected\n\n");
      clients.add(res);
      req.on("close", () => clients.delete(res));
      return;
    }
    if (rest === "__spectacular/selection" && req.method === "POST") {
      const chunks = [];
      for await (const c of req) chunks.push(c);
      try {
        onSelection?.(JSON.parse(Buffer.concat(chunks).toString("utf8")));
        return send(res, 204, "");
      } catch {
        return send(res, 400, "expected a JSON selection");
      }
    }
    if (specText !== undefined && rest === encodeURIComponent(specName)) return send(res, 200, specText, TYPES[".json"]);
    const file = inside(root, rest);
    return file ? sendFile(res, file) : send(res, 404, "not found");
  });

  let watcher;
  if (watching) {
    // Reload on the spec, or the data file the spec names now (re-read after each change).
    const targets = async () => {
      const spec = JSON.parse(await readFile(resolve(specFile), "utf8").catch(() => "null"));
      const data = localData(spec, root);
      return new Set([specName, ...(data ? [relative(root, data)] : [])]);
    };
    let watched = await targets();
    let timer;
    watcher = watch(root, { recursive: true }, (_, name) => {
      if (!name || !watched.has(name)) return;
      clearTimeout(timer);
      timer = setTimeout(async () => {
        watched = await targets();
        for (const c of clients) c.write("event: reload\ndata: {}\n\n");
        onReload?.(name);
      }, 200);
    });
  }

  await new Promise((ok, fail) => server.once("error", fail).listen(port, "127.0.0.1", ok));
  const origin = `http://127.0.0.1:${server.address().port}`;
  return {
    origin,
    specPath: prefix + encodeURIComponent(specName),
    close() {
      watcher?.close();
      for (const c of clients) c.end();
      server.closeAllConnections();
      return new Promise((ok) => server.close(ok));
    },
  };
}
