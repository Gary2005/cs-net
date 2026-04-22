importScripts("wasm/wasm_exec.js");

onmessage = (event) => {
  console.log("received event: ", event);
  var demoData = event.data.data;
  var filename = event.data.filename;
  var debugParser = Boolean(event.data.debugParser);
  console.log("file: ", filename);
  if (demoData instanceof Uint8Array) {
    try {
      const parseError = globalThis.wasmParseDemo(filename, demoData, async function (data) {
        if (data instanceof Uint8Array) {
          postMessage(data);
          // const msg = proto.Message.deserializeBinary(data).toObject()
          // messageBus.emit(msg)
        } else {
          console.log(
            "[message] text data received from server, this is weird. We're using protobufs ?!?!?",
            data
          );
          postMessage(JSON.parse(data));
        }
      }, { debugParser });

      if (parseError) {
        postMessage({
          type: "parseError",
          message: String(parseError),
          filename,
        });
      }
    } catch (error) {
      postMessage({
        type: "parseError",
        message: error instanceof Error ? error.message : String(error),
        filename,
      });
    }
  }
};

async function loadWasm() {
  try {
    const go = new globalThis.Go();
    const response = await fetch("wasm/csdemoparser.wasm");
    if (!response.ok) {
      throw new Error(`Failed to fetch wasm/csdemoparser.wasm (${response.status} ${response.statusText})`);
    }

    const result = await WebAssembly.instantiateStreaming(response, go.importObject);
    go.run(result.instance);
    console.log("should be loaded now");
    postMessage("ready");
  } catch (error) {
    postMessage({
      type: "wasmInitError",
      message: error instanceof Error ? error.message : String(error),
    });
  }
}
loadWasm();
