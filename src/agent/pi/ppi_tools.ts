// Registers the round's tools from the spec file the Python tool server wrote (PPI_TOOLS_SPEC) and
// forwards each call to it (PPI_TOOL_URL). All tool logic lives in src/agent/tools.py.
import { readFileSync } from "node:fs";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";

interface Spec {
	name: string;
	description: string;
	terminate: boolean;
	parameters: Record<string, unknown>;
}

export default function (pi: ExtensionAPI) {
	const url = process.env.PPI_TOOL_URL;
	const specs: Spec[] = JSON.parse(readFileSync(process.env.PPI_TOOLS_SPEC!, "utf8"));
	for (const spec of specs) {
		pi.registerTool({
			name: spec.name,
			label: spec.name,
			description: spec.description,
			parameters: Type.Unsafe(spec.parameters),
			async execute(_toolCallId, params, signal) {
				const response = await fetch(url + spec.name, { method: "POST", body: JSON.stringify(params), signal });
				const body = (await response.json()) as { ok: boolean; text: string };
				if (!body.ok) throw new Error(body.text);
				return { content: [{ type: "text", text: body.text }], details: undefined, terminate: spec.terminate };
			},
		});
	}
}
