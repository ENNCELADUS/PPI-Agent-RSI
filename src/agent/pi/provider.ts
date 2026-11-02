import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { deepseekProvider } from "@earendil-works/pi-ai/providers/deepseek";

export default function provider(pi: ExtensionAPI) {
	const id = process.env.PPI_MODEL_ID;
	const baseUrl = process.env.LLM_API_BASE;
	if (!id || !baseUrl || !process.env.LLM_API_KEY?.trim()) {
		throw new Error("Custom provider requires PPI_MODEL_ID, LLM_API_BASE and LLM_API_KEY");
	}
	if (!URL.canParse(baseUrl)) {
		throw new Error("LLM_API_BASE must be a valid HTTP(S) URL");
	}
	const endpoint = new URL(baseUrl);
	if (!["http:", "https:"].includes(endpoint.protocol) || endpoint.username || endpoint.password) {
		throw new Error("LLM_API_BASE must be an HTTP(S) URL without embedded credentials");
	}
	const model = deepseekProvider().getModels().find((m) => m.id === "deepseek-v4-pro")!;
	pi.registerProvider("custom", {
		baseUrl,
		apiKey: "$LLM_API_KEY",
		api: model.api,
		models: [{ ...model, id, baseUrl }],
	});
}
