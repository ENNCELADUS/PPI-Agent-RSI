// Reuse pi's streaming/tool-call parser, changing only the HTTP route.
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { streamSimple as streamResponses } from "@earendil-works/pi-ai/api/openai-responses";
import { streamSimple } from "@earendil-works/pi-ai/api/openai-completions";

export default function gateway(pi: ExtensionAPI) {
    const baseUrl = process.env.GATEWAY_API_BASE;
    if (!baseUrl?.trim()) throw new Error("GATEWAY_API_BASE is required");
    const endpoint = new URL(baseUrl);
    if (!["http:", "https:"].includes(endpoint.protocol) || endpoint.username || endpoint.password) {
        throw new Error("GATEWAY_API_BASE must be an HTTP(S) URL without credentials");
    }
    const gatewayFetch: typeof fetch = (input, init) => {
        const url = new URL(input instanceof Request ? input.url : String(input));
        const root = baseUrl.replace(/\/$/, "");
        const expected = new URL(root + (url.pathname.endsWith("/responses") ? "/responses" : "/chat/completions"));
        if (url.href !== expected.href) throw new Error("Unexpected gateway request route");
        url.pathname = url.pathname.replace(/\/chat\/completions$/, "/start").replace(/\/responses$/, "/response");
        return fetch(url, init);
    };
    pi.registerProvider("gateway", {
        baseUrl,
        apiKey: "$GATEWAY_API_KEY",
        api: "openai-completions",
        streamSimple: (model, context, options) => model.id === "gpt-6.1-sol"
            ? streamResponses({ ...model, api: "openai-responses" }, context, { ...options, fetch: gatewayFetch })
            : streamSimple({ ...model, api: "openai-completions" }, context, { ...options, fetch: gatewayFetch }),
        models: ["gpt-6.1-sol", "deepseek-pro", "kimi"].map(id => ({
            id, name: id, input: ["text"], reasoning: true,
            // Conservative local budgets; gateway aliases do not advertise model limits or prices.
            contextWindow: 128000, maxTokens: 16384,
            cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
            compat: {
                supportsStore: false, supportsDeveloperRole: false,
                supportsReasoningEffort: id !== "kimi", supportsUsageInStreaming: true,
                maxTokensField: "max_completion_tokens",
                thinkingFormat: id === "deepseek-pro" ? "deepseek" : "openai",
                requiresReasoningContentOnAssistantMessages: id !== "gpt-6.1-sol",
            },
        })),
    });
}
