import * as dotenv from 'dotenv';
import { OpenAI } from 'openai';
import Anthropic from '@anthropic-ai/sdk';
dotenv.config(); // Load environment variables from .env file

interface Config {
    model?: string;
    temperature?: number;
    max_tokens?: number;
    subclass_name: string;
}

interface LLMConstructor {
    new(config?: Config): LLM;
    subclasses: { [key: string]: LLMConstructor };
}

interface OpenAIChatAPIOptions {
    model?: string;
    temperature?: number;
    max_tokens?: number;
}

interface AnthropicChatAPIOptions {
    model?: string;
    temperature?: number;
    max_tokens?: number;
}

class LLM {
    static subclasses: { [key: string]: LLMConstructor } = {};

    constructor() {
        const constructor = this.constructor as LLMConstructor;
        constructor.subclasses[this.constructor.name] = this.constructor as typeof LLM;
    }

    toDict() {
        return {
            'subclass_name': this.constructor.name,
        };
    }

    static fromDict(config: Config) {
        const subclass_name = config.subclass_name;
        const Subclass = this.subclasses[subclass_name];

        if (Subclass) {
            return new Subclass(config);
        } else {
            throw new Error(`Unknown subclass: ${subclass_name}`);
        }
    }

    makeLLMCall(chatMessages: any, systemMessage?: string) {
        throw new Error('makeLLMCall must be implemented by subclasses');
    }
}

class OpenAIChatAPI extends LLM {
    private client: OpenAI;
    private model: string;
    private temperature: number;
    private max_tokens: number;

    constructor({ model = "gpt-3.5-turbo", temperature = 0.2, max_tokens = 1000 }: OpenAIChatAPIOptions = {}) {
        super();
        const openai = new OpenAI(); // defaults to process.env["OPEN_API_KEY"]
        this.client = openai;
        this.model = model;
        this.temperature = temperature;
        this.max_tokens = max_tokens;
    }

    async makeLLMCall(chatMessages: OpenAI.ChatCompletionMessage[]): Promise<string> {

        const params: OpenAI.Chat.ChatCompletionCreateParams = {
            model: this.model,
            messages: chatMessages,
            max_tokens: this.max_tokens,
            temperature: this.temperature,
        };
        const chatCompletion: OpenAI.Chat.ChatCompletion = await this.client.chat.completions.create(params);

        const llmOutput = chatCompletion.choices && chatCompletion.choices[0]?.message?.content?.trim();
        return llmOutput || '';
    }

    toDict(): { subclass_name: string; model: string; temperature: number; max_tokens: number } {
        const baseDict = super.toDict();
        return {
            ...baseDict,
            model: this.model,
            temperature: this.temperature,
            max_tokens: this.max_tokens
        };
    }
}

class AnthropicChatAPI extends LLM {
    private client: Anthropic;
    private model: string;
    private temperature: number;
    private max_tokens: number;

    constructor({ model = "claude-3-haiku-20240307", temperature = 0.2, max_tokens = 1000 }: AnthropicChatAPIOptions = {}) {
        super();
        const anthropic = new Anthropic(); // defaults to process.env["ANTHROPIC_API_KEY"]
        this.client = anthropic;
        this.model = model;
        this.temperature = temperature;
        this.max_tokens = max_tokens;
    }

    async makeLLMCall(chatMessages: Anthropic.MessageParam[], systemMessage: string): Promise<string> {
        //let systemMessage = "";
        //let numSystemMessages = 0;
        //let normalChatMessages: Anthropic.MessageParam[] = [];

        /*chatMessages.forEach(message => {
            if (message.role === "system") {
                if (numSystemMessages > 0) {
                    throw new Error("ERROR: more than one system message detected");
                }
                systemMessage = message.content;
                numSystemMessages += 1;
            } else {
                normalChatMessages.push(message);
            }
        });

        if (normalChatMessages.length === 0) {
            normalChatMessages.push({ role: "user", content: "" });
        }*/

        const params: Anthropic.MessageCreateParams = {
            messages: chatMessages,
            model: this.model,
            max_tokens: this.max_tokens,
            temperature: this.temperature,
            system: systemMessage,
        };
        const message: Anthropic.Message = await this.client.messages.create(params);

        return message.content[0].text;
    }

    toDict(): { subclass_name: string; model: string; temperature: number; max_tokens: number } {
        const baseDict = super.toDict();
        return {
            ...baseDict,
            model: this.model,
            temperature: this.temperature,
            max_tokens: this.max_tokens
        };
    }
}

module.exports = { LLM, OpenAIChatAPI, AnthropicChatAPI };