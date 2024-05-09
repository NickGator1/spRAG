import { CohereClient } from "cohere-ai";
import * as dotenv from 'dotenv';
dotenv.config(); // Load environment variables from .env file

interface Config {
    model?: string;
    subclass_name: string;
}

interface RerankerConstructor {
    new(config?: Config): Reranker;
    subclasses: { [key: string]: RerankerConstructor };
}

interface RerankerSearchResult {
    metadata: {
        chunk_header: string;
        chunk_text: string;
    };
}

interface SearchResults {
    metadata: {
        chunk_header: string;
        chunk_text: string;
    };
}

class Reranker {
    static subclasses: { [key: string]: RerankerConstructor } = {};

    constructor() {
        const constructor = this.constructor as typeof Reranker;
        constructor.subclasses[constructor.name] = constructor;
    }

    static toDict(config: Config): Reranker {
        const subclass_name = config.subclass_name;
        const Subclass = this.subclasses[subclass_name];
        if (Subclass) {
            return new Subclass(config); // Create an instance of the subclass instead of the abstract class
        } else {
            throw new Error(`Unknown subclass: ${subclass_name}`);
        }
    }

    async rerankSearchResults(query: string, searchResults: SearchResults[]): Promise<RerankerSearchResult[]> {
        // Add implementation here
        throw new Error('rerankSearchResults must be implemented by subclasses');
    }
}

class CohereReranker extends Reranker {
    private model: string;
    private client: CohereClient;

    constructor(model: string = "rerank-english-v3.0") {
        super();
        this.model = model;
        const cohere = new CohereClient({
            token: process.env.CO_API_KEY,
        });
        this.client = cohere;
    }

    async rerankSearchResults(query: string, searchResults: SearchResults[]): Promise<RerankerSearchResult[]> {
        const documents = searchResults.map(result => `[${result.metadata.chunk_header}]\n${result.metadata.chunk_text}`);
        const rerankedResults = await this.client.rerank({
            model: this.model,
            query: query,
            documents: documents
        });
        const rerankedIndices = rerankedResults.results.map(result => result.index);
        const rerankedSearchResults = rerankedIndices.map(index => searchResults[index]);
        return rerankedSearchResults;
    }

    toDict(): { subclass_name: string; model: string } {
        const baseDict = Reranker.toDict({model: this.model, subclass_name: this.constructor.name});
        return {
            ...baseDict,
            model: this.model,
            subclass_name: this.constructor.name,
        };
    }
}

class NoReranker extends Reranker {
    async rerankSearchResults(query: string, searchResults: SearchResults[]): Promise<RerankerSearchResult[]> {
        return Promise.resolve(searchResults);
    }
}

module.exports = { Reranker, CohereReranker, NoReranker };