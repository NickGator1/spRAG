import * as math from 'mathjs';

interface SearchResult {
    metadata: {
        doc_id: string;
        chunk_index: number;
    };
}

interface RerankerSearchResult {
    start: number;
    end: number;
}

function getBestSegments(allRelevanceValues: number[][], documentSplits: number[], maxLength: number, overallMaxLength: number, minimumValue: number): RerankerSearchResult[] {
    let bestSegments: RerankerSearchResult[] = [];
    let totalLength = 0;
    let rvIndex = 0;
    let badRvIndices: number[] = [];

    while (totalLength < overallMaxLength) {
        if (rvIndex >= allRelevanceValues.length) {
            rvIndex = 0;
        }
        if (badRvIndices.length >= allRelevanceValues.length) {
            break;
        }
        if (badRvIndices.includes(rvIndex)) {
            rvIndex++;
            continue;
        }

        const relevanceValues = allRelevanceValues[rvIndex];
        let bestSegment: RerankerSearchResult | null = null;
        let bestValue = -Infinity;

        for (let start = 0; start < relevanceValues.length; start++) {
            if (relevanceValues[start] < 0) continue;
            for (let end = start + 1; end <= Math.min(start + maxLength, relevanceValues.length); end++) {
                if (relevanceValues[end - 1] < 0) continue;
                if (bestSegments.some(seg => start < seg.end && end > seg.start)) continue;
                if (documentSplits.some(split => start < split && end > split)) continue;
                if (totalLength + (end - start) > overallMaxLength) continue;

                const segmentValue = math.sum(relevanceValues.slice(start, end));
                if (segmentValue > bestValue) {
                    bestValue = segmentValue;
                    bestSegment = { start, end };
                }
            }
        }

        if (!bestSegment || bestValue < minimumValue) {
            badRvIndices.push(rvIndex);
            rvIndex++;
            continue;
        }

        bestSegments.push(bestSegment);
        totalLength += bestSegment.end - bestSegment.start;
        rvIndex++;
    }

    return bestSegments;
}

function convertRankToValue(rank: number, irrelevantChunkPenalty: number, decayRate: number = 20): number {
    return Math.exp(-rank / decayRate) - irrelevantChunkPenalty;
}

function getMetaDocument(allRankedResults: SearchResult[][], topKForDocumentSelection: number = 7): [number[], Record<string, number>, string[]] {
    let topDocumentIds: string[] = [];
    allRankedResults.forEach(rankedResults => {
        topDocumentIds.push(...rankedResults.slice(0, topKForDocumentSelection).map(result => result.metadata.doc_id));
    });
    let uniqueDocumentIds = Array.from(new Set(topDocumentIds));

    let documentSplits: number[] = [];
    let documentStartPoints: Record<string, number> = {};
    uniqueDocumentIds.forEach(documentId => {
        let maxChunkIndex = -1;
        allRankedResults.forEach(rankedResults => {
            rankedResults.forEach(result => {
                if (result.metadata.doc_id === documentId) {
                    maxChunkIndex = Math.max(maxChunkIndex, result.metadata.chunk_index);
                }
            });
        });
        documentStartPoints[documentId] = documentSplits.length > 0 ? documentSplits[documentSplits.length - 1] : 0;
        documentSplits.push(maxChunkIndex + (documentSplits.length > 0 ? documentSplits[documentSplits.length - 1] + 1 : 1));
    });

    return [documentSplits, documentStartPoints, uniqueDocumentIds];
}

function getRelevanceValues(allRankedResults: SearchResult[][], metaDocumentLength: number, documentStartPoints: Record<string, number>, uniqueDocumentIds: string[], irrelevantChunkPenalty: number, decayRate: number = 20): number[][] {
    let allRelevanceValues: number[][] = [];
    allRankedResults.forEach(rankedResults => {
        let relevanceRanks = Array(metaDocumentLength).fill(1000);
        rankedResults.forEach((result, rank) => {
            const documentId = result.metadata.doc_id;
            if (!uniqueDocumentIds.includes(documentId)) return;
            const chunkIndex = result.metadata.chunk_index;
            const metaDocumentIndex = documentStartPoints[documentId] + chunkIndex;
            relevanceRanks[metaDocumentIndex] = rank;
        });
        let relevanceValues = relevanceRanks.map(rank => convertRankToValue(rank, irrelevantChunkPenalty, decayRate));
        allRelevanceValues.push(relevanceValues);
    });
    return allRelevanceValues;
}

module.exports = { getBestSegments, convertRankToValue, getMetaDocument, getRelevanceValues };