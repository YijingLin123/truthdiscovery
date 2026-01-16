pragma circom 2.1.5;

template TFIDFVector(VOCAB_SIZE) {
    signal input counts[VOCAB_SIZE];
    signal input total_count;
    signal input idf[VOCAB_SIZE];

    signal input inv_total;
    signal output tfidf[VOCAB_SIZE];

    // total_count = sum(counts)
    signal sum_values[VOCAB_SIZE + 1];
    sum_values[0] <== 0;
    for (var i = 0; i < VOCAB_SIZE; i++) {
        sum_values[i + 1] <== sum_values[i] + counts[i];
    }
    sum_values[VOCAB_SIZE] === total_count;

    // inv_total is inverse of total_count (requires total_count != 0)
    total_count * inv_total === 1;

    // tfidf[j] = counts[j] * idf[j] * inv_total  (split)
    signal prod[VOCAB_SIZE];
    for (var j = 0; j < VOCAB_SIZE; j++) {
        prod[j] <== counts[j] * idf[j];
        tfidf[j] <== prod[j] * inv_total;
    }
}
