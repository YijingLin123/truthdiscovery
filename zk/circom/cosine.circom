pragma circom 2.1.5;

template DotProduct(V) {
    signal input vecA[V];
    signal input vecB[V];
    signal output result;

    signal acc[V + 1];
    acc[0] <== 0;
    for (var i = 0; i < V; i++) {
        acc[i + 1] <== acc[i] + vecA[i] * vecB[i];
    }
    result <== acc[V];
}

template VectorNormSq(V) {
    signal input vec[V];
    signal output norm2;

    signal acc[V + 1];
    acc[0] <== 0;
    for (var i = 0; i < V; i++) {
        acc[i + 1] <== acc[i] + vec[i] * vec[i];
    }
    norm2 <== acc[V];
}

// similarity must satisfy: (similarity + scale) * (na2*nb2) = 2 * scale * (dot^2)
template CosineSimilaritySq(V) {
    signal input vectorA[V];
    signal input vectorB[V];
    signal input similarity;
    signal input scale; // SCALE e.g., 100000000

    component dot = DotProduct(V);
    component na2c = VectorNormSq(V);
    component nb2c = VectorNormSq(V);

    for (var i = 0; i < V; i++) {
        dot.vecA[i] <== vectorA[i];
        dot.vecB[i] <== vectorB[i];
        na2c.vec[i] <== vectorA[i];
        nb2c.vec[i] <== vectorB[i];
    }

    // dot2 = dot^2
    signal dot2;
    dot2 <== dot.result * dot.result;

    // denom2 = na2 * nb2
    signal denom2;
    denom2 <== na2c.norm2 * nb2c.norm2;

    // left = (similarity + scale) * denom2
    signal left;
    left <== (similarity + scale) * denom2;

    // right = 2 * scale * dot2
    signal right;
    right <== 2 * scale * dot2;

    left === right;
}
