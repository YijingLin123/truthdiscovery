pragma circom 2.1.5;

template SigmoidApprox() {
    signal input x;
    signal input half_const;
    signal input quarter_const;
    signal input inv48_const;
    signal output y;

    signal x_sq;
    signal x_cu;
    x_sq <== x * x;
    x_cu <== x_sq * x;

    signal term_quad;
    signal term_cubic;
    term_quad <== quarter_const * x;
    term_cubic <== inv48_const * x_cu;
    y <== half_const + term_quad - term_cubic;
}

template TruthFinderIteration(N) {
    signal input similarities[N][N];
    signal input gamma;
    signal input rho;
    signal input trust_in[N];
    signal input half_const;
    signal input quarter_const;
    signal input inv48_const;

    signal output trust_out[N];
    signal output confidence[N];

    signal support[N];
    signal raw[N];
    component sig[N];

    // 2D: contrib[i][j] is the contribution from j to i
    signal contrib[N][N];
    // prefix sums per i
    signal acc[N][N + 1];

    for (var i = 0; i < N; i++) {
        acc[i][0] <== 0;

        for (var j = 0; j < N; j++) {
            if (i != j) {
                contrib[i][j] <== trust_in[j] * similarities[j][i];
            } else {
                contrib[i][j] <== 0;
            }
            acc[i][j + 1] <== acc[i][j] + contrib[i][j];
        }

        support[i] <== acc[i][N];

        raw[i] <== trust_in[i] + rho * support[i];

        sig[i] = SigmoidApprox();
        sig[i].x <== gamma * raw[i];
        sig[i].half_const <== half_const;
        sig[i].quarter_const <== quarter_const;
        sig[i].inv48_const <== inv48_const;

        confidence[i] <== sig[i].y;
        trust_out[i] <== confidence[i];
    }
}
