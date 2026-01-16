pragma circom 2.1.5;

// Top-level circuit wiring TF-IDF, cosine similarities, and TruthFinder iteration.

include "tfidf.circom";
include "cosine.circom";
include "truthfinder.circom";

template TruthFinderTopCore() {}

template TruthFinderTop() {
    signal input counts[4][90];
    signal input total_counts[4];
    signal input inv_total_counts[4];
    signal input idf[90];

    signal input similarities[4][4];
    signal input gamma;
    signal input rho;
    signal input half_const;
    signal input quarter_const;
    signal input inv48_const;
    signal input scale;
    signal input trust_init[4];

    signal output trust_final[4];

    component tfidf_nodes[4];
    for (var i = 0; i < 4; i++) {
        tfidf_nodes[i] = TFIDFVector(90);
        for (var j = 0; j < 90; j++) {
            tfidf_nodes[i].counts[j] <== counts[i][j];
            tfidf_nodes[i].idf[j] <== idf[j];
        }
        tfidf_nodes[i].total_count <== total_counts[i];
        tfidf_nodes[i].inv_total   <== inv_total_counts[i];
    }

    component cos_nodes[4][4];
    for (var a = 0; a < 4; a++) {
        for (var b = 0; b < 4; b++) {
            cos_nodes[a][b] = CosineSimilaritySq(90);
            for (var k = 0; k < 90; k++) {
                cos_nodes[a][b].vectorA[k] <== tfidf_nodes[a].tfidf[k];
                cos_nodes[a][b].vectorB[k] <== tfidf_nodes[b].tfidf[k];
            }
            cos_nodes[a][b].similarity <== similarities[a][b];
            cos_nodes[a][b].scale <== scale;
        }
    }

    component tf_iterations[5];
    for (var iter = 0; iter < 5; iter++) {
        tf_iterations[iter] = TruthFinderIteration(4);
        for (var r = 0; r < 4; r++) {
            for (var c = 0; c < 4; c++) {
                tf_iterations[iter].similarities[r][c] <== similarities[r][c];
            }
            if (iter == 0) {
                tf_iterations[iter].trust_in[r] <== trust_init[r];
            } else {
                tf_iterations[iter].trust_in[r] <== tf_iterations[iter - 1].trust_out[r];
            }
        }
        tf_iterations[iter].gamma <== gamma;
        tf_iterations[iter].rho <== rho;
        tf_iterations[iter].half_const <== half_const;
        tf_iterations[iter].quarter_const <== quarter_const;
        tf_iterations[iter].inv48_const <== inv48_const;
    }

    for (var t = 0; t < 4; t++) {
        trust_final[t] <== tf_iterations[4].trust_out[t];
    }
}

component main = TruthFinderTop();
