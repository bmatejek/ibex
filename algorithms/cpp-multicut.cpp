#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
// for integer linear program
#include "andres/ilp/gurobi.hxx"
#include "andres/graph/multicut/ilp.hxx"

// for kernighan-lin method
#include "andres/graph/graph.hxx"
#include "andres/graph/multicut/kernighan-lin.hxx"

// for greedy-additive method
//#include "andres/graph/graph.hxx"
#include "andres/graph/multicut/greedy-additive.hxx"


unsigned char *CppMulticut(unsigned long nvertices, unsigned long nedges, unsigned long *vertex_ones, unsigned long *vertex_twos, double *edge_weights, double threshold, int algorithm)
{
    andres::graph::Graph<> graph;
    std::vector<double> weights(nedges);

    graph.insertVertices(nvertices);
    for (unsigned long ie = 0; ie < nedges; ++ie) {
        graph.insertEdge(vertex_ones[ie], vertex_twos[ie]);

        // a low beta value encourages not merging (high threshold)
        double beta = threshold;
        weights[ie] = log(edge_weights[ie] / (1.0 - edge_weights[ie])) + log((1 - beta) / beta);
    }

    std::vector<char> edge_labels(nedges, 1);
    edge_labels[0] = 0;

    // run the desired multicut algorithm
    if (algorithm == 0) andres::graph::multicut::ilp<andres::ilp::Gurobi>(graph, weights, edge_labels, edge_labels);
    else if (algorithm == 1) andres::graph::multicut::kernighanLin(graph, weights, edge_labels, edge_labels);
    else if (algorithm == 2) andres::graph::multicut::greedyAdditiveEdgeContraction(graph, weights, edge_labels);

    unsigned char *collapsed_edges = new unsigned char[nedges];
    for (unsigned long ie = 0; ie < nedges; ++ie) {
        collapsed_edges[ie] = edge_labels[ie];
    }

    return collapsed_edges;
}