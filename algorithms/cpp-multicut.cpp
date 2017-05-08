#include <stdio.h>
#include <stdlib.h>
#include <vector>
/*#include "andres/ilp/gurobi.hxx"
#include "andres/graph/multicut/ilp.hxx"*/

#include <andres/graph/graph.hxx>
#include "andres/graph/multicut/kernighan-lin.hxx"


unsigned char *CppMulticut(unsigned long nvertices, unsigned long nedges, unsigned long *vertex_ones, unsigned long *vertex_twos, double *edge_weights, double threshold)
{
    andres::graph::Graph<> graph;
    std::vector<double> weights(nedges);

    graph.insertVertices(nvertices);
    for (unsigned long ie = 0; ie < nedges; ++ie) {
        graph.insertEdge(vertex_ones[ie], vertex_twos[ie]);
        weights[ie] = (threshold - edge_weights[ie]);
    }

    std::vector<char> edge_labels(nedges, 1);

    //andres::graph::multicut::ilp<andres::ilp::Gurobi>(graph, weights, edge_labels, edge_labels);
    andres::graph::multicut::kernighanLin(graph, weights, edge_labels, edge_labels);

    unsigned char *collapsed_edges = new unsigned char[nedges];
    for (unsigned long ie = 0; ie < nedges; ++ie) {
        collapsed_edges[ie] = edge_labels[ie];
    }

    return collapsed_edges;
}