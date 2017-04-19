#include <stdio.h>
#include <string.h>
#include <vector>
//#include "andres/ilp/gurobi.hxx"
#include "andres/graph/multicut/ilp.hxx"
#include "andres/graph/graph.hxx"
#include "andres/graph/multicut/greedy-additive.hxx"
#include <ctime>



// global variables
static int print_verbose = 0;
static char *prefix = NULL;


static int ParseArgs(int argc, char **argv)
{
    argc--;
    argv++;
    while (argc > 0) {
        if ((*argv)[0] == '-') {
            if (!strcmp(*argv, "-v")) print_verbose = 1;
            else { fprintf(stderr, "Invalid program argument: %s\n", *argv); return 0; }
        }
        else {
            if (!prefix) { prefix = *argv; }
            else { fprintf(stderr, "Invalid program argument: %s\n", *argv); return 0; }
        }

        argv++;
        argc--;
    }

    // make sure a prefix is given
    if (!prefix) { fprintf(stderr, "Need to supply prefix\n"); return 0; }

    // return OK status
    return 1;
}



int main(int argc, char **argv)
{
    // parse the arguments
    if (!ParseArgs(argc, argv)) exit(-1);

    // open the multicut file
    // TODO FIX HARD CODED 400nm
    char multicut_filename[4096];
    sprintf(multicut_filename, "multicut/%s_skeleton_400nm.graph", prefix);

    // open file
    FILE *fp = fopen(multicut_filename, "rb");
    if (!fp) { fprintf(stderr, "Failed to read %s\n", multicut_filename); return 0; }

    // read in the data
    unsigned long nvertices;
    unsigned long nedges;

    if (fread(&nvertices, sizeof(unsigned long), 1, fp) != 1)  { fprintf(stderr, "Failed to read %s\n", multicut_filename); return 0; }
    if (fread(&nedges, sizeof(unsigned long), 1, fp) != 1)  { fprintf(stderr, "Failed to read %s\n", multicut_filename); return 0; }

    andres::graph::Graph<> graph;
    std::vector<double> weights(nedges);

    int nmerges_predicted = 0;
    int nsplits_predicted = 0;

    // add the vertices to the graph
    graph.insertVertices(nvertices);
    for (int ie = 0; ie < nedges; ++ie) {
        unsigned long label_one;
        unsigned long label_two;
        double edge_weight;

        if (fread(&label_one, sizeof(unsigned long), 1, fp) != 1) { fprintf(stderr, "Failed to read %s\n", multicut_filename); return 0; }
        if (fread(&label_two, sizeof(unsigned long), 1, fp) != 1) { fprintf(stderr, "Failed to read %s\n", multicut_filename); return 0; }
        if (fread(&edge_weight, sizeof(double), 1, fp) != 1) { fprintf(stderr, "Failed to read %s\n", multicut_filename); return 0; }

        graph.insertEdge(label_one, label_two);
        weights[ie] = (0.5 - edge_weight);
        
        if (edge_weight > 0.5) nmerges_predicted++;
        else nsplits_predicted++;
    }

    printf("%d %d\n", nmerges_predicted, nsplits_predicted);

    // close the file
    fclose(fp);

    std::clock_t start_time = std::clock();

    std::vector<char> edge_labels(nedges, 1);
    andres::graph::multicut::greedyAdditiveEdgeContraction(graph, weights, edge_labels);
    printf("%lf\n", (std::clock() - start_time) / (double)CLOCKS_PER_SEC);

    int nmerges = 0;
    int nsplits = 0;

    // output the edge weight results
    for (int iv = 0; iv < nedges; ++iv) {
        if (edge_labels[iv]) nmerges++;
        else nsplits++;
    }

    printf("%d %d\n", nmerges, nsplits);

    // output a series of merges that need to occur
    char output_filename[4096];
    sprintf(output_filename, "multicut/%s_multicut_output.graph", prefix);

    FILE *output_fp = fopen(output_filename, "wb");
    if (!output_fp) { fprintf(stderr, "Failed to write to %s\n", output_filename); return 0; }

    fwrite(&nedges, sizeof(unsigned long), 1, output_fp);
    fwrite(&(edge_labels[0]), sizeof(char), nedges, output_fp);

    // close file
    fclose(output_fp);

    // return success
    return 1;
}
