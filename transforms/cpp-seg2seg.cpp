#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <queue>
#include <set>
#include <map>



#define IB_Z 0
#define IB_Y 1
#define IB_X 2



void CppMapLabels(long *segmentation, long *mapping, unsigned long nentries)
{
  for (unsigned long iv = 0; iv < nentries; ++iv) {
    segmentation[iv] = mapping[segmentation[iv]];
  }
}



long *CppRemoveSmallConnectedComponents(long *segmentation, int threshold, unsigned long nentries)
{
  if (threshold == 0) return segmentation;
  
  long *thresholded_segmentation = new long[nentries];

  // find the maximum label
  long max_segment_label = 0;
  for (unsigned long iv = 0; iv < nentries; ++iv) {
    if (segmentation[iv] > max_segment_label) max_segment_label = segmentation[iv];
  }
  max_segment_label++;
  
  // create a counter array for the number of voxels
  int *nvoxels_per_segment = new int[max_segment_label];
  for (long iv = 0; iv < max_segment_label; ++iv) {
    nvoxels_per_segment[iv] = 0;
  }
  
  // count the number of voxels per segment
  for (unsigned long iv = 0; iv < nentries; ++iv) {
    nvoxels_per_segment[segmentation[iv]]++;
  }
  
  // create the array for the updated segmentation
  for (unsigned long iv = 0; iv < nentries; ++iv) {
    if (nvoxels_per_segment[segmentation[iv]] < threshold) thresholded_segmentation[iv] = 0;
    else thresholded_segmentation[iv] = segmentation[iv];
  }
  
  // free memory
  delete[] nvoxels_per_segment;
  
  return thresholded_segmentation;
}



static unsigned long row_size;
static unsigned long sheet_size;



void IndexToIndicies(long iv, long &ix, long &iy, long &iz)
{
  iz = iv / sheet_size;
  iy = (iv - iz * sheet_size) / row_size;
  ix = iv % row_size;
}



long IndiciesToIndex(long ix, long iy, long iz)
{
  return iz * sheet_size + iy * row_size + ix;
}



long *CppForceConnectivity(long *segmentation, long zres, long yres, long xres)
{
  // create the new components array
  long nentries = zres * yres * xres;
  long *components = new long[nentries];
  for (long iv = 0; iv < nentries; ++iv)
    components[iv] = 0;

  // set global variables
  row_size = xres;
  sheet_size = yres * xres;


  // create the queue of labels
  std::queue<unsigned long> pixels = std::queue<unsigned long>();

  long current_index = 0;
  long current_label = 1;

  while (current_index < nentries) {
    // set this component and add to the queue
    components[current_index] = current_label;
    pixels.push(current_index);

    // iterate over all pixels in the queue
    while (pixels.size()) {
      // remove this pixel from the queue
      unsigned long pixel = pixels.front();
      pixels.pop();
 
      // add the six neighbors to the queue
      long iz, iy, ix;
      IndexToIndicies(pixel, ix, iy, iz);

      for (long iw = -1; iw <= 1; ++iw) {
        if (iz + iw < 0 or iz + iw > zres - 1) continue;
        for (long iv = -1; iv <= 1; ++iv) {
          if (iy + iv < 0 or iy + iv > yres - 1) continue;
          for (long iu = -1; iu <= 1; ++iu) {
            if (ix + iu < 0 or ix + iu > xres - 1) continue;
            long neighbor = IndiciesToIndex(ix + iu, iy + iv, iz + iw);
            if (segmentation[pixel] == segmentation[neighbor] && !components[neighbor]) {
              components[neighbor] = current_label;
              pixels.push(neighbor);
            }
          }
        }
      }
    }
    current_label++;

    // if the current index is already labeled, continue
    while (current_index < nentries && components[current_index]) current_index++;
  }

  // create a list of mappings
  long max_segment = 0;
  long max_component = 0;
  for (long iv = 0; iv < nentries; ++iv) {
    if (segmentation[iv] > max_segment) max_segment = segmentation[iv];
    if (components[iv] > max_component) max_component = components[iv];
  }
  max_segment++;
  max_component++;

  std::set<long> *seg2comp = new std::set<long>[max_segment];
  for (long iv = 0; iv < max_segment; ++iv)
    seg2comp[iv] = std::set<long>();

  // see if any segments have multiple components
  for (long iv = 0; iv < nentries; ++iv) {
    seg2comp[segmentation[iv]].insert(components[iv]);
  }

  long overflow = max_segment;
  long *comp2seg = new long[max_component];
  for (long iv = 1; iv < max_segment; ++iv) {
    if (seg2comp[iv].size() == 1) {
      // get the component for this segment
      long component = *(seg2comp[iv].begin());
      comp2seg[component] = iv;
    }
    else {
      // iterate over the set
      for (std::set<long>::iterator it = seg2comp[iv].begin(); it != seg2comp[iv].end(); ++it) {
        long component = *it;

        // one of the components keeps the label
        if (it == seg2comp[iv].begin()) comp2seg[component] = iv;
        // set the component to start at max_segment and increment
        else {
          comp2seg[component] = overflow;
          ++overflow;
        }
      }
    }
  }

  // update the segmentation
  for (long iv = 0; iv < nentries; ++iv) {
    if (!segmentation[iv]) components[iv] = 0;
    else components[iv] = comp2seg[components[iv]];
  }

  // free memory
  delete[] seg2comp;
  delete[] comp2seg;

  return components;
}



void CppTopologicalDownsample(const char *prefix, long *segmentation, long input_resolution[3], long output_resolution[3], long input_zres, long input_yres, long input_xres, bool benchmark)
{
  // get the number of entries 
  long nentries = input_zres * input_yres * input_xres;
  
  // get downsample ratios
  float zdown = ((float) output_resolution[IB_Z]) / input_resolution[IB_Z];
  float ydown = ((float) output_resolution[IB_Y]) / input_resolution[IB_Y];
  float xdown = ((float) output_resolution[IB_X]) / input_resolution[IB_X];

  // get the output resolution size
  long output_zres = (long) ceil(input_zres / zdown);
  long output_yres = (long) ceil(input_yres / ydown);
  long output_xres = (long) ceil(input_xres / xdown);
  long output_sheet_size = output_yres * output_xres;
  long output_row_size = output_xres;

  long max_segment = 0;
  for (long iv = 0; iv < nentries; ++iv)
    if (segmentation[iv] > max_segment) max_segment = segmentation[iv];
  max_segment++;

  std::set<long> *downsample_sets = new std::set<long>[max_segment];
  for (long iv = 0; iv < max_segment; ++iv)
    downsample_sets[iv] = std::set<long>();

  long index = 0;
  for (long iz = 0; iz < input_zres; ++iz) {
    for (long iy = 0; iy < input_yres; ++iy) {
      for (long ix = 0; ix < input_xres; ++ix, ++index) {
        long segment = segmentation[index];
        if (!segment) continue;

        long iw = (long) (iz / zdown);
        long iv = (long) (iy / ydown);
        long iu = (long) (ix / xdown);

        long downsample_index = iw * output_sheet_size + iv * output_row_size + iu;
        downsample_sets[segment].insert(downsample_index);
      }
    }
  }

  char output_filename[4096];
  if (benchmark) sprintf(output_filename, "topological/benchmarks/%s-topological-downsample-%ldx%ldx%ld.bytes", prefix, output_resolution[IB_X], output_resolution[IB_Y], output_resolution[IB_Z]);
  else sprintf(output_filename, "topological/%s-topological-downsample-%ldx%ldx%ld.bytes", prefix, output_resolution[IB_X], output_resolution[IB_Y], output_resolution[IB_Z]);

  // open the output file
  FILE *fp = fopen(output_filename, "wb");
  if (!fp) { fprintf(stderr, "Failed to write to %s\n", output_filename); exit(-1); }

  // write the number of segments
  fwrite(&output_zres, sizeof(long), 1, fp);
  fwrite(&output_yres, sizeof(long), 1, fp);
  fwrite(&output_xres, sizeof(long), 1, fp);
  fwrite(&max_segment, sizeof(long), 1, fp);

  // output values for downsampling
  for (long iv = 0; iv < max_segment; ++iv) {
    // write the size for this set
    long nelements = downsample_sets[iv].size();
    fwrite(&nelements, sizeof(long), 1, fp);
    for (std::set<long>::iterator it = downsample_sets[iv].begin(); it != downsample_sets[iv].end(); ++it) {
      long element = *it;
      fwrite(&element, sizeof(long), 1, fp);
    }
  }

  // close the file
  fclose(fp);

  // free memory
  delete[] downsample_sets;
}



void CppTopologicalUpsample(const char *prefix, long *segmentation, long input_resolution[3], long output_resolution[3], long input_zres, long input_yres, long input_xres, bool benchmark)
{
  // get downsample ratios
  float zdown = ((float) output_resolution[IB_Z]) / input_resolution[IB_Z];
  float ydown = ((float) output_resolution[IB_Y]) / input_resolution[IB_Y];
  float xdown = ((float) output_resolution[IB_X]) / input_resolution[IB_X];

  // get the output resolution size
  long output_zres = (long) ceil(input_zres / zdown);
  long output_yres = (long) ceil(input_yres / ydown);
  long output_xres = (long) ceil(input_xres / xdown);
  long output_nentries = output_zres * output_yres * output_xres;
  long output_sheet_size = output_yres * output_xres;
  long output_row_size = output_xres;

  std::map<long, long> *meanx = new std::map<long, long>[output_nentries];
  std::map<long, long> *meany = new std::map<long, long>[output_nentries];
  std::map<long, long> *meanz = new std::map<long, long>[output_nentries];
  std::map<long, long> *ndownsampled_voxels = new std::map<long, long>[output_nentries];
  for (long iv = 0; iv < output_nentries; ++iv) {
    meanx[iv] = std::map<long, long>();
    meany[iv] = std::map<long, long>();
    meanz[iv] = std::map<long, long>();
    ndownsampled_voxels[iv] = std::map<long, long>();
  }

  long index = 0;
  for (long iz = 0; iz < input_zres; ++iz) {
    for (long iy = 0; iy < input_yres; ++iy) {
      for (long ix = 0; ix < input_xres; ++ix, ++index) {
        long segment = segmentation[index];

        long iw = (long) (iz / zdown);
        long iv = (long) (iy / ydown);
        long iu = (long) (ix / xdown);

        long downsample_index = iw * output_sheet_size + iv * output_row_size + iu;

        // update the mean values
        meanz[downsample_index][segment] += iz;
        meany[downsample_index][segment] += iy;
        meanx[downsample_index][segment] += ix;
        ndownsampled_voxels[downsample_index][segment]++;
      }
    }
  }

  char input_filename[4096];
  if (benchmark) sprintf(input_filename, "topological/benchmarks/%s-topological-downsample-%ldx%ldx%ld.bytes", prefix, output_resolution[IB_X], output_resolution[IB_Y], output_resolution[IB_Z]);
  else sprintf(input_filename, "topological/%s-topological-downsample-%ldx%ldx%ld.bytes", prefix, output_resolution[IB_X], output_resolution[IB_Y], output_resolution[IB_Z]);

  // open the input file
  FILE *rfp = fopen(input_filename, "rb");
  if (!rfp) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }

  char output_filename[4096];
  if (benchmark) sprintf(output_filename, "topological/benchmarks/%s-topological-upsample-%ldx%ldx%ld.bytes", prefix, output_resolution[IB_X], output_resolution[IB_Y], output_resolution[IB_Z]);
  else sprintf(output_filename, "topological/%s-topological-upsample-%ldx%ldx%ld.bytes", prefix, output_resolution[IB_X], output_resolution[IB_Y], output_resolution[IB_Z]);

  // open the output file
  FILE *wfp = fopen(output_filename, "wb");
  if (!wfp) { fprintf(stderr, "Failed to write %s\n", output_filename); exit(-1); }

  // write the number of segments
  long read_zres, read_yres, read_xres, max_segment;
  if (fread(&read_zres, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }
  assert (read_zres == output_zres);
  if (fread(&read_yres, sizeof(long), 1, rfp) != 1)  { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }
  assert (read_yres == output_yres);
  if (fread(&read_xres, sizeof(long), 1, rfp) != 1)  { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }
  assert (read_xres == output_xres);
  if (fread(&max_segment, sizeof(long), 1, rfp) != 1)  { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }

  // write the output file size of the upsample version
  fwrite(&input_zres, sizeof(long), 1, wfp);
  fwrite(&input_yres, sizeof(long), 1, wfp);
  fwrite(&input_xres, sizeof(long), 1, wfp);
  fwrite(&max_segment, sizeof(long), 1, wfp);

  // go through all the segments and write the upsampled location
  for (long label = 0; label < max_segment; ++label) {
    long nelements;
    if (fread(&nelements, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }
    fwrite(&nelements, sizeof(long), 1, wfp);
    for (long ie = 0; ie < nelements; ++ie) {
      long element;
      if (fread(&element, sizeof(long), 1, rfp) != 1) { fprintf(stderr, "Failed to read %s\n", input_filename); exit(-1); }
      long upsamplez = meanz[element][label] / ndownsampled_voxels[element][label];
      long upsampley = meany[element][label] / ndownsampled_voxels[element][label];
      long upsamplex = meanx[element][label] / ndownsampled_voxels[element][label];

      long upsample_index = upsamplez * input_xres * input_yres + upsampley * input_xres + upsamplex;
      fwrite(&upsample_index, sizeof(long), 1, wfp);
    }
  }
  
  // close the files
  fclose(rfp);
  fclose(wfp);
}
