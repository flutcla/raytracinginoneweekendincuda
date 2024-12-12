#include "dynasoar.h"

struct AllArgs {
  vec3* fb;
  int max_x;
  int max_y;
  int ns;
  camera** cam;
  hitable** world;
  curandState* rand_state;
};

class Area;

using AllocatorT = SoaAllocator</*num_objs=*/ 4194304 * 8, Area>;

__global__ void initialize(AllArgs* args, int min_x, int max_x, int min_y, int max_y);

class Area : public AllocatorT::Base
{
public:
  declare_field_types(Area, AllArgs*, int, int, int, int, int, int)

public:
  Field<Area, 0> args;
  Field<Area, 1> x_begin;
  Field<Area, 2> x_end;
  Field<Area, 3> y_begin;
  Field<Area, 4> y_end;
  Field<Area, 5> state;
  Field<Area, 6> depth;

public:
  __device__ Area(
    AllArgs* args,
    int x_begin,
    int x_end,
    int y_begin,
    int y_end)
    : args(args),
    x_begin(x_begin),
    x_end(x_end),
    y_begin(y_begin),
    y_end(y_end) {
    state = 0;
    depth = 0;
  }

  __device__ void render();

  __device__ void calcStep();

  __device__ void calcStepSerial();

  __device__ void calcEager();
};
