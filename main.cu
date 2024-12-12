#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"

#include "main.h"

#define num_objects 40

bool USE_DYNASOAR = false;
bool DO_SERAL = false;

__device__ AllocatorT* device_allocator;        // device side
AllocatorHandle<AllocatorT>* allocator_handle;  // host side

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
  if (result) {
    std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
      file << ":" << line << " '" << func << "' \n";
    // Make sure we call CUDA Device Reset before exiting
    cudaDeviceReset();
    exit(99);
  }
}

// Matching the C++ code would recurse enough into color() calls that
// it was blowing up the stack, so we have to turn this into a
// limited-depth loop instead.  Later code in the book limits to a max
// depth of 50, so we adapt this a few chapters early on the GPU.
__device__ vec3 color(const ray& r, hitable** world, curandState* local_rand_state) {
  ray cur_ray = r;
  vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
  for (int i = 0; i < 50; i++) {
    hit_record rec;
    if ((*world)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
      ray scattered;
      vec3 attenuation;
      if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
        cur_attenuation *= attenuation;
        cur_ray = scattered;
      }
      else {
        return vec3(0.0, 0.0, 0.0);
      }
    }
    else {
      vec3 unit_direction = unit_vector(cur_ray.direction());
      float t = 0.5f * (unit_direction.y() + 1.0f);
      vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
      return cur_attenuation * c;
    }
  }
  return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState* rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curand_init(1984, 0, 0, rand_state);
  }
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j * max_x + i;
  // Original: Each thread gets same seed, a different sequence number, no offset
  // curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
  // BUGFIX, see Issue#2: Each thread gets different seed, same sequence for
  // performance improvement of about 2x!
  curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* fb, int max_x, int max_y, int ns, camera** cam, hitable** world, curandState* rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if ((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j * max_x + i;
  curandState local_rand_state = rand_state[pixel_index];
  vec3 col(0, 0, 0);
  for (int s = 0; s < ns; s++) {
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
    ray r = (*cam)->get_ray(u, v, &local_rand_state);
    col += color(r, world, &local_rand_state);
  }
  rand_state[pixel_index] = local_rand_state;
  col /= float(ns);
  col[0] = sqrt(col[0]);
  col[1] = sqrt(col[1]);
  col[2] = sqrt(col[2]);
  fb[pixel_index] = col;
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_camera, int nx, int ny, curandState* rand_state) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    curandState local_rand_state = *rand_state;
    d_list[0] = new sphere(vec3(0, -1000.0, -1), 1000,
      new lambertian(vec3(0.5, 0.5, 0.5)));
    int i = 1;
    int n = num_objects / 2;
    for (int a = -n; a < n; a++) {
      for (int b = -n; b < n; b++) {
        float choose_mat = RND;
        vec3 center(a + RND, 0.2, b + RND);
        if (choose_mat < 0.4f) {
          d_list[i++] = new sphere(center, 0.2,
            new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
        }
        else if (choose_mat < 0.8f) {
          d_list[i++] = new sphere(center, 0.2,
            new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.0));
        }
        else {
          d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
        }
      }
    }
    d_list[i++] = new sphere(vec3(0, 1, 0), 1.0, new dielectric(1.5));
    d_list[i++] = new sphere(vec3(-4, 1, 0), 1.0, new lambertian(vec3(0.4, 0.2, 0.1)));
    d_list[i++] = new sphere(vec3(4, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));
    *rand_state = local_rand_state;
    *d_world = new hitable_list(d_list, num_objects * num_objects + 1 + 3);

    vec3 lookfrom(13, 2, 3);
    vec3 lookat(0, 2, 0);
    float dist_to_focus = 10.0; (lookfrom - lookat).length();
    float aperture = 0.1;
    *d_camera = new camera(lookfrom,
      lookat,
      vec3(0, 1, 0),
      30.0,
      float(nx) / float(ny),
      aperture,
      dist_to_focus);
  }
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_camera) {
  for (int i = 0; i < num_objects * num_objects + 1 + 3; i++) {
    delete ((sphere*)d_list[i])->mat_ptr;
    delete d_list[i];
  }
  delete* d_world;
  delete* d_camera;
}

int main() {
  int nx = 1200;
  int ny = 800;
  int ns = 10;
  int tx = 8;
  int ty = 8;

  std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  int num_pixels = nx * ny;
  size_t fb_size = num_pixels * sizeof(vec3);

  // allocate FB
  vec3* fb;
  checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));

  // allocate random state
  curandState* d_rand_state;
  checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));
  curandState* d_rand_state2;
  checkCudaErrors(cudaMalloc((void**)&d_rand_state2, 1 * sizeof(curandState)));

  // we need that 2nd random state to be initialized for the world creation
  rand_init << <1, 1 >> > (d_rand_state2);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  // make our world of hitables & the camera
  hitable** d_list;
  int num_hitables = num_objects * num_objects + 1 + 3;
  checkCudaErrors(cudaMalloc((void**)&d_list, num_hitables * sizeof(hitable*)));
  hitable** d_world;
  checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hitable*)));
  camera** d_camera;
  checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));
  create_world << <1, 1 >> > (d_list, d_world, d_camera, nx, ny, d_rand_state2);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  clock_t start, stop;
  start = clock();
  // Render our buffer
  dim3 blocks(nx / tx + 1, ny / ty + 1);
  dim3 threads(tx, ty);
  render_init << <blocks, threads >> > (nx, ny, d_rand_state);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());

  if (USE_DYNASOAR)
  {

    AllArgs h_args = { fb, nx, ny, ns, d_camera, d_world, d_rand_state };
    AllArgs* d_args;
    cudaMalloc(&d_args, sizeof(AllArgs));
    cudaMemcpy(d_args, &h_args, sizeof(AllArgs), cudaMemcpyHostToDevice);

    // DynaSOAr 初期化
    allocator_handle = new AllocatorHandle<AllocatorT>();
    AllocatorT* dev_ptr = allocator_handle->device_pointer();
    cudaMemcpyToSymbol(device_allocator, &dev_ptr, sizeof(AllocatorT*), 0,
      cudaMemcpyHostToDevice);
    cudaDeviceSetLimit(cudaLimitStackSize, 4096);

    initialize << <1, 1 >> > (d_args, 0, nx - 1, 0, ny - 1);

    int core_count = 3840;
    int cm = 8;
    // レンダリング
    for (int i = 1; i < 100000; i++)
    {
      std::cerr << "=+=+=+=+= Iteration " << i << " =+=+=+=+=" << std::endl;
      int area_rough_count = allocator_handle->count_allocated_object_roughly<Area>(true);
      std::cerr << "area_rough_count: " << area_rough_count << std::endl;
      if (area_rough_count == 0)
      {
        break;
      }
      else if (DO_SERAL)
      {
        std::cerr << "Parallel do Area calcStepSerial" << std::endl;
        allocator_handle->parallel_do_bounded_by_count<Area, &Area::calcStepSerial>(core_count * cm);
      }
      else if (area_rough_count > core_count * cm)
      {
        std::cerr << "Parallel do Area calcStep" << std::endl;
        allocator_handle->parallel_do_bounded_by_count<Area, &Area::calcStep>(core_count * cm);
      }
      else
      {
        std::cerr << "Parallel do Area calcEager" << std::endl;
        allocator_handle->parallel_do<Area, &Area::calcEager>();
      }
    }
  }
  else
  {
    render << <blocks, threads >> > (fb, nx, ny, ns, d_camera, d_world, d_rand_state);
  }

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  stop = clock();
  double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
  std::cerr << "took " << timer_seconds << " seconds.\n";

  // Output FB as Image
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny - 1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {
      size_t pixel_index = j * nx + i;
      int ir = int(255.99 * fb[pixel_index].r());
      int ig = int(255.99 * fb[pixel_index].g());
      int ib = int(255.99 * fb[pixel_index].b());
      std::cout << ir << " " << ig << " " << ib << "\n";
    }
  }

  // clean up
  checkCudaErrors(cudaDeviceSynchronize());
  free_world << <1, 1 >> > (d_list, d_world, d_camera);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_camera));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_rand_state));
  checkCudaErrors(cudaFree(d_rand_state2));
  checkCudaErrors(cudaFree(fb));

  cudaDeviceReset();
}

__device__ void Area::render()
{
  AllArgs* args = (AllArgs*)this->args;
  int max_x = args->max_x;
  int max_y = args->max_y;
  int ns = args->ns;
  camera** cam = args->cam;
  hitable** world = args->world;
  curandState* rand_state = args->rand_state;
  vec3* fb = args->fb;
  int i = x_begin;
  int j = y_begin;
  if ((i >= max_x) || (j >= max_y))
  {
    return;
  }
  int pixel_index = j * max_x + i;
  curandState local_rand_state = rand_state[pixel_index];
  vec3 col(0, 0, 0);
  for (int s = 0; s < ns; s++) {
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
    ray r = (*cam)->get_ray(u, v, &local_rand_state);
    col += color(r, world, &local_rand_state);
  }
  rand_state[pixel_index] = local_rand_state;
  col /= float(ns);
  col[0] = sqrt(col[0]);
  col[1] = sqrt(col[1]);
  col[2] = sqrt(col[2]);
  fb[pixel_index] = col;
  return;
}

__device__ void Area::calcStep()
{
  if (x_begin == x_end && y_begin == y_end)
  {
    render();
    destroy(device_allocator, this);
    return;
  }

  if (y_begin < y_end)
  {
    int y_mid = (y_begin + y_end) / 2;
    switch (state)
    {
    case 0:
      new(device_allocator) Area(args, x_begin, x_end, y_begin, y_mid);
      state++;
      break;
    case 1:
      y_begin = y_mid + 1;
      state = 0;
      break;
    }
  }
  else if (x_begin < x_end)
  {
    int x_mid = (x_begin + x_end) / 2;
    switch (state)
    {
    case 0:
      new(device_allocator) Area(args, x_begin, x_mid, y_begin, y_end);
      state++;
      break;
    case 1:
      x_begin = x_mid + 1;
      state = 0;
      break;
    }
  }
}

__device__ void Area::calcStepSerial()
{
  if (depth++ >= 5)
  {
    depth = 0;
    return;
  }

  if (x_begin == x_end && y_begin == y_end)
  {
    render();
    destroy(device_allocator, this);
    return;
  }

  if (y_begin < y_end)
  {
    int y_mid = (y_begin + y_end) / 2;
    switch (state)
    {
    case 0:
    {
      Area* area = new(device_allocator) Area(args, x_begin, x_end, y_begin, y_mid);
      state++;
      area->depth = depth;
      area->calcStepSerial();
      break;
    }
    case 1:
    {
      y_begin = y_mid + 1;
      state = 0;
      calcStepSerial();
      break;
    }
    }
  }
  else if (x_begin < x_end)
  {
    int x_mid = (x_begin + x_end) / 2;
    switch (state)
    {
    case 0:
    {
      Area* area = new(device_allocator) Area(args, x_begin, x_mid, y_begin, y_end);
      state++;
      area->depth = depth + 1;
      area->calcStepSerial();
      break;
    }
    case 1:
    {
      x_begin = x_mid + 1;
      state = 0;
      calcStepSerial();
      break;
    }
    }
  }
}

__device__ void Area::calcEager()
{
  if (x_begin == x_end && y_begin == y_end)
  {
    render();
    destroy(device_allocator, this);
  }
  else if (x_begin == x_end)
  {
    int y_mid = (y_begin + y_end) / 2;
    new(device_allocator) Area(args, x_begin, x_end, y_begin, y_mid);
    y_begin = y_mid + 1;
  }
  else if (y_begin == y_end)
  {
    int x_mid = (x_begin + x_end) / 2;
    new(device_allocator) Area(args, x_begin, x_mid, y_begin, y_end);
    x_begin = x_mid + 1;
  }
  else
  {
    int x_mid = (x_begin + x_end) / 2;
    int y_mid = (y_begin + y_end) / 2;
    new(device_allocator) Area(args, x_begin, x_mid, y_begin, y_mid);
    new(device_allocator) Area(args, x_begin, x_mid, y_mid + 1, y_end);
    new(device_allocator) Area(args, x_mid + 1, x_end, y_begin, y_mid);
    x_begin = x_mid + 1;
    y_begin = y_mid + 1;
  }
}

__global__ void initialize(AllArgs* args, int min_x, int max_x, int min_y, int max_y)
{
  new(device_allocator) Area(args, min_x, max_x, min_y, max_y);
}
