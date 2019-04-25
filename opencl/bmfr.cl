/*  The MIT License (MIT)
 *  
 *  Copyright (c) 2019 Matias Koskela / Tampere University
 *  Copyright (c) 2018 Kalle Immonen / Tampere University of Technology
 *  
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 *  of this software and associated documentation files (the "Software"), to deal
 *  in the Software without restriction, including without limitation the rights
 *  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *  copies of the Software, and to permit persons to whom the Software is
 *  furnished to do so, subject to the following conditions:
 *  
 *  The above copyright notice and this permission notice shall be included in
 *  all copies or substantial portions of the Software.
 *  
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *  THE SOFTWARE.
 */

// Unrolled parallel reduction. Works only with 256 inputs
static inline void parallel_reduction_sum(
      __local float *result,
      __local float *sum_vec,
      const int start_index) {

   const int id = get_local_id(0);
   if(id < 64)
      sum_vec[id] += sum_vec[id+ 64] + sum_vec[id + 128] + sum_vec[id + 192];
   barrier(CLK_LOCAL_MEM_FENCE);
   if(id < 8)
      sum_vec[id] += sum_vec[id + 8] + sum_vec[id + 16] + sum_vec[id + 24] +
         sum_vec[id + 32] + sum_vec[id + 40] + sum_vec[id + 48] + sum_vec[id + 56];
   barrier(CLK_LOCAL_MEM_FENCE);
   if(id == 0){
      *result = sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3] +
         sum_vec[4] + sum_vec[5] + sum_vec[6] + sum_vec[7];
   }
   barrier(CLK_LOCAL_MEM_FENCE);
}

static inline void parallel_reduction_min(
      __local float *result,
      __local float *sum_vec) {

   const int id = get_local_id(0);
   if(id < 64)
      sum_vec[id] = fmin(fmin(fmin(sum_vec[id], sum_vec[id+ 64]),
         sum_vec[id + 128]), sum_vec[id + 192]);
   barrier(CLK_LOCAL_MEM_FENCE);
   if(id < 8)
      sum_vec[id] = fmin(fmin(fmin(fmin(fmin(fmin(fmin(sum_vec[id], sum_vec[id + 8]),
         sum_vec[id + 16]), sum_vec[id + 24]), sum_vec[id + 32]), sum_vec[id + 40]),
         sum_vec[id + 48]), sum_vec[id + 56]);
   barrier(CLK_LOCAL_MEM_FENCE);
   if(id == 0){
      *result = fmin(fmin(fmin(fmin(fmin(fmin(fmin(sum_vec[0], sum_vec[1]), sum_vec[2]),
         sum_vec[3]), sum_vec[4]), sum_vec[5]), sum_vec[6]), sum_vec[7]);
   }
   barrier(CLK_LOCAL_MEM_FENCE);

}

static inline void parallel_reduction_max(
      __local float *result,
      __local float *sum_vec) {

   const int id = get_local_id(0);
   if(id < 64)
      sum_vec[id] = fmax(fmax(fmax(sum_vec[id], sum_vec[id+ 64]),
         sum_vec[id + 128]), sum_vec[id + 192]);
   barrier(CLK_LOCAL_MEM_FENCE);
   if(id < 8)
      sum_vec[id] = fmax(fmax(fmax(fmax(fmax(fmax(fmax(sum_vec[id], sum_vec[id + 8]),
         sum_vec[id + 16]), sum_vec[id + 24]), sum_vec[id + 32]), sum_vec[id + 40]),
         sum_vec[id + 48]), sum_vec[id + 56]);
   barrier(CLK_LOCAL_MEM_FENCE);
   if(id == 0){
      *result = fmax(fmax(fmax(fmax(fmax(fmax(fmax(sum_vec[0], sum_vec[1]), sum_vec[2]),
         sum_vec[3]), sum_vec[4]), sum_vec[5]), sum_vec[6]), sum_vec[7]);
   }
   barrier(CLK_LOCAL_MEM_FENCE);
}

// Helper defines used in IN_ACCESS define
#define BLOCK_EDGE_HALF (BLOCK_EDGE_LENGTH / 2)
#define HORIZONTAL_BLOCKS (WORKSET_WIDTH / BLOCK_EDGE_LENGTH)
#define BLOCK_INDEX_X (group_id % (HORIZONTAL_BLOCKS + 1))
#define BLOCK_INDEX_Y (group_id / (HORIZONTAL_BLOCKS + 1))
#define IN_BLOCK_INDEX (BLOCK_INDEX_Y * (HORIZONTAL_BLOCKS + 1) + BLOCK_INDEX_X)
#define FEATURE_START (feature_buffer * BLOCK_PIXELS)
#define IN_ACCESS (IN_BLOCK_INDEX * buffers * BLOCK_PIXELS + \
   FEATURE_START + sub_vector * 256 + id)


#if COMPRESSED_R
#define R_SIZE (R_EDGE * (R_EDGE + 1) / 2)
#define R_ROW_START (R_SIZE - (R_EDGE - y) * (R_EDGE - y + 1) / 2)
#define R_ACCESS (R_ROW_START + x - y)
// Reduces unused values in the begining of each row
// 00 01 02 03 04 05
// 11 12 13 14 15 22
// 23 24 25 33 34 35
// 44 45 55
#else
#define R_ACCESS (x * R_EDGE + y)
// Here - means unused value
// Note: "unused" values are still set to 0 so some operations can be done to
// every element in a row or column
//    0  1  2  3  4  5 x
// 0 00 01 02 03 04 05
// 1  - 11 12 13 14 15
// 2  -  - 22 23 24 25
// 3  -  -  - 33 34 35
// 4  -  -  -  - 44 45
// 5  -  -  -  -  - 55
// y
#endif

static inline float3 load_r_mat(
      __local const float3* r_mat,
       const int x,
       const int y){
   return r_mat[R_ACCESS];
}

static inline void store_r_mat(
      __local float3* r_mat,
      const int x,
      const int y,
      const float3 value){
   r_mat[R_ACCESS] = value;
}

static inline void store_r_mat_broadcast(
      __local float3* r_mat,
      const int x,
      const int y,
      const float value){
   r_mat[R_ACCESS] = value;
}

static inline void store_r_mat_channel(
      __local float3* r_mat,
      const int x,
      const int y,
      const int channel,
      const float value){
   if(channel == 0)
      r_mat[R_ACCESS].x = value;
   else if(channel == 1)
      r_mat[R_ACCESS].y = value;
   else // channel == 2
      r_mat[R_ACCESS].z = value;
}

// Random generator from here http://asgerhoedt.dk/?p=323
static inline float random(unsigned int a) {
   a = (a+0x7ed55d16) + (a<<12);
   a = (a^0xc761c23c) ^ (a>>19);
   a = (a+0x165667b1) + (a<<5);
   a = (a+0xd3a2646c) ^ (a<<9);
   a = (a+0xfd7046c5) + (a<<3);
   a = (a^0xb55a4f09) ^ (a>>16);

   return convert_float(a) / convert_float(UINT_MAX);
}

static inline float add_random(
      const float value,
      const int id,
      const int sub_vector,
      const int feature_buffer,
      const int frame_number){
   return value + NOISE_AMOUNT * 2.f * (random(id + sub_vector * LOCAL_SIZE +
      feature_buffer * BLOCK_EDGE_LENGTH * BLOCK_EDGE_LENGTH +
      frame_number * BUFFER_COUNT * BLOCK_EDGE_LENGTH * BLOCK_EDGE_LENGTH) - 0.5f);
}

float3 RGB_to_YCoCg(float3 rgb) {
   return (float3){
      dot(rgb, (float3){ 1.f, 2.f,  1.f}),
      dot(rgb, (float3){ 2.f, 0.f, -2.f}),
      dot(rgb, (float3){-1.f, 2.f, -1.f})
   };
}

float3 YCoCg_to_RGB(float3 YCoCg) {
   return (float3){
      dot(YCoCg, (float3){0.25f,  0.25f, -0.25f}),
      dot(YCoCg, (float3){0.25f,    0.f,  0.25f}),
      dot(YCoCg, (float3){0.25f, -0.25f, -0.25f})
   };
}

static inline float scale(float value, float min, float max) {
   if(fabs(max - min) > 1.0f){
      return (value - min) / (max - min);
   }
   return value - min;
}

// Simple mirroring of image index if it is out of bounds.
// NOTE: Works only if index is less than one size out of bounds.
static inline int mirror(int index, int size){
   if(index < 0)
      index = abs(index) - 1;
   else if(index >= size)
      index = 2 * size - index - 1;

   return index;
}
static inline int2 mirror2(int2 index, int2 size){
   index.x = mirror(index.x, size.x);
   index.y = mirror(index.y, size.y);

   return index;
}

static inline void store_float3(
   __global float* restrict buffer,
   const int index,
   const float3 value){

   buffer[index * 3 + 0] = value.x;
   buffer[index * 3 + 1] = value.y;
   buffer[index * 3 + 2] = value.z;
}

// This is significantly slower the the inline function on Vega FE
//#define store_float3(buffer, index, value) \
//   buffer[(index) * 3 + 0] = value.x; \
//   buffer[(index) * 3 + 1] = value.y; \
//   buffer[(index) * 3 + 2] = value.z;

#define load_float3(buffer, index) ((float3)\
   {buffer[(index) * 3], buffer[(index) * 3 + 1], buffer[(index) * 3 + 2]})

// This gives on Vega FE warning about breaking the restrict keyword of the kernel
//static inline float3 load_float3(
//   __global float* restrict buffer,
//   const int index){
//
//   return (float3){
//      buffer[index * 3 + 0],
//      buffer[index * 3 + 1],
//      buffer[index * 3 + 2]
//   };
//}

#if USE_HALF_PRECISION_IN_TMP_DATA

#define LOAD(buffer, index) vload_half(index, buffer)
#define STORE(buffer, index, value) vstore_half(value, index, buffer)

#else

#define LOAD(buffer, index) buffer[index]
#define STORE(buffer, index, value) buffer[index] = value

#endif

#define BLOCK_OFFSETS_COUNT 16
__constant int2 BLOCK_OFFSETS[BLOCK_OFFSETS_COUNT] = {
   (int2){ -14, -14 },
   (int2){   4,  -6 },
   (int2){  -8,  14 },
   (int2){   8,   0 },
   (int2){ -10,  -8 },
   (int2){   2,  12 },
   (int2){  12, -12 },
   (int2){ -10,   0 },
   (int2){  12,  14 },
   (int2){  -8, -16 },
   (int2){   6,   6 },
   (int2){  -2,  -2 },
   (int2){   6, -14 },
   (int2){ -16,  12 },
   (int2){  14,  -4 },
   (int2){  -6,   4 }
};

#if ADD_REQD_WG_SIZE
__attribute__((reqd_work_group_size(LOCAL_WIDTH, LOCAL_HEIGHT, 1)))
#endif
__kernel void accumulate_noisy_data(
      __global float2* restrict out_prev_frame_pixel,
      __global unsigned char* restrict accept_bools,
      const __global float* restrict current_normals,
      const __global float* restrict previous_normals,
      const __global float* restrict current_positions,
      const __global float* restrict previous_positions,
      __global float* restrict current_noisy,
      const __global float* restrict previous_noisy,
      const __global unsigned char* restrict previous_spp,
      __global unsigned char* restrict current_spp,
#if USE_HALF_PRECISION_IN_TMP_DATA
      __global half* restrict tmp_data,
#else
      __global float* restrict tmp_data,
#endif
      const float16 prev_frame_camera_matrix,
      const float2 pixel_offset,
      const int frame_number) {

   const int2 gid = {get_global_id(0), get_global_id(1)};

   // Mirror indexed of the input. x and y are always less than one size out of
   // bounds if image dimensions are bigger than BLOCK_EDGE_LENGTH
   const int2 pixel_without_mirror = gid - BLOCK_EDGE_HALF
      + BLOCK_OFFSETS[frame_number % BLOCK_OFFSETS_COUNT];
   const int2 pixel = mirror2(pixel_without_mirror, (int2){IMAGE_WIDTH, IMAGE_HEIGHT});
   const int linear_pixel = pixel.y * IMAGE_WIDTH + pixel.x;

   float4 world_position = (float4){0.f, 0.f, 0.f, 1.f};
   world_position.xyz = load_float3(current_positions, linear_pixel);
   float3 normal = load_float3(current_normals, linear_pixel);
   float3 current_color = load_float3(current_noisy, linear_pixel);

   // Default previous frame pixel is the same pixel
   float2 prev_frame_pixel_f = convert_float2(pixel);

   // This is changed to non zero if previous frame is not discarded completely
   unsigned char store_accept = 0x00;

   // Blend_alpha 1.f means that only current frame color is used. The value
   // is is changed if sample from previous frame can be used
   float blend_alpha = 1.f;
   float3 previous_color = (float3){0.f, 0.f, 0.f};

   float sample_spp = 0.f;
   if(frame_number > 0){

      // Matrix multiplication and normalization to 0..1
      // TODO: transpose camera matrix somewhere else if it hits the performance
      // NOTE: not enough to test performance by changing s048c to s0123 here because it
      // produces prev_frame_pixels outside screen and removes many memory accesses
      float2 prev_frame_uv;
      prev_frame_uv.x = dot(prev_frame_camera_matrix.s048c, world_position);
      prev_frame_uv.y = dot(prev_frame_camera_matrix.s159d, world_position);
      // No need for z-buffer in accumulation of the noisy data
      //prev_frame_pixel.z = dot(prev_frame_camera_matrix.s26ae, world_position);
      prev_frame_uv /= dot(prev_frame_camera_matrix.s37bf, world_position);
      prev_frame_uv += 1.f;
      prev_frame_uv /= 2.f;

      // Change to pixel indexes and apply offset
      prev_frame_pixel_f = prev_frame_uv * (float2){IMAGE_WIDTH, IMAGE_HEIGHT};
      prev_frame_pixel_f -= (float2){
         pixel_offset.x, 1 - pixel_offset.y
      };
      int2 prev_frame_pixel = convert_int2_rtn(prev_frame_pixel_f);

      // These are needed for  the bilinear sampling
      int2 offsets[4];
      offsets[0] = (int2){0, 0};
      offsets[1] = (int2){1, 0};
      offsets[2] = (int2){0, 1};
      offsets[3] = (int2){1, 1};
      float2 prev_pixel_fract = prev_frame_pixel_f - convert_float2(prev_frame_pixel);
      float2 one_minus_prev_pixel_fract = 1.f - prev_pixel_fract;
      float weights[4];
      weights[0] = one_minus_prev_pixel_fract.x * one_minus_prev_pixel_fract.y;
      weights[1] = prev_pixel_fract.x           * one_minus_prev_pixel_fract.y;
      weights[2] = one_minus_prev_pixel_fract.x * prev_pixel_fract.y;
      weights[3] = prev_pixel_fract.x           * prev_pixel_fract.y;
      float total_weight = 0.f;

      // Bilinear sampling
      for(int i = 0; i < 4; ++i){

         int2 sample_location = prev_frame_pixel + offsets[i];
         int linear_sample_location = sample_location.y * IMAGE_WIDTH + sample_location.x;

         // Check if previous frame color can be used based on its screen location
         if(sample_location.x >= 0 && sample_location.y >= 0 &&
            sample_location.x < IMAGE_WIDTH && sample_location.y < IMAGE_HEIGHT){

            // Fetch previous frame world position
            float3 prev_world_position =
               load_float3(previous_positions, linear_sample_location);

            // Compute world distance squared
            float3 position_difference = prev_world_position - world_position.xyz;
            float position_distance_squared =
               dot(position_difference, position_difference);

            // World position distance discard
            if(position_distance_squared < convert_float(POSITION_LIMIT_SQUARED)){

               // Fetch previous frame normal
               float3 prev_normal = load_float3(previous_normals, linear_sample_location);

               // Distance of the normals
               // NOTE: could use some other distance metric (e.g. angle), but we use hard
               // experimentally found threshold -> means that the metric doesn't matter.
               float3 normal_difference = prev_normal - normal;
               float normal_distance_squared = dot(normal_difference, normal_difference);

               if(normal_distance_squared < convert_float(NORMAL_LIMIT_SQUARED)){

                  // Pixel passes all tests so store it to accept bools
                  store_accept |= 1 << i;

                  sample_spp += weights[i] *
                     convert_float(previous_spp[linear_sample_location]);

                  previous_color += weights[i] *
                     load_float3(previous_noisy, linear_sample_location);

                  total_weight += weights[i];
               }
            }
         }
      }

      if(total_weight > 0.f){
         previous_color /= total_weight;
         sample_spp /= total_weight;

         // Blend_alpha is dymically decided so that the result is average
         // of all samples until the cap defined by BLEND_ALPHA is reached
         blend_alpha = 1.f / (sample_spp + 1.f);
         blend_alpha = fmax(blend_alpha, BLEND_ALPHA);
      }
   }

   // Store new spp
   unsigned char new_spp = 1;
   if(blend_alpha < 1.f){
      if(sample_spp > 254.f){
         new_spp = 255;
      }else{
         // _sat is just extra causion because sample_spp should be less equal than 254
         new_spp = convert_uchar_sat_rte(sample_spp) + 1;
      }
   }
   current_spp[linear_pixel] = new_spp;

   float3 new_color = blend_alpha * current_color +
      (1.f - blend_alpha) * previous_color;

   // The set of feature buffers used in the fitting
   float features[BUFFER_COUNT] = {
      FEATURE_BUFFERS,
      new_color.x,
      new_color.y,
      new_color.z
   };

   for(int feature_num = 0; feature_num < BUFFER_COUNT; ++feature_num) {
      const int x_in_block = gid.x % BLOCK_EDGE_LENGTH;
      const int y_in_block = gid.y % BLOCK_EDGE_LENGTH;
      const int x_block = gid.x / BLOCK_EDGE_LENGTH;
      const int y_block = gid.y / BLOCK_EDGE_LENGTH;
      const unsigned int location_in_data = feature_num * BLOCK_PIXELS +
         x_in_block + y_in_block * BLOCK_EDGE_LENGTH +
         x_block * BLOCK_PIXELS * BUFFER_COUNT +
         y_block * (WORKSET_WITH_MARGINS_WIDTH / BLOCK_EDGE_LENGTH) *
         BLOCK_PIXELS * BUFFER_COUNT;

      float store_value = features[feature_num];

      if(isnan(store_value))
         store_value = 0.0f;

#if USE_HALF_PRECISION_IN_TMP_DATA
      store_value = max(min(store_value, 65504.f), -65504.f);
#endif

      STORE(tmp_data, location_in_data, store_value);
   }

   if(pixel_without_mirror.x >= 0 && pixel_without_mirror.x < IMAGE_WIDTH &&
      pixel_without_mirror.y >= 0 && pixel_without_mirror.y < IMAGE_HEIGHT){

      store_float3(current_noisy, linear_pixel, new_color);
      out_prev_frame_pixel[linear_pixel] = prev_frame_pixel_f;
      accept_bools[linear_pixel] = store_accept;
   }
}

#if ADD_REQD_WG_SIZE
__attribute__((reqd_work_group_size(256, 1, 1)))
#endif
__kernel void fitter(
      __local float *sum_vec,
      __local float *u_vec,
      __local float3 *r_mat,
      __global float* restrict weights,
      __global float* restrict mins_maxs,
#if USE_HALF_PRECISION_IN_TMP_DATA
      __global half* restrict tmp_data,
#else
      __global float* restrict tmp_data,
#endif
      const int frame_number) {

   __local float u_length_squared, dot, block_min, block_max, vec_length;
   float prod = 0.0f;

   const int group_id = get_group_id(0);
   const int id = get_local_id(0);
   const int buffers = BUFFER_COUNT;

   // Scales world positions to 0..1 in a block
   for(int feature_buffer = FEATURES_NOT_SCALED; feature_buffer < buffers - 3; ++feature_buffer){

      // Find maximum and minimum of the whole block
      float tmp_max = -INFINITY;
      float tmp_min = INFINITY;
      for(int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector){
         float value = LOAD(tmp_data, IN_ACCESS);
         tmp_max = fmax(value, tmp_max);
         tmp_min = fmin(value, tmp_min);
      }
      sum_vec[id] = tmp_max;
      barrier(CLK_LOCAL_MEM_FENCE);

      parallel_reduction_max(&block_max, sum_vec);

      sum_vec[id] = tmp_min;
      barrier(CLK_LOCAL_MEM_FENCE);

      parallel_reduction_min(&block_min, sum_vec);

      if(id == 0){
         const int index = (group_id * FEATURES_SCALED + feature_buffer - FEATURES_NOT_SCALED) * 2;
         mins_maxs[index + 0] = block_min;
         mins_maxs[index + 1] = block_max;
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      for(int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector){
         float scaled_value = scale(LOAD(tmp_data, IN_ACCESS), block_min, block_max);
         STORE(tmp_data, IN_ACCESS, scaled_value);
      }
   }

   // Non square matrices require processing every column. Otherwise result is
   // OKish, but R is not upper triangular matrix
   int limit = buffers == BLOCK_PIXELS ? buffers - 1 : buffers;

   // Compute R
   for(int col = 0; col < limit; col++) {
      int col_limited = min(col, buffers - 3);

      // Load new column into memory
      int feature_buffer = col;
      float tmp_sum_value = 0.f;
      for(int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector){
         float tmp = LOAD(tmp_data, IN_ACCESS);

         const int index = id + sub_vector * LOCAL_SIZE;
         u_vec[index] = tmp;
         if(index >= col_limited + 1){
            tmp_sum_value += tmp * tmp;
         }
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      // Find length of vector in A's column with reduction sum function
      sum_vec[id] = tmp_sum_value;
      barrier(CLK_LOCAL_MEM_FENCE);
      parallel_reduction_sum(&vec_length, sum_vec, col_limited + 1);

      // NOTE: GCN Opencl compiler can do some optimization with this because if
      // initially wanted col_limited is used to select wich WI runs which branch
      // it is slower. However using col produces the same result.
      float r_value;
      if(id < col){

         // Copy u_vec value
         r_value = u_vec[id];

      }else if(id == col){

         u_length_squared = vec_length;
         vec_length = sqrt(vec_length + u_vec[col_limited] * u_vec[col_limited]);
         u_vec[col_limited] -= vec_length;
         u_length_squared += u_vec[col_limited] * u_vec[col_limited];
         // (u_length_squared is now updated length squared)
         r_value = vec_length;

      }else if(id > col){ //Could have "&& id <  R_EDGE" but this is little bit faster

         // Last values on every column are zeros
         r_value = 0.0f;

      }

      int id_limited = min(id, buffers - 3);
      if(col < buffers - 3)
         store_r_mat_broadcast(r_mat, col_limited, id_limited, r_value);
      else
         store_r_mat_channel(r_mat, col_limited, id_limited, col - buffers + 3, r_value);
      barrier(CLK_LOCAL_MEM_FENCE);

      // Transform further columns of A
      // NOTE: three last columns are three color channels of noisy data. However,
      // they all need to be transfomed as they were column indexed (buffers - 3)
      for(int feature_buffer = col_limited+1; feature_buffer < buffers; feature_buffer++){

         // Starts by computing dot product with reduction sum function
#if CACHE_TMP_DATA
         // No need to load tmp_data twice because each WI first copies value for
         // dot product computation and then modifies the same value
         float tmp_data_private_cache[(BLOCK_EDGE_LENGTH *
            BLOCK_EDGE_LENGTH) / LOCAL_SIZE];
#endif
         float tmp_sum_value = 0.f;
         for(int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector){

            const int index = id + sub_vector * LOCAL_SIZE;
            if(index >= col_limited){

               float tmp = LOAD(tmp_data, IN_ACCESS);

               // Add noise on the first time values are loaded
               // (does not add noise to constant buffer and noisy image data)
               if(col == 0 && feature_buffer < buffers - 3){
                  tmp = add_random(tmp, id, sub_vector, feature_buffer, frame_number);
               }

#if CACHE_TMP_DATA
               tmp_data_private_cache[sub_vector] = tmp;
#endif
               tmp_sum_value += tmp * u_vec[index];
            }
         }

         sum_vec[id] = tmp_sum_value;
         barrier(CLK_LOCAL_MEM_FENCE);
         parallel_reduction_sum(&dot, sum_vec, col_limited);

         for (int sub_vector = 0; sub_vector < BLOCK_PIXELS / LOCAL_SIZE; ++sub_vector) {
             const int index = id + sub_vector * LOCAL_SIZE;
             if (index >= col_limited) {
#if CACHE_TMP_DATA
               float store_value = tmp_data_private_cache[sub_vector];
#else
               float store_value = LOAD(tmp_data, IN_ACCESS);
               store_value =
                  add_random(store_value, id, sub_vector, feature_buffer, frame_number);
#endif
               store_value -= 2 * u_vec[index] * dot / u_length_squared;
               STORE(tmp_data, IN_ACCESS, store_value);
            }
         }
         barrier(CLK_GLOBAL_MEM_FENCE);
      }
   }

   // Back substitution
   local float3 divider;
   for(int i = R_EDGE - 2; i >= 0; i--){
      if(id == 0)
         divider = load_r_mat(r_mat, i, i);
      barrier(CLK_LOCAL_MEM_FENCE);
#if COMPRESSED_R
      if(id < R_EDGE && id >= i){
#else
      // First values are always zero if R !COMPRESSED_R and
      // "&& id >= i" makes not compressed code run little bit slower
      if(id < R_EDGE){
#endif
         float3 value = load_r_mat(r_mat, id, i);
         store_r_mat(r_mat, id, i, value / divider);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
      if(id == 0) //Optimization proposal: parallel reduction
         for(int j = i + 1; j < R_EDGE - 1; j++){
            float3 value = load_r_mat(r_mat, R_EDGE - 1, i);
            float3 value2 = load_r_mat(r_mat, j, i);
            store_r_mat(r_mat, R_EDGE - 1, i, value - value2);
         }
      barrier(CLK_LOCAL_MEM_FENCE);
#if COMPRESSED_R
      if(id < R_EDGE && i >= id){
#else
      if(id < R_EDGE){
#endif
         float3 value = load_r_mat(r_mat, i, id);
         float3 value2 = load_r_mat(r_mat, R_EDGE - 1, i);
         store_r_mat(r_mat, i, id, value * value2);
      }
      barrier(CLK_LOCAL_MEM_FENCE);
   }

   if(id < buffers - 3){
      // Store weights
      const int index = group_id * (buffers - 3) + id;
      const float3 weight = load_r_mat(r_mat, R_EDGE - 1, id);
      store_float3(weights, index, weight);
   }
}


__kernel void weighted_sum(
      const __global float* restrict weights,
      const __global float* restrict mins_maxs,
      __global float* restrict output,
      const __global float* restrict current_normals,
      const __global float* restrict current_positions,
      const __global float* restrict current_noisy, // Only used for debugging
      const int frame_number){

   const int2 pixel = (int2){get_global_id(0), get_global_id(1)};
   const int linear_pixel = pixel.y * IMAGE_WIDTH + pixel.x;
   if(pixel.x >= IMAGE_WIDTH || pixel.y >=  IMAGE_HEIGHT)
      return;

   // Load weights and min_max which this pixel should use.
   const int2 offset = BLOCK_OFFSETS[frame_number % BLOCK_OFFSETS_COUNT];
   const int2 offset_pixel = pixel + BLOCK_EDGE_HALF - offset;
   const int group_index = (offset_pixel.x / BLOCK_EDGE_LENGTH) +
      (offset_pixel.y / BLOCK_EDGE_LENGTH) *
      (WORKSET_WITH_MARGINS_WIDTH / BLOCK_EDGE_LENGTH);

   // Load feature buffers
   float3 world_position = load_float3(current_positions, linear_pixel);
   float3 normal = load_float3(current_normals, linear_pixel);
   float features[BUFFER_COUNT - 3] = {
      FEATURE_BUFFERS
   };

   // Weighted sum of the feature buffers
   float3 color = (float3){0.f, 0.f, 0.f};
   for(int feature_buffer = 0; feature_buffer < BUFFER_COUNT - 3; feature_buffer++){
      float feature = features[feature_buffer];

      // Scale world position buffers
      if (feature_buffer >= FEATURES_NOT_SCALED) {
         const int min_max_index = (group_index * FEATURES_SCALED + feature_buffer - FEATURES_NOT_SCALED) * 2;
         feature =
            scale(feature, mins_maxs[min_max_index + 0], mins_maxs[min_max_index + 1]);
      }

      // Load weight and sum
      float3 weight =
         load_float3(weights, group_index * (BUFFER_COUNT - 3) + feature_buffer);
      color += weight * feature;
   }

   // Remove negative values from every component of the fitting results
   color = color < 0.f ? 0.f : color;

   // !!!!!
   // Uncomment this for debugging. Removes fitting completely.
   //color = load_float3(current_noisy, linear_pixel);

   // Store resutls
   store_float3(output, linear_pixel, color);
}


__kernel void accumulate_filtered_data(
      const __global float* restrict filtered_frame,
      const __global float2* restrict in_prev_frame_pixel,
      const __global unsigned char* restrict accept_bools,
      const __global float* restrict albedo,
      __global float* restrict tone_mapped_frame,
      const __global unsigned char* restrict current_spp,
      const __global float* restrict accumulated_prev_frame,
      __global float* restrict accumulated_frame,
      const int frame_number){

   // Return if out of image
   const int2 pixel = (int2){get_global_id(0), get_global_id(1)};
   const int linear_pixel = pixel.y * IMAGE_WIDTH + pixel.x;
   if(pixel.x >= IMAGE_WIDTH || pixel.y >=  IMAGE_HEIGHT)
      return;

   float3 filtered_color = load_float3(filtered_frame, linear_pixel);
   float3 prev_color = (float3){0.f, 0.f, 0.f};
   float blend_alpha = 1.f;

   //!!!!!!
   // Add "&& false" to debug other kernels (removes accumulation completely)
   if(frame_number > 0){

      // Accept tells which bilinear pixels were accepted in the first accum kernel
      const unsigned char accept = accept_bools[linear_pixel];

      if(accept > 0){ // If any prev frame sample is accepted

         // Bilinear sampling
         const float2 prev_frame_pixel_f =
            in_prev_frame_pixel[linear_pixel];
         const int2 prev_frame_pixel = convert_int2_rtn(prev_frame_pixel_f);
         const float2 prev_pixel_fract = prev_frame_pixel_f -
            convert_float2(prev_frame_pixel);
         const float2 one_minus_prev_pixel_fract = 1.f - prev_pixel_fract;
         float total_weight = 0.f;

         // Accept tells if the sample is acceptable based on world position and normal
         if(accept & 0x01){
            float weight = one_minus_prev_pixel_fract.x * one_minus_prev_pixel_fract.y;
            total_weight += weight;
            int linear_sample_location =
               prev_frame_pixel.y * IMAGE_WIDTH + prev_frame_pixel.x;
            prev_color += weight * load_float3(accumulated_prev_frame,
               linear_sample_location);
         }
         if(accept & 0x02){
            float weight = prev_pixel_fract.x * one_minus_prev_pixel_fract.y;
            total_weight += weight;
            int linear_sample_location =
               prev_frame_pixel.y * IMAGE_WIDTH + prev_frame_pixel.x + 1;
            prev_color += weight * load_float3(accumulated_prev_frame,
               linear_sample_location);
         }
         if(accept & 0x04){
            float weight = one_minus_prev_pixel_fract.x * prev_pixel_fract.y;
            total_weight += weight;
            int linear_sample_location =
               (prev_frame_pixel.y + 1) * IMAGE_WIDTH + prev_frame_pixel.x;
            prev_color +=  weight * load_float3(accumulated_prev_frame,
               linear_sample_location);
         }
         if(accept & 0x08){
            float weight = prev_pixel_fract.x * prev_pixel_fract.y;
            total_weight += weight;
            int linear_sample_location =
               (prev_frame_pixel.y + 1) * IMAGE_WIDTH + prev_frame_pixel.x + 1;
            prev_color += weight * load_float3(accumulated_prev_frame,
               linear_sample_location);
         }

         if(total_weight > 0.f){

            // Blend_alpha is dymically decided so that the result is average
            // of all samples until the cap defined by SECOND_BLEND_ALPHA is reached
            blend_alpha = 1.f / convert_float(current_spp[linear_pixel]);
            blend_alpha = fmax(blend_alpha, SECOND_BLEND_ALPHA);

            prev_color /= total_weight;
         }
      }
   }

   // Mix with colors and store results
   float3 accumulated_color = blend_alpha * filtered_color +
      (1.f - blend_alpha) * prev_color;
   store_float3(accumulated_frame, linear_pixel, accumulated_color);

   // Remodulate albedo and tone map
   float3 my_albedo = load_float3(albedo, linear_pixel);
   const float3 tone_mapped_color = clamp(
      powr(max(0.f, my_albedo * accumulated_color), 0.454545f), 0.f, 1.f);

   store_float3(tone_mapped_frame, linear_pixel, tone_mapped_color);
}


__kernel void taa(
      const __global float2* restrict in_prev_frame_pixel,
      const __global float* restrict new_frame,
      __global float* restrict result_frame,
      const __global float* restrict prev_frame,
      const int frame_number){

   // Return if out of image
   const int2 pixel = (int2){get_global_id(0), get_global_id(1)};
   const int linear_pixel = pixel.y * IMAGE_WIDTH + pixel.x;
   if(pixel.x >= IMAGE_WIDTH || pixel.y >=  IMAGE_HEIGHT)
      return;

   float3 my_new_color = load_float3(new_frame, linear_pixel);

   // Loads value which tells where this pixel was in the previous frame.
   // The value is already calculated in accumulate_noisy_data
   const float2 prev_frame_pixel_f =
      in_prev_frame_pixel[linear_pixel];
   int2 prev_frame_pixel = convert_int2_rtn(prev_frame_pixel_f);

   //!!!!!!
   // Add "|| true" to debug other kernels (removes taa)
   // Return if all sampled pixels are going to be out of image area
   if(frame_number == 0 ||
      prev_frame_pixel.x < -1 || prev_frame_pixel.y < -1 ||
      prev_frame_pixel.x >= IMAGE_WIDTH || prev_frame_pixel.y >= IMAGE_HEIGHT){

      store_float3(result_frame, linear_pixel, my_new_color);
      return;
   }


   float3 minimum_box = INFINITY;
   float3 minimum_cross = INFINITY;
   float3 maximum_box = -INFINITY;
   float3 maximum_cross = -INFINITY;
   for(int y = -1; y < 2; ++y){
      for(int x = -1; x < 2; ++x){
         int2 sample_location = pixel + (int2){x, y};
         if(sample_location.x >= 0 && sample_location.y >= 0 &&
            sample_location.x < IMAGE_WIDTH && sample_location.y < IMAGE_HEIGHT){

            float3 sample_color;
            if(x == 0 && y == 0)
               sample_color = my_new_color;
            else
               sample_color = load_float3(
                  new_frame, sample_location.x + sample_location.y * IMAGE_WIDTH);

            sample_color = RGB_to_YCoCg(sample_color);

            if(x == 0 || y == 0){
               minimum_cross = fmin(minimum_cross, sample_color);
               maximum_cross = fmax(maximum_cross, sample_color);
            }
            minimum_box = fmin(minimum_box, sample_color);
            maximum_box = fmax(maximum_box, sample_color);
         }
      }
   }

   // Bilinear sampling of previous frame.
   // NOTE: WI has already returned if the sampling location is complety out of image
   float3 prev_color = (float3){0.f, 0.f, 0.f};
   float total_weight = 0;
   float2 pixel_fract = prev_frame_pixel_f - convert_float2(prev_frame_pixel);
   float2 one_minus_pixel_fract = 1.f - pixel_fract;

   if(prev_frame_pixel.y >= 0){
      if(prev_frame_pixel.x >= 0){
         float weight = one_minus_pixel_fract.x * one_minus_pixel_fract.y;
         prev_color += weight *
            load_float3(prev_frame,
               prev_frame_pixel.y * IMAGE_WIDTH + prev_frame_pixel.x);
         total_weight += weight;
      }
      if(prev_frame_pixel.x < IMAGE_WIDTH - 1){
         float weight = pixel_fract.x * one_minus_pixel_fract.y;
         prev_color += weight *
            load_float3(prev_frame,
               prev_frame_pixel.y * IMAGE_WIDTH + prev_frame_pixel.x + 1);
         total_weight += weight;
      }
   }
   if(prev_frame_pixel.y < IMAGE_HEIGHT - 1){
      if(prev_frame_pixel.x >= 0){
         float weight = one_minus_pixel_fract.x * pixel_fract.y;
         prev_color += weight *
            load_float3(prev_frame,
               (prev_frame_pixel.y + 1) * IMAGE_WIDTH + prev_frame_pixel.x);
         total_weight += weight;
      }
      if(prev_frame_pixel.x < IMAGE_WIDTH - 1){
         float weight = pixel_fract.x * pixel_fract.y;
         prev_color += weight *
            load_float3(prev_frame,
               (prev_frame_pixel.y + 1) * IMAGE_WIDTH + prev_frame_pixel.x + 1);
         total_weight += weight;
      }
   }

   prev_color /= total_weight; // Total weight can be less than one on the edges
   float3 prev_color_ycocg = RGB_to_YCoCg(prev_color);

   // NOTE: Some references use more complicated methods to move the previous frame color
   // to the YCoCg space AABB
   float3 minimum = (minimum_box + minimum_cross) / 2.f;
   float3 maximum = (maximum_box + maximum_cross) / 2.f;
   float3 prev_color_rgb = YCoCg_to_RGB(clamp(prev_color_ycocg, minimum, maximum));

   float3 result_color = TAA_BLEND_ALPHA * my_new_color +
      (1.f - TAA_BLEND_ALPHA) * prev_color_rgb;
   store_float3(result_frame, linear_pixel, result_color);
}
