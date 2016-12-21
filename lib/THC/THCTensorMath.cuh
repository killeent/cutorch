#ifndef THC_TENSORMATH_CUH
#define THC_TENSORMATH_CUH

// Copy the kth diagonal of a matrix B to a vector A.
template <typename T>
__global__ void THCTensor_copyFromDiagonal(T* a, T* b, ptrdiff_t start, ptrdiff_t size, ptrdiff_t strideSum, ptrdiff_t strideA) {
  for (ptrdiff_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < size;
       linearIndex += gridDim.x * blockDim.x) {
    const ptrdiff_t bOffset = start + strideSum * linearIndex;
    a[strideA * linearIndex] = b[bOffset];
  }
}

// Copy vector B to the kth diagonal of a matrix A
template <typename T>
__global__ void THCTensor_copyToDiagonal(T* a, T* b, ptrdiff_t start, ptrdiff_t size, ptrdiff_t strideSum, ptrdiff_t strideB) {
  for (ptrdiff_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < size;
       linearIndex += gridDim.x * blockDim.x) {
    const ptrdiff_t aOffset = start + strideSum * linearIndex;
    a[aOffset] = b[strideB * linearIndex];
  }
}

// Limited by the maximum size of kernel arguments (4kb)
#define CAT_ARRAY_KERNEL_BATCH_SIZE 16

template <typename T>
struct CatArrayKernelParam {
  // Tensors to copy into the output parameter
  /* CatArrayKernelParam(TensorInfo<T, unsigned int>* ins, int* offs, int* cts, int n); */

  TensorInfo<T, unsigned int> inputs[CAT_ARRAY_KERNEL_BATCH_SIZE];

  // The offsets along the dimension in the output tensor where we
  // should begin the copy, for each tensor
  int offsets[CAT_ARRAY_KERNEL_BATCH_SIZE];

  // The of the dimension for each input tensor, used to specify
  // how to narrow the output tensor
  int dimSizes[CAT_ARRAY_KERNEL_BATCH_SIZE];

  // Number of elements in each tensor, for the grid-stride loop
  // bound
  int nElements[CAT_ARRAY_KERNEL_BATCH_SIZE];

  // Actual number of tensors in this param (may be less than the
  // batch size)
  int count;

  /* __host__ __device__ void debugString() const; */
};

// Limited by the 4kb argument limit for kernels. The larger KERNEL_MAX,
// the smaller the buffer can be. At this setting, can support concatenating
// 128 (1D Tensors), 128 (2D Tensors), 106 (3D Tensors), 80 (4D Tensors), etc.
#define CAT_ARRAY_KERNEL_MAX 128
#define CAT_ARRAY_STRIDE_BUFFER_SIZE 320

template <typename T>
struct CatArrayKernelParam2 {
  // This is the array of pointers to the Tensors we are concatenating into the result
  // Tensor.
  T *data[CAT_ARRAY_KERNEL_MAX];

  // Number of dimensions in input Tensors.
  int dims;

  // A buffer to store stride information for all the input Tensors. In particular, this
  // array stores count * dims entries: The first dims entries are the stride information
  // for the first tensor, arranged in the same format as a Tensor (i.e. outer dimension
  // first)
  int strides[CAT_ARRAY_STRIDE_BUFFER_SIZE];

  // The offsets along the dimension in the output Tensor where we should begin the 
  // copy, for each Tensor.
  int offsets[CAT_ARRAY_KERNEL_MAX];

  // The size at the concatenation dimension for each input Tensor, used to specify
  // how to narrow the output tensor, as well as determining the offset for indices
  // into the input tensors.
  int dimSizes[CAT_ARRAY_KERNEL_MAX];

  // Number of elements in each tensor, for the grid-stride loop bound
  int nElements[CAT_ARRAY_KERNEL_MAX];

  // Actual number of tensors in this param (may be less than the  max size)
  int count;

  __host__ __device__ void debugString() const;
};

template <typename T>
void
CatArrayKernelParam2<T>::debugString() const {
  printf("Kernel Param: %d elements, with %d dimensions\n", count, dims);
  printf("---------------------------------------------\n");
  for (int i = 0; i < count; ++i) {
    printf("Input Tensor %d: dataptr %p, offset %d, dimSize %d, nElements %d\n", 
        i, data[i], offsets[i], dimSizes[i], nElements[i]);
    printf("Strides:");
    for (int j = 0; j < dims; ++j) {
      printf(" %d,", strides[(i*dims) + j]);
    }
    printf("\n");
  }
}

template <typename T>
struct CatArrayKernelOffsetCalc {
  // Utility function to map and index to offset for a particular input tensor, and particular
  // index, in the kernel. This is the same as any other dynamic indexing approach (see e.g.
  // IndexToOffset) but we take advantage of the fact that for the catArray problem, the size
  // of the dimensions of every input Tensor must be the same for the dimensions we *are not*
  // concatenating along. Hence we can borrow them from the TensorInfo for the result.
  static inline __device__ unsigned int indexToOffset(
      const TensorInfo<T, unsigned int> result, 
      const CatArrayKernelParam2<T> param, 
      const unsigned int concatDim,
      const unsigned int tensorIndex, 
      unsigned int linearIndex) {
    assert(entry < param.count);
    /* printf("Generating offset for tensor %d, linearIndex %d\n", tensorIndex, linearIndex); */

    unsigned int offset = 0;

    // Calculate offset into strides buffer - 
    int bufOffset = result.dims * tensorIndex;

    for (int i = param.dims - 1; i >= 0; --i) {
      unsigned int curDimSize = i == concatDim ? param.dimSizes[tensorIndex] : result.sizes[i];
      unsigned int curDimIndex = linearIndex % curDimSize;
      unsigned int curDimOffset = curDimIndex * param.strides[i + bufOffset];
      /* printf("i: %d, li: %d, cds: %d, cdi: %d, cdo: %d\n", i, linearIndex, curDimSize, curDimIndex, curDimOffset); */
      offset += curDimOffset;

      linearIndex /= curDimSize;
    }
    /* printf("Result: %d\n", offset); */
    /* printf("-----------------------------------------------\n"); */

    return offset;
  }
};

/* template <typename T> */
/* inline __device__ unsigned int */
/* CatArrayKernelParam2<T>::indexToOffset(unsigned int entry, unsigned int index) { */
/*   assert(entry < count); */

/*   unsigned int offset = 0; */

/*   // Calculate offset into sizes/strides buffer */
/*   int bufOffset = dims * index; */

/*   for (int i = dims - 1; i >= 0; --i) { */
/*     unsigned int curDimIndex = index % sizes[i + bufOffset]; */
/*     unsigned int curDimOffset = curDimIndex * strides[i + bufOffset]; */
/*     offset += curDimOffset; */

/*     index /= sizes[i + bufOffset]; */
/*   } */

/*   return offset; */
/* } */

/* template <typename T> */
/* CatArrayKernelParam<T>::CatArrayKernelParam(TensorInfo<T, unsigned int>* ins, int* offs, int* cts, int n) { */
/*   for (int i = 0; i < n; ++i) { */
/*     inputs[i] = ins[i]; */
/*     offsets[i] = offs[i]; */
/*     counts[i] = cts[i]; */
/*   } */
/*   count = n; */
/* } */

/* template <typename T> */
/* void */
/* CatArrayKernelParam<T>::debugString() const { */
/*   printf("================================\n"); */
/*   printf("CatArrayKernelParam: (%d associated TensorInfos)\n", count); */
/*   printf("--------------------------------------\n"); */
/*   for (int i = 0; i < count; ++i) { */
/*     inputs[i].debugString(); */
/*     printf("Associated offset: %d, dimSize: %d, nElements: %d\n", */
/*         offsets[i], dimSizes[i], nElements[i]); */
/*     printf("--------------------------------------\n"); */
/*   } */
/*   printf("================================\n"); */
/* } */

template <typename T>
__global__ void catArrayBatchedCopy(TensorInfo<T, unsigned int> result, const CatArrayKernelParam<T> param, int dimension) {
  // A block is responsible for the ith tensor in the param if its blockDim.y = i, so let's narrow
  // the result TensorInfo according to the offset, dimSize for that tensor
  /* result.debugString(); */
  /* param.debugString(); */
  /* printf("offset: %d\n, dimSize: %d\n", param.offsets[blockIdx.y], param.dimSizes[blockIdx.y]); */
  /* printf("%p\n", &param); */
  TensorInfo<T, unsigned int> nt = result.newNarrow(dimension, param.offsets[blockIdx.y], param.dimSizes[blockIdx.y]);
  /* nt.debugString(); */

  // Now follow the normal pointwise op code, where the the linear index is determined by thread/block x values
  for (unsigned int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < param.nElements[blockIdx.y];
       linearIndex += gridDim.x * blockDim.x) {
    const unsigned int resultOffset = IndexToOffset<T, unsigned int, -1>::get(linearIndex, nt);
    const unsigned int srcOffset = IndexToOffset<T, unsigned int, -1>::get(linearIndex, param.inputs[blockIdx.y]);
    /* printf("Copying from input %d, offset: %d to result offset: %d\n", blockIdx.y, srcOffset, resultOffset); */
    nt.data[resultOffset] = param.inputs[blockIdx.y].data[srcOffset];
  }
}

template <typename T>
__global__ void catArrayBatchedCopy2(TensorInfo<T, unsigned int> result, const CatArrayKernelParam2<T> param, int dimension) {
  // A block is responsible for the ith tensor in the param if its blockDim.y = i, so let's narrow
  // the result TensorInfo according to the offset, dimSize for that tensor
  TensorInfo<T, unsigned int> nt = result.newNarrow(dimension, param.offsets[blockIdx.y], param.dimSizes[blockIdx.y]);
  nt.debugString();
  param.debugString();

  // Now follow the normal pointwise op code, where the the linear index is determined by thread/block x values
  for (unsigned int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < param.nElements[blockIdx.y];
       linearIndex += gridDim.x * blockDim.x) {
    const unsigned int resultOffset = IndexToOffset<T, unsigned int, -1>::get(linearIndex, nt);
    const unsigned int srcOffset = CatArrayKernelOffsetCalc<T>::indexToOffset(result, param, dimension, blockIdx.y, linearIndex);
    /* printf("Accessing dest address: %p, Source Address: %p\n", &nt.data[resultOffset], &param.data[blockIdx.y][srcOffset]); */
    /* printf("resultOffset: %d, srcOffset: %d\n", resultOffset, srcOffset); */
    /* nt.data[resultOffset] = param.data[blockIdx.y][srcOffset]; */
  }
}

#endif
