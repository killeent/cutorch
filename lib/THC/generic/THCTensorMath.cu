#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/THCTensorMath.cu"
#else

THC_API void
THCTensor_(fill)(THCState* state, THCTensor *self_, real value)
{
  THAssert(THCTensor_(checkGPU)(state, 1, self_));

  if (!THC_pointwiseApply1(
        state, self_, TensorFillOp<real>(value))) {
    THArgCheck(false, 1, CUTORCH_DIM_WARNING);
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(zero)(THCState *state, THCTensor *self_)
{
  THAssert(THCTensor_(checkGPU)(state, 1, self_));
  if (THCTensor_(isContiguous)(state, self_)) {
    THCudaCheck(cudaMemsetAsync(THCTensor_(data)(state, self_),
                                0,
                                sizeof(real) * THCTensor_(nElement)(state, self_),
                                THCState_getCurrentStream(state)));
  } else {
    if (!THC_pointwiseApply1(
          state, self_,
          TensorFillOp<real>(ScalarConvert<int, real>::to(0)))) {
      THArgCheck(false, 1, CUTORCH_DIM_WARNING);
    }
  }

  THCudaCheck(cudaGetLastError());
}

THC_API void
THCTensor_(zeros)(THCState *state, THCTensor *r_, THLongStorage *size)
{
  THAssert(THCTensor_(checkGPU)(state, 1, r_));
  THCTensor_(resize)(state, r_, size, NULL);
  THCTensor_(zero)(state, r_);
}

THC_API void
THCTensor_(ones)(THCState *state, THCTensor *r_, THLongStorage *size)
{
  THAssert(THCTensor_(checkGPU)(state, 1, r_));
  THCTensor_(resize)(state, r_, size, NULL);
  THCTensor_(fill)(state, r_, ScalarConvert<int, real>::to(1));
}

THC_API void
THCTensor_(reshape)(THCState *state, THCTensor *r_, THCTensor *t, THLongStorage *size)
{
  THAssert(THCTensor_(checkGPU)(state, 2, r_, t));
  THCTensor_(resize)(state, r_, size, NULL);
  THCTensor_(copy)(state, r_, t);
}

ptrdiff_t
THCTensor_(numel)(THCState *state, THCTensor *t)
{
  return THCTensor_(nElement)(state, t);
}

void THCTensor_(cat)(THCState *state, THCTensor *result,
		     THCTensor *ta, THCTensor *tb, int dimension)
{
  THCTensor* inputs[2];
  inputs[0] = ta;
  inputs[1] = tb;
  THCTensor_(catArray)(state, result, inputs, 2, dimension);
}

void THCTensor_(catArray)(THCState *state, THCTensor *result,
			  THCTensor **inputs, int numInputs, int dimension)
{
  THLongStorage *size;
  int i, j, k, cohortMax;
  /* int index; */
  long offset;
  int ndim = dimension + 1;
  for (i = 0; i < numInputs; i++)
  {
    ndim = THMax(ndim, THCTensor_(nDimension)(state, inputs[i]));
  }

  THArgCheck(numInputs > 0, 3, "invalid number of inputs %d", numInputs);
  THArgCheck(dimension >= 0, 4, "invalid dimension %d", dimension+1);

  size = THLongStorage_newWithSize(ndim);
  for(i = 0; i < ndim; i++)
  {
    long dimSize = i < THCTensor_(nDimension)(state, inputs[0])
                       ? THCTensor_(size)(state, inputs[0], i)
                       : 1;
    if (i == dimension)
    {
      for (j = 1; j < numInputs; j++)
      {
        dimSize += i < THCTensor_(nDimension)(state, inputs[j])
                       ? THCTensor_(size)(state, inputs[j], i)
                       : 1;
      }
    }
    else
    {
      for (j = 1; j < numInputs; j++)
      {
        if (dimSize != (i < THCTensor_(nDimension)(state, inputs[j])
			? THCTensor_(size)(state, inputs[j], i)
			: 1)) {
          THLongStorage_free(size);
          THError("inconsistent tensor sizes");
        }
      }
    }
    size->data[i] = dimSize;
  }

  THCTensor_(resize)(state, result, size, NULL);
  THLongStorage_free(size);

  // We parallelize the copy if all 3 conditions pass:
  //
  // 1. There is more than one input tensor
  // 2. All input tensors can use 32-bit indexing
  // 3. All input tensors are on the same device

  if (numInputs > 1 &&
      TensorUtils<THCTensor>::all32BitIndexable(state, inputs, numInputs) &&
      TensorUtils<THCTensor>::allSameDevice(state, inputs, numInputs)) {

    // For debugging purposes
    /* THCTensor_(fill)(state, result, 0); */

    // First, define a TensorInfo for the result tensor to pass to the kernel
    TensorInfo<real, unsigned int> rst = getTensorInfo<THCTensor, unsigned int>(state, result);
    rst.debugString();

    // Now, we need to set up our loop bounds. In particular, we want to stuff as many
    // Tensors at once into the kernel call. We are bound by two conditions:
    //
    // 1. The maximum number of Tensors in the kernel param (CAT_ARRAY_KERNEL_MAX)
    // 2. The maximum amount of stride information we can pass to the kernel param
    // (CAT_ARRAY_STIRDE_BUFFER_SIZE / nDim)

    /* int strideLimit = CAT_ARRAY_STRIDE_BUFFER_SIZE / ndim; */
    int strideLimit = 1;
    int batchSize = CAT_ARRAY_KERNEL_MAX <= strideLimit ? CAT_ARRAY_KERNEL_MAX : strideLimit;

    // Now loop, handling this batchSize amount of Tensors in each kernel call. We need
    // to prepare the kernel param for each batch.
    offset = 0;
    for (i = 0; i < numInputs; i += batchSize) {
      CatArrayKernelParam2<real> param;
      param.dims = ndim;
      cohortMax = 0;
      for (j = 0; j < batchSize && (i+j) < numInputs; ++j) {
        // Copy over data pointer
        param.data[j] = THCTensor_(data)(state, inputs[i+j]);

        // Initialize strides for this Tensor, offset by the number of Tensors
        // prior to this one * the number of dimensions
        for (k = 0; k < ndim; ++k) {
          param.strides[(j*ndim) + k] = THCTensor_(stride)(state, inputs[i+j], k);
        }

        param.offsets[j] = offset;
        param.dimSizes[j] = THCTensor_(size)(state, inputs[i+j], dimension);
        offset += param.dimSizes[j];
        param.nElements[j] = THCTensor_(nElement)(state, inputs[i+j]);
        cohortMax = cohortMax > param.nElements[j] ? cohortMax : param.nElements[j];
      }

      param.count = j;
      param.debugString();

      // Next, let's consider how we set our kernel launch parameters.
      // We borrow from THCApply, which the kernel's internal indexing
      // is based on.
      dim3 applyBlock = getApplyBlock();

      // We also re-use the applyGrid - but note that we use the maximum number of
      // elements for a given tensor in this grouping to determine the count
      dim3 applyGrid;
      getApplyGrid(state, cohortMax, applyGrid);

      // Next, we set our grid's y component to be the number of tensors in
      // the batch. This will allow the kernel to determine which input
      // tensor it is responsible for copying
      applyGrid.y = j;

      // Actually launch the kernel 
      /* catArrayBatchedCopy2<real><<<applyGrid, applyBlock, 0, THCState_getCurrentStream(state)>>>( */
      /*   rst, param, dimension); */
      catArrayBatchedCopy2<real><<<dim3(1, j), 1, 0, THCState_getCurrentStream(state)>>>(
        rst, param, dimension);
      THCudaCheck(cudaGetLastError());
  }

    // The basic strategy is as follows: We batch copies of the remaining input
    // tensors by passing a struct to the kernel containing TensorInfos for a fixed
    // number of inputs, along with information to handle the narrow, copy within the kernel call
    /* index = 0; */
    /* for (i = 0; i < numInputs; i += CAT_ARRAY_KERNEL_BATCH_SIZE) { */
    /*   CatArrayKernelParam<real> param; */
    /*   cohortMax = 0; */
    /*   for (j = 0; j < CAT_ARRAY_KERNEL_BATCH_SIZE && (i+j) < numInputs; ++j) { */
    /*     param.inputs[j] = getTensorInfo<THCTensor, unsigned int>(state, inputs[i+j]); */
    /*     param.offsets[j] = index; */
    /*     param.dimSizes[j] = dimension < THCTensor_(nDimension)(state, inputs[i+j]) */
    /*            ? THCTensor_(size)(state, inputs[i+j], dimension) */
    /*            : 1; */
    /*     index += param.dimSizes[j]; */
    /*     param.nElements[j] = THCTensor_(nElement)(state, inputs[i+j]); */
    /*     cohortMax = cohortMax > param.nElements[j] ? cohortMax : param.nElements[j]; */
    /*   } */
    /*   param.count = j; */

    /*   // Next, let's consider how we set our kernel launch parameters. */
    /*   // We borrow from THCApply, which the kernel's internal indexing */
    /*   // is based on. */
    /*   dim3 applyBlock = getApplyBlock(); */

    /*   // We also re-use the applyGrid - but note that we use the maximum number of */
    /*   // elements for a given tensor in this grouping to determine the count */
    /*   dim3 applyGrid; */
    /*   getApplyGrid(state, cohortMax, applyGrid); */

    /*   // Next, we set our grid's y component to be the number of tensors in */
    /*   // the batch. This will allow the kernel to determine which input */
    /*   // tensor it is responsible for copying */
    /*   applyGrid.y = j; */

    /*   // Actually launch the kernel */
    /*   catArrayBatchedCopy<real><<<applyGrid, applyBlock, 0, THCState_getCurrentStream(state)>>>(rst, param, dimension); */
    /* } */
  } else {
    offset = 0;
    for (j = 0; j < numInputs; j++)
    {
      long dimSize = dimension < THCTensor_(nDimension)(state, inputs[j])
               ? THCTensor_(size)(state, inputs[j], dimension)
               : 1;
      THCTensor *nt = THCTensor_(newWithTensor)(state, result);
      THCTensor_(narrow)(state, nt, NULL, dimension, offset, dimSize);
      THCTensor_(copy)(state, nt, inputs[j]);
      THCTensor_(free)(state, nt);
      offset += dimSize;
    }
  }
}

void THCTensor_(nonzero)(THCState* state, THCudaLongTensor *tensor,
                          THCTensor *self)
{
  THAssert(THCTensor_(checkGPU)(state, 1, self  ));
  THAssert(THCudaLongTensor_checkGPU(state, 1, tensor));

  using namespace thrust::placeholders;

  self = THCTensor_(newContiguous)(state, self);
  thrust::device_ptr<real> self_data(THCTensor_(data)(state, self));

  int num_dim = THCTensor_(nDimension)(state, self);
  long N = THCTensor_(nElement)(state, self);

  THCudaLongTensor_resize2d(state, tensor, N, num_dim);
  tensor = THCudaLongTensor_newContiguous(state, tensor);
  thrust::device_ptr<long> tensor_data(THCudaLongTensor_data(state, tensor));

  thrust::counting_iterator<long> idxfirst(0);
  thrust::counting_iterator<long> idxlast = idxfirst + N;

  typedef thrust::device_ptr<long> Iter;
  strided_range<Iter> strided_tensor(tensor_data,
                                     tensor_data+N*num_dim, num_dim);

#if CUDA_VERSION >= 7000
  cudaStream_t stream = THCState_getCurrentStream(state);
#endif

  strided_range<Iter>::iterator dend = thrust::copy_if(
#if CUDA_VERSION >= 7000
    thrust::cuda::par.on(stream),
#endif
    idxfirst,
    idxlast,
    self_data,
    strided_tensor.begin(),
    NonZeroOp<real>()
  );

  long num_nonzeros = thrust::distance(strided_tensor.begin(), dend);

  long div = 1;
  for (int dim = num_dim-1; dim >= 0; dim--) {
    strided_range<Iter> stride_dim(tensor_data+dim,
                                   tensor_data+N*num_dim, num_dim);
    thrust::transform(
#if CUDA_VERSION >= 7000
      thrust::cuda::par.on(stream),
#endif
      strided_tensor.begin(),
      strided_tensor.end(),
      stride_dim.begin(),
      idx_functor(div, self->size[dim])
    );
    div *= self->size[dim];
  }

  THCudaLongTensor_resize2d(state, tensor, num_nonzeros, num_dim);

  THCTensor_(free)(state, self);
  THCudaLongTensor_free(state, tensor);

  THCudaCheck(cudaGetLastError());
}

void THCTensor_(diag)(THCState *state, THCTensor *self_, THCTensor *src_, long k){
  THAssert(THCTensor_(checkGPU)(state, 2, self_, src_));
  int nDimension = THCTensor_(nDimension)(state, src_);
  THArgCheck((nDimension == 2) || (nDimension == 1), 1, "expected a matrix or a vector");
  if (nDimension == 2) {
    long stride0 = THCTensor_(stride)(state, src_, 0);
    long stride1 = THCTensor_(stride)(state, src_, 1);
    long size0 = THCTensor_(size)(state, src_, 0);
    long size1 = THCTensor_(size)(state, src_, 1);
    long size = (k > 0) ? min((long long)size0, (long long)size1 - k) : min((long long)size0 + k, (long long)size1);
    THCTensor_(resize1d)(state, self_, size);
    long strideSelf = THCTensor_(stride)(state, self_, 0);
    const dim3 threads(min((long long)THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock, (long long)size));
    dim3 grid(min((long long)1024, (long long)THCCeilDiv(size, (long)threads.x)));
    long start = (k >= 0 ? k * stride1 : -k * stride0);
    THCTensor_copyFromDiagonal<real><<<grid, threads, 0, THCState_getCurrentStream(state)>>>
    (THCTensor_(data)(state, self_), THCTensor_(data)(state, src_), start, size, stride0 + stride1, strideSelf);
  } else {
    ptrdiff_t totalElements = THCTensor_(nElement)(state, src_);
    ptrdiff_t size = (k > 0) ? totalElements + k : totalElements - k;
    long strideSrc = THCTensor_(stride)(state, src_, 0);
    THCTensor_(resize2d)(state, self_, size, size);
    THCTensor_(zero)(state, self_);
    long stride0 = THCTensor_(stride)(state, self_, 0);
    long stride1 = THCTensor_(stride)(state, self_, 1);
    const dim3 threads(min((long long)THCState_getCurrentDeviceProperties(state)->maxThreadsPerBlock, (long long)size));
    dim3 grid(min((long long)1024, (long long)THCCeilDiv(size, (ptrdiff_t)threads.x)));
    ptrdiff_t start = (k >= 0 ? k * stride1 : -k * stride0);
    THCTensor_copyToDiagonal<real><<<grid, threads, 0, THCState_getCurrentStream(state)>>>
    (THCTensor_(data)(state, self_), THCTensor_(data)(state, src_), start, totalElements, stride0 + stride1, strideSrc);
  }
  THCudaCheck(cudaGetLastError());
}

accreal THCTensor_(trace)(THCState *state, THCTensor *src_) {
  THAssert(THCTensor_(checkGPU)(state, 1, src_));
  THArgCheck((src_->nDimension == 2), 1, "expected a matrix");
  THCTensor *diag = THCTensor_(new)(state);
  THCTensor_(diag)(state, diag, src_, 0);
  accreal trace = THCTensor_(sumall)(state, diag);
  THCTensor_(free)(state, diag);
  return trace;
}

#endif
