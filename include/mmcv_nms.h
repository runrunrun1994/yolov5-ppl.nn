// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef __ST_PPL_KERNEL_X86_FP32_MMCV_NMS_H_
#define __ST_PPL_KERNEL_X86_FP32_MMCV_NMS_H_
#include "ppl/common/retcode.h"


ppl::common::RetCode mmcv_nms_ndarray_fp32(
        const float *boxes,
        const float *scores,
        const uint32_t num_boxes_in,
        const float iou_threshold,
        const int64_t offset,
        int64_t *dst,
        int64_t *num_boxes_out);

#endif //! __ST_PPL_KERNEL_X86_FP32_MMCV_NMS_H_
