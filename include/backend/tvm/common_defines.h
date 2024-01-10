#ifndef _H_SIMPLE_AI_BACKEND_TVM_COMMON_DEFINES_H_
#define _H_SIMPLE_AI_BACKEND_TVM_COMMON_DEFINES_H_

#include "dlpack/dlpack.h"
#include "tvm/runtime/module.h"
#include "tvm/runtime/vm/vm.h"
#include "tvm/runtime/registry.h"
#include "tvm/runtime/packed_func.h"

namespace simple_ai {
namespace backend {
namespace tvm {

using TVMModule = ::tvm::runtime::Module;

constexpr const char* kDefaultExecutorType = "vm";
constexpr const char* kVMExecutorType = "vm";
constexpr const char* kGraphExecutorType = "graph";

constexpr const char* kDefaultTarget = "llvm";
constexpr const char* kLLVMTarget = "llvm";

constexpr const char* kCPUTarget = "cpu";
constexpr const char* kGPUTarget = "gpu";

constexpr const char* kDefaultTuningType = "AutoTVM";
constexpr const char* kAutotvmTuningType = "AutoTVM";
constexpr const char* kAnsorTuningType = "Ansor";

constexpr const unsigned int kDefaultOptLevel = 3;

}    // namespace tvm
}    // namespace backend
}    // namespace simple_ai

#endif