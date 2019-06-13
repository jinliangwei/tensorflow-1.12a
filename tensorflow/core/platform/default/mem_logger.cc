#include "tensorflow/core/platform/default/mem_logger.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/stacktrace.h"
#include <unistd.h>
#include <errno.h>
#include <string.h>

namespace tensorflow {
namespace internal {

const char* MemLogger::kAllocTypeUnknown = "Unknown";
const char* MemLogger::kAllocTypePersistent = "Persistent";
const char* MemLogger::kAllocTypeTemporary = "Temporary";
const char* MemLogger::kAllocTypeOutput = "Output";
const char* MemLogger::kAllocTypeCopy = "Copy";
const char* MemLogger::kAllocTypeRecvCopy = "RecvCopy";
const char* MemLogger::kAllocTypeRecvCopySrc = "RecvCopySrc";
const char* MemLogger::kAllocTypeInitScratchBuffers = "InitScratchBuffers";

thread_local std::string MemLogger::op_name_ = "unknown";
thread_local std::string MemLogger::op_type_ = "unknown";
thread_local MemLogger::AllocType MemLogger::alloc_type_ = MemLogger::AllocType::kUnknown;

MemLogger::MemLogger(const std::string &path_prefix,
                     bool log_to_stderr,
                     const std::string &allocator_name):kLogToStderr(log_to_stderr) {
  std::string log_file_name = std::string("memory_log.") + allocator_name;
  char hostname[100];
  gethostname(hostname, 100);

  static EnvTime* env_time = tensorflow::EnvTime::Default();
  uint64 now_micros = env_time->NowMicros();
  time_t now_seconds = static_cast<time_t>(now_micros / 1000000);
  int32 micros_remainder = static_cast<int32>(now_micros % 1000000);
  const size_t time_buffer_size = 30;
  char time_buffer[time_buffer_size];
  strftime(time_buffer, time_buffer_size, ".%Y%m%d-%H%M%S.",
           localtime(&now_seconds));
  VLOG(3) << __func__ << " time_buffer = " << time_buffer;
  log_file_name += "." + std::string(hostname) + std::string(time_buffer) + std::to_string(micros_remainder);
  log_file_path_ = path_prefix + "/" + log_file_name;

  VLOG(3) << __func__ << " log_file_path = " << log_file_path_;
  FILE *log_file = fopen(log_file_path_.c_str(), "w");
  CHECK(log_file != nullptr) << strerror(errno);
  fclose(log_file);
}

void MemLogger::LogSessionRunStart(int64_t time_stamp, int64_t step_id) const {
  FILE *log_file = fopen(log_file_path_.c_str(), "a");
  CHECK(log_file != nullptr) << strerror(errno);
  fprintf(log_file, "[%ld] SessionRun Start counter:%d\n",
          time_stamp, step_id);
  fclose(log_file);
}

void MemLogger::LogSessionRunEnd(int64_t time_stamp, int64_t step_id) const {
  FILE *log_file = fopen(log_file_path_.c_str(), "a");
  CHECK(log_file != nullptr) << strerror(errno);
  fprintf(log_file, "[%ld] SessionRun End counter:%d\n",
          time_stamp, step_id);
  fclose(log_file);
}

void MemLogger::LogAlloc(uintptr_t ptr,
                         size_t size, size_t allocator_bytes_in_use,
                         int64_t time_stamp) const {
  FILE *log_file = fopen(log_file_path_.c_str(), "a");
  CHECK(log_file != nullptr) << strerror(errno);
  fprintf(log_file, "[%ld] Allocate type:%s op_name:%s op_type:%s ptr:%ld bytes:%ld bytes_in_use:%ld\n",
          time_stamp, AllocTypeToString(alloc_type_), op_name_.c_str(), op_type_.c_str(), ptr, size, allocator_bytes_in_use);
  fclose(log_file);
  if (kLogToStderr) {
    char buff[400];
    sprintf(buff, "[%ld] Allocate type:%s op_name:%s op_type:%s ptr:%ld bytes:%ld bytes_in_use:%ld\n",
          time_stamp, AllocTypeToString(alloc_type_), op_name_.c_str(), op_type_.c_str(), ptr, size, allocator_bytes_in_use);
    LOG(INFO) << buff << " allocator = " << (void*) this;
  }
}

void MemLogger::LogDealloc(uintptr_t ptr, size_t size,
                           size_t allocator_bytes_in_use, int64_t time_stamp) const {
  FILE *log_file = fopen(log_file_path_.c_str(), "a");
  CHECK(log_file != nullptr) << strerror(errno);
  fprintf(log_file, "[%ld] Deallocate ptr:%ld bytes:%ld bytes_in_use:%ld\n",
          time_stamp, ptr, size, allocator_bytes_in_use);
  fclose(log_file);
  if (kLogToStderr) {
    char buff[100];
    sprintf(buff, "[%ld] Deallocate ptr:%ld bytes:%ld bytes_in_use:%ld\n",
            time_stamp, ptr, size, allocator_bytes_in_use);
    LOG(INFO) << buff;
  }
}

void MemLogger::SetAllocationInfo(AllocType alloc_type) {
  VLOG(3) << __func__ << " alloc_type = " << static_cast<int>(alloc_type);
  alloc_type_ = alloc_type;
}

void MemLogger::ResetAllocationInfo() {
  VLOG(3) << __func__;
  alloc_type_ = AllocType::kUnknown;
}

const char *MemLogger::AllocTypeToString(AllocType alloc_type) {
  switch (alloc_type) {
    case AllocType::kUnknown:
      return kAllocTypeUnknown;
    case AllocType::kPersistent:
      return kAllocTypePersistent;
    case AllocType::kTemporary:
      return kAllocTypeTemporary;
    case AllocType::kOutput:
      return kAllocTypeOutput;
    case AllocType::kCopy:
      return kAllocTypeCopy;
    case AllocType::kRecvCopy:
      return kAllocTypeRecvCopy;
    case AllocType::kRecvCopySrc:
      return kAllocTypeRecvCopySrc;
    case AllocType::kInitScratchBuffers:
      return kAllocTypeInitScratchBuffers;
    default:
      LOG(FATAL) << __func__ << " unknown AllocType = " << static_cast<int>(alloc_type);
  }
  return nullptr;
}

void MemLogger::SetOperationInfo(const std::string &op_name,
                                 const std::string &op_type) {
  VLOG(3) << __func__ << " op_name = " << op_name
            << " op_type = " << op_type;
  op_name_ = op_name;
  op_type_ = op_type;
}

void MemLogger::ResetOperationInfo() {
  VLOG(3) << __func__;
  op_name_ = "unknown";
  op_type_ = "unknown";
}

}
}
