#ifndef TENSORFLOW_PLATFORM_DEFAULT_MEM_LOGGER_H_
#define TENSORFLOW_PLATFORM_DEFAULT_MEM_LOGGER_H_
#include <string>
#include <cstdint>
namespace tensorflow {
namespace internal {
class MemLogger {
public:
  enum class AllocType {
    kUnknown = 0,
 kPersistent = 1,
  kTemporary = 2,
     kOutput = 3,
       kCopy = 4,
   kRecvCopy = 5,
   kRecvCopySrc = 6,
kInitScratchBuffers = 7
  };

  MemLogger(const std::string &path_prefix, bool log_to_stderr,
            const std::string &allocator_name);
  virtual ~MemLogger() { }
  void LogSessionRunStart(int64_t time_stamp, int64_t step_id) const;
  void LogSessionRunEnd(int64_t time_stamp, int64_t step_id) const;
  void LogAlloc(uintptr_t ptr, size_t size, size_t allocator_bytes_in_use,
                int64_t time_stamp) const;
  void LogDealloc(uintptr_t ptr, size_t size, size_t allocator_bytes_in_use, int64_t time_stamp) const;
  void SetOperationInfo(const std::string &op_name, const std::string &op_type);
  void ResetOperationInfo();
  void SetAllocationInfo(AllocType alloc_type);
  void ResetAllocationInfo();

  static const char *AllocTypeToString(AllocType alloc_type);

private:
  const bool kLogToStderr;
  std::string log_file_path_;
  static thread_local std::string op_name_;
  static thread_local std::string op_type_;
  static thread_local AllocType alloc_type_;
  static const char *kAllocTypeUnknown;
  static const char *kAllocTypePersistent;
  static const char *kAllocTypeTemporary;
  static const char *kAllocTypeOutput;
  static const char *kAllocTypeCopy;
  static const char *kAllocTypeRecvCopy;
  static const char *kAllocTypeRecvCopySrc;
  static const char *kAllocTypeInitScratchBuffers;
};
}
}
#endif
