# CMake generated Testfile for 
# Source directory: /mnt/2UB102/HPCC_FPGA-master/STREAM/build/_deps/extern_cxxopts-src/test
# Build directory: /mnt/2UB102/HPCC_FPGA-master/STREAM/build/_deps/extern_cxxopts-build/test
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(options "options_test")
set_tests_properties(options PROPERTIES  _BACKTRACE_TRIPLES "/mnt/2UB102/HPCC_FPGA-master/STREAM/build/_deps/extern_cxxopts-src/test/CMakeLists.txt;4;add_test;/mnt/2UB102/HPCC_FPGA-master/STREAM/build/_deps/extern_cxxopts-src/test/CMakeLists.txt;0;")
add_test(find-package-test "/usr/bin/ctest" "-C" "--build-and-test" "/mnt/2UB102/HPCC_FPGA-master/STREAM/build/_deps/extern_cxxopts-src/test/find-package-test" "/mnt/2UB102/HPCC_FPGA-master/STREAM/build/_deps/extern_cxxopts-build/test/find-package-test" "--build-generator" "Unix Makefiles" "--build-makeprogram" "/usr/bin/gmake" "--build-options" "-DCMAKE_CXX_COMPILER=/usr/bin/c++" "-DCMAKE_BUILD_TYPE=" "-Dcxxopts_DIR=/mnt/2UB102/HPCC_FPGA-master/STREAM/build/_deps/extern_cxxopts-build")
set_tests_properties(find-package-test PROPERTIES  _BACKTRACE_TRIPLES "/mnt/2UB102/HPCC_FPGA-master/STREAM/build/_deps/extern_cxxopts-src/test/CMakeLists.txt;7;add_test;/mnt/2UB102/HPCC_FPGA-master/STREAM/build/_deps/extern_cxxopts-src/test/CMakeLists.txt;0;")
add_test(add-subdirectory-test "/usr/bin/ctest" "-C" "--build-and-test" "/mnt/2UB102/HPCC_FPGA-master/STREAM/build/_deps/extern_cxxopts-src/test/add-subdirectory-test" "/mnt/2UB102/HPCC_FPGA-master/STREAM/build/_deps/extern_cxxopts-build/test/add-subdirectory-test" "--build-generator" "Unix Makefiles" "--build-makeprogram" "/usr/bin/gmake" "--build-options" "-DCMAKE_CXX_COMPILER=/usr/bin/c++" "-DCMAKE_BUILD_TYPE=")
set_tests_properties(add-subdirectory-test PROPERTIES  _BACKTRACE_TRIPLES "/mnt/2UB102/HPCC_FPGA-master/STREAM/build/_deps/extern_cxxopts-src/test/CMakeLists.txt;21;add_test;/mnt/2UB102/HPCC_FPGA-master/STREAM/build/_deps/extern_cxxopts-src/test/CMakeLists.txt;0;")
