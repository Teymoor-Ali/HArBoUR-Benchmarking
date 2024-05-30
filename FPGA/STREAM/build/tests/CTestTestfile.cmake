# CMake generated Testfile for 
# Source directory: /mnt/2UB102/HPCC_FPGA-master/STREAM/tests
# Build directory: /mnt/2UB102/HPCC_FPGA-master/STREAM/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_unit_stream_kernels_single_emulate_xilinx "/mnt/2UB102/HPCC_FPGA-master/STREAM/build/bin/STREAM_FPGA_test_xilinx" "-f" "stream_kernels_single_emulate.xclbin")
set_tests_properties(test_unit_stream_kernels_single_emulate_xilinx PROPERTIES  WORKING_DIRECTORY "/mnt/2UB102/HPCC_FPGA-master/STREAM/build/bin" _BACKTRACE_TRIPLES "/mnt/2UB102/HPCC_FPGA-master/cmake/unitTestTargets.cmake;32;add_test;/mnt/2UB102/HPCC_FPGA-master/cmake/unitTestTargets.cmake;0;;/mnt/2UB102/HPCC_FPGA-master/STREAM/tests/CMakeLists.txt;7;include;/mnt/2UB102/HPCC_FPGA-master/STREAM/tests/CMakeLists.txt;0;")
