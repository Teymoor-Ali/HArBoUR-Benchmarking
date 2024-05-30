# CMake generated Testfile for 
# Source directory: /mnt/2UB102/HPCC_FPGA-master/STREAM/src/host
# Build directory: /mnt/2UB102/HPCC_FPGA-master/STREAM/build/src/host
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_xilinx_host_executable "/mnt/2UB102/HPCC_FPGA-master/STREAM/build/bin/STREAM_FPGA_xilinx" "-h")
set_tests_properties(test_xilinx_host_executable PROPERTIES  _BACKTRACE_TRIPLES "/mnt/2UB102/HPCC_FPGA-master/STREAM/src/host/CMakeLists.txt;30;add_test;/mnt/2UB102/HPCC_FPGA-master/STREAM/src/host/CMakeLists.txt;0;")
subdirs("../../lib/hpccbase")
