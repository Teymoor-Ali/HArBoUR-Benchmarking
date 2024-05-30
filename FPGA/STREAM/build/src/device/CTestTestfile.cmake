# CMake generated Testfile for 
# Source directory: /mnt/2UB102/HPCC_FPGA-master/STREAM/src/device
# Build directory: /mnt/2UB102/HPCC_FPGA-master/STREAM/build/src/device
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_single_emulation_xilinx "/mnt/2UB102/HPCC_FPGA-master/STREAM/build/bin/STREAM_FPGA_xilinx" "-f" "stream_kernels_single_emulate.xclbin" "-n" "1" "-s" "65536")
set_tests_properties(test_single_emulation_xilinx PROPERTIES  WORKING_DIRECTORY "/mnt/2UB102/HPCC_FPGA-master/STREAM/build/bin" _BACKTRACE_TRIPLES "/mnt/2UB102/HPCC_FPGA-master/STREAM/src/device/CMakeLists.txt;22;add_test;/mnt/2UB102/HPCC_FPGA-master/STREAM/src/device/CMakeLists.txt;0;")
add_test(test_output_parsing_xilinx "/mnt/2UB102/HPCC_FPGA-master/STREAM/../scripts/evaluation/execute_and_parse.sh" "./STREAM_FPGA_xilinx" "-s" "65536" "-f" "stream_kernels_single_emulate.xclbin" "-n" "1")
set_tests_properties(test_output_parsing_xilinx PROPERTIES  WORKING_DIRECTORY "/mnt/2UB102/HPCC_FPGA-master/STREAM/build/bin" _BACKTRACE_TRIPLES "/mnt/2UB102/HPCC_FPGA-master/STREAM/src/device/CMakeLists.txt;24;add_test;/mnt/2UB102/HPCC_FPGA-master/STREAM/src/device/CMakeLists.txt;0;")
add_test(test_emulation_mpi_xilinx "mpirun" "-n" "2" "./STREAM_FPGA_xilinx" "-f" "stream_kernels_single_emulate.xclbin" "-n" "1" "-s" "65536")
set_tests_properties(test_emulation_mpi_xilinx PROPERTIES  WORKING_DIRECTORY "/mnt/2UB102/HPCC_FPGA-master/STREAM/build/bin" _BACKTRACE_TRIPLES "/mnt/2UB102/HPCC_FPGA-master/STREAM/src/device/CMakeLists.txt;27;add_test;/mnt/2UB102/HPCC_FPGA-master/STREAM/src/device/CMakeLists.txt;0;")
