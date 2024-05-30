# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.22.1)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(extern_googletest-populate NONE)



include(ExternalProject)
ExternalProject_Add(extern_googletest-populate
                     "UPDATE_DISCONNECTED" "False" "URL" "https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz" "URL_HASH" "SHA256=b4870bf121ff7795ba20d20bcdd8627b8e088f2d1dab299a031c1034eddc93d5"
                    SOURCE_DIR          "/mnt/2UB102/HPCC_FPGA-master/STREAM/build/_deps/extern_googletest-src"
                    BINARY_DIR          "/mnt/2UB102/HPCC_FPGA-master/STREAM/build/_deps/extern_googletest-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)


