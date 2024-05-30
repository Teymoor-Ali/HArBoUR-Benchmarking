# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.22.1)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(extern_cxxopts-populate NONE)



include(ExternalProject)
ExternalProject_Add(extern_cxxopts-populate
                     "UPDATE_DISCONNECTED" "False" "URL" "https://github.com/jarro2783/cxxopts/archive/v2.2.1.tar.gz" "URL_HASH" "SHA256=984aa3c8917d649b14d7f6277104ce38dd142ce378a9198ec926f03302399681"
                    SOURCE_DIR          "/mnt/2UB102/HPCC_FPGA-master/STREAM/build/_deps/extern_cxxopts-src"
                    BINARY_DIR          "/mnt/2UB102/HPCC_FPGA-master/STREAM/build/_deps/extern_cxxopts-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)


