project(PRETUS-VITAL-LUNG-Plugins)
cmake_minimum_required(VERSION 3.16)
## #################################################################
## Qt required for signals and slots (even if no GUI is required)
## #################################################################

find_package(Qt5 REQUIRED Gui Core Widgets Concurrent Xml)

## #################################################################
## CMAKE generic options / settings
## #################################################################
set(CMAKE_CXX_STANDARD 11)
option(BUILD_SHARED_LIBS "Build shared libraries, required for the plug-ins" ON)
if (BUILD_SHARED_LIBS)
    set(PROJECT_BUILD_TYPE SHARED)
else()
    set(PROJECT_BUILD_TYPE STATIC)
endif()
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(PLUGIN_SCRIPT_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}/Plugins" )
if(WIN32)
    set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/bin)
else()
    set(LIBRARY_OUTPUT_PATH  ${CMAKE_BINARY_DIR}/lib)
endif(WIN32)
if (APPLE)
    set(GUI_EXECUTABLE MACOSX_BUNDLE)
endif()

file(WRITE ${PROJECT_BINARY_DIR}/VERSION "${PROJECT_VERSION}")

if (UNIX AND NOT APPLE)
find_package(Qt5 REQUIRED X11Extras PrintSupport)
endif()
ADD_DEFINITIONS(-DQT_NO_KEYWORDS) # needed because python has something called "slots" and may class with Qt definitions

file(GLOB SOURCES "*.h" "*.cxx" "*.md")

# find and include the plug-in library
find_library(PLUGIN_LIBRARY
    NAMES plugin
    HINTS "${CMAKE_PREFIX_PATH}/"
)

find_path(PLUGIN_INCLUDE_DIR NAMES Plugin.h)
include_directories(${PLUGIN_INCLUDE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}) # need this for the generated files

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

option(BUILD_PLUGIN_LUSCLASSP "Build plugin to classify 5 artefacts in LUS, pytorch implementation" OFF)
if (BUILD_PLUGIN_LUSCLASSP)
  add_subdirectory(Plugin_lusclassificationp)
endif()


