# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

if(EXISTS "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/src/rapidxml-populate-stamp/rapidxml-populate-gitclone-lastrun.txt" AND EXISTS "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/src/rapidxml-populate-stamp/rapidxml-populate-gitinfo.txt" AND
  "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/src/rapidxml-populate-stamp/rapidxml-populate-gitclone-lastrun.txt" IS_NEWER_THAN "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/src/rapidxml-populate-stamp/rapidxml-populate-gitinfo.txt")
  message(VERBOSE
    "Avoiding repeated git clone, stamp file is up to date: "
    "'/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/src/rapidxml-populate-stamp/rapidxml-populate-gitclone-lastrun.txt'"
  )
  return()
endif()

# Even at VERBOSE level, we don't want to see the commands executed, but
# enabling them to be shown for DEBUG may be useful to help diagnose problems.
cmake_language(GET_MESSAGE_LOG_LEVEL active_log_level)
if(active_log_level MATCHES "DEBUG|TRACE")
  set(maybe_show_command COMMAND_ECHO STDOUT)
else()
  set(maybe_show_command "")
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/rapidxml"
  RESULT_VARIABLE error_code
  ${maybe_show_command}
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/rapidxml'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"
            clone --no-checkout --config "advice.detachedHead=false" "https://gitlab.inria.fr/sibr/libs/rapidxml.git" "rapidxml"
    WORKING_DIRECTORY "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml"
    RESULT_VARIABLE error_code
    ${maybe_show_command}
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(NOTICE "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://gitlab.inria.fr/sibr/libs/rapidxml.git'")
endif()

execute_process(
  COMMAND "/usr/bin/git"
          checkout "069e87f5ec5ce1745253bd64d89644d6b894e516" --
  WORKING_DIRECTORY "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/rapidxml"
  RESULT_VARIABLE error_code
  ${maybe_show_command}
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: '069e87f5ec5ce1745253bd64d89644d6b894e516'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git" 
            submodule update --recursive --init 
    WORKING_DIRECTORY "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/rapidxml"
    RESULT_VARIABLE error_code
    ${maybe_show_command}
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/rapidxml'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/src/rapidxml-populate-stamp/rapidxml-populate-gitinfo.txt" "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/src/rapidxml-populate-stamp/rapidxml-populate-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  ${maybe_show_command}
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/src/rapidxml-populate-stamp/rapidxml-populate-gitclone-lastrun.txt'")
endif()
