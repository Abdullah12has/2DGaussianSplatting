# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/rapidxml")
  file(MAKE_DIRECTORY "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/rapidxml")
endif()
file(MAKE_DIRECTORY
  "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/build"
  "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix"
  "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/tmp"
  "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/src/rapidxml-populate-stamp"
  "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/src"
  "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/src/rapidxml-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/src/rapidxml-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/cluster/51/koubaa/mahdi/2DGaussianSplatting/Gaussian-Splatting-Monitor/SIBR_viewers/extlibs/rapidxml/subbuild/rapidxml-populate-prefix/src/rapidxml-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
