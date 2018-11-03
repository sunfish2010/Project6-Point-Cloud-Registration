set(SOURCE_FILES
    "utilityCore.hpp"
    "utilityCore.cpp"
    "glslUtility.hpp"
    "glslUtility.cpp"
        )

cuda_add_library(util
   ${SOURCE_FILES}
   OPTIONS -arch=sm_52
   )
