set(SOURCE_FILES
    "registration.cu"
    "registration.h"
    "registrationTools.h"
    )

cuda_add_library(src
    ${SOURCE_FILES}
    OPTIONS -arch=sm_20
    )
