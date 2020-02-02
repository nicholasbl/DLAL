TARGET = linalgtest

TEMPLATE = app

CONFIG += c++2a

CONFIG -= app_bundle qt

macx {
    INCLUDEPATH += /usr/local/include

    QMAKE_CXXFLAGS += -fsanitize=address
    QMAKE_LFLAGS += -fsanitize=address
}

SOURCES += \
    main.cpp \
    matrix.cpp \
    matrix_operations.cpp \
    matrix_transform.cpp \
    quat.cpp \
    tmatrix.cpp \
    vector.cpp

HEADERS += \
    doctest.h \
    include/linear_algebra.h \
    include/mat.h \
    include/mat_detail.h \
    include/mat_operations.h \
    include/mat_transforms.h \
    include/packed_mat.h \
    include/packed_vec.h \
    include/quat.h \
    include/tmatrix.h \
    include/vec.h \
    include/vec_trig.h \
    test_common.h
